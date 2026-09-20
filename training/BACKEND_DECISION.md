# Adapter backend decision: AMD Radeon RX 9070

Status: **proposed; local non-training dry-run passed; training not authorized**
Reviewed: 2026-09-20

## Decision

Propose Unsloth QLoRA for `gpt-oss-20b` in a dedicated Ubuntu 24.04 WSL2 distribution whose virtual disk lives under `E:\LLM_HOME\wsl\LocalPilot-Training`. Select Python 3.12, AMD's published multi-architecture ROCm 7.14.1 wheels for `gfx1201`, PyTorch 2.12, bitsandbytes 0.50.2, and Unsloth 2026.9.7. Start with rank 8, 1,040-token sequences, micro-batch 1, gradient accumulation 4, BF16 compute, Unsloth checkpointing, and requested embedding offload. The 1,040-token limit is the smallest 16-token-aligned window above the exact 1,036-token maximum of the fully rendered corpus, including native tool schemas; no example is truncated. This exact combination passed the non-training dry-run on this machine on 2026-09-20. Model loading and forward/backward peak memory remain deliberately unmeasured until the separately authorized, watched first batch.

This is a plausible first adapter path with a narrow memory margin. The RX 9070 is a 16 GB-class RDNA4 card and the machine has 31.93 GiB system RAM. Unsloth reports a 14 GB VRAM QLoRA requirement, compared with 44 GB for LoRA over unquantized BF16 base weights. Those are upstream figures, not an RX 9070 measurement. A local dry-run must establish package, GPU, data, model-cache, tokenizer, and adapter readiness. It cannot prove peak training memory without a later, separately authorized training smoke run.

OpenAI describes `gpt-oss-20b` as a 21B-total/3.6B-active MoE model, natively MXFP4, Apache 2.0, and intended for local use. The model can run in about 16 GB, but inference memory is not a training-memory guarantee. [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-oss-20b), [OpenAI release details](https://openai.com/index/introducing-gpt-oss/)

AMD publishes ROCm 7.14.1 / PyTorch 2.12 wheels for `gfx1201`, the RX 9070 architecture. This is a selected documented stack, not a promise to follow the newest release. The WSL matrix lists RX 9070 and Ubuntu 24.04; ROCDXG 1.2.1 supports ROCm 7.14 and retains the RX 9070 support of its predecessor. The ROCDXG repository has since moved into `rocm-systems`, but the pinned 1.2.1 release remains available. [AMD PyTorch install](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html), [AMD WSL matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html), [ROCDXG support](https://github.com/ROCm/librocdxg)

Unsloth supports AMD across Windows, WSL, and Linux and publishes an AMD `gpt-oss-20b` fine-tuning notebook. Its `gpt-oss` guide explains the special NF4 conversion needed for memory-efficient training of the released MXFP4 model and documents export to merged HF/GGUF weights. [Unsloth `gpt-oss` guide](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune), [AMD fine-tuning notebook](https://github.com/unslothai/notebooks/blob/main/nb/AMD-gpt-oss-%2820B%29-Fine-tuning.ipynb), [Unsloth AMD installation](https://unsloth.ai/docs/get-started/install/amd)

Use the explicit prequantized training artifact `unsloth/gpt-oss-20b-unsloth-bnb-4bit` at `093fba6992ef5a7152481afec0bdfca1ac486998`, with `use_exact_model_name=True` and `fast_inference=False`. Unsloth otherwise remaps model names according to quantization settings, which can invalidate a revision pinned to a different repository. This is an Unsloth-derived NF4 training artifact of `openai/gpt-oss-20b`, not the local Ollama MXFP4 blob. Its configuration records NF4, double quantization, and BF16 compute. The proposed q/k/v/o/gate/up/down projection targets and rank 8 follow the AMD notebook; the actual loaded model must expose them. [Pinned training artifact configuration](https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit/blob/093fba6992ef5a7152481afec0bdfca1ac486998/config.json), [Unsloth loader](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py)

bitsandbytes publishes AMD ROCm builds for `gfx1201`; use 0.50.2. The released Unsloth AMD extra requires at least 0.50.0 because older AMD 4-bit decoding had correctness bugs. Unsloth 2026.9.7 admits PyTorch 2.12 (`torch>=2.4,<2.13`); the earlier 2026.9.2 proposal was rejected by pip because it capped PyTorch below 2.12. Pair 2026.9.7 with Unsloth Zoo 2026.9.6. Its dependencies cap Transformers at 5.5 and TRL at 0.24, so unpinned latest HF packages are inappropriate here. Proposed pins remain Transformers 4.56.2 and TRL 0.22.2 from the AMD notebook, PEFT 0.18.0, and datasets 4.3.0. Package resolution and an ignored local environment lock are still required. [bitsandbytes installation matrix](https://huggingface.co/docs/bitsandbytes/installation), [Unsloth 2026.9.7 package metadata](https://pypi.org/pypi/unsloth/2026.9.7/json), [Transformers PEFT integration](https://huggingface.co/docs/transformers/peft)

## Unsupported or rejected primary paths

- **CUDA/NVIDIA recipes:** not applicable to this AMD machine.
- **Plain BF16 LoRA/full fine-tuning of 20B:** exceeds the 16 GB GPU by a wide margin. Unsloth lists about 44 GB for BF16 LoRA; full tuning is higher.
- **Generic Transformers/PEFT QLoRA directly over the released MXFP4 weights:** rejected as the primary path because MXFP4 training needs the specialized conversion/backward handling described by Unsloth.
- **Native Windows as the first training environment:** AMD now ships Windows PyTorch and Unsloth has Windows AMD support, but the native AMD path is newer and AMD's historical matrix warns that only PyTorch—not the complete ROCm stack—is supported there. WSL2 keeps the Linux Triton/toolchain path and is the lower-risk first run. Native Windows remains a fallback experiment after WSL succeeds.
- **Direct Ollama Safetensors adapter import:** Ollama's documented Safetensors adapter architectures do not include `gptoss`, and its own guide recommends non-QLoRA adapters because quantization methods differ. Export a merged model from Unsloth, convert/export to GGUF, and create a separate Ollama model instead. Keep the exact base identity. [Ollama import guide](https://github.com/ollama/ollama/blob/main/docs/import.mdx)

## Resource and deployment expectations

- VRAM: declared minimum 14 GiB; 13.5 GiB in the config is an **unmeasured planning estimate**, not a measured peak. Leave headroom for the Windows desktop and stop if the free-memory check fails. A local GPU operation check does not measure the model's forward/backward peak.
- RAM: allocate 28 GB to WSL and require at least 27.0 GiB visible after guest overhead. On this machine the 28 GB WSL cap exposes about 27.4 GiB inside the guest; the readiness floor accounts for that virtualization overhead while preserving the intended roughly 4 GiB host reserve. Close memory-heavy applications first. Default WSL allocation is only half the host RAM. Embedding offload is requested, but actual placement must be inspected after loading; do not assume heavy CPU offload makes a larger model fit. [Microsoft WSL memory settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
- Storage: the dedicated distro places PyTorch, its environment, and the Hugging Face cache on `E:`. Put the merged checkout on `E:` too if its ignored `training/outputs/` and `training/reports/` should live there. A checkout on `C:` keeps those repo-relative outputs on `C:`. Allow at least 100 GiB free for the environment, model cache, and potential exports. A 20.9B BF16 export alone is roughly 42 GB before temporary copies; merge/export may require a larger-memory machine even when QLoRA fits locally.
- Deployment: save the adapter first; only after evaluation should a separately named merged GGUF be imported into Ollama. Never overwrite `gpt-oss:20b`.

## Current-machine finding

The existing default WSL distro remains untouched. The dedicated `LocalPilot-Training` distro is Ubuntu 24.04.4 with Python 3.12 and lives under `E:\LLM_HOME\wsl`. Its isolated virtual environment contains the exact ROCm 7.14.1, AMD PyTorch 2.12, Triton 3.8.0, Unsloth 2026.9.7, Unsloth Zoo 2026.9.6, and pinned Hugging Face stack recorded above. The pinned 12,525,742,660-byte model snapshot is present in the distro's E:-backed cache and all four shard hashes were verified after copying.

The offline project dry-run passed on 2026-09-20 with 14.34 GiB clean free VRAM on `gfx1201`, about 27.4 GiB visible system RAM, zero held-out collisions, and all 16,417 expanded examples rendered without truncation (maximum 1,036 tokens). Its ignored local report records `training_performed: false`. The tracked config deliberately remains `proposed`; no model weights were loaded for training and no optimizer step ran.

## Exact post-merge setup and dry-run commands

Run these in an elevated Windows PowerShell after confirming a compatible AMD Windows driver and current WSL are installed. The named distro is new; do not use these commands to replace an existing distro:

```powershell
New-Item -ItemType Directory -Force E:\LLM_HOME\wsl\LocalPilot-Training | Out-Null
wsl --install Ubuntu-24.04 --name LocalPilot-Training --location E:\LLM_HOME\wsl\LocalPilot-Training --version 2
```

Finish the Ubuntu first-run user setup. In **WSL Settings**, set the memory limit to 28 GB and swap to 8 GB under `E:\LLM_HOME\wsl\swap.vhdx`, preserving other existing settings. Alternatively, edit the existing `%UserProfile%\.wslconfig` to set these keys under its existing `[wsl2]` section (do not replace unrelated settings):

```ini
memory=28GB
swap=8GB
swapFile=E:\\LLM_HOME\\wsl\\swap.vhdx
```

These settings affect all WSL2 distributions. Save any work in running WSL sessions before `wsl --shutdown`, then restart the training distro. Swap is contingency space, not a substitute for training RAM.

Run the remaining commands from Windows PowerShell. Set `$localPilotRepo` to the merged checkout; the example keeps repo-relative outputs on `E:`. The setup downloads software and later the pinned model weights, but performs no training. The ROCm SDK prerequisite is explicitly installed into the E-backed virtual environment. [AMD ROCm 7.14.1 installation](https://rocm.docs.amd.com/en/docs-7.14.1/install/rocm.html), [ROCDXG 1.2.1 release](https://github.com/ROCm/librocdxg/releases/tag/v1.2.1)

```powershell
$localPilotRepo = 'E:\LLM_HOME\src\localpilot'
git clone https://github.com/n47h4ni3l/localpilot.git $localPilotRepo
# If that checkout already exists, skip clone; use its clean, merged main.
wsl --shutdown
wsl -d LocalPilot-Training -- bash -lc "sudo apt update && sudo apt install -y python3.12-venv python3-pip python3-wheel python3-argcomplete build-essential cmake git wget libnuma-dev"
wsl -d LocalPilot-Training -- bash -lc "python3.12 -m venv ~/.venvs/localpilot-training && ~/.venvs/localpilot-training/bin/python -m pip install --upgrade pip wheel"
wsl -d LocalPilot-Training -- bash -lc "~/.venvs/localpilot-training/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'rocm[libraries,device-gfx1201]==7.14.1'"
wsl -d LocalPilot-Training -- bash -lc "wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.1/rocdxg-roct_1.2.1_amd64.deb -O /tmp/rocdxg.deb && sudo apt install -y /tmp/rocdxg.deb"
wsl -d LocalPilot-Training -- bash -lc "wget -q https://github.com/ROCm/librocdxg/releases/download/v1.2.1/rocdxg-amd-smi-lib_1.2.1_amd64.deb -O /tmp/rocdxg-amd-smi-lib.deb && sudo apt install -y /tmp/rocdxg-amd-smi-lib.deb"
wsl -d LocalPilot-Training -- bash -lc "~/.venvs/localpilot-training/bin/python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ 'torch[device-gfx1201]==2.12.0+rocm7.14.1' 'torchvision[device-gfx1201]==0.27.0+rocm7.14.1' 'torchaudio==2.11.0+rocm7.14.1'"
wsl -d LocalPilot-Training -- bash -lc "~/.venvs/localpilot-training/bin/python -m pip install 'unsloth[amd]==2026.9.7' 'unsloth_zoo==2026.9.6' 'bitsandbytes==0.50.2' 'transformers==4.56.2' 'trl==0.22.2' 'peft==0.18.0' 'datasets==4.3.0' 'torch==2.12.0+rocm7.14.1' 'torchvision==0.27.0+rocm7.14.1' 'torchaudio==2.11.0+rocm7.14.1'"
wsl -d LocalPilot-Training -- bash -lc "source ~/.venvs/localpilot-training/bin/activate && source /etc/profile.d/rocdxg-amd-smi-lib.sh && pip check && rocm-sdk version && rocm-sdk targets | grep -q gfx1201 && amd-smi version && rocminfo | grep -q gfx1201 && python -m bitsandbytes"
wsl -d LocalPilot-Training --cd $localPilotRepo -- bash -lc "source ~/.venvs/localpilot-training/bin/activate && source /etc/profile.d/rocdxg-amd-smi-lib.sh && HF_HOME=~/.cache/huggingface-localpilot python training/scripts/train_adapter.py --dry-run --allow-downloads"
```

The second package installation retains explicit AMD Torch versions so the resolver must fail on an incompatible requirement instead of silently replacing them with CUDA/CPU wheels. The torchaudio 2.11 / torch 2.12 combination above is the one AMD publishes, not an inferred version pairing. Activating the virtual environment is required for bitsandbytes to find the matching `rocminfo` executable. The separate AMD SMI ROCDXG package is the supported WSL management path; the ROCm 7.14.1 SDK's aggregate `rocm-sdk test` still invokes legacy `rocm-smi`, which expects a bare-metal `amdgpu` kernel module and therefore produces a WSL-only false failure even when `amd-smi`, HIP, BF16 compute, and the project dry-run pass. If dependency resolution, `pip check`, or a GPU check fails, keep `status: proposed` and record the failure; do not bypass it with `--no-deps`. Preserve the resolved package versions from the dry-run locally before any later run.

Do not run `--train` while the tracked config is `proposed`. A passing ignored dry-run report must match the exact config and dataset digests; then a separate reviewed change may mark the config `approved_after_dry_run`.

## Rollback

Leave the existing Ollama `gpt-oss:20b` model and LocalPilot configuration unchanged. If WSL QLoRA cannot stay below the resource ceiling, use a rented AMD/NVIDIA Linux GPU with at least 24 GB for the same pinned corpus/config, or validate pipeline mechanics on a smaller 7-8B adapter while keeping `gpt-oss:20b` as the production model. Neither fallback is promoted without the same held-out Eval v1 and Evolution Execution gates.
