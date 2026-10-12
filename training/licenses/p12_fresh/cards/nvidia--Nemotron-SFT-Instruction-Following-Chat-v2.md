---
language:
- en
license:
- odc-by
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: reasoning_off
    path: data/reasoning_off.jsonl
  - split: reasoning_on
    path: data/reasoning_on.jsonl
---

## Dataset Description:
The Nemotron-Instruction-Following-Chat-v2 dataset is designed to broadly strengthen the model’s interactive capabilities, including open-ended chat and precise instruction following.  
The dataset is a refreshed version of [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) with synthetic dialogues generated from [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking), [GLM-4.6](https://huggingface.co/zai-org/GLM-4.6), [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507), [GPT-OSS-120b](https://huggingface.co/openai/gpt-oss-120b), [Kimi-K2-Instruct-0905](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905), and [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507).


This dataset is ready for commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created on: 11/20/2025  
Last Modified on: 11/20/2025

## License/Terms of Use: 
This dataset is licensed under the Open Data Commons Attribution License (ODC-By).

## Intended Usage:
This dataset is intended to be used by the community to continue to improve the Instruction Following and Chat capabilities of models. The data may be freely used to train and evaluate. The dataset can be used in addition to [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1).

## Dataset Characterization
**Data Collection Method**  
* Hybrid: Human, Synthetic, Automated

**Labeling Method**
* Hybrid: Human, Synthetic, Automated

## Dataset Format
Modality: Text  
Format: JSONL  
Structure: Text + Metadata

## Dataset Quantification
| Subset | Samples |
|--------|---------|
| chat   | 1,998,568 |

Total Disk Size: ~15GB


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)