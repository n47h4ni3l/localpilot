# Fresh P1.2 corpus attribution and transformations

The examples are attributed individually by repository, pinned revision,
source file, row index, original-record SHA-256 and acquisition-file SHA-256.
The original selected records are retained in
`training/sources/p12_fresh_source_records.jsonl`. The acquisition lock records
the immutable upstream URLs and file or bounded-prefix hashes. Only upstream
training splits are used; no public test partition or private holdout is read.

The following authors/publishers supplied the data:

- **BigCode**: `bigcode/self-oss-instruct-sc2-exec-filter-50k`, Open Data
  Commons Attribution License 1.0. Its card describes execution-validated
  responses generated from Python-function seeds in the MultiPL-T pipeline.
  We use the task-specific instruction and response, excluding generation
  few-shot examples. Selected code is checked again using its included
  assertions in confined, time-bounded CPU subprocesses. This does not
  establish complete specification correctness or independent expert approval.
- **Hugging Face Smol Models Research**: `HuggingFaceTB/smoltalk`, specifically
  its newly generated Smol-Magpie-Ultra, Smol-rewrite, Smol-summarize and
  Smol-constraints subsets, Apache License 2.0. No other component of the
  heterogeneous SmolTalk mixture is imported under that blanket license.
- **Hugging Face Smol Models Research**:
  `HuggingFaceTB/everyday-conversations-llama3.1-2k`, Apache License 2.0.
  We remove the initial stock greeting pair and retain the substantive
  conversation verbatim. Records promising unsupported actions are excluded.
- **NVIDIA Corporation**:
  `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2`, Open Data Commons
  Attribution License 1.0. We use only its `reasoning_off` release, downloaded
  as a bounded, hash-recorded prefix of the pinned file. Empty system messages
  are omitted; remaining user and assistant text is preserved.
- **ACE Team**: `Team-ACE/ToolACE`, Apache License 2.0. Completed single-call
  trajectories are converted into the trainer's native structured calls and
  results. Tool names become consistent identifiers and schema `dict` / `list`
  types become JSON Schema `object` / `array`. The source's incompatible
  textual call-transport instructions are replaced by native-tool instructions;
  task requests, arguments, results and final answers are retained. Parallel,
  unfinished, blank or unsupported traces are excluded.

Fifty self-improvement examples are explicitly **source-derived adaptations**,
not verbatim upstream conversations. A selected fresh BigCode implementation
receives one controlled operator mutation inside a function. The original
assertions run unchanged and must fail against the mutant and pass against
the corrected implementation. The prompt shows the objective, candidate,
checks and observed failure; the answer supplies the corrected implementation
and a brief repair explanation. Actual execution outcomes remain in metadata. These are Tier C, `synthetic_validated`, with
independent semantic review pending. No model-generated hidden reasoning or
production mutation is involved.

Complete upstream dataset cards are bundled under `cards/`. Full license
texts are bundled as `Apache-2.0.txt` and `ODC-BY-1.0.txt` using the SPDX
license-list-data copies of the standard license texts. The corresponding
publisher pages are <https://www.apache.org/licenses/LICENSE-2.0> and
<https://opendatacommons.org/licenses/by/1-0/>.

Selection, deduplication, normalization and adaptations are dated 12 October
2026, transformation `p12-fresh-v2`, by the LocalPilot owner-directed Codex
task. Neither source attribution nor passing code assertions is a claim of
independent review, model improvement, training readiness or permission to train.
