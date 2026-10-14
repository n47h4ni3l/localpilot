# Nestra training lineage

Nestra uses one cumulative accepted adapter lineage over the exact frozen GPT-OSS base.

An accepted sequence is:

```text
openai/gpt-oss-20b
  -> Package 1 adapter
  -> nestra:20b-p1
  -> Package 2 continues the Package 1 adapter
  -> nestra:20b-p2
  -> ...
```

## Verified P1.1 to P1.2 continuation (14 October 2026)

The training machine's accepted manifests are now recorded byte-for-byte:
[`package-2-production-20261013.json`](package-2-production-20261013.json)
is the 51-step P1 continuation later named **P1.1**;
[`package-3.json`](package-3.json) is **P1.2**, the subsequent 1,185-step,
three-epoch continuation over the fresh 1,000-record corpus. Package numbers
identify cumulative adapter generations, independently of the model's short
version label. The completed recipe is
[`qlora_p12_currentprod_20261013.yaml`](../configs/qlora_p12_currentprod_20261013.yaml).
The old `qlora_p12_curated_1000.yaml` preparation recipe and the earlier
wrong-parent experiment are preserved as historical records.

The content-addressed [verification receipt](receipts/p11-p12-7b89b458db509a1008406231fa41c7dba9cc3b9ce3a4601122e3a8a69daa872a.json)
binds both accepted manifests, the actual preflight parent, source corpus
and canonical rendered dataset identities, config, run identities, natural
completion and final checkpoint markers, GGUFs and deployed aliases. Its
filename contains the SHA256 of its exact bytes. On 14 October, both retained
GGUFs and every parent/final adapter and required checkpoint file were
rehashed against those receipts. A read-only Ollama inventory confirmed
`nestra:p1.2` and the retained `nestra:p1.1` rollback. Separately observed
desktop configuration selects P1.2 for all three selectors in Guide-only mode.

The original P1.1 manifest records its old deployment name `nestra:20b-p1`;
the promotion receipt explicitly maps that same digest to `nestra:p1.1`.
Neither its historic name nor its hashes were rewritten. Artifact paths in
the accepted records refer to retained local files; weights, raw evaluation
material and private conversations are not published here.

Owner acceptance and deployment were verified on 13 October at
11:15:27 UTC. The exact timestamp of the owner's chat decision is not in the
local deployment receipt and is explicitly unknown. `benchmark_promoted`
remains false. The bounded Guide-only comparison and single execution
fixture support the recorded owner decision, not broad benchmark superiority.

The deployed GGUF/Ollama model is never used as a training source. New packages load the exact pinned GPT-OSS training snapshot and then load the previous accepted adapter with PEFT in trainable mode. A new package creates a fresh Trainer, optimizer and scheduler; only an in-package `--resume` restores optimizer/scheduler/RNG state.

## Accepted-head manifest

After the owner accepts a completed candidate as the working lineage head, run `training/scripts/record_lineage.py`. The resulting manifest binds:

- exact frozen base identity and revision;
- completed training config/output and natural-completion marker;
- final adapter file sizes and SHA256 values;
- deployed Ollama model and digest;
- GGUF SHA256;
- immediately previous rollback model and digest;
- original frozen GPT-OSS baseline/control model and digest;
- final checkpoint completion-marker hash plus checkpoint file inventory;
- retained Eval v1 and Evolution Execution aggregate evidence; and
- the explicit owner decision.

Owner acceptance as the working lineage and formal benchmark promotion are separate fields. A mixed benchmark result may be retained and accepted for continued remediation without being relabeled as benchmark-promoted.

For Package 2 and later, add:

```json
"lineage": {
  "package": 2,
  "parent_manifest": "training/lineage/package-1.json"
}
```

The dry-run refuses a missing, altered, unaccepted, wrong-package or architecture-mismatched parent. The parent adapter hashes are checked again immediately before model loading. Never edit an old accepted manifest to point at a newer package; preserve immutable package records and create the next one after owner acceptance.
