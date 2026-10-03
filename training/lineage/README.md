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

The deployed GGUF/Ollama model is never used as a training source. New packages load the exact pinned GPT-OSS training snapshot and then load the previous accepted adapter with PEFT in trainable mode. A new package creates a fresh Trainer, optimizer and scheduler; only an in-package `--resume` restores optimizer/scheduler/RNG state.

## Accepted-head manifest

After the owner accepts a completed candidate as the working lineage head, run `training/scripts/record_lineage.py`. The resulting manifest binds:

- exact frozen base identity and revision;
- completed training config/output and natural-completion marker;
- final adapter file sizes and SHA256 values;
- deployed Ollama model and digest;
- GGUF SHA256;
- rollback model and digest;
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
