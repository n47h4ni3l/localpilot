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

## P1.2 operational report and verification gap (14 October 2026)

The owner reports that **P1.2 completed training as a continuation of
P1.1 and was promoted on 13 October 2026**. The currently checked-in
`package-1.json` records the earlier P1 lineage only. The separate
`training/configs/qlora_p12_curated_1000.yaml` is a frozen preparation
configuration whose status note refers to P1 as the proposed parent and
to a 12 October dry-run; it is **not evidence of P1.2's actual input
adapter or successful promotion**. The repository does not yet contain
a verified P1.1 -> P1.2 immutable run/promotion manifest.

Keep these distinctions explicit: the owner report establishes the stated
operational status, while GitHub alone cannot verify the trained parent,
actual checkpoint, deployed digest or rollback identity. Do not rewrite
historic configurations or package-1 hashes to make them appear current.
Obtain the actual training-PC receipts and create new content-addressed
lineage records for the P1.1 parent and promoted P1.2, including exact
adapter/config/completion hashes, checkpoint inventory, GGUF/Ollama model
identity and digest, owner acceptance, evaluation status, and rollback.
If artefacts are inaccessible, mark individual fields **unverified** rather
than guessing. Promotion does not itself prove benchmark superiority.

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
