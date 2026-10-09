# Post-repair P1 sanity protocol

The new runner `training/scripts/run_postrepair_sanity.py` writes a fresh report and refuses to overwrite an existing file. It does not import the old 48-case report or write to the 684-case matrix. Eight independently authored scenarios have unique `lp-postrepair-v2-*` IDs; each runs three times in each arm (48 cells). Alternating arm order reduces consistent warmup/order advantage. This is descriptive scaffold evaluation, not a model promotion benchmark.

Both arms use exactly the deployment model and SHA-256 recorded in `training/lineage/package-1.json`. The evaluator requires a clean checkout at an explicit full revision, checks model identity before every cell, freezes prompts/rubrics/inference settings in the report, and rejects incompatible resumes. Operators should check the identity again after completion. Current executable runtime files must match the repaired runtime at `4589eb3f5c70d1614decfabce027546720be3ee1`; evaluator-only changes are recorded separately.

Each scaffold cell has a fresh agent and unique empty temporary data directory. The source is a tracked `git archive HEAD` with the entire training tree removed. Only repository tree/read/search/dependency tools are exposed. Web, machine-state, Library, memory-summary, GitHub and action tools are absent; embeddings, SystemSense, self-development and Library are disabled. Direct cells receive no tools or personal memory. Request/response model evidence, tool messages and audit events are retained. No results enter learning, feedback or training. Model weights and production configuration are unchanged.

Run from the evaluator checkout with Python 3.12:

```powershell
python training/scripts/run_postrepair_sanity.py --expected-evaluator-revision FULL_COMMIT_SHA --output NEW_ABSOLUTE_REPORT_PATH
```

Check running training/model traffic first. Preserve any original scheduled-worker state if pausing it for isolation, and restore that state after all calls, including failures. A halted run may resume only with unchanged revision, task digest, model and settings. Independently review evidence-grounded correctness, conceptual versus live claims, historical versus current evidence, lineage reasoning, unfamiliar script safety, and refusal/withholding behavior; automated counts do not establish quality.

The inherited-tensor mismatch variant is a separate six-cell supplement with a distinct task digest and suite. A public-web probe is an ordinary isolated interaction outside benchmark mode, using only real `search_public_web` and `fetch_public_https` tools and fresh ephemeral state. Asking unfamiliar manufacturer details without an instruction to search tests voluntary selection. Save actual calls, returned source content and digests, response, timing and errors; an offered tool alone is not a successful lookup. Its restricted tool surface is not evidence of the full production surface working. Do not fold the probe or supplement into the 48-case scores.
