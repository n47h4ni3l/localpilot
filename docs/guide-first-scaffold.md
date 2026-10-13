# Guide-only Nestra runtime

LocalPilot supplies one advisory operator path. Nestra owns the reasoning, tool choices and final wording. The guide offers useful methods before inference; it never reads, scores, rewrites, appends to or withholds a nonempty model-authored final answer.

## Configuration

Guide-only is the default for new configurations and the only supported runtime. Existing configurations using `scaffold_mode = "guide_first"` continue to work. Selecting the retired `strict` path is rejected, including when constructing an agent directly. Future local evaluation runs also use Guide-only.

## Execution and resource boundaries

The guide grants no action authority. The owner's no-web/no-tool choices, registered tool policy, destructive-action confirmation, candidate confinement and hard resource/tool ceilings remain in force. The soft research budget is advice; a notebook checkpoint is optional. Permission checks govern actual execution rather than judging prose.

Source-linked memory and library records can help Nestra choose research, without mandatory source reads before she answers. The optional hosted mentor is registered only when the owner enables it; it cannot be contacted automatically. Existing source and privacy boundaries still apply to reading, learning and transmission.

## Generation and transport recovery

Blank or interrupted inference can use the existing bounded same-model rendering continuation within measured context headroom. It does not assess answer quality, replace a refusal, or reroute to a different model. Valid rendered text is delivered unchanged, including whitespace. Genuine tool transport failures remain distinct from error-like text quoted inside a source.

## Historical evidence

Past experiments, frozen models and raw evaluation results remain historical evidence under their original identities and source revisions. Removing the retired runtime does not relabel those results as Guide-only or establish quality improvement. Future comparisons must identify their exact source revision and model digest.
