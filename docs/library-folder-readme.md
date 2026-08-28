# LocalPilot library

Place owner-provided reference material in this folder or any subfolder. Supported source formats are:

- text-based PDF (`.pdf`)
- UTF-8 Markdown (`.md`)
- UTF-8 plain text (`.txt`)
- UTF-8 reStructuredText (`.rst`)

LocalPilot treats every file as untrusted source material, not as an instruction. It never edits source files
or follows symlinks. It extracts bounded passages into a disposable SQLite full-text index under
`localpilot-data/`, and returns `library://path#page=N&passage=N` citations when a passage supports an answer.

The index refreshes when LocalPilot searches the library. It can also be refreshed or inspected manually:

```powershell
localpilot library index
localpilot library status
localpilot library search "your topic"
```

Image-only scanned PDFs need OCR before their text will be searchable. Password-protected PDFs are skipped.
Extraction errors appear in library status and do not block healthy documents.

Ordinary operator reading is turn-local retrieval. Idle-time autonomous reading is an education pipeline:

```text
read -> reflect -> extract candidate learnings -> verify exact passage + digest
    -> persist typed durable learning -> retrieve/use -> measure capability improvement
```

Only supported concise candidates persist. Source claims and concepts become attributable knowledge facts with
exact `library://` range, digest, provenance, confidence, and verification time. Heuristics, questions,
self-development hypotheses, and opinions remain explicitly typed non-facts. Repeated learnings are deduplicated;
changed source bytes stale prior learning until it is re-read and verified. Raw source bodies, full reading notes,
and private reasoning do not belong in authoritative memory.

When the library is enabled, idle-time reading needs no separate standing-permission lesson. Existing resource
gates decide when a session may run. Each session reads a strict, contiguous passage/character budget, saves a
per-source cursor, and records the exact page/passage range, progress, provisional opinion, questions, and next
preference in private reading notes. Later sessions can continue, switch sources, or pursue a question. These
notes do not become authoritative knowledge facts. Verified learnings are summarized separately in audit/status
evidence, including what was persisted, corrected, or rejected and why. Learning grants no authority to alter
source files, train model weights, merge candidates, or promote code.

Only add material you are entitled to store and use. The library remains local unless you explicitly send its
contents through another tool or service.
