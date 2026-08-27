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

Reading a source is turn-local retrieval. It does not silently write durable facts or alter model weights.
Future library study may distill bounded, source-linked claims into durable learning only with provenance,
staleness handling, held-out evaluation, and rollback. Raw source bodies and private reasoning do not belong in
durable memory.

Only add material you are entitled to store and use. The library remains local unless you explicitly send its
contents through another tool or service.
