# Chat files and activity

The existing desktop chat supports expandable tool steps, local attachments and
returned files. Tool steps record actual execution status, bounded input/result
previews and elapsed time. They remain attached to the assistant message across
history reloads. They are an activity log, not a representation of private model
reasoning. Local runtime errors render immediately in the chat.

Use **Attach**, drag files into the conversation, or paste an image/file from the
clipboard. **Screenshot** captures the current displays only on a deliberate
click, then attaches a preview for the next message. Remove a pending attachment
to omit it from the message. Unsent text and attachments stay with their chat.

Supported inputs: UTF-8 text and code, CSV/JSON/Markdown, text PDFs, DOCX, PNG,
JPEG and WebP. Limits are four files per message, 10 MiB per file, 12 million
pixels per image and 64 stored files per conversation. PDF extraction and local
Windows OCR run in time-limited, memory-limited child processes. Extraction
failures are shown rather than represented as successful reading. OCR reads
text; it does not provide object recognition, visual understanding or OCR for
scanned PDF pages. No remote vision model is called.

Nestra can read same-chat files by ID, create text/code/CSV/JSON, perform exact
text replacements into a new file, produce a simple DOCX, and crop/resize/annotate
an image using supplied coordinates. Arbitrary filesystem paths and execution
of uploaded code are not supported. PDF/DOCX edits can be returned as a new
document; exact replacement preserves source formatting only for text files.
The original bytes remain unchanged. Edited artifacts retain a reference to
their source, exposed through **Original** on the file card.

Returned file cards offer **Preview**, **Open** and **Save as**. Open uses a
working copy and opens code in a text editor; it never executes an attachment.
Save as copies bytes to a new owner-selected path and refuses to overwrite an
existing file. Filenames, IDs, session boundaries, formats, sizes and hashes are
validated. All artifacts remain local under the configured data directory.

This feature does not change model selection, Guide-only response generation,
training, existing safety settings or autonomous promotion rules.
