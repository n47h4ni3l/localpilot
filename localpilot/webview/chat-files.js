(function () {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_FILES = 4;
  const SAFE_IMAGES = /^(image\/(png|jpeg|gif|webp|bmp))$/i;
  const TEXT_EXTENSIONS = /\.(txt|md|markdown|csv|tsv|json|jsonl|log|py|js|cjs|mjs|ts|tsx|jsx|css|html?|xml|ya?ml|toml|ini|cfg|sh|ps1|sql|rst)$/i;

  function sizeLabel(size) {
    const bytes = Number(size) || 0;
    return bytes < 1024 ? bytes + " B" : bytes < 1024 * 1024 ? Math.ceil(bytes / 1024) + " KB" : (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function fileStatus(file) {
    if (file._uploading) return file._capture ? "Taking screenshot…" : "Attaching…";
    const status = String(file.text_status || (file.extraction && file.extraction.status) || "").toLowerCase();
    if (status === "extracted" || status === "text" || status === "ready") return "Text extracted";
    if (status === "ocr") return "Text extracted with OCR";
    if (status === "processing" || status === "pending") return "Reading text…";
    if (status === "empty" || status === "no_text") return "No readable text found";
    if (status === "failed" || status === "error") return "Text extraction failed; file attached";
    if (/^image\//i.test(file.mime || "")) return "Image attached; visual understanding unavailable";
    if (status === "unsupported" || status === "unavailable") return "File attached; text extraction unavailable";
    return "File attached";
  }

  function pendingStatus(file) {
    if (file._uploading) return file._capture ? "Capturing…" : "Attaching…";
    const status = String(file.text_status || (file.extraction && file.extraction.status) || "").toLowerCase();
    if (["extracted", "text", "ready", "ocr"].includes(status)) return "Ready";
    if (status === "processing" || status === "pending") return "Reading text…";
    if (status === "empty" || status === "no_text") return "No readable text";
    if (status === "failed" || status === "error") return "Text unavailable";
    return /^image\//i.test(file.mime || "") ? "Image attached" : "Attached";
  }

  function init(options) {
    const document = window.document;
    const api = options.api;
    const bridge = options.bridge;
    const states = new Map();
    const renderedFiles = new WeakMap();
    let currentSessionId = options.getSessionId();
    let previewUrl = null;
    let previewGeneration = 0;
    let previewReturnFocus = null;
    const input = document.getElementById("chat-file-input");
    const attachButton = document.getElementById("chat-attach-btn");
    const screenshotButton = document.getElementById("chat-screenshot-btn");
    const pending = document.getElementById("chat-pending-files");
    const feedback = document.getElementById("chat-file-feedback");
    const composer = document.getElementById("composer-input");
    const dropTarget = document.getElementById("panel") || document.getElementById("message-stream");

    function getState(id) {
      if (!states.has(id)) states.set(id, {items: [], busy: 0, feedback: "", error: false});
      return states.get(id);
    }

    function route(sessionId, suffix) {
      return "/v1/sessions/" + encodeURIComponent(sessionId) + "/files" + (suffix || "");
    }

    function element(tag, className, text) {
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined) el.textContent = text;
      return el;
    }

    function button(label, onClick, className) {
      const el = element("button", className || "chat-file-action", label);
      el.type = "button";
      el.addEventListener("click", onClick);
      return el;
    }

    function setFeedback(sessionId, message, error) {
      const state = getState(sessionId);
      state.feedback = message;
      state.error = !!error;
      if (sessionId === currentSessionId) refresh();
    }

    function notify() {
      if (typeof options.onPendingChange === "function") options.onPendingChange();
    }

    function refresh() {
      const state = getState(currentSessionId);
      if (pending) {
        pending.replaceChildren();
        state.items.forEach(function (file) { pending.appendChild(makeCard(file, currentSessionId, true)); });
        pending.hidden = !state.items.length;
      }
      if (feedback) {
        feedback.textContent = state.feedback;
        feedback.hidden = !state.feedback;
        feedback.classList.toggle("is-error", state.error);
      }
      if (attachButton) attachButton.disabled = !currentSessionId || state.items.length >= MAX_FILES;
      if (screenshotButton) screenshotButton.disabled = !currentSessionId || state.items.length >= MAX_FILES || state.busy > 0;
      notify();
    }

    const dialog = element("dialog", "chat-file-preview");
    dialog.setAttribute("aria-labelledby", "chat-file-preview-heading");
    const dialogHeader = element("div", "chat-file-preview__header");
    const dialogHeading = element("h3", "", "File preview");
    dialogHeading.id = "chat-file-preview-heading";
    const dialogBody = element("div", "chat-file-preview__body");
    const dialogClose = button("Close preview", closePreview);
    dialogHeader.appendChild(dialogHeading);
    dialogHeader.appendChild(dialogClose);
    dialog.appendChild(dialogHeader);
    dialog.appendChild(dialogBody);
    document.body.appendChild(dialog);

    function releasePreview() {
      previewGeneration += 1;
      if (previewUrl) window.URL.revokeObjectURL(previewUrl);
      previewUrl = null;
      dialogBody.replaceChildren();
    }

    function closePreview() {
      if (typeof dialog.close === "function") dialog.close();
      else dialog.removeAttribute("open");
      releasePreview();
      if (previewReturnFocus && typeof previewReturnFocus.focus === "function") previewReturnFocus.focus();
      previewReturnFocus = null;
    }

    dialog.addEventListener("close", releasePreview);
    dialog.addEventListener("cancel", function (event) { event.preventDefault(); closePreview(); });

    async function preview(file, sessionId, trigger) {
      releasePreview();
      const generation = previewGeneration;
      previewReturnFocus = trigger;
      dialogHeading.textContent = file.name || "File preview";
      dialogBody.appendChild(element("p", "", "Loading preview…"));
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
      dialogClose.focus();
      try {
        const content = await api("GET", route(sessionId, "/" + encodeURIComponent(file.id) + "/content"));
        if (generation !== previewGeneration) return;
        if ((content.data_base64 || "").length > Math.ceil(MAX_BYTES / 3) * 4) throw new Error("This preview exceeds the 10 MB limit.");
        const raw = window.atob(content.data_base64 || "");
        if (raw.length > MAX_BYTES) throw new Error("This preview exceeds the 10 MB limit.");
        const bytes = new Uint8Array(raw.length);
        for (let index = 0; index < raw.length; index += 1) bytes[index] = raw.charCodeAt(index);
        const mime = String(content.mime || file.mime || "application/octet-stream");
        dialogBody.replaceChildren();
        if (SAFE_IMAGES.test(mime)) {
          previewUrl = window.URL.createObjectURL(new window.Blob([bytes], {type: mime}));
          const image = element("img", "chat-file-preview__image");
          image.src = previewUrl;
          image.alt = file.name || "Attached image";
          dialogBody.appendChild(image);
        } else if (/^text\//i.test(mime) || TEXT_EXTENSIONS.test(file.name || "") || /^(application\/(json|xml))$/i.test(mime)) {
          const previewBytes = bytes.subarray(0, 256 * 1024);
          const text = new window.TextDecoder("utf-8", {fatal: false}).decode(previewBytes);
          dialogBody.appendChild(element("pre", "chat-file-preview__text", text));
          if (bytes.length > previewBytes.length) dialogBody.appendChild(element("p", "", "Showing the first 256 KB. Open or Save as to read the complete file."));
        } else {
          dialogBody.appendChild(element("p", "", "A preview is unavailable for this file type. Use Open or Save as to view it in a suitable app."));
        }
      } catch (error) {
        if (generation !== previewGeneration) return;
        dialogBody.replaceChildren(element("p", "chat-file-error", "Could not load this preview. " + error.message));
      }
    }

    async function externalFileAction(method, file, sessionId, trigger) {
      trigger.disabled = true;
      try {
        if (typeof bridge !== "function") throw new Error("Desktop file actions are unavailable in this window.");
        const result = await bridge(method, {session_id: sessionId, file_id: file.id});
        if (!result) throw new Error("Desktop file actions are unavailable in this window.");
        if (result.cancelled || result.canceled) return;
        if (result.ok === false || result.error) throw new Error(result.error || "The file action could not finish.");
        setFeedback(currentSessionId, method === "save_chat_file" ? "File saved." : "File opened.", false);
      } catch (error) {
        setFeedback(currentSessionId, error.message || "The file action could not finish.", true);
      } finally {
        trigger.disabled = false;
      }
    }

    function makeCard(file, sessionId, isPending) {
      const card = element("div", "chat-file-card" + (isPending ? " chat-file-card--pending" : ""));
      const details = element("div", "chat-file-card__details");
      const name = element("strong", "chat-file-card__name", file.name || "Attached file");
      name.setAttribute("title", file.name || "Attached file");
      details.appendChild(name);
      const fullStatus = sizeLabel(file.size) + " · " + fileStatus(file) +
        (file.extraction && file.extraction.truncated ? " · Text extraction limited" : "");
      const meta = element("span", "chat-file-card__meta", isPending ? sizeLabel(file.size) + " · " + pendingStatus(file) : fullStatus);
      meta.setAttribute("title", fullStatus);
      meta.setAttribute("aria-label", fullStatus);
      details.appendChild(meta);
      card.appendChild(details);
      const actions = element("div", "chat-file-card__actions");
      if (!file._uploading && file.id) {
        const previewButton = button("Preview", function () { return preview(file, sessionId, previewButton); });
        previewButton.setAttribute("aria-label", "Preview " + file.name);
        actions.appendChild(previewButton);
        if (!isPending) {
          const openButton = button("Open", function () { return externalFileAction("open_chat_file", file, sessionId, openButton); });
          const saveButton = button("Save as…", function () { return externalFileAction("save_chat_file", file, sessionId, saveButton); });
          openButton.setAttribute("aria-label", "Open " + file.name);
          saveButton.setAttribute("aria-label", "Save " + file.name + " as");
          actions.appendChild(openButton);
          actions.appendChild(saveButton);
          if (file.source_file_id) {
            const originalButton = button("Original", async function () {
              try {
                const result = await api("GET", route(sessionId, "/" + encodeURIComponent(file.source_file_id)));
                await preview(result.file, sessionId, originalButton);
              } catch (error) { setFeedback(sessionId, error.message || "The original file could not be opened.", true); }
            });
            originalButton.setAttribute("aria-label", "Preview original file for " + file.name);
            actions.appendChild(originalButton);
          }
        }
      }
      if (isPending) {
        const remove = button("×", function () {
          const state = getState(sessionId);
          state.items = state.items.filter(function (item) { return item !== file; });
          if (!state.items.length && !state.busy && !state.error) state.feedback = "";
          if (sessionId === currentSessionId) {
            refresh();
            if (attachButton) attachButton.focus();
          }
        }, "chat-file-action chat-file-action--remove");
        const removeLabel = "Remove " + (file.name || "attached file") + " from this message";
        remove.setAttribute("aria-label", removeLabel);
        remove.setAttribute("title", removeLabel);
        actions.appendChild(remove);
      }
      card.appendChild(actions);
      return card;
    }

    function readBase64(file) {
      return new Promise(function (resolve, reject) {
        const reader = new window.FileReader();
        reader.onerror = function () { reject(new Error("The selected file could not be read.")); };
        reader.onabort = function () { reject(new Error("Reading the selected file was cancelled.")); };
        reader.onload = function () {
          const value = reader.result;
          if (typeof value !== "string" || value.indexOf(",") < 0) reject(new Error("The selected file could not be encoded."));
          else resolve(value.slice(value.indexOf(",") + 1));
        };
        reader.readAsDataURL(file);
      });
    }

    async function addFiles(fileList) {
      const sessionId = currentSessionId;
      if (!sessionId) return;
      const state = getState(sessionId);
      const files = Array.from(fileList || []);
      if (!files.length) return;
      if (files.length + state.items.length > MAX_FILES) {
        setFeedback(sessionId, "Attach up to 4 files to one message.", true);
        return;
      }
      if (files.some(function (file) { return file.size > MAX_BYTES; })) {
        setFeedback(sessionId, "Each file must be 10 MB or smaller.", true);
        return;
      }
      const queued = files.map(function (file) {
        return {name: file.name || "Pasted image.png", mime: file.type || "application/octet-stream", size: file.size, _uploading: true};
      });
      state.items.push.apply(state.items, queued);
      state.busy += 1;
      setFeedback(sessionId, "Attaching files…", false);
      try {
        // Read and upload one file at a time rather than retaining a batch of
        // base64 strings alongside the original clipboard/drop buffers.
        for (let index = 0; index < files.length; index += 1) {
          const item = queued[index];
          if (!state.items.includes(item)) continue;
          try {
            const data = await readBase64(files[index]);
            const response = await api("POST", route(sessionId), {name: item.name, data_base64: data});
            if (!response.file || !response.file.id) throw new Error("The server did not return an attached file.");
            const position = state.items.indexOf(item);
            if (position >= 0) state.items[position] = response.file;
          } catch (error) {
            state.items = state.items.filter(function (entry) { return entry !== item; });
            setFeedback(sessionId, "Could not attach " + item.name + ". " + error.message, true);
          }
          if (sessionId === currentSessionId) refresh();
        }
      } finally {
        state.busy -= 1;
        if (!state.error && !state.busy) state.feedback = "";
        if (sessionId === currentSessionId) refresh();
      }
    }

    async function captureScreenshot() {
      const sessionId = currentSessionId;
      if (!sessionId) return;
      const state = getState(sessionId);
      if (state.busy || state.items.length >= MAX_FILES) return;
      const item = {name: "Screenshot", size: 0, _uploading: true, _capture: true};
      state.items.push(item);
      state.busy += 1;
      setFeedback(sessionId, "Taking a screenshot of your desktop…", false);
      try {
        const response = await api("POST", "/v1/sessions/" + encodeURIComponent(sessionId) + "/screenshot", {});
        if (!response.file || !response.file.id) throw new Error("The server did not return a screenshot.");
        const position = state.items.indexOf(item);
        if (position >= 0) state.items[position] = response.file;
        setFeedback(sessionId, "", false);
      } catch (error) {
        state.items = state.items.filter(function (entry) { return entry !== item; });
        setFeedback(sessionId, "Could not take a screenshot. " + error.message, true);
      } finally {
        state.busy -= 1;
        if (sessionId === currentSessionId) refresh();
      }
    }

    if (attachButton && input) attachButton.addEventListener("click", function () { input.click(); });
    if (input) input.addEventListener("change", function () {
      const files = Array.from(input.files || []);
      input.value = "";
      return addFiles(files);
    });
    if (screenshotButton) screenshotButton.addEventListener("click", captureScreenshot);
    if (composer) composer.addEventListener("paste", function (event) {
      const data = event.clipboardData;
      const files = data && data.files ? Array.from(data.files) : [];
      if (!files.length) return;
      event.preventDefault();
      return addFiles(files);
    });
    if (dropTarget) {
      function isFileDrag(event) { return event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files"); }
      dropTarget.addEventListener("dragover", function (event) {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        dropTarget.classList.add("chat-files-drop-target");
      });
      dropTarget.addEventListener("dragleave", function (event) {
        if (!dropTarget.contains(event.relatedTarget)) dropTarget.classList.remove("chat-files-drop-target");
      });
      dropTarget.addEventListener("drop", function (event) {
        dropTarget.classList.remove("chat-files-drop-target");
        if (!isFileDrag(event)) return;
        event.preventDefault();
        return addFiles(event.dataTransfer.files);
      });
    }

    function pendingItems(sessionId) {
      return getState(sessionId === undefined ? currentSessionId : sessionId).items.filter(function (file) { return file.id && !file._uploading; }).slice();
    }

    const controller = {
      pendingItems: pendingItems,
      pendingIds: function (sessionId) { return pendingItems(sessionId).map(function (file) { return file.id; }); },
      hasPending: function () { return getState(currentSessionId).items.length > 0; },
      hasDraft: function () { return Array.from(states.values()).some(function (state) { return state.items.length > 0 || state.busy > 0; }); },
      isBusy: function () { return getState(currentSessionId).busy > 0; },
      clearPending: function (sessionId, sentIds) {
        const id = sessionId === undefined ? currentSessionId : sessionId;
        const state = getState(id);
        if (!sentIds && state.busy) return;
        if (sentIds) {
          const sent = new Set(sentIds);
          state.items = state.items.filter(function (file) { return !sent.has(file.id); });
        } else state.items = [];
        state.feedback = ""; state.error = false;
        if (id === currentSessionId) refresh();
        else notify();
      },
      restorePending: function (files) {
        const state = getState(currentSessionId);
        const ids = new Set(state.items.map(function (file) { return file.id; }));
        (files || []).forEach(function (file) {
          if (file && file.id && !ids.has(file.id) && state.items.length < MAX_FILES) { state.items.push(file); ids.add(file.id); }
        });
        refresh();
      },
      sessionChanged: function (sessionId) {
        if (sessionId !== currentSessionId) closePreview();
        currentSessionId = sessionId;
        refresh();
      },
      renderMessage: function (container, message) {
        const files = message.files || [];
        const signature = JSON.stringify(files);
        const previous = renderedFiles.get(container);
        if (previous && previous.signature === signature) return;
        if (previous) previous.list.remove();
        if (!files.length) return;
        const list = element("div", "chat-message-files");
        list.setAttribute("aria-label", message.role === "assistant" ? "Returned files" : "Attached files");
        const sessionId = message.session_id || currentSessionId;
        files.forEach(function (file) { list.appendChild(makeCard(file, file.session_id || sessionId, false)); });
        container.appendChild(list);
        renderedFiles.set(container, {signature: signature, list: list});
      },
    };
    refresh();
    return controller;
  }

  window.LocalPilotFiles = {init: init};
}());
