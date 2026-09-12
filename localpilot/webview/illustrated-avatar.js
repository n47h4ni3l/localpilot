"use strict";

/* LocalPilot illustrated avatar: per-state hand-drawn sprite sheets.
 *
 * app.js owns runtime state and the original pixel canvas remains the fail-safe.
 * Each completed state uses its own independently drawn sprite sheet. Frames are
 * cropped from the generated sheet, alpha-trimmed, and fitted into the avatar
 * without stretching. Transition sheets are intentionally deferred; for now the
 * WebView crossfades between the outgoing and incoming live loops.
 */
(function () {
  "use strict";

  const MANIFEST_URL = "avatar/anim/animation-manifest.json";
  const REQUIRED_STATES = [
    "idle", "listening", "thinking", "researching", "working", "speaking",
    "success", "uncertain", "error", "learning", "restarting", "sleeping", "offline",
  ];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvases = [
    document.getElementById("avatar-dock"),
    document.getElementById("avatar-header"),
  ].filter(Boolean);

  if (!canvases.length) return;

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function normalizeState(state) {
    return REQUIRED_STATES.includes(state) ? state : "error";
  }

  function frameForElapsed(stateSpec, assetSpec, elapsedMs) {
    if (reducedMotion) return Number(stateSpec.representative_frame || 0);
    const count = Number(assetSpec.frames);
    const step = Math.floor(Math.max(0, elapsedMs) / Number(stateSpec.frame_ms));
    return step % Math.max(1, count);
  }

  function approximateCellRect(image, asset, frameIndex) {
    const columns = Number(asset.columns);
    const rows = Number(asset.rows);
    const index = Number(frameIndex) % Number(asset.frames);
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x0 = Math.round(column * image.naturalWidth / columns);
    const x1 = Math.round((column + 1) * image.naturalWidth / columns);
    const y0 = Math.round(row * image.naturalHeight / rows);
    const y1 = Math.round((row + 1) * image.naturalHeight / rows);
    return { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
  }

  function alphaTrimRect(image, asset, frameIndex) {
    const rect = approximateCellRect(image, asset, frameIndex);
    const scratch = document.createElement("canvas");
    scratch.width = rect.width;
    scratch.height = rect.height;
    const context = scratch.getContext("2d", { willReadFrequently: true });
    context.clearRect(0, 0, rect.width, rect.height);
    context.drawImage(
      image,
      rect.x, rect.y, rect.width, rect.height,
      0, 0, rect.width, rect.height
    );
    const pixels = context.getImageData(0, 0, rect.width, rect.height).data;
    let left = rect.width;
    let top = rect.height;
    let right = -1;
    let bottom = -1;
    for (let y = 0; y < rect.height; y += 1) {
      for (let x = 0; x < rect.width; x += 1) {
        const alpha = pixels[(y * rect.width + x) * 4 + 3];
        if (alpha <= 2) continue;
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < top) top = y;
        if (y > bottom) bottom = y;
      }
    }
    if (right < left || bottom < top) return rect;
    return {
      x: rect.x + left,
      y: rect.y + top,
      width: right - left + 1,
      height: bottom - top + 1,
    };
  }

  function drawFittedFrame(canvas, image, sourceRect) {
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);
    const availableWidth = Math.max(1, width - 4);
    const availableHeight = Math.max(1, height - 4);
    const scale = Math.min(
      availableWidth / sourceRect.width,
      availableHeight / sourceRect.height
    );
    const drawWidth = Math.max(1, sourceRect.width * scale);
    const drawHeight = Math.max(1, sourceRect.height * scale);
    const x = (width - drawWidth) / 2;
    const y = height - drawHeight - 2;
    context.drawImage(
      image,
      sourceRect.x, sourceRect.y, sourceRect.width, sourceRect.height,
      x, y, drawWidth, drawHeight
    );
  }

  class IllustratedAvatarLayer {
    constructor(canvas, manifest, assets) {
      this.canvas = canvas;
      this.manifest = manifest;
      this.assets = assets;
      this.transitionMs = Number(manifest.transition_ms);
      this.enabled = false;
      this.state = "restarting";
      this.previousState = "restarting";
      this.stateStart = performance.now();
      this.previousStateStart = this.stateStart;
      this.transitionStart = this.stateStart - this.transitionMs;

      this.root = document.createElement("span");
      this.root.className = "illustrated-avatar-layer";
      this.root.setAttribute("aria-hidden", "true");
      Object.assign(this.root.style, {
        position: "absolute",
        display: "block",
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: "2",
      });

      this.previous = this.makeCanvas("illustrated-avatar-frame illustrated-avatar-frame--previous");
      this.current = this.makeCanvas("illustrated-avatar-frame illustrated-avatar-frame--current");
      this.root.appendChild(this.previous);
      this.root.appendChild(this.current);

      const parent = canvas.parentElement;
      if (parent && window.getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
      }
      if (parent) parent.appendChild(this.root);
      this.syncGeometry();
    }

    makeCanvas(className) {
      const frame = document.createElement("canvas");
      frame.className = className;
      frame.width = 128;
      frame.height = 128;
      Object.assign(frame.style, {
        position: "absolute",
        inset: "0",
        width: "100%",
        height: "100%",
        display: "block",
        pointerEvents: "none",
        transform: "none",
        willChange: "opacity",
      });
      return frame;
    }

    spec(state) {
      return this.manifest.states[normalizeState(state)];
    }

    asset(state) {
      const spec = this.spec(state);
      return this.assets[spec.asset];
    }

    syncGeometry() {
      const width = this.canvas.offsetWidth || this.canvas.width;
      const height = this.canvas.offsetHeight || this.canvas.height;
      this.root.style.left = this.canvas.offsetLeft + "px";
      this.root.style.top = this.canvas.offsetTop + "px";
      this.root.style.width = width + "px";
      this.root.style.height = height + "px";
    }

    setFrame(element, state, frameIndex) {
      const asset = this.asset(state);
      const index = Number(frameIndex) % asset.frames.length;
      drawFittedFrame(element, asset.image, asset.frames[index]);
    }

    enable(state, now) {
      this.canvas.style.opacity = "0";
      this.enabled = true;
      this.state = normalizeState(state);
      this.previousState = this.state;
      this.stateStart = now;
      this.previousStateStart = now;
      this.transitionStart = now - this.transitionMs;
      this.syncGeometry();
      const spec = this.spec(this.state);
      this.setFrame(this.current, this.state, Number(spec.representative_frame));
      this.current.style.opacity = this.state === "offline" ? "0.72" : "1";
      this.previous.style.opacity = "0";
    }

    setState(next, now) {
      next = normalizeState(next);
      if (next === this.state) return;
      this.previousState = this.state;
      this.previousStateStart = this.stateStart;
      this.state = next;
      this.stateStart = now;
      this.transitionStart = now;
    }

    drawStateFrame(element, state, elapsedMs) {
      const spec = this.spec(state);
      const asset = this.asset(state);
      this.setFrame(element, state, frameForElapsed(spec, asset.spec, elapsedMs));
    }

    draw(now) {
      if (!this.enabled) return;

      if (reducedMotion) {
        const spec = this.spec(this.state);
        this.setFrame(this.current, this.state, Number(spec.representative_frame));
        this.current.style.opacity = this.state === "offline" ? "0.72" : "1";
        this.previous.style.opacity = "0";
        return;
      }

      const transitionProgress = (now - this.transitionStart) / Math.max(1, this.transitionMs);
      if (transitionProgress < 1) {
        const p = clamp01(transitionProgress);
        this.drawStateFrame(this.previous, this.previousState, now - this.previousStateStart);
        this.drawStateFrame(this.current, this.state, now - this.stateStart);
        this.previous.style.opacity = String(1 - p);
        this.current.style.opacity = String(p * (this.state === "offline" ? 0.72 : 1));
        return;
      }

      this.previous.style.opacity = "0";
      this.drawStateFrame(this.current, this.state, now - this.stateStart);
      this.current.style.opacity = this.state === "offline" ? "0.72" : "1";
    }
  }

  function currentState() {
    return normalizeState(document.documentElement.dataset.state || "restarting");
  }

  async function loadImage(url) {
    return new Promise(function (resolve, reject) {
      const image = new Image();
      image.onload = function () { resolve(image); };
      image.onerror = reject;
      image.src = url;
    });
  }

  async function loadManifestAndSheets() {
    const response = await fetch(MANIFEST_URL, { cache: "no-store" });
    if (!response.ok) throw new Error("animation manifest unavailable");
    const manifest = await response.json();
    if (
      Number(manifest.version) !== 6 ||
      Number(manifest.frame_size) !== 128 ||
      !manifest.states ||
      !manifest.assets
    ) {
      throw new Error("invalid animation manifest");
    }

    for (const state of REQUIRED_STATES) {
      const spec = manifest.states[state];
      if (!spec || typeof spec.asset !== "string" || !manifest.assets[spec.asset]) {
        throw new Error("missing animation state: " + state);
      }
      const asset = manifest.assets[spec.asset];
      if (
        Number(asset.columns) < 1 ||
        Number(asset.rows) < 1 ||
        Number(asset.frames) !== Number(asset.columns) * Number(asset.rows) ||
        Number(spec.frame_ms) < 50 ||
        Number(spec.frame_ms) > 1000 ||
        Number(spec.representative_frame) < 0 ||
        Number(spec.representative_frame) >= Number(asset.frames)
      ) {
        throw new Error("invalid animation state: " + state);
      }
    }

    const assets = {};
    for (const [name, spec] of Object.entries(manifest.assets)) {
      if (typeof spec.file !== "string" || !/^[A-Za-z0-9._-]+\.png$/.test(spec.file)) {
        throw new Error("invalid animation asset path");
      }
      const image = await loadImage("avatar/anim/" + spec.file);
      if (image.naturalWidth < Number(spec.columns) * 32 || image.naturalHeight < Number(spec.rows) * 32) {
        throw new Error("invalid animation sheet dimensions: " + name);
      }
      const frames = [];
      for (let index = 0; index < Number(spec.frames); index += 1) {
        frames.push(alphaTrimRect(image, spec, index));
      }
      assets[name] = { image: image, frames: frames, spec: spec };
    }
    return { manifest: manifest, assets: assets };
  }

  loadManifestAndSheets().then(function (loaded) {
    const layers = canvases.map(function (canvas) {
      return new IllustratedAvatarLayer(canvas, loaded.manifest, loaded.assets);
    });

    const observer = new MutationObserver(function (records) {
      if (!records.some(function (record) { return record.attributeName === "data-state"; })) return;
      const state = currentState();
      const now = performance.now();
      layers.forEach(function (layer) { layer.setState(state, now); });
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-state"] });

    window.addEventListener("resize", function () {
      layers.forEach(function (layer) { layer.syncGeometry(); });
    });

    const state = currentState();
    const now = performance.now();
    layers.forEach(function (layer) { layer.enable(state, now); });

    function animate(timestamp) {
      layers.forEach(function (layer) { layer.draw(timestamp); });
      requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }).catch(function (error) {
    // Deliberately leave app.js' known-good pixel avatar visible.
    console.warn("LocalPilot: illustrated animation unavailable; using pixel fallback.", error);
  });
})();
