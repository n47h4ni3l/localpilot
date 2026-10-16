(function (global) {
  "use strict";

  const LABELS = {
    get_system_summary: "Checked system information",
    get_system_sense_summary: "Checked SystemSense",
    get_system_sense_history: "Checked SystemSense history",
    get_systemsense_watch_report: "Checked a SystemSense watch",
    get_storage_summary: "Checked disk space",
    get_top_processes: "Checked running processes",
    get_startup_items: "Checked startup items",
    get_active_power_plan: "Checked power settings",
    get_defender_summary: "Checked Defender status",
    get_device_problem_summary: "Checked device issues",
    get_library_summary: "Checked the library",
    search_library: "Searched the library",
    read_library_passage: "Read a library passage",
    get_recent_library_reading_notes: "Checked recent reading notes",
    get_learning_memory_summary: "Checked learning memory",
    search_public_web: "Searched the web",
    fetch_public_https: "Read a web page",
    list_repository_tree: "Listed project files",
    read_repository_file: "Read a project file",
    search_repository: "Searched the project",
    get_repository_status: "Checked Git status",
    list_github_pull_requests: "Listed pull requests",
    get_github_pull_request: "Read a pull request",
    get_github_pull_request_diff: "Read pull request changes",
    list_github_issues: "Listed issues",
    get_github_issue: "Read an issue",
    open_windows_app: "Opened an app",
    open_windows_settings: "Opened Settings",
    set_active_power_plan: "Changed power plan",
    restore_power_plan: "Restored power plan",
  };
  const STATES = { running: "In progress", succeeded: "Succeeded", failed: "Failed", interrupted: "Interrupted" };

  function label(tool) { return LABELS[tool] || String(tool || "Unknown tool"); }

  // A stable invocation ID distinguishes repeated calls to the same tool and
  // makes history replay and overlapping poll responses idempotent.
  function upsert(steps, incoming) {
    const result = Array.isArray(steps) ? steps.slice() : [];
    if (!incoming || incoming.id == null || !incoming.tool) return result;
    const id = String(incoming.id);
    const index = result.findIndex(function (step) { return String(step.id) === id; });
    if (index < 0) result.push(Object.assign({}, incoming));
    else if (result[index].status !== "running" && incoming.status === "running") return result;
    else result[index] = Object.assign({}, result[index], incoming);
    return result;
  }

  function preview(value) {
    if (value == null) return "";
    const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    return text.length > 3000 ? text.slice(0, 3000) + "\n… Preview truncated" : text;
  }

  function duration(step) {
    const ms = Number(step.duration_ms);
    if (step.duration_ms == null || !Number.isFinite(ms) || ms < 0) return "";
    return ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s";
  }

  function addPreview(document, container, title, value) {
    const content = preview(value);
    if (!content) return;
    const heading = document.createElement("div");
    heading.className = "chat-activity__heading";
    heading.textContent = title;
    const pre = document.createElement("pre");
    pre.className = "chat-activity__preview";
    pre.tabIndex = 0;
    pre.setAttribute("aria-label", title);
    pre.textContent = content;
    container.appendChild(heading);
    container.appendChild(pre);
  }

  function render(container, steps) {
    const document = container.ownerDocument || global.document;
    const list = Array.isArray(steps) ? steps : [];
    const signature = JSON.stringify(list);
    if (container.activitySignature === signature) return;
    container.activitySignature = signature;
    const previous = container.querySelector(".chat-activity");
    const wasOpen = !!(previous && previous.open);
    const openRows = new Set(previous ? Array.from(previous.querySelectorAll(".chat-activity__step"))
      .filter(function (row) { return row.open; }).map(function (row) { return row.dataset.stepId; }) : []);
    const focused = document.activeElement;
    const focusedRow = previous && Array.from(previous.querySelectorAll(".chat-activity__step"))
      .find(function (row) { return row.contains(focused); });
    const focusSummary = previous && focused === previous.firstElementChild;
    container.replaceChildren();
    if (!list.length) return;
    const details = document.createElement("details");
    details.className = "chat-activity";
    details.open = wasOpen;
    const summary = document.createElement("summary");
    const failed = list.filter(function (step) { return step.status === "failed" || step.status === "interrupted"; }).length;
    const running = list.some(function (step) { return step.status === "running"; });
    summary.textContent = list.length + " step" + (list.length === 1 ? "" : "s") +
      (running ? " · In progress" : "") + (failed ? " · " + failed + " unsuccessful" : "");
    details.appendChild(summary);
    list.forEach(function (step) {
      const row = document.createElement("details");
      row.className = "chat-activity__step";
      row.dataset.stepId = String(step.id);
      row.dataset.status = STATES[step.status] ? step.status : "interrupted";
      row.open = openRows.has(row.dataset.stepId);
      const heading = document.createElement("summary");
      heading.textContent = label(step.tool) + " — " + (STATES[step.status] || "Unknown status") +
        (duration(step) ? " · " + duration(step) : "");
      row.appendChild(heading);
      const tool = document.createElement("div");
      tool.className = "chat-activity__tool";
      tool.textContent = step.tool;
      row.appendChild(tool);
      addPreview(document, row, "Requested", step.arguments_preview);
      addPreview(document, row, "Result", step.result_preview);
      addPreview(document, row, "Error", step.error);
      if (!step.arguments_preview && !step.result_preview && !step.error) {
        const empty = document.createElement("div");
        empty.className = "chat-activity__tool";
        empty.textContent = step.status === "running" ? "Waiting for the tool result." : "No additional details were recorded.";
        row.appendChild(empty);
      }
      details.appendChild(row);
    });
    container.appendChild(details);
    if (focusSummary) summary.focus();
    else if (focusedRow) {
      const replacement = Array.from(details.querySelectorAll(".chat-activity__step"))
        .find(function (row) { return row.dataset.stepId === focusedRow.dataset.stepId; });
      if (replacement) replacement.firstElementChild.focus();
    }
  }

  global.LocalPilotActivity = { render: render, upsert: upsert, label: label };
})(window);
