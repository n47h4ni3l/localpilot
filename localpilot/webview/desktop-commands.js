"use strict";

(function () {
  const input = document.getElementById("composer-input");
  const send = document.getElementById("send-btn");
  const composerWrap = document.querySelector(".composer-wrap");
  const historyNew = document.getElementById("history-new");
  if (!input || !send || !composerWrap || !historyNew) return;

  const commands = [
    { name: "/help", usage: "", description: "Show available LocalPilot commands" },
    { name: "/status", usage: "", description: "Current runtime, resources, evolution and Git status" },
    { name: "/doctor", usage: "", description: "Check LocalPilot prerequisites and runtime health" },
    { name: "/teach", usage: "<lesson>", description: "Save an explicit durable owner teaching" },
    { name: "/evolve", usage: "[--force]", description: "Run a self-development cycle; --force bypasses the idle gate" },
    { name: "/clear", usage: "", description: "Start a fresh conversation; keep durable learning" },
  ];

  const menu = document.createElement("div");
  menu.id = "command-menu";
  menu.className = "command-menu";
  menu.setAttribute("role", "listbox");
  menu.setAttribute("aria-label", "LocalPilot commands");
  menu.hidden = true;
  composerWrap.prepend(menu);

  let matches = [];
  let selected = 0;

  function commandPrefix() {
    const value = input.value.trimStart();
    if (!value.startsWith("/") || value.includes("\n") || /\s/.test(value)) return null;
    return value.toLowerCase();
  }

  function closeMenu() {
    menu.hidden = true;
    menu.replaceChildren();
    input.removeAttribute("aria-activedescendant");
    matches = [];
    selected = 0;
  }

  function setSelected(index) {
    if (!matches.length) return;
    selected = (index + matches.length) % matches.length;
    Array.from(menu.children).forEach(function (option, optionIndex) {
      const active = optionIndex === selected;
      option.classList.toggle("is-selected", active);
      option.setAttribute("aria-selected", String(active));
    });
    input.setAttribute("aria-activedescendant", "command-option-" + selected);
  }

  function choose(index) {
    const command = matches[index];
    if (!command) return;
    input.value = command.name + (command.usage ? " " : "");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    closeMenu();
    input.focus();
  }

  function renderMenu() {
    const prefix = commandPrefix();
    if (prefix === null) {
      closeMenu();
      return;
    }
    matches = commands.filter(function (command) {
      return command.name.startsWith(prefix) || prefix === "/";
    });
    if (!matches.length) {
      closeMenu();
      return;
    }
    selected = Math.min(selected, matches.length - 1);
    menu.replaceChildren();
    matches.forEach(function (command, index) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "command-menu__item" + (index === selected ? " is-selected" : "");
      option.id = "command-option-" + index;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", String(index === selected));

      const commandText = document.createElement("span");
      commandText.className = "command-menu__command";
      commandText.textContent = command.name + (command.usage ? " " + command.usage : "");
      const description = document.createElement("span");
      description.className = "command-menu__description";
      description.textContent = command.description;
      option.appendChild(commandText);
      option.appendChild(description);

      option.addEventListener("pointerdown", function (event) {
        // Keep focus in the composer so selecting a command feels like
        // autocompletion rather than opening a second control surface.
        event.preventDefault();
      });
      option.addEventListener("click", function () { choose(index); });
      option.addEventListener("mouseenter", function () { setSelected(index); });
      menu.appendChild(option);
    });
    menu.hidden = false;
    setSelected(selected);
  }

  function executeLocalCommand(text) {
    const command = text.trim().toLowerCase();
    if (command !== "/clear" && command !== "/new") return false;
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    closeMenu();
    historyNew.click();
    return true;
  }

  input.addEventListener("input", function () {
    selected = 0;
    renderMenu();
  });

  input.addEventListener("keydown", function (event) {
    const text = input.value.trim();

    if (event.key === "Escape" && !menu.hidden) {
      event.preventDefault();
      event.stopImmediatePropagation();
      closeMenu();
      return;
    }

    if (!menu.hidden && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
      event.preventDefault();
      event.stopImmediatePropagation();
      setSelected(selected + (event.key === "ArrowDown" ? 1 : -1));
      return;
    }

    if (!menu.hidden && event.key === "Tab") {
      event.preventDefault();
      event.stopImmediatePropagation();
      choose(selected);
      return;
    }

    if (!menu.hidden && event.key === "Enter" && !event.shiftKey) {
      const selectedCommand = matches[selected];
      const typedHead = text.split(/\s+/, 1)[0].toLowerCase();
      if (selectedCommand && typedHead !== selectedCommand.name) {
        event.preventDefault();
        event.stopImmediatePropagation();
        choose(selected);
        return;
      }
    }

    if (event.key === "Enter" && !event.shiftKey && executeLocalCommand(text)) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);

  send.addEventListener("click", function (event) {
    if (!executeLocalCommand(input.value)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);

  input.addEventListener("blur", function () {
    // Pointer selection uses pointerdown preventDefault, but the short delay
    // also keeps the menu stable for keyboard/mouse hand-offs.
    setTimeout(function () {
      if (document.activeElement !== input) closeMenu();
    }, 100);
  });
})();
