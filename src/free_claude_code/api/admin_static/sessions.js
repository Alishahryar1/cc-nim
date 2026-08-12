(() => {
  const api = window.FCCAdminApi;
  const PAGE_SIZE = 20;
  const state = {
    active: false,
    snapshot: null,
    sessionId: null,
    renameSessionId: null,
    page: 1,
    query: "",
    socket: null,
    socketSessionId: null,
    suspendedSessionId: null,
    poller: null,
    terminal: null,
    fitAddon: null,
    resizeObserver: null,
  };

  const byId = (id) => document.getElementById(id);

  function sessions() {
    return state.snapshot?.sessions || [];
  }

  function currentSession() {
    return sessions().find((session) => session.id === state.sessionId) || null;
  }

  function mount(container) {
    container.innerHTML = `
      <div class="session-nav-tools">
        <label class="session-search">
          <span class="sr-only">Search sessions</span>
          <input id="sessionSearch" type="search" placeholder="Search sessions" autocomplete="off" />
        </label>
        <button id="newSessionButton" class="session-add-button" type="button" aria-label="New session" title="New session">+</button>
      </div>
      <div id="sessionNavList" class="session-nav-list" aria-label="Coding sessions"></div>
      <div id="sessionPagination" class="session-pagination" hidden>
        <button id="previousSessionPage" type="button" aria-label="Previous sessions">‹</button>
        <span id="sessionPageLabel"></span>
        <button id="nextSessionPage" type="button" aria-label="Next sessions">›</button>
      </div>
    `;
    byId("newSessionButton").addEventListener("click", openSessionDialog);
    byId("sessionSearch").addEventListener("input", (event) => {
      state.query = event.target.value.trim().toLocaleLowerCase();
      state.page = 1;
      renderSidebar();
    });
    byId("previousSessionPage").addEventListener("click", () => {
      state.page -= 1;
      renderSidebar();
    });
    byId("nextSessionPage").addEventListener("click", () => {
      state.page += 1;
      renderSidebar();
    });
    renderSidebar();
  }

  async function activate() {
    if (state.active) return;
    state.active = true;
    await refresh();
    state.poller = window.setInterval(() => refresh({ quiet: true }), 2000);
  }

  function deactivate() {
    if (!state.active) return;
    state.active = false;
    window.clearInterval(state.poller);
    state.poller = null;
    closeSocket();
  }

  async function refresh({ quiet = false } = {}) {
    try {
      state.snapshot = await api("/admin/api/sessions");
      reconcileSelection();
      render();
    } catch (error) {
      if (!quiet) showMessage(error.message, "error");
    }
  }

  function reconcileSelection() {
    if (!sessions().some((session) => session.id === state.sessionId)) {
      state.sessionId = sessions()[0]?.id || null;
      state.suspendedSessionId = null;
    }
  }

  function render() {
    renderSidebar();
    renderWorkspace();
    if (
      state.active &&
      state.sessionId &&
      state.socketSessionId !== state.sessionId &&
      state.suspendedSessionId !== state.sessionId
    ) {
      connectTerminal();
    }
  }

  function matchingGroups() {
    const groups = new Map();
    sessions().forEach((session) => {
      const searchable = [
        session.name,
        session.project_name,
        session.path,
        harnessLabel(session.harness),
      ]
        .join(" ")
        .toLocaleLowerCase();
      if (state.query && !searchable.includes(state.query)) return;
      if (!groups.has(session.path)) {
        groups.set(session.path, {
          name: session.project_name,
          path: session.path,
          sessions: [],
        });
      }
      groups.get(session.path).sessions.push(session);
    });
    return Array.from(groups.values());
  }

  function renderSidebar() {
    const list = byId("sessionNavList");
    const create = byId("newSessionButton");
    if (!list || !create) return;
    list.innerHTML = "";
    const snapshot = state.snapshot;
    const hasHarness = snapshot?.harnesses.some((harness) => harness.available);
    create.disabled = !snapshot?.available || !hasHarness;
    byId("emptyCreateSessionButton").disabled = create.disabled;
    if (snapshot && !snapshot.available) {
      showMessage(snapshot.message || "Browser Sessions is unavailable.", "error");
    }

    const groups = matchingGroups();
    const flattened = groups.flatMap((group) => group.sessions);
    const pageCount = Math.max(1, Math.ceil(flattened.length / PAGE_SIZE));
    state.page = Math.min(Math.max(1, state.page), pageCount);
    const first = (state.page - 1) * PAGE_SIZE;
    const pageSessions = flattened.slice(first, first + PAGE_SIZE);
    const pageGroups = groupSessions(pageSessions);

    if (pageGroups.length === 0) {
      const empty = document.createElement("p");
      empty.className = "session-nav-empty";
      empty.textContent = state.query ? "No matching sessions" : "No sessions yet";
      list.appendChild(empty);
    } else {
      pageGroups.forEach((group) => list.appendChild(renderGroup(group)));
    }

    const pagination = byId("sessionPagination");
    pagination.hidden = pageCount <= 1;
    byId("sessionPageLabel").textContent = `${state.page} / ${pageCount}`;
    byId("previousSessionPage").disabled = state.page === 1;
    byId("nextSessionPage").disabled = state.page === pageCount;
  }

  function groupSessions(items) {
    const groups = new Map();
    items.forEach((session) => {
      if (!groups.has(session.path)) {
        groups.set(session.path, {
          name: session.project_name,
          path: session.path,
          sessions: [],
        });
      }
      groups.get(session.path).sessions.push(session);
    });
    return Array.from(groups.values());
  }

  function renderGroup(group) {
    const section = document.createElement("section");
    section.className = "session-folder-group";
    const heading = document.createElement("div");
    heading.className = "session-folder-heading";
    heading.title = group.path;
    const name = document.createElement("strong");
    name.textContent = group.name;
    const path = document.createElement("span");
    path.textContent = group.path;
    heading.append(name, path);
    section.appendChild(heading);
    group.sessions.forEach((session) => section.appendChild(renderSessionRow(session)));
    return section;
  }

  function renderSessionRow(session) {
    const row = document.createElement("div");
    row.className = `session-nav-row${session.id === state.sessionId ? " active" : ""}`;

    const select = document.createElement("button");
    select.type = "button";
    select.className = "session-nav-select";
    select.setAttribute("aria-current", session.id === state.sessionId ? "page" : "false");
    const dot = document.createElement("span");
    dot.className = `session-state-dot ${session.state}`;
    dot.title = session.state;
    const copy = document.createElement("span");
    copy.className = "session-nav-copy";
    const label = document.createElement("strong");
    label.textContent = session.name;
    const harness = document.createElement("span");
    harness.textContent = harnessLabel(session.harness);
    copy.append(label, harness);
    select.append(dot, copy);
    select.addEventListener("click", () => selectSession(session.id));

    const menu = document.createElement("details");
    menu.className = "session-row-menu";
    const summary = document.createElement("summary");
    summary.textContent = "⋯";
    summary.setAttribute("aria-label", `Actions for ${session.name}`);
    summary.title = `Actions for ${session.name}`;
    const actions = document.createElement("div");
    actions.className = "session-row-menu-actions";
    const transitioning = ["starting", "stopping"].includes(session.state);
    const lifecycle = menuButton(
      session.state === "running" || session.state === "starting" ? "Stop" : "Start",
      () => sessionAction(
        session.id,
        session.state === "running" || session.state === "starting" ? "stop" : "start",
      ),
    );
    lifecycle.disabled = transitioning;
    const rename = menuButton("Rename", () => openRenameDialog(session));
    rename.disabled = transitioning;
    const remove = menuButton("Delete", () => deleteSession(session), "danger");
    remove.disabled = transitioning;
    actions.append(lifecycle, rename, remove);
    actions.addEventListener("click", () => {
      menu.open = false;
    });
    menu.append(summary, actions);
    row.append(select, menu);
    return row;
  }

  function menuButton(label, action, kind = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    if (kind) button.className = kind;
    button.addEventListener("click", action);
    return button;
  }

  function renderWorkspace() {
    const session = currentSession();
    const empty = byId("sessionsEmpty");
    const panel = byId("terminalPanel");
    if (!session) {
      empty.hidden = false;
      panel.hidden = true;
      return;
    }

    empty.hidden = true;
    panel.hidden = false;
    byId("sessionName").textContent = session.name;
    byId("sessionHarness").textContent = harnessLabel(session.harness);
    byId("sessionPath").textContent = session.path;
    byId("sessionPath").title = session.path;
    const status = byId("sessionStatus");
    status.textContent = session.state;
    status.className = `status-pill ${statusKind(session.state)}`;
    updateTerminalInput(session);
  }

  function updateTerminalInput(session) {
    if (state.terminal) state.terminal.options.disableStdin = session?.state !== "running";
    const running = session?.state === "running";
    const starting = session?.state === "starting";
    const overlay = byId("terminalClosed");
    overlay.hidden = running || starting;
    byId("terminalClosedMessage").textContent =
      session?.detail || "Terminal is closed. Start again to resume this session.";
    byId("restartSessionButton").disabled = !session || starting || session.state === "stopping";
  }

  function selectSession(sessionId) {
    state.suspendedSessionId = null;
    if (state.sessionId === sessionId) {
      render();
      return;
    }
    state.sessionId = sessionId;
    closeSocket();
    resetTerminal();
    render();
  }

  function ensureTerminal() {
    if (state.terminal) return;
    state.terminal = new window.Terminal({
      cursorBlink: true,
      fontFamily: '"Cascadia Mono", "SFMono-Regular", Consolas, monospace',
      fontSize: 14,
      scrollback: 5000,
      theme: {
        background: "#08090d",
        foreground: "#e5e7eb",
        cursor: "#f59e0b",
        selectionBackground: "#374151",
      },
    });
    state.fitAddon = new window.FitAddon.FitAddon();
    state.terminal.loadAddon(state.fitAddon);
    state.terminal.open(byId("terminal"));
    state.terminal.onData((data) => {
      if (state.socket?.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify({ type: "input", data }));
      }
    });
    state.resizeObserver = new ResizeObserver(() => fitTerminal());
    state.resizeObserver.observe(byId("terminalViewport"));
  }

  function resetTerminal() {
    if (!state.terminal) return;
    state.terminal.reset();
    state.terminal.clear();
  }

  function connectTerminal() {
    closeSocket();
    ensureTerminal();
    resetTerminal();
    const sessionId = state.sessionId;
    if (!sessionId) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(
      `${protocol}//${window.location.host}/admin/api/sessions/${encodeURIComponent(sessionId)}/terminal`,
    );
    socket.binaryType = "arraybuffer";
    state.socket = socket;
    state.socketSessionId = sessionId;
    socket.addEventListener("open", () => fitTerminal());
    socket.addEventListener("message", (event) => {
      if (socket !== state.socket) return;
      if (event.data instanceof ArrayBuffer) {
        state.terminal.write(new Uint8Array(event.data));
        return;
      }
      handleControlMessage(event.data);
    });
    socket.addEventListener("close", () => {
      if (socket === state.socket) {
        state.socket = null;
        state.socketSessionId = null;
      }
    });
  }

  function handleControlMessage(raw) {
    let event;
    try {
      event = JSON.parse(raw);
    } catch {
      return;
    }
    const session = currentSession();
    if (session && event.state) {
      session.state = event.state;
      session.detail = event.detail;
      renderSidebar();
      renderWorkspace();
    }
    if (event.type === "replaced") {
      state.suspendedSessionId = state.sessionId;
      showMessage("Terminal opened in another tab. Select this session to take control.", "warn");
    }
    if (event.type === "deleted") {
      state.suspendedSessionId = state.sessionId;
      showMessage("Session was deleted.", "warn");
    }
    if (event.type === "error") {
      showMessage(event.detail || "Terminal connection ended.", "error");
    }
  }

  function fitTerminal() {
    if (!state.fitAddon || byId("terminalPanel").hidden) return;
    window.requestAnimationFrame(() => {
      state.fitAddon.fit();
      if (state.socket?.readyState === WebSocket.OPEN) {
        state.socket.send(
          JSON.stringify({
            type: "resize",
            columns: state.terminal.cols,
            rows: state.terminal.rows,
          }),
        );
      }
    });
  }

  function closeSocket() {
    const socket = state.socket;
    state.socket = null;
    state.socketSessionId = null;
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
  }

  function openSessionDialog() {
    const select = byId("newSessionHarness");
    select.innerHTML = "";
    (state.snapshot?.harnesses || []).forEach((harness) => {
      const option = document.createElement("option");
      option.value = harness.id;
      option.textContent = harness.available
        ? harnessLabel(harness.id)
        : `${harnessLabel(harness.id)} — unavailable`;
      option.disabled = !harness.available;
      option.dataset.message = harness.message || "";
      select.appendChild(option);
    });
    const first = Array.from(select.options).find((option) => !option.disabled);
    select.value = first?.value || "";
    byId("createSessionButton").disabled = !first;
    updateHarnessHint();
    byId("sessionDialog").showModal();
    byId("newSessionPath").focus();
  }

  function updateHarnessHint() {
    const option = byId("newSessionHarness").selectedOptions[0];
    byId("harnessHint").textContent = option?.dataset.message || "";
  }

  async function submitSession(event) {
    event.preventDefault();
    const suppliedName = byId("newSessionName").value.trim();
    try {
      const session = await api("/admin/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          path: byId("newSessionPath").value,
          harness: byId("newSessionHarness").value,
          name: suppliedName || null,
        }),
      });
      byId("sessionDialog").close();
      byId("sessionForm").reset();
      state.query = "";
      state.page = 1;
      byId("sessionSearch").value = "";
      state.sessionId = session.id;
      state.suspendedSessionId = null;
      closeSocket();
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  function openRenameDialog(session) {
    state.renameSessionId = session.id;
    byId("renameSessionName").value = session.name;
    byId("renameDialog").showModal();
    byId("renameSessionName").select();
  }

  async function submitRename(event) {
    event.preventDefault();
    if (!state.renameSessionId) return;
    try {
      await api(`/admin/api/sessions/${state.renameSessionId}`, {
        method: "PATCH",
        body: JSON.stringify({ name: byId("renameSessionName").value }),
      });
      state.renameSessionId = null;
      byId("renameDialog").close();
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  async function sessionAction(sessionId, action) {
    try {
      await api(`/admin/api/sessions/${sessionId}/${action}`, {
        method: "POST",
        body: "{}",
      });
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  async function deleteSession(session) {
    const confirmed = window.confirm(
      `Delete “${session.name}” from FCC? Its process will stop, but its native ${harnessLabel(session.harness)} history will be kept.`,
    );
    if (!confirmed) return;
    try {
      await api(`/admin/api/sessions/${session.id}`, { method: "DELETE" });
      if (state.sessionId === session.id) {
        closeSocket();
        state.sessionId = null;
        state.suspendedSessionId = null;
        resetTerminal();
      }
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  function harnessLabel(harness) {
    return { claude: "Claude Code", pi: "Pi" }[harness] || harness;
  }

  function statusKind(status) {
    if (status === "running") return "ok";
    if (["starting", "stopping"].includes(status)) return "warn";
    if (status === "failed") return "error";
    return "neutral";
  }

  function showMessage(message, kind = "") {
    const area = byId("sessionsMessage");
    area.textContent = message;
    area.className = `sessions-message ${kind}`.trim();
  }

  byId("emptyCreateSessionButton").addEventListener("click", openSessionDialog);
  byId("restartSessionButton").addEventListener("click", () => {
    if (state.sessionId) sessionAction(state.sessionId, "start");
  });
  byId("sessionForm").addEventListener("submit", submitSession);
  byId("renameForm").addEventListener("submit", submitRename);
  byId("newSessionHarness").addEventListener("change", updateHarnessHint);
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => byId(button.dataset.closeDialog).close());
  });

  window.FCCSessions = { mount, activate, deactivate };
})();
