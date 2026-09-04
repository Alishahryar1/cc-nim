(() => {
  "use strict";

  const RECONNECT_DELAYS_MS = [250, 500, 1000, 2000, 5000];
  const state = {
    api: null,
    active: false,
    routeGeneration: 0,
    socketGeneration: 0,
    currentSession: null,
    socket: null,
    reconnectAttempt: 0,
    reconnectTimer: null,
    terminal: null,
    fitAddon: null,
    resizeObserver: null,
    disposables: [],
    resizeTimer: null,
    lastSentRows: null,
    lastSentColumns: null,
    applyingCanonicalSize: false,
    role: null,
  };

  const root = () => document.getElementById("terminalRoot");

  function routedSessionId(path) {
    const match = /^\/admin\/terminal\/([^/?#]+)\/?$/.exec(path);
    if (!match) return null;
    try {
      return decodeURIComponent(match[1]);
    } catch {
      return null;
    }
  }

  async function initialize(api) {
    state.api = api;
    if (state.active) await route(window.location.pathname);
  }

  async function activate(path) {
    state.active = true;
    if (state.api) await route(path);
  }

  function deactivate() {
    if (!state.active && !state.terminal && !state.socket) return;
    state.active = false;
    state.routeGeneration += 1;
    teardownDetail();
  }

  async function route(path) {
    const generation = ++state.routeGeneration;
    teardownDetail();
    const sessionId = routedSessionId(path);
    renderLoading();
    try {
      if (sessionId) {
        const session = await state.api(
          `/admin/api/terminal/sessions/${encodeURIComponent(sessionId)}`,
        );
        if (!isCurrentRoute(generation)) return;
        renderDetail(session);
        return;
      }
      const payload = await state.api("/admin/api/terminal/sessions");
      if (!isCurrentRoute(generation)) return;
      if (payload.available === false) {
        renderFailure(payload.error || "Rerun the FCC installer and restart FCC.");
        return;
      }
      renderLibrary(payload.sessions || []);
    } catch (error) {
      if (!isCurrentRoute(generation)) return;
      if (sessionId && error.status === 404) {
        window.history.replaceState({}, "", "/admin/terminal");
        renderLibrary([], "That Terminal Session no longer exists.");
        return;
      }
      renderFailure(error.message || "Could not load Terminal Sessions.");
    }
  }

  function isCurrentRoute(generation) {
    return state.active && generation === state.routeGeneration;
  }

  function renderLoading() {
    const container = root();
    if (!container) return;
    container.innerHTML = '<div class="terminal-empty"><p>Loading terminals…</p></div>';
  }

  function renderFailure(message) {
    const container = root();
    if (!container) return;
    container.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "terminal-empty";
    const heading = document.createElement("h3");
    heading.textContent = "Terminal Sessions unavailable";
    const detail = document.createElement("p");
    detail.textContent = message;
    const retry = button("Retry", "secondary-button", () => {
      void route(window.location.pathname);
    });
    empty.append(heading, detail, retry);
    container.appendChild(empty);
  }

  function renderLibrary(sessions, notice = "") {
    const container = root();
    if (!container) return;
    container.innerHTML = "";

    const library = document.createElement("div");
    library.className = "terminal-library";
    const content = document.createElement("div");
    content.className = "terminal-library-content";
    const header = document.createElement("header");
    header.className = "terminal-library-header";
    const copy = document.createElement("div");
    const heading = document.createElement("h2");
    heading.textContent = "Terminal Sessions";
    const description = document.createElement("p");
    description.textContent =
      "Run a native shell that stays alive while you navigate or close this page.";
    copy.append(heading, description);
    const create = button("New terminal", "primary-button", async () => {
      create.disabled = true;
      try {
        const session = await state.api("/admin/api/terminal/sessions", {
          method: "POST",
        });
        openSession(session.id);
      } catch (error) {
        create.disabled = false;
        renderLibrary(sessions, error.message || "Could not create a terminal.");
      }
    });
    header.append(copy, create);
    content.appendChild(header);

    if (notice) {
      const message = document.createElement("p");
      message.className = "terminal-notice warn";
      message.textContent = notice;
      content.appendChild(message);
    }

    if (sessions.length === 0) {
      const empty = document.createElement("div");
      empty.className = "terminal-empty";
      const emptyHeading = document.createElement("h3");
      emptyHeading.textContent = "No terminals yet";
      const emptyText = document.createElement("p");
      emptyText.textContent = "Create one to open your system shell.";
      empty.append(emptyHeading, emptyText);
      content.appendChild(empty);
    } else {
      const list = document.createElement("div");
      list.className = "terminal-session-list";
      sessions.forEach((session) => list.appendChild(sessionCard(session)));
      content.appendChild(list);
    }

    library.appendChild(content);
    container.appendChild(library);
  }

  function sessionCard(session) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "terminal-card";
    card.addEventListener("click", () => openSession(session.id));
    const heading = document.createElement("div");
    heading.className = "terminal-card-heading";
    const name = document.createElement("strong");
    name.textContent = session.name;
    const status = statusPill(session.status);
    heading.append(name, status);
    const meta = document.createElement("p");
    meta.textContent = sessionMeta(session);
    card.append(heading, meta);
    return card;
  }

  function sessionMeta(session) {
    if (session.status === "exited") {
      if (session.error) return session.error;
      return session.exit_code == null
        ? "Shell exited"
        : `Shell exited with code ${session.exit_code}`;
    }
    return `Started ${new Date(session.created_at).toLocaleString()}`;
  }

  function renderDetail(session) {
    const container = root();
    if (!container) return;
    state.currentSession = session;
    container.innerHTML = "";

    const shell = document.createElement("div");
    shell.className = "terminal-session-shell";
    const header = document.createElement("header");
    header.className = "terminal-detail-header";
    const identity = document.createElement("div");
    identity.className = "terminal-detail-identity";
    const back = button("← Terminals", "ghost-button", openLibrary);
    const name = document.createElement("input");
    name.className = "terminal-name";
    name.value = session.name;
    name.maxLength = 100;
    name.dataset.terminalName = "";
    name.setAttribute("aria-label", "Terminal name");
    const status = statusPill(session.status);
    status.dataset.terminalStatus = "";
    identity.append(back, name, status);

    const actions = document.createElement("div");
    actions.className = "terminal-detail-actions";
    const stop = button("Stop", "secondary-button", async () => {
      stop.disabled = true;
      try {
        updateSession(
          await state.api(
            `/admin/api/terminal/sessions/${encodeURIComponent(session.id)}/stop`,
            { method: "POST" },
          ),
        );
      } catch (error) {
        stop.disabled = false;
        setNotice(error.message || "Could not stop the terminal.", "error");
      }
    });
    stop.dataset.terminalStop = "";
    const remove = button("Delete", "danger-button", async () => {
      if (!window.confirm(`Delete “${state.currentSession?.name || session.name}”?`)) {
        return;
      }
      remove.disabled = true;
      try {
        await state.api(
          `/admin/api/terminal/sessions/${encodeURIComponent(session.id)}`,
          { method: "DELETE" },
        );
        openLibrary();
      } catch (error) {
        remove.disabled = false;
        setNotice(error.message || "Could not delete the terminal.", "error");
      }
    });
    actions.append(stop, remove);
    header.append(identity, actions);

    const notice = document.createElement("p");
    notice.className = "terminal-notice";
    notice.dataset.terminalNotice = "";
    const stage = document.createElement("div");
    stage.className = "terminal-stage";
    stage.setAttribute("aria-label", `${session.name} terminal`);
    const terminalHost = document.createElement("div");
    terminalHost.className = "terminal-host";
    stage.appendChild(terminalHost);
    shell.append(header, notice, stage);
    container.appendChild(shell);

    name.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        name.blur();
      } else if (event.key === "Escape") {
        name.value = state.currentSession?.name || session.name;
        name.blur();
      }
    });
    name.addEventListener("change", () => void renameSession(name));

    updateSession(session);
    openTerminal(terminalHost, session);
  }

  async function renameSession(input) {
    const session = state.currentSession;
    if (!session) return;
    const value = input.value.trim();
    if (!value || value === session.name) {
      input.value = session.name;
      return;
    }
    input.disabled = true;
    try {
      const updated = await state.api(
        `/admin/api/terminal/sessions/${encodeURIComponent(session.id)}`,
        { method: "PATCH", body: JSON.stringify({ name: value }) },
      );
      updateSession(updated);
      input.value = updated.name;
    } catch (error) {
      input.value = state.currentSession?.name || session.name;
      setNotice(error.message || "Could not rename the terminal.", "error");
    } finally {
      input.disabled = false;
    }
  }

  function openTerminal(host, session) {
    if (!window.Terminal || !window.FitAddon?.FitAddon) {
      setNotice("The terminal emulator could not be loaded.", "error");
      return;
    }
    const terminal = new window.Terminal({
      cursorBlink: true,
      cursorStyle: "bar",
      fontFamily:
        'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
      fontSize: 14,
      lineHeight: 1.15,
      scrollback: 10000,
      theme: {
        background: "#050608",
        foreground: "#f3f4f6",
        cursor: "#10b981",
        selectionBackground: "#334155",
      },
    });
    const fitAddon = new window.FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(host);
    state.terminal = terminal;
    state.fitAddon = fitAddon;

    state.disposables.push(
      terminal.onData(sendInput),
      terminal.onResize(({ cols, rows }) => {
        if (!state.applyingCanonicalSize) {
          sendResize(rows, cols);
        }
      }),
    );

    const claim = () => sendClaim();
    for (const eventName of ["keydown", "paste", "beforeinput", "pointerdown"]) {
      host.addEventListener(eventName, claim, true);
    }
    state.disposables.push({
      dispose: () => {
        for (const eventName of [
          "keydown",
          "paste",
          "beforeinput",
          "pointerdown",
        ]) {
          host.removeEventListener(eventName, claim, true);
        }
      },
    });

    const focusAndResize = () => {
      fitTerminal();
      sendResize(terminal.rows, terminal.cols);
    };
    host.addEventListener("focusin", focusAndResize);
    window.addEventListener("focus", focusAndResize);
    document.addEventListener("visibilitychange", focusAndResize);
    state.disposables.push({
      dispose: () => {
        host.removeEventListener("focusin", focusAndResize);
        window.removeEventListener("focus", focusAndResize);
        document.removeEventListener("visibilitychange", focusAndResize);
      },
    });
    state.resizeObserver = new ResizeObserver(() => {
      window.clearTimeout(state.resizeTimer);
      state.resizeTimer = window.setTimeout(() => {
        fitTerminal();
        sendProposedSize();
      }, 50);
    });
    state.resizeObserver.observe(host);

    fitTerminal();
    terminal.focus();
    updateTerminalInput(session.status);
    connectSocket(session.id);
  }

  function fitTerminal() {
    if (!state.terminal || !state.fitAddon || !state.active) return;
    const host = root()?.querySelector(".terminal-host");
    if (!host || host.clientWidth === 0 || host.clientHeight === 0) return;
    try {
      if (state.role !== "observer") state.fitAddon.fit();
    } catch {
      setNotice("The terminal could not fit the available space.", "error");
    }
  }

  function connectSocket(sessionId) {
    clearReconnect();
    state.lastSentRows = null;
    state.lastSentColumns = null;
    const generation = ++state.socketGeneration;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const dimensions = proposedDimensions();
    const query = new URLSearchParams({
      rows: String(dimensions.rows),
      columns: String(dimensions.columns),
    });
    const socket = new WebSocket(
      `${protocol}//${window.location.host}/admin/api/terminal/sessions/${encodeURIComponent(sessionId)}/attach?${query}`,
    );
    socket.binaryType = "arraybuffer";
    state.socket = socket;

    socket.addEventListener("open", () => {
      if (!isCurrentSocket(generation, sessionId)) return;
      state.reconnectAttempt = 0;
      setNotice("");
      fitTerminal();
      sendProposedSize();
    });
    socket.addEventListener("message", (event) => {
      if (!isCurrentSocket(generation, sessionId)) return;
      if (event.data instanceof ArrayBuffer) {
        state.terminal?.write(new Uint8Array(event.data));
        return;
      }
      handleControlMessage(event.data, sessionId);
    });
    socket.addEventListener("close", (event) => {
      if (!isCurrentSocket(generation, sessionId)) return;
      state.socket = null;
      if (event.code === 4404) {
        openLibrary();
        return;
      }
      scheduleReconnect(sessionId);
    });
    socket.addEventListener("error", () => {
      if (isCurrentSocket(generation, sessionId)) {
        setNotice("Terminal connection interrupted. Reconnecting…", "warn");
      }
    });
  }

  function isCurrentSocket(generation, sessionId) {
    return (
      state.active &&
      state.currentSession?.id === sessionId &&
      generation === state.socketGeneration
    );
  }

  function handleControlMessage(raw, sessionId) {
    let message;
    try {
      message = JSON.parse(raw);
    } catch {
      setNotice("Received an invalid terminal control message.", "error");
      return;
    }
    if (message.type === "attached") {
      state.role = message.role;
      updateSession(message.session);
      return;
    }
    if (message.type === "reset") {
      state.role = message.role;
      state.terminal?.reset();
      return;
    }
    if (message.type === "state") {
      updateSession(message.session);
      return;
    }
    if (message.type === "deleted") {
      if (state.currentSession?.id === sessionId) openLibrary();
      return;
    }
    if (message.type === "error") {
      setNotice(message.message || "Terminal operation failed.", "error");
    }
  }

  function sendInput(data) {
    if (
      state.currentSession?.status !== "running" ||
      state.socket?.readyState !== WebSocket.OPEN
    ) {
      return;
    }
    for (const chunk of utf8Chunks(data, 64 * 1024)) {
      state.socket.send(JSON.stringify({ type: "input", data: chunk }));
    }
  }

  function sendClaim() {
    if (
      state.currentSession?.status === "running" &&
      state.socket?.readyState === WebSocket.OPEN
    ) {
      state.socket.send(JSON.stringify({ type: "claim" }));
    }
  }

  function sendResize(rows, columns) {
    if (state.socket?.readyState !== WebSocket.OPEN) return;
    if (state.lastSentRows === rows && state.lastSentColumns === columns) return;
    state.lastSentRows = rows;
    state.lastSentColumns = columns;
    state.socket.send(JSON.stringify({ type: "resize", rows, columns }));
  }

  function sendProposedSize() {
    const dimensions = proposedDimensions();
    sendResize(dimensions.rows, dimensions.columns);
  }

  function proposedDimensions() {
    const proposed = state.fitAddon?.proposeDimensions?.();
    return {
      rows: proposed?.rows || state.terminal?.rows || 24,
      columns: proposed?.cols || state.terminal?.cols || 80,
    };
  }

  function utf8Chunks(value, limit) {
    const encoder = new TextEncoder();
    const chunks = [];
    let current = [];
    let size = 0;
    for (const character of value) {
      const characterSize = encoder.encode(character).byteLength;
      if (size + characterSize > limit && current.length > 0) {
        chunks.push(current.join(""));
        current = [];
        size = 0;
      }
      current.push(character);
      size += characterSize;
    }
    if (current.length > 0) chunks.push(current.join(""));
    return chunks;
  }

  function scheduleReconnect(sessionId) {
    if (!state.active || state.currentSession?.id !== sessionId) return;
    const delay =
      RECONNECT_DELAYS_MS[
        Math.min(state.reconnectAttempt, RECONNECT_DELAYS_MS.length - 1)
      ];
    state.reconnectAttempt += 1;
    setNotice("Terminal connection interrupted. Reconnecting…", "warn");
    state.reconnectTimer = window.setTimeout(() => connectSocket(sessionId), delay);
  }

  function clearReconnect() {
    if (state.reconnectTimer != null) {
      window.clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
  }

  function updateSession(session, { syncDimensions = true } = {}) {
    if (!session || state.currentSession?.id !== session.id) return;
    state.currentSession = session;
    const status = root()?.querySelector("[data-terminal-status]");
    if (status) {
      status.className = `terminal-status ${session.status}`;
      status.textContent = session.status;
    }
    const name = root()?.querySelector("[data-terminal-name]");
    if (name && document.activeElement !== name) name.value = session.name;
    const stop = root()?.querySelector("[data-terminal-stop]");
    if (stop) stop.disabled = session.status !== "running";
    if (
      syncDimensions &&
      state.terminal &&
      (state.terminal.rows !== session.rows ||
        state.terminal.cols !== session.columns)
    ) {
      state.applyingCanonicalSize = true;
      try {
        state.terminal.resize(session.columns, session.rows);
      } finally {
        state.applyingCanonicalSize = false;
      }
    }
    updateTerminalInput(session.status);
    if (session.error) setNotice(session.error, "error");
  }

  function updateTerminalInput(status) {
    if (!state.terminal) return;
    state.terminal.options.disableStdin = status !== "running";
    state.terminal.options.cursorBlink = status === "running";
  }

  function setNotice(message, kind = "") {
    const notice = root()?.querySelector("[data-terminal-notice]");
    if (!notice) return;
    notice.textContent = message;
    notice.className = `terminal-notice ${kind}`.trim();
  }

  function openSession(sessionId) {
    window.history.pushState(
      {},
      "",
      `/admin/terminal/${encodeURIComponent(sessionId)}`,
    );
    void route(window.location.pathname);
  }

  function openLibrary() {
    window.history.pushState({}, "", "/admin/terminal");
    void route(window.location.pathname);
  }

  function teardownDetail() {
    clearReconnect();
    window.clearTimeout(state.resizeTimer);
    state.resizeTimer = null;
    state.socketGeneration += 1;
    const socket = state.socket;
    state.socket = null;
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close();
    state.resizeObserver?.disconnect();
    state.resizeObserver = null;
    state.disposables.forEach((disposable) => disposable.dispose());
    state.disposables = [];
    state.terminal?.dispose();
    state.terminal = null;
    state.fitAddon = null;
    state.currentSession = null;
    state.reconnectAttempt = 0;
    state.lastSentRows = null;
    state.lastSentColumns = null;
    state.applyingCanonicalSize = false;
    state.role = null;
  }

  function statusPill(status) {
    const pill = document.createElement("span");
    pill.className = `terminal-status ${status}`;
    pill.textContent = status;
    return pill;
  }

  function button(label, className, action) {
    const element = document.createElement("button");
    element.type = "button";
    element.className = className;
    element.textContent = label;
    element.addEventListener("click", action);
    return element;
  }

  window.TerminalSessions = { initialize, activate, deactivate };
})();
