(() => {
  const api = window.FCCAdminApi;
  const state = {
    active: false,
    snapshot: null,
    projectId: null,
    sessionId: null,
    socket: null,
    socketSessionId: null,
    suspendedSessionId: null,
    poller: null,
    terminal: null,
    fitAddon: null,
    resizeObserver: null,
  };

  const byId = (id) => document.getElementById(id);

  function currentProject() {
    return state.snapshot?.projects.find((project) => project.id === state.projectId) || null;
  }

  function currentSession() {
    return currentProject()?.sessions.find((session) => session.id === state.sessionId) || null;
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
      const snapshot = await api("/admin/api/sessions");
      state.snapshot = snapshot;
      reconcileSelection();
      render();
    } catch (error) {
      if (!quiet) showMessage(error.message, "error");
    }
  }

  function reconcileSelection() {
    const projects = state.snapshot?.projects || [];
    if (!projects.some((project) => project.id === state.projectId)) {
      state.projectId = projects[0]?.id || null;
      state.sessionId = null;
    }
    const project = currentProject();
    if (!project?.sessions.some((session) => session.id === state.sessionId)) {
      state.sessionId = project?.sessions[0]?.id || null;
      state.suspendedSessionId = null;
    }
  }

  function render() {
    renderProjects();
    renderSessionTabs();
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

  function renderProjects() {
    const list = byId("projectList");
    list.innerHTML = "";
    const snapshot = state.snapshot;
    byId("addProjectButton").disabled = snapshot ? !snapshot.available : true;
    if (!snapshot?.available) {
      showMessage(snapshot?.message || "Browser Sessions is unavailable.", "error");
    }
    (snapshot?.projects || []).forEach((project) => {
      const row = document.createElement("div");
      row.className = `project-row${project.id === state.projectId ? " active" : ""}`;

      const select = document.createElement("button");
      select.type = "button";
      select.className = "project-select";
      select.title = project.path;
      const name = document.createElement("strong");
      name.textContent = project.name;
      const count = document.createElement("span");
      count.textContent = `${project.sessions.length} session${project.sessions.length === 1 ? "" : "s"}`;
      select.append(name, count);
      select.addEventListener("click", () => selectProject(project.id));

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "project-remove";
      remove.textContent = "×";
      remove.title = `Remove ${project.name}`;
      remove.setAttribute("aria-label", `Remove ${project.name}`);
      remove.addEventListener("click", () => deleteProject(project));
      row.append(select, remove);
      list.appendChild(row);
    });
  }

  function renderSessionTabs() {
    const tabs = byId("sessionTabs");
    tabs.innerHTML = "";
    const project = currentProject();
    byId("newSessionButton").disabled = !project || !state.snapshot?.available;
    (project?.sessions || []).forEach((session) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = `session-tab${session.id === state.sessionId ? " active" : ""}`;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(session.id === state.sessionId));
      const dot = document.createElement("span");
      dot.className = `session-state-dot ${session.state}`;
      const label = document.createElement("span");
      label.textContent = session.name;
      tab.append(dot, label);
      tab.addEventListener("click", () => selectSession(session.id));
      tabs.appendChild(tab);
    });
  }

  function renderWorkspace() {
    const project = currentProject();
    const session = currentSession();
    const empty = byId("sessionsEmpty");
    const panel = byId("terminalPanel");
    if (!project) {
      empty.hidden = false;
      empty.querySelector("h3").textContent = "Add a project to begin";
      empty.querySelector("p").textContent =
        "Choose a local folder, then run Claude Code, Codex, or Pi in it.";
      panel.hidden = true;
      return;
    }
    if (!session) {
      empty.hidden = false;
      empty.querySelector("h3").textContent = "Create a session";
      empty.querySelector("p").textContent =
        `Start a native coding harness inside ${project.path}.`;
      panel.hidden = true;
      return;
    }

    empty.hidden = true;
    panel.hidden = false;
    byId("sessionName").textContent = session.name;
    byId("sessionHarness").textContent = harnessLabel(session.harness);
    const status = byId("sessionStatus");
    status.textContent = session.state;
    status.className = `status-pill ${statusKind(session.state)}`;
    byId("sessionDetail").textContent = session.detail || currentProject().path;

    const running = session.state === "running";
    const transitioning = ["starting", "stopping"].includes(session.state);
    byId("startSessionButton").hidden = running || transitioning;
    byId("stopSessionButton").hidden = !running && session.state !== "starting";
    byId("stopSessionButton").disabled = transitioning;
    byId("renameSessionButton").disabled = transitioning;
    byId("deleteSessionButton").disabled = transitioning;
    updateTerminalInput(session);
  }

  function updateTerminalInput(session) {
    if (!state.terminal) return;
    const running = session?.state === "running";
    state.terminal.options.disableStdin = !running;
    const overlay = byId("terminalClosed");
    overlay.hidden = running || session?.state === "starting";
    overlay.textContent = session?.detail || "Terminal is closed. Start again to resume.";
  }

  function selectProject(projectId) {
    if (state.projectId === projectId) return;
    state.projectId = projectId;
    state.sessionId = currentProject()?.sessions[0]?.id || null;
    state.suspendedSessionId = null;
    closeSocket();
    resetTerminal();
    render();
  }

  function selectSession(sessionId) {
    state.sessionId = sessionId;
    state.suspendedSessionId = null;
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
      renderSessionTabs();
      renderWorkspace();
    }
    if (event.type === "replaced") {
      state.suspendedSessionId = state.sessionId;
      showMessage("Terminal opened in another tab. Click its tab to take control.", "warn");
    }
    if (event.type === "deleted") {
      state.suspendedSessionId = state.sessionId;
      showMessage("Session was deleted.", "warn");
    }
    if (event.type === "error") showMessage(event.detail || "Terminal connection ended.", "error");
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

  async function submitProject(event) {
    event.preventDefault();
    const path = byId("projectPath").value;
    try {
      const project = await api("/admin/api/projects", {
        method: "POST",
        body: JSON.stringify({ path }),
      });
      byId("projectDialog").close();
      byId("projectForm").reset();
      state.projectId = project.id;
      state.sessionId = null;
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
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
    byId("newSessionName").focus();
  }

  function updateHarnessHint() {
    const option = byId("newSessionHarness").selectedOptions[0];
    byId("harnessHint").textContent = option?.dataset.message || "";
  }

  async function submitSession(event) {
    event.preventDefault();
    if (!state.projectId) return;
    try {
      const session = await api(`/admin/api/projects/${state.projectId}/sessions`, {
        method: "POST",
        body: JSON.stringify({
          name: byId("newSessionName").value,
          harness: byId("newSessionHarness").value,
        }),
      });
      byId("sessionDialog").close();
      byId("sessionForm").reset();
      state.sessionId = session.id;
      state.suspendedSessionId = null;
      closeSocket();
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  function openRenameDialog() {
    const session = currentSession();
    if (!session) return;
    byId("renameSessionName").value = session.name;
    byId("renameDialog").showModal();
    byId("renameSessionName").select();
  }

  async function submitRename(event) {
    event.preventDefault();
    if (!state.sessionId) return;
    try {
      await api(`/admin/api/sessions/${state.sessionId}`, {
        method: "PATCH",
        body: JSON.stringify({ name: byId("renameSessionName").value }),
      });
      byId("renameDialog").close();
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  async function sessionAction(action) {
    if (!state.sessionId) return;
    try {
      await api(`/admin/api/sessions/${state.sessionId}/${action}`, {
        method: "POST",
        body: "{}",
      });
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  async function deleteSession() {
    const session = currentSession();
    if (!session || !window.confirm(`Delete session “${session.name}”?`)) return;
    try {
      await api(`/admin/api/sessions/${session.id}`, { method: "DELETE" });
      closeSocket();
      state.sessionId = null;
      state.suspendedSessionId = null;
      resetTerminal();
      await refresh();
    } catch (error) {
      showMessage(error.message, "error");
    }
  }

  async function deleteProject(project) {
    if (!window.confirm(`Remove project “${project.name}” and its FCC sessions?`)) return;
    try {
      await api(`/admin/api/projects/${project.id}`, { method: "DELETE" });
      if (state.projectId === project.id) {
        closeSocket();
        state.projectId = null;
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
    return { claude: "Claude Code", codex: "Codex", pi: "Pi" }[harness] || harness;
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

  byId("addProjectButton").addEventListener("click", () => {
    byId("projectDialog").showModal();
    byId("projectPath").focus();
  });
  byId("newSessionButton").addEventListener("click", openSessionDialog);
  byId("renameSessionButton").addEventListener("click", openRenameDialog);
  byId("startSessionButton").addEventListener("click", () => sessionAction("start"));
  byId("stopSessionButton").addEventListener("click", () => sessionAction("stop"));
  byId("deleteSessionButton").addEventListener("click", deleteSession);
  byId("projectForm").addEventListener("submit", submitProject);
  byId("sessionForm").addEventListener("submit", submitSession);
  byId("renameForm").addEventListener("submit", submitRename);
  byId("newSessionHarness").addEventListener("change", updateHarnessHint);
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => byId(button.dataset.closeDialog).close());
  });

  window.FCCSessions = { activate, deactivate };
})();
