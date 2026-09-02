(() => {
  const state = {
    api: null,
    initialized: false,
    bootstrap: null,
    detail: null,
    items: new Map(),
    interactions: new Map(),
    libraryItems: [],
    libraryCursor: null,
    libraryQuery: "",
    routeVersion: 0,
    requestVersion: 0,
    feed: null,
    feedLastId: 0,
    feedGeneration: null,
    feedStatus: "connecting",
    eventBuffer: null,
    feedTimer: null,
    pendingCreateOperation: null,
    draft: "",
    draftSessionId: null,
    modelComboboxes: new Set(),
  };

  const WORK_EVENTS = [
    "session.created",
    "session.updated",
    "session.deleted",
    "session.status",
    "timeline.item",
    "interaction.created",
    "interaction.resolved",
    "operation.updated",
    "work.disconnected",
    "work.warning",
  ];

  const ACTIVE_STATUSES = new Set([
    "working",
    "waiting_for_approval",
    "waiting_for_input",
    "stopping",
    "deleting",
  ]);

  const root = () => document.getElementById("workRoot");

  function node(tag, className = "", text = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text) element.textContent = text;
    return element;
  }

  function button(label, className, action) {
    const element = node("button", className, label);
    element.type = "button";
    element.addEventListener("click", action);
    return element;
  }

  function uuid() {
    return crypto.randomUUID();
  }

  function workIsVisible() {
    return document.getElementById("view-work")?.classList.contains("active") || false;
  }

  function routedThreadId(path) {
    const match = path.match(/^\/admin\/work\/([^/]+)$/);
    return match ? decodeURIComponent(match[1]) : null;
  }

  async function initialize(api) {
    if (state.initialized) return;
    state.initialized = true;
    state.api = api;
    connectFeed();
    if (workIsVisible()) renderLoading();
  }

  function activate(path) {
    void route(path);
  }

  function connectFeed() {
    window.clearTimeout(state.feedTimer);
    state.feedTimer = null;
    state.feed?.close();
    state.feedStatus = state.feedLastId ? "reconnecting" : "connecting";
    const feed = new EventSource("/admin/api/work/events");
    state.feed = feed;
    feed.addEventListener("feed.ready", (event) => {
      if (state.feed === feed) void synchronizeFeed(event);
    });
    feed.addEventListener("feed.resync_required", () => {
      if (state.feed === feed) restartFeed();
    });
    WORK_EVENTS.forEach((eventName) => {
      feed.addEventListener(eventName, (event) => {
        if (state.feed === feed) receiveEvent(eventName, event);
      });
    });
    feed.addEventListener("error", () => {
      if (state.feed !== feed) return;
      state.feedStatus = "reconnecting";
      refreshComposerState();
    });
  }

  async function synchronizeFeed(event) {
    let payload;
    const cursor = Number.parseInt(event.lastEventId, 10);
    try {
      payload = JSON.parse(event.data);
    } catch {
      restartFeed();
      return;
    }
    if (
      !Number.isSafeInteger(cursor) ||
      cursor < 0 ||
      payload?.cursor !== cursor ||
      typeof payload?.generation !== "string"
    ) {
      restartFeed();
      return;
    }
    if (state.feedGeneration && state.feedGeneration !== payload.generation) {
      clearSharedState();
    }
    state.feedGeneration = payload.generation;
    state.feedLastId = cursor;
    state.feedStatus = "synchronizing";
    const buffer = [];
    state.eventBuffer = buffer;
    try {
      await refresh(window.location.pathname);
    } finally {
      if (state.eventBuffer !== buffer) return;
      state.eventBuffer = null;
    }
    buffer.forEach(applyEvent);
    state.feedStatus = "live";
    refreshComposerState();
  }

  function receiveEvent(type, event) {
    let payload;
    const id = Number.parseInt(event.lastEventId, 10);
    try {
      payload = JSON.parse(event.data);
    } catch {
      restartFeed();
      return;
    }
    if (!Number.isSafeInteger(id) || id <= 0 || !payload) {
      restartFeed();
      return;
    }
    if (id <= state.feedLastId) return;
    if (id !== state.feedLastId + 1) {
      restartFeed();
      return;
    }
    state.feedLastId = id;
    const workEvent = { type, payload };
    if (state.eventBuffer) state.eventBuffer.push(workEvent);
    else applyEvent(workEvent);
  }

  function restartFeed() {
    state.feed?.close();
    state.feed = null;
    state.feedStatus = "reconnecting";
    state.eventBuffer = null;
    window.clearTimeout(state.feedTimer);
    state.feedTimer = window.setTimeout(connectFeed, 250);
    refreshComposerState();
  }

  function clearSharedState() {
    state.detail = null;
    state.items.clear();
    state.interactions.clear();
    state.libraryItems = [];
    state.libraryCursor = null;
  }

  function applyEvent({ type, payload }) {
    const currentId = state.detail?.summary?.thread_id;
    if (type === "timeline.item") {
      if (payload.thread_id === currentId) {
        state.items.set(itemKey(payload), payload);
        renderTimeline({ follow: true });
      }
      return;
    }
    if (type === "interaction.created") {
      if (payload.thread_id === currentId) {
        state.interactions.set(payload.interaction_id, payload);
        renderTimeline({ follow: true });
        setCurrentStatus(
          payload.kind === "user_input" ? "waiting_for_input" : "waiting_for_approval",
        );
      }
      return;
    }
    if (type === "interaction.resolved") {
      state.interactions.delete(payload.interaction_id);
      if (payload.thread_id === currentId) renderTimeline({ follow: false });
      return;
    }
    if (type === "session.status") {
      updateLibraryStatus(payload.thread_id, payload.status);
      if (payload.thread_id === currentId) setCurrentStatus(payload.status);
      return;
    }
    if (type === "operation.updated") {
      applyOperation(payload);
      return;
    }
    if (type === "session.deleted") {
      state.libraryItems = state.libraryItems.filter(
        (session) => session.thread_id !== payload.thread_id,
      );
      if (payload.thread_id === currentId) goLibrary();
      else if (!state.detail && workIsVisible()) renderLibraryItems();
      return;
    }
    if (type === "session.created" || type === "session.updated") {
      if (payload.thread_id === currentId) void loadDetail(currentId, state.routeVersion);
      else if (!state.detail && workIsVisible()) void loadLibrary(true, state.routeVersion);
      return;
    }
    if (type === "work.disconnected") {
      setNotice(payload.message || "Codex disconnected.", "error");
      void refresh(window.location.pathname);
      return;
    }
    if (type === "work.warning") setNotice(payload.message || "Codex warning.", "warn");
  }

  function applyOperation(payload) {
    if (payload.operation_id === state.pendingCreateOperation) {
      if (payload.state === "completed" && payload.thread_id) {
        state.pendingCreateOperation = null;
        openSession(payload.thread_id);
      } else if (payload.state === "failed" || payload.state === "interrupted") {
        state.pendingCreateOperation = null;
        setNotice(payload.error_message || "Could not create the Work session.", "error");
      }
    }
    if (payload.thread_id !== state.detail?.summary?.thread_id) return;
    if (payload.state === "failed" || payload.state === "interrupted") {
      setNotice(payload.error_message || "The Work operation did not complete.", "error");
    }
  }

  function updateLibraryStatus(threadId, status) {
    const session = state.libraryItems.find((item) => item.thread_id === threadId);
    if (session) session.status = status;
    if (!state.detail && workIsVisible()) renderLibraryItems();
  }

  function setCurrentStatus(status) {
    if (!state.detail) return;
    state.detail.summary.status = status;
    const element = document.getElementById("workStatus");
    if (element) {
      element.textContent = statusLabel(status);
      element.className = `work-status ${status}`;
    }
    refreshComposerState();
  }

  async function route(path) {
    const version = ++state.routeVersion;
    const threadId = routedThreadId(path);
    if (!threadId) {
      state.detail = null;
      state.items.clear();
      state.interactions.clear();
      renderLibrary();
      await loadLibrary(true, version);
      return;
    }
    renderLoading();
    await loadDetail(threadId, version);
  }

  async function refresh(path = window.location.pathname) {
    const version = state.routeVersion;
    try {
      state.bootstrap = await state.api("/admin/api/work/bootstrap");
    } catch (error) {
      if (version === state.routeVersion && workIsVisible()) renderUnavailable(error.message);
      return;
    }
    if (!workIsVisible() || version !== state.routeVersion) return;
    const threadId = routedThreadId(path);
    if (threadId) await loadDetail(threadId, version);
    else {
      renderLibrary();
      await loadLibrary(true, version);
    }
  }

  function renderLoading() {
    root()?.replaceChildren(node("div", "work-empty", "Loading Work Sessions…"));
  }

  function renderUnavailable(message) {
    const container = node("section", "work-empty");
    container.append(
      node("h3", "", "Work Sessions unavailable"),
      node("p", "", message || "Install or update Codex and restart FCC."),
    );
    root()?.replaceChildren(container);
  }

  function renderLibrary() {
    const container = root();
    if (!container) return;
    const shell = node("section", "work-library");
    const header = node("header", "work-library-header");
    const copy = node("div");
    copy.append(
      node("h2", "", "Work Sessions"),
      node("p", "", "Run Codex in a local project and keep working across tabs."),
    );
    const create = button("+ New session", "primary-button", showCreateDialog);
    create.dataset.testid = "work-new";
    header.append(copy, create);
    const search = node("input", "work-search");
    search.type = "search";
    search.value = state.libraryQuery;
    search.placeholder = "Search Work sessions";
    search.setAttribute("aria-label", "Search Work sessions");
    let timer = null;
    search.addEventListener("input", () => {
      state.libraryQuery = search.value;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => loadLibrary(true, state.routeVersion), 180);
    });
    const notice = node("div", "work-notice");
    notice.id = "workNotice";
    notice.hidden = true;
    const list = node("div", "work-project-list");
    list.id = "workProjectList";
    const more = button("Load more", "secondary-button work-load-more", () =>
      loadLibrary(false, state.routeVersion),
    );
    more.id = "workLoadMore";
    more.hidden = true;
    shell.append(header, search, notice, list, more);
    container.replaceChildren(shell);
    renderLibraryItems();
  }

  async function loadLibrary(reset, version) {
    const requestVersion = ++state.requestVersion;
    if (reset) {
      state.libraryCursor = null;
      state.libraryItems = [];
      const list = document.getElementById("workProjectList");
      if (list) list.replaceChildren(node("div", "work-empty", "Loading sessions…"));
    }
    try {
      const params = new URLSearchParams({ query: state.libraryQuery, limit: "25" });
      if (!reset && state.libraryCursor) params.set("cursor", state.libraryCursor);
      const page = await state.api(`/admin/api/work/sessions?${params}`);
      if (version !== state.routeVersion || requestVersion !== state.requestVersion) return;
      state.libraryItems = reset
        ? page.sessions
        : [...state.libraryItems, ...page.sessions];
      state.libraryCursor = page.next_cursor;
      renderLibraryItems();
    } catch (error) {
      if (version !== state.routeVersion) return;
      setNotice(error.message, "error");
      renderLibraryItems();
    }
  }

  function renderLibraryItems() {
    const list = document.getElementById("workProjectList");
    if (!list) return;
    list.replaceChildren();
    if (!state.libraryItems.length) {
      list.appendChild(
        node(
          "div",
          "work-empty",
          state.libraryQuery ? "No matching Work sessions." : "Start your first Work session.",
        ),
      );
    } else {
      const groups = new Map();
      state.libraryItems.forEach((session) => {
        if (!groups.has(session.cwd)) groups.set(session.cwd, []);
        groups.get(session.cwd).push(session);
      });
      groups.forEach((sessions, cwd) => {
        const group = node("section", "work-project-group");
        group.appendChild(node("div", "work-project-path", cwd));
        const sessionList = node("div", "work-session-list");
        sessions.forEach((session) => sessionList.appendChild(renderSessionCard(session)));
        group.appendChild(sessionList);
        list.appendChild(group);
      });
    }
    const more = document.getElementById("workLoadMore");
    if (more) more.hidden = !state.libraryCursor;
  }

  function renderSessionCard(session) {
    const card = button("", "work-session-card", () => openSession(session.thread_id));
    card.dataset.threadId = session.thread_id;
    const title = node("strong", "", session.title || "New Work Session");
    const preview = node("p", "", session.preview || "No messages yet");
    const meta = node("div", "work-card-meta");
    const status = node(
      "span",
      `work-status ${session.status}`,
      statusLabel(session.status),
    );
    const time = node("span", "work-muted", formatTime(session.updated_at_ms));
    meta.append(status, time);
    card.append(title, preview, meta);
    return card;
  }

  function showCreateDialog() {
    if (!state.bootstrap?.available) {
      setNotice(state.bootstrap?.reason || "Install or update Codex first.", "error");
      return;
    }
    const dialog = node("dialog", "work-dialog");
    const title = node("h3", "", "New Work Session");
    const label = node("label");
    label.appendChild(node("span", "", "Absolute project folder path"));
    const input = node("input");
    input.type = "text";
    input.placeholder = navigator.platform.startsWith("Win")
      ? "C:\\path\\to\\project"
      : "/path/to/project";
    input.autocomplete = "off";
    label.appendChild(input);
    const recent = node("div", "work-recent-projects");
    (state.bootstrap?.recent_projects || []).forEach((path) => {
      recent.appendChild(button(path, "secondary-button", () => (input.value = path)));
    });
    const message = node("div", "work-notice");
    message.hidden = true;
    const actions = node("div", "work-dialog-actions");
    actions.append(
      button("Cancel", "secondary-button", () => dialog.close()),
      button("Create", "primary-button", async () => {
        message.hidden = true;
        const operationId = uuid();
        state.pendingCreateOperation = operationId;
        try {
          const acknowledgement = await state.api("/admin/api/work/sessions", {
            method: "POST",
            body: JSON.stringify({ cwd: input.value, operation_id: operationId }),
          });
          dialog.close();
          setNotice("Creating Codex session…");
          if (acknowledgement.state === "completed" && acknowledgement.thread_id) {
            state.pendingCreateOperation = null;
            openSession(acknowledgement.thread_id);
          }
        } catch (error) {
          state.pendingCreateOperation = null;
          message.textContent = error.message;
          message.className = "work-notice error";
          message.hidden = false;
        }
      }),
    );
    dialog.append(title, label, recent, message, actions);
    document.body.appendChild(dialog);
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    dialog.showModal();
    input.focus();
  }

  function openSession(threadId) {
    window.history.pushState({}, "", `/admin/work/${encodeURIComponent(threadId)}`);
    void route(window.location.pathname);
  }

  function goLibrary() {
    window.history.pushState({}, "", "/admin/work");
    void route(window.location.pathname);
  }

  async function loadDetail(threadId, version) {
    try {
      const detail = await state.api(`/admin/api/work/sessions/${encodeURIComponent(threadId)}`);
      if (version !== state.routeVersion || routedThreadId(window.location.pathname) !== threadId)
        return;
      state.detail = detail;
      state.items = new Map(
        [...(detail.turns?.items || []), ...(detail.live_items || [])].map((item) => [
          itemKey(item),
          item,
        ]),
      );
      state.interactions = new Map(
        (detail.interactions || []).map((interaction) => [
          interaction.interaction_id,
          interaction,
        ]),
      );
      loadDraft(threadId);
      renderDetail();
    } catch (error) {
      if (version !== state.routeVersion) return;
      renderUnavailable(error.message);
    }
  }

  function renderDetail() {
    const detail = state.detail;
    const container = root();
    if (!detail || !container) return;
    const shell = node("section", "work-session-shell");
    const header = renderDetailHeader(detail);
    const notice = node("div", "work-notice");
    notice.id = "workNotice";
    notice.hidden = true;
    const timeline = node("div", "work-timeline");
    timeline.id = "workTimeline";
    const composer = renderComposer();
    shell.append(header, notice, timeline, composer);
    container.replaceChildren(shell);
    renderTimeline({ follow: true });
    refreshComposerState();
  }

  function renderDetailHeader(detail) {
    const header = node("header", "work-session-header");
    const row = node("div", "work-header-row");
    row.appendChild(button("← Work", "work-back-button", goLibrary));
    const title = node("input", "work-title");
    title.value = detail.summary.title;
    title.maxLength = 200;
    title.setAttribute("aria-label", "Work session name");
    title.addEventListener("keydown", (event) => {
      if (event.key === "Enter") title.blur();
    });
    title.addEventListener("blur", () => {
      if (title.value !== state.detail?.summary?.title) void renameSession(title.value);
    });
    row.append(title, button("Delete", "danger-button", deleteSession));
    const meta = node("div", "work-session-meta");
    const status = node(
      "span",
      `work-status ${detail.summary.status}`,
      statusLabel(detail.summary.status),
    );
    status.id = "workStatus";
    meta.append(node("code", "", detail.summary.cwd), status);
    if (!detail.summary.project_available)
      meta.appendChild(node("span", "work-status failed", "Project unavailable"));
    if (!detail.summary.session_available)
      meta.appendChild(
        button("Remove from Work", "secondary-button", removeMissingSession),
      );
    header.append(row, meta, renderControls(detail));
    return header;
  }

  function renderControls(detail) {
    const controls = node("div", "work-controls");
    controls.append(
      renderModelControl(detail),
      renderSelectControl(
        "Thinking",
        "workEffort",
        reasoningValues(detail),
        detail.settings.reasoning_effort,
        (value) => updateSettings({ reasoning_effort: value || null }),
      ),
      renderSelectControl(
        "Mode",
        "workMode",
        collaborationValues(detail),
        detail.settings.collaboration_mode,
        (value) => updateSettings({ collaboration_mode: value || null }),
        true,
      ),
      renderSelectControl(
        "Permissions",
        "workPermissions",
        permissionValues(detail),
        detail.settings.permission_profile,
        (value) => updateSettings({ permission_profile: value || null }),
        true,
      ),
    );
    return controls;
  }

  function renderModelControl(detail) {
    const group = node("div", "work-control work-model-control");
    const label = node("label", "", "Model");
    label.htmlFor = "workModel";
    const input = node("input", "work-model-input");
    input.id = "workModel";
    input.type = "text";
    input.autocomplete = "off";
    input.value = detail.settings.model || "";
    const values = () => modelValues(detail);
    let committed = input.value;
    const combobox = new window.FccModelCombobox(input, {
      listboxId: "work-model-options",
      label: "model",
      values,
      emptyMessage: () => (values().length ? "No matching models." : "No models available."),
      registry: state.modelComboboxes,
      onSelect: (model) => {
        committed = model;
        const native = modelRecord(detail, model);
        void updateSettings({
          model,
          reasoning_effort: native?.defaultReasoningEffort || null,
        });
      },
      onClose: () => (input.value = committed),
    });
    group.append(label, combobox.element);
    return group;
  }

  function renderSelectControl(
    labelText,
    id,
    values,
    selected,
    onChange,
    includeDefault = false,
  ) {
    const group = node("label", "work-control");
    group.appendChild(node("span", "", labelText));
    const select = node("select");
    select.id = id;
    if (includeDefault) {
      const option = node("option", "", "Default");
      option.value = "";
      select.appendChild(option);
    }
    values.forEach((value) => {
      const option = node("option", "", humanize(value));
      option.value = value;
      select.appendChild(option);
    });
    select.value = selected || "";
    select.disabled = values.length === 0 && !includeDefault;
    select.addEventListener("change", () => onChange(select.value));
    group.appendChild(select);
    return group;
  }

  function modelValues(detail) {
    return (detail.controls?.models || [])
      .map((model) => model.model || model.id)
      .filter((value) => typeof value === "string");
  }

  function modelRecord(detail, modelId = detail.settings.model) {
    return (detail.controls?.models || []).find(
      (model) => (model.model || model.id) === modelId,
    );
  }

  function reasoningValues(detail) {
    return (modelRecord(detail)?.supportedReasoningEfforts || [])
      .map((entry) =>
        typeof entry === "string" ? entry : entry?.reasoningEffort,
      )
      .filter((value) => typeof value === "string");
  }

  function collaborationValues(detail) {
    return (detail.controls?.collaboration_modes || [])
      .map((mode) => mode.name)
      .filter((value) => typeof value === "string");
  }

  function permissionValues(detail) {
    return (detail.controls?.permission_profiles || [])
      .filter((profile) => profile.allowed !== false)
      .map((profile) => profile.id)
      .filter((value) => typeof value === "string");
  }

  async function updateSettings(updates) {
    if (!state.detail) return;
    try {
      const record = await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(state.detail.summary.thread_id)}/settings`,
        {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: state.detail.summary.revision,
            updates,
          }),
        },
      );
      state.detail.settings = record.settings;
      state.detail.summary.revision = record.revision;
      renderDetail();
    } catch (error) {
      setNotice(error.message, "error");
      await loadDetail(state.detail.summary.thread_id, state.routeVersion);
    }
  }

  async function renameSession(name) {
    if (!state.detail) return;
    try {
      const record = await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(state.detail.summary.thread_id)}/name`,
        {
          method: "PUT",
          body: JSON.stringify({
            expected_revision: state.detail.summary.revision,
            name,
          }),
        },
      );
      state.detail.summary.title = name.trim().replace(/\s+/g, " ");
      state.detail.summary.revision = record.revision;
    } catch (error) {
      setNotice(error.message, "error");
      await loadDetail(state.detail.summary.thread_id, state.routeVersion);
    }
  }

  function renderComposer() {
    const wrapper = node("div", "work-composer");
    const textarea = node("textarea");
    textarea.id = "workComposer";
    textarea.rows = 2;
    textarea.placeholder = "Ask Codex to work in this project";
    textarea.value = state.draft;
    textarea.setAttribute("aria-label", "Work message");
    textarea.addEventListener("input", () => {
      state.draft = textarea.value;
      saveDraft();
      resizeComposer(textarea);
      refreshComposerState();
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!document.getElementById("workSend")?.disabled) void sendMessage();
      }
    });
    const actions = node("div", "work-composer-actions");
    const status = node("span", "work-composer-status");
    status.id = "workComposerStatus";
    const send = button("Send", "primary-button", sendMessage);
    send.id = "workSend";
    const stop = button("Stop", "danger-button", stopTurn);
    stop.id = "workStop";
    stop.hidden = true;
    actions.append(status, send, stop);
    wrapper.append(textarea, actions);
    window.requestAnimationFrame(() => resizeComposer(textarea));
    return wrapper;
  }

  function resizeComposer(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > textarea.clientHeight ? "auto" : "hidden";
  }

  function refreshComposerState() {
    const detail = state.detail;
    const textarea = document.getElementById("workComposer");
    const send = document.getElementById("workSend");
    const stop = document.getElementById("workStop");
    const status = document.getElementById("workComposerStatus");
    if (!detail || !textarea || !send || !stop || !status) return;
    const active = ACTIVE_STATUSES.has(detail.summary.status);
    const unavailable =
      !detail.summary.project_available || !detail.summary.session_available;
    textarea.disabled = unavailable || detail.summary.status === "deleting";
    send.hidden = active;
    stop.hidden = !active || detail.summary.status === "deleting";
    send.disabled =
      state.feedStatus !== "live" || unavailable || !state.draft.trim();
    stop.disabled = state.feedStatus !== "live" || detail.summary.status === "stopping";
    status.textContent =
      state.feedStatus === "live"
        ? active
          ? statusLabel(detail.summary.status)
          : ""
        : "Reconnecting…";
  }

  async function sendMessage() {
    if (!state.detail || !state.draft.trim()) return;
    const textarea = document.getElementById("workComposer");
    const text = state.draft;
    try {
      await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(state.detail.summary.thread_id)}/turns`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_revision: state.detail.summary.revision,
            operation_id: uuid(),
            text,
          }),
        },
      );
      state.draft = "";
      saveDraft();
      if (textarea) {
        textarea.value = "";
        resizeComposer(textarea);
        textarea.focus({ preventScroll: true });
      }
      setCurrentStatus("working");
    } catch (error) {
      setNotice(error.message, "error");
      textarea?.focus({ preventScroll: true });
    }
  }

  async function stopTurn() {
    if (!state.detail) return;
    try {
      await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(state.detail.summary.thread_id)}/stop`,
        {
          method: "POST",
          body: JSON.stringify({ operation_id: uuid() }),
        },
      );
      setCurrentStatus("stopping");
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  async function deleteSession() {
    if (!state.detail) return;
    if (!window.confirm("Delete this native Codex session and its history?")) return;
    try {
      await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(state.detail.summary.thread_id)}/delete`,
        {
          method: "POST",
          body: JSON.stringify({ operation_id: uuid() }),
        },
      );
      setCurrentStatus("deleting");
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  async function removeMissingSession() {
    if (!state.detail) return;
    if (!window.confirm("Remove this missing Codex session from Work Sessions?")) return;
    try {
      await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(state.detail.summary.thread_id)}/remove`,
        { method: "POST", body: "{}" },
      );
      goLibrary();
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  function renderTimeline({ follow }) {
    const timeline = document.getElementById("workTimeline");
    if (!timeline) return;
    const shouldFollow = follow && nearBottom(timeline);
    const scrollTop = timeline.scrollTop;
    timeline.replaceChildren();
    if (state.detail?.turns?.next_cursor) {
      timeline.appendChild(
        button("Load older activity", "secondary-button work-load-more", loadOlder),
      );
    }
    const items = [...state.items.values()];
    items.forEach((item) => timeline.appendChild(renderTimelineItem(item)));
    [...state.interactions.values()].forEach((interaction) =>
      timeline.appendChild(renderInteraction(interaction)),
    );
    if (!items.length && !state.interactions.size) {
      timeline.appendChild(
        node("div", "work-empty", "Tell Codex what you want to do in this project."),
      );
    }
    if (shouldFollow) timeline.scrollTop = timeline.scrollHeight;
    else timeline.scrollTop = scrollTop;
  }

  function renderTimelineItem(item) {
    if (item.kind === "userMessage") {
      return node("article", "work-timeline-item work-user-message", item.text || "");
    }
    if (item.kind === "agentMessage") {
      const message = node("article", "work-timeline-item work-agent-message");
      if (typeof item.html === "string") message.innerHTML = item.html;
      else message.textContent = item.text || "";
      return message;
    }
    const active = item.status && !["completed", "failed", "declined"].includes(item.status);
    const details = node(
      "details",
      `work-timeline-item work-activity${active ? " active" : ""}`,
    );
    details.open = Boolean(active || ["reasoning", "plan"].includes(item.kind));
    const summary = node("summary", "", activityLabel(item.kind, item.status));
    const content = node("div");
    if (typeof item.html === "string" && ["reasoning", "plan"].includes(item.kind)) {
      content.className = "work-agent-message";
      content.innerHTML = item.html;
    } else {
      const pre = node("pre", "", item.text || JSON.stringify(item.payload, null, 2));
      content.appendChild(pre);
    }
    details.append(summary, content);
    return details;
  }

  function renderInteraction(interaction) {
    const card = node("section", "work-timeline-item work-interaction");
    card.appendChild(node("h4", "", interaction.title));
    const payload = interaction.payload || {};
    if (payload.reason) card.appendChild(node("p", "", payload.reason));
    if (payload.command) {
      const command = Array.isArray(payload.command)
        ? payload.command.join(" ")
        : String(payload.command);
      card.appendChild(node("pre", "", command));
    }
    if (interaction.kind === "user_input") {
      card.appendChild(renderQuestions(interaction));
      return card;
    }
    const actions = node("div", "work-interaction-actions");
    if (interaction.kind === "command_approval") {
      const decisions = Array.isArray(payload.available_decisions)
        ? payload.available_decisions
        : ["accept", "decline", "cancel"];
      decisions.forEach((decision) =>
        actions.appendChild(
          button(humanize(decision), decision.includes("accept") ? "primary-button" : "secondary-button", () =>
            answerInteraction(interaction, { decision }),
          ),
        ),
      );
    } else if (interaction.kind === "file_change_approval") {
      ["accept", "acceptForSession", "decline", "cancel"].forEach((decision) =>
        actions.appendChild(
          button(humanize(decision), decision.startsWith("accept") ? "primary-button" : "secondary-button", () =>
            answerInteraction(interaction, { decision }),
          ),
        ),
      );
    } else {
      actions.append(
        button("Allow this turn", "primary-button", () =>
          answerInteraction(interaction, { decision: "accept", scope: "turn" }),
        ),
        button("Allow this session", "secondary-button", () =>
          answerInteraction(interaction, { decision: "accept", scope: "session" }),
        ),
        button("Decline", "secondary-button", () =>
          answerInteraction(interaction, { decision: "decline", scope: "turn" }),
        ),
      );
    }
    card.appendChild(actions);
    return card;
  }

  function renderQuestions(interaction) {
    const form = node("form");
    const questions = interaction.payload?.questions || [];
    questions.forEach((question) => {
      const group = node("label", "work-question");
      group.appendChild(node("strong", "", question.question || question.header || question.id));
      const input = node("input");
      input.name = question.id;
      input.type = question.isSecret ? "password" : "text";
      if (Array.isArray(question.options)) {
        input.placeholder = question.options.map((option) => option.label).join(" · ");
      }
      group.appendChild(input);
      form.appendChild(group);
    });
    const submit = button("Submit answers", "primary-button", () => {});
    submit.type = "submit";
    form.appendChild(submit);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const answers = {};
      questions.forEach((question) => {
        answers[question.id] = [String(data.get(question.id) || "")];
      });
      void answerInteraction(interaction, { answers });
    });
    return form;
  }

  async function answerInteraction(interaction, value) {
    try {
      await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(interaction.thread_id)}/interactions/${interaction.interaction_id}`,
        { method: "POST", body: JSON.stringify({ value }) },
      );
      state.interactions.delete(interaction.interaction_id);
      renderTimeline({ follow: false });
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  async function loadOlder() {
    if (!state.detail?.turns?.next_cursor) return;
    const threadId = state.detail.summary.thread_id;
    const cursor = state.detail.turns.next_cursor;
    try {
      const page = await state.api(
        `/admin/api/work/sessions/${encodeURIComponent(threadId)}/turns?cursor=${encodeURIComponent(cursor)}&limit=50`,
      );
      page.items.forEach((item) => {
        const key = itemKey(item);
        if (!state.items.has(key)) state.items.set(key, item);
      });
      state.detail.turns.next_cursor = page.next_cursor;
      renderTimeline({ follow: false });
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  function setNotice(message, kind = "") {
    const notice = document.getElementById("workNotice");
    if (!notice) return;
    notice.textContent = message;
    notice.className = `work-notice ${kind}`.trim();
    notice.hidden = !message;
  }

  function itemKey(item) {
    return `${item.turn_id}:${item.item_id}`;
  }

  function loadDraft(threadId) {
    if (state.draftSessionId === threadId) return;
    state.draftSessionId = threadId;
    state.draft = sessionStorage.getItem(`fcc.work.draft.${threadId}`) || "";
  }

  function saveDraft() {
    if (!state.draftSessionId) return;
    const key = `fcc.work.draft.${state.draftSessionId}`;
    if (state.draft) sessionStorage.setItem(key, state.draft);
    else sessionStorage.removeItem(key);
  }

  function nearBottom(element) {
    return element.scrollHeight - element.scrollTop - element.clientHeight < 120;
  }

  function statusLabel(status) {
    const labels = {
      ready: "Ready",
      working: "Working",
      waiting_for_approval: "Waiting for approval",
      waiting_for_input: "Waiting for input",
      stopping: "Stopping",
      deleting: "Deleting",
      completed: "Completed",
      interrupted: "Interrupted",
      failed: "Failed",
      disconnected: "Disconnected",
    };
    return labels[status] || humanize(status);
  }

  function activityLabel(kind, status) {
    const label = {
      reasoning: "Thinking",
      plan: "Plan",
      commandExecution: "Command",
      fileChange: "File changes",
      mcpToolCall: "Tool call",
      dynamicToolCall: "Tool call",
      webSearch: "Web search",
      imageView: "Image",
      imageGeneration: "Image generation",
      contextCompaction: "Context compacted",
      diff: "Turn diff",
      codexActivity: "Codex activity",
    }[kind] || humanize(kind);
    return status ? `${label} · ${humanize(status)}` : label;
  }

  function humanize(value) {
    if (!value) return "Default";
    return String(value)
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .replace(/_/g, " ")
      .replace(/^./, (letter) => letter.toUpperCase());
  }

  function formatTime(value) {
    if (!Number.isFinite(value)) return "";
    return new Date(value).toLocaleString();
  }

  window.WorkSessions = { initialize, activate, refresh };
})();
