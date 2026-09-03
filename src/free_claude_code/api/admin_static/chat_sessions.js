(() => {
  const state = {
    api: null,
    initialized: false,
    bootstrap: null,
    bootstrapRequestVersion: 0,
    session: null,
    turns: [],
    compaction: null,
    nextBefore: null,
    context: null,
    contextError: "",
    libraryCursor: null,
    libraryItems: [],
    libraryQuery: "",
    libraryRequestVersion: 0,
    libraryLoadMore: null,
    olderLoad: null,
    operation: null,
    activeOperations: new Map(),
    pendingCommands: new Map(),
    sessionSummaries: new Map(),
    sessionRevisions: new Map(),
    deletedSessions: new Set(),
    retiredTitleInputs: new WeakSet(),
    draft: "",
    draftSessionId: null,
    draftOperationId: null,
    draftSubmittedText: "",
    routeVersion: 0,
    estimateTimer: null,
    estimateVersion: 0,
    feed: null,
    feedStatus: "connecting",
    feedLastId: 0,
    feedSyncVersion: 0,
    eventBuffer: null,
    feedRestartTimer: null,
    modelComboboxes: new Set(),
  };

  const CHAT_EVENT_TYPES = [
    "session.created",
    "session.updated",
    "session.deleted",
    "preferences.updated",
    "operation.started",
    "turn.started",
    "segment.started",
    "segment.delta",
    "segment.completed",
    "compaction.started",
    "compaction.progress",
    "compaction.completed",
    "compaction.failed",
    "compaction.stopped",
    "turn.completed",
    "turn.failed",
    "turn.stopped",
    "operation.failed",
  ];

  const root = () => document.getElementById("chatRoot");

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

  function setNotice(message, kind = "") {
    const area = document.getElementById("chatNotice");
    if (!area) return;
    area.textContent = message;
    area.className = `chat-notice ${kind}`.trim();
    area.hidden = !message;
  }

  async function initialize(api) {
    if (state.initialized) return;
    state.api = api;
    state.initialized = true;
    connectEventFeed();
    if (chatIsVisible()) {
      renderLoading();
    }
  }

  function mutationsReady() {
    return state.feedStatus === "live";
  }

  function connectEventFeed() {
    window.clearTimeout(state.feedRestartTimer);
    state.feedRestartTimer = null;
    state.feed?.close();
    state.feedStatus = state.feedLastId ? "reconnecting" : "connecting";
    refreshFeedState();
    const feed = new EventSource("/admin/api/chat/events");
    state.feed = feed;
    feed.addEventListener("feed.ready", (event) => {
      if (state.feed === feed) void synchronizeFeed(event);
    });
    feed.addEventListener("feed.resync_required", () => {
      if (state.feed === feed) restartEventFeed();
    });
    CHAT_EVENT_TYPES.forEach((type) => {
      feed.addEventListener(type, (event) => {
        if (state.feed === feed) receiveChatEvent(type, event);
      });
    });
    feed.addEventListener("error", () => {
      if (state.feed !== feed) return;
      state.feedSyncVersion += 1;
      state.feedStatus = "reconnecting";
      refreshFeedState();
      if (!state.bootstrap) void refresh(window.location.pathname);
    });
  }

  function restartEventFeed() {
    state.feedSyncVersion += 1;
    state.feed?.close();
    state.feed = null;
    state.feedStatus = "reconnecting";
    refreshFeedState();
    window.clearTimeout(state.feedRestartTimer);
    state.feedRestartTimer = window.setTimeout(connectEventFeed, 250);
  }

  async function synchronizeFeed(event) {
    const cursor = Number.parseInt(event.lastEventId, 10);
    let snapshot;
    try {
      snapshot = JSON.parse(event.data);
    } catch {
      restartEventFeed();
      return;
    }
    if (
      !Number.isSafeInteger(cursor) ||
      cursor < 0 ||
      snapshot?.cursor !== cursor ||
      !Array.isArray(snapshot?.active_operations)
    ) {
      restartEventFeed();
      return;
    }
    const syncVersion = ++state.feedSyncVersion;
    state.feedLastId = cursor;
    state.feedStatus = "synchronizing";
    refreshFeedState();
    const snapshotOperationIds = installActiveSnapshot(snapshot.active_operations);
    if (!snapshotOperationIds) {
      restartEventFeed();
      return;
    }
    await withBufferedEvents(() => refresh(window.location.pathname));
    if (syncVersion !== state.feedSyncVersion) return;
    settlePendingAfterSnapshot(snapshotOperationIds);
    state.feedStatus = "live";
    refreshFeedState();
  }

  async function withBufferedEvents(loadSnapshot) {
    if (state.eventBuffer) {
      await loadSnapshot();
      return;
    }
    const buffer = [];
    state.eventBuffer = buffer;
    try {
      await loadSnapshot();
    } finally {
      if (state.eventBuffer !== buffer) return;
      state.eventBuffer = null;
      buffer.forEach(applyChatEvent);
    }
  }

  function receiveChatEvent(type, event) {
    let payload;
    const id = Number.parseInt(event.lastEventId, 10);
    try {
      payload = JSON.parse(event.data);
    } catch {
      restartEventFeed();
      return;
    }
    if (!Number.isSafeInteger(id) || id <= 0 || !payload) {
      restartEventFeed();
      return;
    }
    if (id <= state.feedLastId) return;
    if (id !== state.feedLastId + 1) {
      restartEventFeed();
      return;
    }
    state.feedLastId = id;
    const chatEvent = { type, id, payload };
    if (state.eventBuffer) {
      state.eventBuffer.push(chatEvent);
    } else {
      applyChatEvent(chatEvent);
    }
  }

  function applyChatEvent(event) {
    const { type, payload } = event;
    if (type === "session.created" || type === "session.updated") {
      applySessionSummary(payload);
      return;
    }
    if (type === "session.deleted") {
      applySessionDeletion(payload.session_id);
      return;
    }
    if (type === "preferences.updated") {
      applyPreferencesUpdate(payload);
      return;
    }
    applyOperationEvent(type, payload);
  }

  function applySessionSummary(payload) {
    const summary = sessionSummaryFromPayload(payload);
    const current = summary ? reduceSessionSummary(summary) : null;
    if (!current) return;
    upsertLibrarySummary(current);
    if (state.session?.id !== current.id) {
      if (!state.session && chatIsVisible()) renderLibraryItems();
      return;
    }
    if (current.revision < state.session.revision) return;
    const controlsChanged =
      state.session.title !== current.title ||
      state.session.model !== current.model ||
      state.session.reasoning !== current.reasoning;
    const titleEdit = captureTitleEdit();
    const selection = captureComposerSelection();
    state.session = {
      ...state.session,
      title: current.title,
      model: current.model,
      reasoning: current.reasoning,
      revision: current.revision,
      created_at: current.created_at,
      updated_at: current.updated_at,
    };
    if (controlsChanged) {
      renderSessionPreservingScroll(titleEdit);
      restoreComposerSelection(selection);
    }
    scheduleEstimate(true);
  }

  function sessionSummaryFromPayload(payload) {
    if (
      typeof payload.session_id !== "string" ||
      typeof payload.title !== "string" ||
      typeof payload.model !== "string" ||
      typeof payload.reasoning !== "string" ||
      !Number.isInteger(payload.revision) ||
      typeof payload.preview !== "string" ||
      !Number.isInteger(payload.created_at) ||
      !Number.isInteger(payload.updated_at)
    )
      return null;
    return {
      id: payload.session_id,
      title: payload.title,
      model: payload.model,
      reasoning: payload.reasoning,
      revision: payload.revision,
      preview: payload.preview,
      created_at: payload.created_at,
      updated_at: payload.updated_at,
    };
  }

  function reduceSessionSummary(summary) {
    if (state.deletedSessions.has(summary.id)) return null;
    const knownRevision = state.sessionRevisions.get(summary.id) || 0;
    if (summary.revision < knownRevision) return null;
    state.sessionRevisions.set(summary.id, summary.revision);
    state.sessionSummaries.set(summary.id, summary);
    return summary;
  }

  function upsertLibrarySummary(summary) {
    state.libraryItems = state.libraryItems.filter((item) => item.id !== summary.id);
    if (matchesLibraryQuery(summary)) state.libraryItems.push(summary);
    sortLibraryItems();
  }

  function matchesLibraryQuery(summary) {
    const query = state.libraryQuery.trim().toLocaleLowerCase();
    return !query || summary.title.toLocaleLowerCase().includes(query);
  }

  function sortLibraryItems() {
    state.libraryItems.sort(
      (left, right) =>
        right.updated_at - left.updated_at || right.id.localeCompare(left.id),
    );
  }

  function applySessionDeletion(sessionId) {
    if (typeof sessionId !== "string") return;
    state.deletedSessions.add(sessionId);
    state.sessionSummaries.delete(sessionId);
    state.sessionRevisions.delete(sessionId);
    const operation = state.activeOperations.get(sessionId);
    if (operation) cancelOperationRender(operation);
    state.activeOperations.delete(sessionId);
    const pending = state.pendingCommands.get(sessionId);
    if (pending) cancelOperationRender(pending);
    state.pendingCommands.delete(sessionId);
    removeDraft(sessionId);
    state.libraryItems = state.libraryItems.filter(
      (session) => session.id !== sessionId,
    );
    if (
      state.session?.id === sessionId ||
      routedSessionId(window.location.pathname) === sessionId
    ) {
      detachVisibleOperation();
      window.history.replaceState({}, "", "/admin/chat");
      void route(window.location.pathname);
    } else if (!state.session && chatIsVisible()) {
      renderLibraryItems();
    }
  }

  function applyPreferencesUpdate(payload) {
    if (!state.bootstrap || !Number.isInteger(payload.updated_at)) return;
    const current = state.bootstrap.preferences;
    if (current && payload.updated_at < current.updated_at) return;
    state.bootstrap.preferences = {
      system_prompt: payload.system_prompt,
      last_model: payload.last_model,
      last_reasoning: payload.last_reasoning,
      updated_at: payload.updated_at,
    };
    if (state.session) scheduleEstimate(true);
  }

  function createOperation(values) {
    const phase = values.phase || "generating";
    return {
      id: values.id,
      sessionId: values.sessionId,
      action: values.action,
      phase,
      segments: values.segments || [],
      sequence: values.sequence || 0,
      status: phase === "compacting" ? "Compacting…" : "Thinking…",
      userText: values.userText || "",
      accepted: Boolean(values.accepted),
      failureMessage: "",
      renderFrame: null,
      returnFocusToComposer: Boolean(values.returnFocusToComposer),
      commandPending: Boolean(values.commandPending),
      serverObserved: Boolean(values.serverObserved),
      ambiguousError: "",
      turnId: values.turnId || null,
      generationId: values.generationId || null,
      regeneration: Boolean(values.regeneration),
      actualModel: values.actualModel || null,
    };
  }

  function operationFromSnapshot(snapshot, existing = null) {
    const operation =
      existing && existing.id === snapshot.operation_id
        ? existing
        : createOperation({
            id: snapshot.operation_id,
            sessionId: snapshot.session_id,
            action: snapshot.kind,
          });
    if (snapshot.operation_sequence < operation.sequence) return operation;
    operation.action = snapshot.kind;
    operation.phase = snapshot.phase;
    operation.status =
      snapshot.phase === "compacting" ? "Compacting…" : "Thinking…";
    operation.userText = snapshot.submitted_text || "";
    operation.turnId = snapshot.turn_id;
    operation.generationId = snapshot.generation_id;
    operation.regeneration = Boolean(snapshot.regeneration);
    operation.actualModel = snapshot.actual_model;
    operation.segments = (snapshot.segments || []).map((segment) => ({
      kind: segment.kind,
      text: segment.text,
      pending: [],
    }));
    operation.sequence = snapshot.operation_sequence;
    operation.accepted = Boolean(snapshot.turn_id || snapshot.generation_id);
    operation.serverObserved = true;
    operation.commandPending = false;
    operation.ambiguousError = "";
    return operation;
  }

  function validOperationKind(value) {
    return ["send", "retry", "regenerate", "compact"].includes(value);
  }

  function validOperationSnapshot(snapshot) {
    return Boolean(
      snapshot &&
        typeof snapshot.session_id === "string" &&
        typeof snapshot.operation_id === "string" &&
        validOperationKind(snapshot.kind) &&
        ["generating", "compacting"].includes(snapshot.phase) &&
        Number.isInteger(snapshot.operation_sequence) &&
        Array.isArray(snapshot.segments),
    );
  }

  function installActiveSnapshot(snapshots) {
    const next = new Map();
    const operationIds = new Set();
    for (const snapshot of snapshots) {
      if (!validOperationSnapshot(snapshot) || next.has(snapshot.session_id)) {
        return null;
      }
      const current = state.activeOperations.get(snapshot.session_id);
      const pending = state.pendingCommands.get(snapshot.session_id);
      const reusable =
        current?.id === snapshot.operation_id
          ? current
          : pending?.id === snapshot.operation_id
            ? pending
            : null;
      const operation = operationFromSnapshot(snapshot, reusable);
      next.set(snapshot.session_id, operation);
      operationIds.add(snapshot.operation_id);
      if (pending?.id === snapshot.operation_id) {
        state.pendingCommands.delete(snapshot.session_id);
      }
    }
    state.activeOperations.forEach((operation, sessionId) => {
      if (next.get(sessionId) !== operation) cancelOperationRender(operation);
    });
    state.activeOperations = next;
    selectVisibleOperation();
    if (!state.session && chatIsVisible()) renderLibraryItems();
    return operationIds;
  }

  function settlePendingAfterSnapshot(operationIds) {
    if (!operationIds) {
      restartEventFeed();
      return;
    }
    [...state.pendingCommands.values()].forEach((pending) => {
      if (pending.commandPending || operationIds.has(pending.id)) return;
      if (pending.serverObserved) {
        clearPendingCommand(pending);
      } else if (pending.ambiguousError) {
        markCommandUncertain(pending);
      }
    });
    selectVisibleOperation();
    if (state.session) renderSessionPreservingScroll();
  }

  function selectVisibleOperation() {
    if (!state.session) {
      state.operation = null;
      return;
    }
    state.operation =
      state.activeOperations.get(state.session.id) ||
      state.pendingCommands.get(state.session.id) ||
      null;
  }

  function applyOperationEvent(type, payload) {
    if (
      typeof payload.session_id !== "string" ||
      typeof payload.operation_id !== "string" ||
      !validOperationKind(payload.kind) ||
      !Number.isInteger(payload.operation_sequence)
    )
      return;
    const active = state.activeOperations.get(payload.session_id);
    const pending = state.pendingCommands.get(payload.session_id);
    const matchingActive = active?.id === payload.operation_id ? active : null;
    const matchingPending = pending?.id === payload.operation_id ? pending : null;
    if (isTerminalEvent(type, payload.kind)) {
      if (
        matchingActive &&
        payload.operation_sequence <= matchingActive.sequence
      )
        return;
      if (matchingActive) matchingActive.sequence = payload.operation_sequence;
      if (matchingPending) matchingPending.serverObserved = true;
      void settleOperation(type, payload, matchingActive || matchingPending);
      return;
    }
    let operation = matchingActive || matchingPending;
    if (!operation) {
      operation = createOperation({
        id: payload.operation_id,
        sessionId: payload.session_id,
        action: payload.kind,
      });
    }
    if (payload.operation_sequence <= operation.sequence) return;
    if (active && active !== operation) cancelOperationRender(active);
    state.activeOperations.set(payload.session_id, operation);
    if (matchingPending) state.pendingCommands.delete(payload.session_id);
    operation.serverObserved = true;
    operation.commandPending = false;
    let structural = true;
    let restoreComposerFocus = false;
    if (type === "operation.started") {
      operation.action = payload.kind;
      operation.phase = payload.phase;
      operation.status =
        payload.phase === "compacting" ? "Compacting…" : "Thinking…";
      operation.userText = payload.submitted_text || "";
    } else if (type === "turn.started") {
      operation.accepted = true;
      operation.phase = "generating";
      operation.status = "Thinking…";
      operation.turnId = payload.turn_id;
      operation.generationId = payload.generation_id;
      operation.regeneration = Boolean(payload.regeneration);
      if (operation.action === "send") confirmSubmittedDraft(operation.id);
      restoreComposerFocus =
        operation.returnFocusToComposer && composerFocusIsUnclaimed();
    } else if (type === "segment.started") {
      operation.segments[payload.ordinal] = {
        kind: payload.segment_kind,
        text: "",
        pending: [],
      };
    } else if (type === "segment.delta") {
      const segment = operation.segments[payload.ordinal];
      if (segment) {
        if (state.operation === operation) {
          segment.pending.push(payload.delta);
          structural = false;
          scheduleLiveRender(operation);
        } else {
          segment.text += payload.delta;
        }
      }
    } else if (type === "compaction.started") {
      operation.phase = "compacting";
      operation.status = "Compacting…";
    } else if (type === "compaction.completed" && operation.action !== "compact") {
      operation.phase = "generating";
      operation.status = "Thinking…";
    }
    operation.sequence = payload.operation_sequence;
    if (state.session?.id === operation.sessionId) state.operation = operation;
    if (structural && state.operation === operation) {
      renderOperationStructure(operation);
      if (restoreComposerFocus) focusComposerAtEnd();
    } else if (!state.session && chatIsVisible()) {
      renderLibraryItems();
    }
  }

  function isTerminalEvent(type, kind) {
    return (
      [
        "turn.completed",
        "turn.failed",
        "turn.stopped",
        "compaction.failed",
        "compaction.stopped",
        "operation.failed",
      ].includes(type) ||
      (type === "compaction.completed" && kind === "compact")
    );
  }

  async function settleOperation(type, payload, operation) {
    const sessionId = payload.session_id;
    const operationId = payload.operation_id;
    if (operation) cancelOperationRender(operation);
    if (state.activeOperations.get(sessionId)?.id === operationId) {
      state.activeOperations.delete(sessionId);
    }
    if (state.pendingCommands.get(sessionId)?.id === operationId) {
      state.pendingCommands.delete(sessionId);
    }
    if (state.session?.id !== sessionId) {
      if (!state.session && chatIsVisible()) renderLibraryItems();
      return;
    }
    selectVisibleOperation();
    if (type === "operation.failed") {
      if (payload.kind === "send") rejectSubmittedDraft(operationId);
      renderSessionPreservingScroll();
      setNotice(payload.message || "Chat operation failed.", "error");
      return;
    }
    await reloadSession();
    if (type === "compaction.failed") {
      setNotice(payload.message || "Compaction failed.", "error");
    }
  }

  function refreshFeedState() {
    if (state.session) refreshComposerState();
    const newChat = document.querySelector('[data-testid="chat-new"]');
    if (newChat) newChat.disabled = !mutationsReady();
  }

  function focusComposerAtEnd() {
    const textarea = document.getElementById("chatComposer");
    restoreComposerSelection(
      textarea
        ? {
            start: textarea.value.length,
            end: textarea.value.length,
            direction: "none",
          }
        : null,
    );
  }

  function draftKey(sessionId) {
    return `fcc.chat.draft.${sessionId}`;
  }

  function loadDraft(sessionId) {
    state.draftSessionId = sessionId;
    state.draft = "";
    state.draftOperationId = null;
    state.draftSubmittedText = "";
    try {
      const stored = JSON.parse(sessionStorage.getItem(draftKey(sessionId)) || "null");
      if (!stored || typeof stored !== "object") return;
      if (typeof stored.text === "string") state.draft = stored.text;
      if (typeof stored.operationId === "string") {
        state.draftOperationId = stored.operationId;
      }
      if (typeof stored.submittedText === "string") {
        state.draftSubmittedText = stored.submittedText;
      }
    } catch {
      // In-memory draft state remains usable when browser storage is unavailable.
    }
  }

  function saveDraft() {
    if (!state.draftSessionId) return;
    try {
      if (!state.draft && !state.draftOperationId) {
        sessionStorage.removeItem(draftKey(state.draftSessionId));
        return;
      }
      sessionStorage.setItem(
        draftKey(state.draftSessionId),
        JSON.stringify({
          text: state.draft,
          operationId: state.draftOperationId,
          submittedText: state.draftSubmittedText,
        }),
      );
    } catch {
      // In-memory draft state remains usable when browser storage is unavailable.
    }
  }

  function removeDraft(sessionId) {
    try {
      sessionStorage.removeItem(draftKey(sessionId));
    } catch {
      // Deletion remains correct even when browser storage is unavailable.
    }
    if (state.draftSessionId === sessionId) {
      state.draft = "";
      state.draftSessionId = null;
      state.draftOperationId = null;
      state.draftSubmittedText = "";
    }
  }

  function confirmSubmittedDraft(operationId) {
    if (state.draftOperationId !== operationId) return;
    if (state.draft === state.draftSubmittedText) state.draft = "";
    state.draftOperationId = null;
    state.draftSubmittedText = "";
    saveDraft();
    const textarea = document.getElementById("chatComposer");
    if (textarea) textarea.value = state.draft;
  }

  function rejectSubmittedDraft(operationId) {
    if (state.draftOperationId !== operationId) return;
    if (!state.draft) state.draft = state.draftSubmittedText;
    state.draftOperationId = null;
    state.draftSubmittedText = "";
    saveDraft();
  }

  function rejectCommand(operation, message) {
    cancelOperationRender(operation);
    if (state.pendingCommands.get(operation.sessionId) === operation) {
      state.pendingCommands.delete(operation.sessionId);
    }
    if (operation.action === "send") rejectSubmittedDraft(operation.id);
    if (state.operation === operation) {
      selectVisibleOperation();
      renderSessionPreservingScroll();
      setNotice(message, "error");
    }
  }

  function clearPendingCommand(operation) {
    if (state.pendingCommands.get(operation.sessionId) === operation) {
      state.pendingCommands.delete(operation.sessionId);
    }
    if (
      operation.action === "send" &&
      state.turns.some((turn) => turn.operation_id === operation.id)
    ) {
      confirmSubmittedDraft(operation.id);
    }
    cancelOperationRender(operation);
  }

  function markCommandUncertain(operation) {
    clearPendingCommand(operation);
    if (state.operation !== operation) return;
    selectVisibleOperation();
    renderSessionPreservingScroll();
    setNotice(
      "The connection was lost before FCC could confirm this command. Check the chat before trying it again.",
      "error",
    );
  }

  function reconcileAmbiguousCommand(operation) {
    if (state.pendingCommands.get(operation.sessionId) !== operation) return;
    restartEventFeed();
  }

  function detachVisibleOperation() {
    if (!state.operation) return;
    cancelOperationRender(state.operation);
    commitPendingDeltas(state.operation);
    state.operation = null;
  }

  async function refresh(path = window.location.pathname) {
    if (!state.initialized) return;
    const requestVersion = ++state.bootstrapRequestVersion;
    const routeVersion = state.routeVersion;
    let bootstrap;
    try {
      bootstrap = await state.api("/admin/api/chat/bootstrap");
    } catch (error) {
      bootstrap = {
        available: false,
        message: error.message,
        models: [],
        preferences: null,
      };
    }
    if (requestVersion !== state.bootstrapRequestVersion) return;
    state.bootstrap = bootstrap;
    if (!chatIsVisible() || routeVersion !== state.routeVersion) return;
    await route(path);
  }

  async function activate(path) {
    if (!state.initialized) {
      renderLoading();
      return;
    }
    state.routeVersion += 1;
    renderLoading();
    await withBufferedEvents(() => refresh(path));
  }

  function chatIsVisible() {
    const view = root()?.closest(".admin-view");
    return Boolean(view && !view.hidden);
  }

  async function route(path) {
    const version = ++state.routeVersion;
    const sessionId = routedSessionId(path);
    if (!sessionId) {
      await showLibrary(version);
      return;
    }
    await showSession(sessionId, version);
  }

  function routedSessionId(path) {
    return path.match(/^\/admin\/chat\/([0-9a-f-]+)$/i)?.[1].toLowerCase() || null;
  }

  function renderLoading() {
    state.modelComboboxes.clear();
    const container = root();
    container.replaceChildren(node("div", "chat-empty", "Loading Chat Sessions…"));
  }

  function renderUnavailable() {
    state.modelComboboxes.clear();
    const container = node("section", "chat-empty");
    container.append(
      node("h3", "", "Chat Sessions unavailable"),
      node(
        "p",
        "",
        state.bootstrap?.message || "Chat storage could not be opened.",
      ),
    );
    root().replaceChildren(container);
  }

  async function showLibrary(version) {
    detachVisibleOperation();
    state.session = null;
    if (!state.bootstrap?.available) {
      renderUnavailable();
      return;
    }
    renderLibraryShell();
    await loadLibrary(true, version);
  }

  function renderLibraryShell() {
    state.modelComboboxes.clear();
    const header = node("header", "chat-library-header");
    const copy = node("div");
    copy.append(
      node("h2", "", "Chat Sessions"),
      node("p", "", "Talk directly to any configured FCC model."),
    );
    const newButton = button("New chat", "primary-button", createSession);
    newButton.dataset.testid = "chat-new";
    newButton.disabled = !mutationsReady();
    header.append(copy, newButton);

    const search = node("input", "chat-search");
    search.type = "search";
    search.placeholder = "Search chats";
    search.value = state.libraryQuery;
    search.setAttribute("aria-label", "Search chats");
    let timer = null;
    search.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        state.libraryQuery = search.value;
        loadLibrary(true, state.routeVersion);
      }, 200);
    });

    const notice = node("div", "chat-notice");
    notice.id = "chatNotice";
    notice.hidden = true;
    const list = node("div", "chat-session-list");
    list.id = "chatSessionList";
    const more = button("Load more", "secondary-button chat-load-more", () =>
      loadLibrary(false, state.routeVersion),
    );
    more.id = "chatLoadMore";
    more.hidden = true;
    root().replaceChildren(header, search, notice, list, more);
  }

  async function loadLibrary(reset, version) {
    const list = document.getElementById("chatSessionList");
    if (!list) return;
    const query = state.libraryQuery;
    const cursor = reset ? null : state.libraryCursor;
    if (!reset && (!cursor || state.libraryLoadMore)) return;
    const requestVersion = reset
      ? ++state.libraryRequestVersion
      : state.libraryRequestVersion;
    const loadMore = reset ? null : {};
    if (reset) {
      state.libraryItems = [];
      state.libraryCursor = null;
      state.libraryLoadMore = null;
      list.replaceChildren(node("div", "chat-empty", "Loading chats…"));
    } else {
      state.libraryLoadMore = loadMore;
      const more = document.getElementById("chatLoadMore");
      if (more) more.disabled = true;
    }
    const params = new URLSearchParams({ query, limit: "25" });
    if (cursor) params.set("cursor", cursor);
    const isCurrent = () =>
      version === state.routeVersion &&
      requestVersion === state.libraryRequestVersion &&
      !state.session &&
      query === state.libraryQuery &&
      (reset || state.libraryCursor === cursor);
    try {
      const page = await state.api(`/admin/api/chat/sessions?${params}`);
      if (!isCurrent()) return;
      const merged = new Map(
        state.libraryItems.map((session) => [session.id, session]),
      );
      page.sessions.forEach((session) => {
        if (state.deletedSessions.has(session.id)) return;
        const current = reduceSessionSummary(session);
        if (current && matchesLibraryQuery(current)) merged.set(current.id, current);
      });
      state.libraryItems = [...merged.values()].filter(matchesLibraryQuery);
      sortLibraryItems();
      state.libraryCursor = page.next_cursor;
      renderLibraryItems();
    } catch (error) {
      if (isCurrent()) setNotice(error.message, "error");
    } finally {
      if (loadMore && state.libraryLoadMore === loadMore) {
        state.libraryLoadMore = null;
        const more = document.getElementById("chatLoadMore");
        if (more) more.disabled = false;
      }
    }
  }

  function renderLibraryItems() {
    const list = document.getElementById("chatSessionList");
    if (!list) return;
    list.replaceChildren();
    if (!state.libraryItems.length) {
      list.appendChild(
        node(
          "div",
          "chat-empty",
          state.libraryQuery ? "No matching chats." : "Start your first chat.",
        ),
      );
    }
    state.libraryItems.forEach((session) => {
      const item = button("", "chat-session-card", () => openSession(session.id));
      const heading = node("strong", "", session.title);
      const preview = node("p", "", session.preview || "No messages yet");
      const meta = node("span", "", `${session.model} · ${relativeTime(session.updated_at)}`);
      item.append(heading, preview, meta);
      const active = state.activeOperations.get(session.id);
      if (active) {
        item.appendChild(
          node(
            "span",
            "chat-session-status",
            active.action === "compact" ? "Compacting…" : "Thinking…",
          ),
        );
      }
      list.appendChild(item);
    });
    const more = document.getElementById("chatLoadMore");
    if (more) {
      more.hidden = !state.libraryCursor;
      more.disabled = Boolean(state.libraryLoadMore);
    }
  }

  async function createSession() {
    if (!mutationsReady()) return;
    setNotice("Creating chat…");
    try {
      const session = await state.api("/admin/api/chat/sessions", {
        method: "POST",
        body: "{}",
      });
      openSession(session.id);
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  function openSession(id) {
    window.history.pushState({}, "", `/admin/chat/${id}`);
    route(window.location.pathname);
  }

  async function showSession(id, version, freshnessRetry = false) {
    if (!state.bootstrap?.available) {
      renderUnavailable();
      return;
    }
    const refreshingVisibleSession =
      state.session?.id === id && Boolean(document.getElementById("chatTranscript"));
    if (!refreshingVisibleSession) renderLoading();
    try {
      const detail = await state.api(`/admin/api/chat/sessions/${id}`);
      if (version !== state.routeVersion) return;
      const titleEdit = refreshingVisibleSession ? captureTitleEdit() : null;
      const composerSelection = refreshingVisibleSession
        ? captureComposerSelection()
        : null;
      if (!applyDetail(detail)) {
        if (!freshnessRetry && !state.deletedSessions.has(id)) {
          await showSession(id, version, true);
        }
        return;
      }
      if (refreshingVisibleSession) {
        renderSessionPreservingScroll(titleEdit);
        restoreComposerSelection(composerSelection);
      } else {
        renderSession();
      }
    } catch (error) {
      if (version !== state.routeVersion) return;
      const empty = node("section", "chat-empty");
      empty.append(
        node("h3", "", "Could not open this chat"),
        node("p", "", error.message),
        button("Back to chats", "secondary-button", goLibrary),
      );
      root().replaceChildren(empty);
    }
  }

  function applyDetail(detail) {
    const sessionId = detail.session.id;
    const knownRevision = state.sessionRevisions.get(sessionId) || 0;
    if (
      state.deletedSessions.has(sessionId) ||
      detail.session.revision < knownRevision ||
      (state.session?.id === sessionId &&
        detail.session.revision < state.session.revision)
    ) {
      return false;
    }
    state.sessionRevisions.set(sessionId, detail.session.revision);
    invalidateEstimate();
    if (state.draftSessionId !== sessionId) {
      loadDraft(sessionId);
    }
    const pendingTurn =
      state.draftOperationId &&
      detail.turns.find((turn) => turn.operation_id === state.draftOperationId);
    if (pendingTurn) confirmSubmittedDraft(state.draftOperationId);
    state.session = detail.session;
    state.turns = detail.turns;
    state.nextBefore = detail.next_before;
    state.compaction = detail.compaction;
    state.context = detail.context;
    state.contextError = detail.context_error || "";
    selectVisibleOperation();
    if (state.draft && !state.operation) scheduleEstimate(true);
    return true;
  }

  function renderSession({ followLatest = true, scrollTop = 0 } = {}) {
    const session = state.session;
    if (!session) return;
    const previousTitle = document.querySelector(".chat-title");
    if (previousTitle instanceof HTMLInputElement) {
      state.retiredTitleInputs.add(previousTitle);
    }
    state.modelComboboxes.clear();
    const shell = node("div", "chat-session-shell");
    const header = renderSessionHeader(session);
    const notice = node("div", "chat-notice");
    notice.id = "chatNotice";
    notice.hidden = true;
    const scroller = node("div", "chat-transcript");
    scroller.id = "chatTranscript";
    scroller.setAttribute("aria-label", "Conversation");
    const composer = renderComposer();
    shell.append(header, notice, scroller, composer);
    root().replaceChildren(shell);
    renderTranscript();
    refreshComposerState();
    if (followLatest) {
      scrollLatest(false);
    } else {
      scroller.scrollTop = scrollTop;
    }
  }

  function renderSessionPreservingScroll(titleEdit = captureTitleEdit()) {
    const scroller = document.getElementById("chatTranscript");
    renderSession({
      followLatest: !scroller || nearBottom(scroller),
      scrollTop: scroller?.scrollTop || 0,
    });
    restoreTitleEdit(titleEdit);
  }

  function renderSessionHeader(session) {
    const header = node("header", "chat-session-header");
    const top = node("div", "chat-header-row");
    top.appendChild(button("← Chats", "chat-back-button", goLibrary));
    const title = node("input", "chat-title");
    title.value = session.title;
    title.maxLength = 200;
    title.setAttribute("aria-label", "Chat title");
    title.addEventListener("blur", () => {
      if (
        title.isConnected &&
        !state.retiredTitleInputs.has(title) &&
        title.value !== state.session?.title
      )
        void updateSession({ title: title.value });
    });
    title.addEventListener("keydown", (event) => {
      if (event.key === "Enter") title.blur();
    });
    top.appendChild(title);
    top.appendChild(button("Delete", "danger-button", deleteSession));

    const controls = node("div", "chat-controls");
    const compact = button("Compact now", "secondary-button", compactSession);
    compact.id = "chatCompact";
    controls.append(
      renderModelControl(session),
      renderReasoningControl(session),
      renderContextMeter(),
      compact,
      button("System prompt", "secondary-button", editSystemPrompt),
    );
    header.append(top, controls);
    return header;
  }

  function renderModelControl(session) {
    const group = node("div", "chat-control chat-model-control");
    const label = node("label", "", "Model");
    label.htmlFor = "chatModel";
    const input = node("input", "chat-model-input");
    input.id = "chatModel";
    input.type = "text";
    input.autocomplete = "off";
    input.value = session.model;
    input.setAttribute("aria-label", "Selected model");
    let committedModel = session.model;
    const availableModels = () =>
      (state.bootstrap?.models || []).map((option) => option.model_ref);
    const combobox = new window.FccModelCombobox(input, {
      listboxId: "chat-model-options",
      label: "model",
      values: availableModels,
      emptyMessage: () =>
        availableModels().length ? "No matching models." : "No models available.",
      registry: state.modelComboboxes,
      onSelect: (model) => {
        committedModel = model;
        void updateSession({ model });
      },
      onClose: () => {
        input.value = committedModel;
      },
    });
    group.append(label, combobox.element);
    return group;
  }

  function renderReasoningControl(session) {
    const group = node("label", "chat-control");
    group.appendChild(node("span", "", "Thinking"));
    const select = node("select");
    select.id = "chatReasoning";
    [
      ["off", "Off"],
      ["low", "Low"],
      ["medium", "Medium"],
      ["high", "High"],
      ["xhigh", "Extra High"],
      ["max", "Max"],
    ].forEach(([value, label]) => {
      const option = node("option", "", label);
      option.value = value;
      select.appendChild(option);
    });
    select.value = session.reasoning;
    select.addEventListener("change", () =>
      updateSession({ reasoning: select.value }),
    );
    group.appendChild(select);
    return group;
  }

  function renderContextMeter() {
    const meter = node("div", "chat-context-meter");
    meter.id = "chatContextMeter";
    updateContextMeter(meter);
    return meter;
  }

  function updateContextMeter(element = document.getElementById("chatContextMeter")) {
    if (!element) return;
    const context = state.context;
    if (!context || context.context_window_tokens === null) {
      element.textContent = "Context: Not reported";
      element.className = "chat-context-meter unknown";
      return;
    }
    const percent = Math.round((context.usage_ratio || 0) * 100);
    element.textContent = `Context: ${percent}% · ${formatCount(context.estimated_input_tokens)} / ${formatCount(context.context_window_tokens)}`;
    element.className = `chat-context-meter${percent >= 85 ? " warn" : ""}`;
  }

  function renderComposer() {
    const wrapper = node("div", "chat-composer");
    const textarea = node("textarea");
    textarea.id = "chatComposer";
    textarea.rows = 2;
    textarea.placeholder = "Message this model";
    textarea.value = state.draft;
    textarea.setAttribute("aria-label", "Message");
    textarea.addEventListener("input", () => {
      state.draft = textarea.value;
      state.draftOperationId = null;
      state.draftSubmittedText = "";
      saveDraft();
      refreshComposerState();
      scheduleEstimate();
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!document.getElementById("chatSend")?.disabled) sendMessage();
      }
    });
    const actions = node("div", "chat-composer-actions");
    const status = node("span", "chat-composer-status");
    status.id = "chatComposerStatus";
    status.setAttribute("aria-live", "polite");
    const send = button("Send", "primary-button", sendMessage);
    send.id = "chatSend";
    const stop = button("Stop", "danger-button", stopOperation);
    stop.id = "chatStop";
    stop.hidden = true;
    actions.append(status, send, stop);
    wrapper.append(textarea, actions);
    return wrapper;
  }

  function resizeComposer(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > textarea.clientHeight ? "auto" : "hidden";
  }

  function composerFocusIsUnclaimed() {
    const active = document.activeElement;
    return !active || active === document.body || active === document.documentElement;
  }

  function captureComposerSelection() {
    const textarea = document.getElementById("chatComposer");
    if (!textarea || document.activeElement !== textarea) return null;
    return {
      start: textarea.selectionStart,
      end: textarea.selectionEnd,
      direction: textarea.selectionDirection,
    };
  }

  function restoreComposerSelection(selection) {
    const textarea = document.getElementById("chatComposer");
    if (!textarea || textarea.disabled || !selection) return;
    const start = Math.min(selection.start, textarea.value.length);
    const end = Math.min(selection.end, textarea.value.length);
    textarea.focus({ preventScroll: true });
    textarea.setSelectionRange(start, end, selection.direction);
  }

  function captureTitleEdit() {
    const input = document.querySelector(".chat-title");
    if (
      !(input instanceof HTMLInputElement) ||
      document.activeElement !== input ||
      input.value === state.session?.title
    )
      return null;
    return {
      value: input.value,
      start: input.selectionStart,
      end: input.selectionEnd,
      direction: input.selectionDirection,
    };
  }

  function restoreTitleEdit(edit) {
    const input = document.querySelector(".chat-title");
    if (!(input instanceof HTMLInputElement) || !edit) return;
    input.value = edit.value;
    input.focus({ preventScroll: true });
    const start = Math.min(edit.start ?? edit.value.length, edit.value.length);
    const end = Math.min(edit.end ?? edit.value.length, edit.value.length);
    input.setSelectionRange(start, end, edit.direction || "none");
  }

  function renderTranscript() {
    const scroller = document.getElementById("chatTranscript");
    if (!scroller) return;
    if (state.operation) commitPendingDeltas(state.operation);
    scroller.replaceChildren();
    if (state.nextBefore) {
      scroller.appendChild(
        button("Load older messages", "secondary-button chat-older", loadOlder),
      );
    }
    if (!state.turns.length && !state.operation) {
      scroller.appendChild(
        node("div", "chat-empty chat-transcript-empty", "Start the conversation."),
      );
      return;
    }
    let dividerRendered = false;
    state.turns.forEach((turn, index) => {
      scroller.appendChild(renderUserMessage(turn));
      const replacingLatest =
        turn.id === state.operation?.turnId ||
        (index === state.turns.length - 1 &&
          !state.operation?.turnId &&
          ["retry", "regenerate"].includes(state.operation?.action));
      if (!replacingLatest) scroller.appendChild(renderAssistantMessage(turn));
      if (
        state.compaction &&
        turn.sequence === state.compaction.covered_through_sequence
      ) {
        scroller.appendChild(renderCompaction());
        dividerRendered = true;
      }
    });
    const operationHasTurn = state.turns.some(
      (turn) => turn.operation_id === state.operation?.id,
    );
    if (
      state.operation?.action === "send" &&
      state.operation.userText &&
      !operationHasTurn
    ) {
      const pending = node("article", "chat-message user-message");
      pending.append(
        node("div", "chat-message-label", "You"),
        node("div", "chat-message-plain", state.operation.userText),
      );
      scroller.appendChild(pending);
    }
    if (state.compaction && !dividerRendered) {
      scroller.insertBefore(renderCompaction(), scroller.firstChild);
    }
    if (state.operation && state.operation.action !== "compact") {
      scroller.appendChild(renderLiveAssistant(state.operation));
    }
  }

  function renderUserMessage(turn) {
    const message = node("article", "chat-message user-message");
    message.append(node("div", "chat-message-label", "You"));
    const body = node("div", "chat-message-plain", turn.user_text);
    message.appendChild(body);
    return message;
  }

  function renderAssistantMessage(turn) {
    const generation = turn.generation;
    const message = node("article", "chat-message assistant-message");
    const label = node("div", "chat-message-label", "Assistant");
    if (
      generation.actual_model &&
      generation.actual_model !== generation.requested_model
    ) {
      label.appendChild(
        node("span", "chat-fallback-label", `Answered by ${generation.actual_model}`),
      );
    }
    message.appendChild(label);
    generation.segments.forEach((segment) => {
      if (segment.kind === "thinking") {
        const details = document.createElement("details");
        details.className = "chat-thinking";
        details.appendChild(node("summary", "", "Thinking"));
        const content = node("div", "chat-markdown");
        content.innerHTML = segment.html;
        details.appendChild(content);
        message.appendChild(details);
      } else {
        const content = node("div", "chat-markdown");
        // The server renders this with raw HTML disabled and safe-link rules.
        content.innerHTML = segment.html;
        message.appendChild(content);
      }
    });
    if (!generation.segments.length && generation.status === "running") {
      message.appendChild(node("p", "chat-muted", "Running in another tab…"));
    }
    if (generation.status !== "completed" && generation.status !== "running") {
      const status = node("div", `chat-generation-status ${generation.status}`);
      status.textContent =
        generation.error_message || `${capitalize(generation.status)} answer`;
      message.appendChild(status);
    }
    if (turn === state.turns[state.turns.length - 1]) {
      const actions = node("div", "chat-message-actions");
      if (["failed", "stopped", "interrupted"].includes(generation.status)) {
        actions.appendChild(button("Retry", "secondary-button", retryMessage));
      }
      if (generation.status === "completed") {
        actions.appendChild(
          button("Regenerate", "secondary-button", regenerateMessage),
        );
      }
      message.appendChild(actions);
    }
    return message;
  }

  function renderLiveAssistant(operation) {
    const message = node("article", "chat-message assistant-message live-message");
    message.appendChild(node("div", "chat-message-label", "Assistant"));
    if (!operation.segments.length || operation.status !== "Thinking…") {
      const status = node(
        "p",
        "chat-muted chat-operation-status",
        operation.status,
      );
      status.setAttribute("aria-live", "polite");
      message.appendChild(status);
    }
    operation.segments.forEach((segment, ordinal) => {
      const content = node("div", "chat-message-plain");
      content.dataset.liveSegment = String(ordinal);
      content.appendChild(document.createTextNode(segment.text));
      if (segment.kind === "thinking") {
        const details = document.createElement("details");
        details.className = "chat-thinking";
        details.open = true;
        details.appendChild(node("summary", "", "Thinking"));
        details.appendChild(content);
        message.appendChild(details);
      } else {
        message.appendChild(content);
      }
    });
    return message;
  }

  function renderCompaction() {
    const details = document.createElement("details");
    details.className = "chat-compaction";
    details.appendChild(node("summary", "", "Earlier conversation compacted"));
    const summary = node("div", "chat-markdown");
    summary.innerHTML = state.compaction.summary_html;
    details.appendChild(summary);
    return details;
  }

  async function loadOlder() {
    const scroller = document.getElementById("chatTranscript");
    if (!scroller || !state.nextBefore || !state.session) return;
    const sessionId = state.session.id;
    const revision = state.session.revision;
    const before = state.nextBefore;
    const routeVersion = state.routeVersion;
    if (
      state.olderLoad?.sessionId === sessionId &&
      state.olderLoad.before === before
    )
      return;
    const request = { sessionId, before };
    state.olderLoad = request;
    const button = scroller.querySelector(".chat-older");
    if (button) button.disabled = true;
    const oldHeight = scroller.scrollHeight;
    const isCurrent = () =>
      routeVersion === state.routeVersion &&
      state.session?.id === sessionId &&
      state.session.revision === revision &&
      state.nextBefore === before &&
      document.getElementById("chatTranscript") === scroller;
    try {
      const page = await state.api(
        `/admin/api/chat/sessions/${sessionId}/turns?before=${before}&limit=50`,
      );
      if (!isCurrent()) return;
      state.turns = [...page.turns, ...state.turns];
      state.nextBefore = page.next_before;
      state.compaction = page.compaction;
      renderTranscript();
      scroller.scrollTop += scroller.scrollHeight - oldHeight;
    } catch (error) {
      if (isCurrent()) setNotice(error.message, "error");
    } finally {
      if (state.olderLoad === request) {
        state.olderLoad = null;
        if (button?.isConnected) button.disabled = false;
      }
    }
  }

  async function updateSession(changes) {
    if (!state.session || !mutationsReady()) return;
    if (state.operation && (changes.model || changes.reasoning)) return;
    const sessionId = state.session.id;
    const expectedRevision = state.session.revision;
    const routeVersion = state.routeVersion;
    try {
      const session = await state.api(
        `/admin/api/chat/sessions/${sessionId}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: expectedRevision,
            ...changes,
          }),
        },
      );
      if (
        routeVersion !== state.routeVersion ||
        state.session?.id !== sessionId ||
        state.session.revision !== expectedRevision ||
        state.deletedSessions.has(sessionId)
      )
        return;
      state.session = session;
      state.sessionRevisions.set(session.id, session.revision);
      renderSessionPreservingScroll();
      scheduleEstimate(true);
    } catch (error) {
      if (routeVersion !== state.routeVersion || state.session?.id !== sessionId)
        return;
      await reloadSession();
      setNotice(error.message, "error");
    }
  }

  async function deleteSession() {
    if (!state.session || !mutationsReady()) return;
    if (!window.confirm(`Permanently delete “${state.session.title}”?`)) return;
    const sessionId = state.session.id;
    try {
      await state.api(`/admin/api/chat/sessions/${sessionId}`, {
        method: "DELETE",
        body: JSON.stringify({ expected_revision: state.session.revision }),
      });
      goLibrary();
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  function goLibrary() {
    detachVisibleOperation();
    window.history.pushState({}, "", "/admin/chat");
    route(window.location.pathname);
  }

  function editSystemPrompt() {
    if (!mutationsReady()) return;
    const current = state.bootstrap.preferences?.system_prompt || "";
    const dialog = document.createElement("dialog");
    dialog.className = "chat-prompt-dialog";
    const form = document.createElement("form");
    form.method = "dialog";
    form.append(
      node("h3", "", "System prompt"),
      node(
        "p",
        "chat-muted",
        "Shared by every Chat Session and applied on the next turn.",
      ),
    );
    const textarea = node("textarea");
    textarea.value = current;
    textarea.rows = 12;
    textarea.setAttribute("aria-label", "System prompt");
    form.appendChild(textarea);
    const actions = node("div", "chat-dialog-actions");
    const reset = button("Reset to default", "secondary-button", async () => {
      try {
        const preferences = await state.api(
          "/admin/api/chat/preferences/system-prompt",
          { method: "DELETE" },
        );
        state.bootstrap.preferences = preferences;
        dialog.close();
        scheduleEstimate(true);
      } catch (error) {
        setNotice(error.message, "error");
      }
    });
    const cancel = button("Cancel", "secondary-button", () => dialog.close());
    const save = button("Save", "primary-button", async () => {
      try {
        const preferences = await state.api(
          "/admin/api/chat/preferences/system-prompt",
          { method: "PUT", body: JSON.stringify({ value: textarea.value }) },
        );
        state.bootstrap.preferences = preferences;
        dialog.close();
        scheduleEstimate(true);
      } catch (error) {
        setNotice(error.message, "error");
      }
    });
    actions.append(reset, cancel, save);
    form.appendChild(actions);
    dialog.appendChild(form);
    dialog.addEventListener("close", () => dialog.remove());
    document.body.appendChild(dialog);
    dialog.showModal();
    textarea.focus();
  }

  function invalidateEstimate() {
    window.clearTimeout(state.estimateTimer);
    state.estimateTimer = null;
    state.estimateVersion += 1;
  }

  function scheduleEstimate(immediate = false) {
    invalidateEstimate();
    const version = state.estimateVersion;
    state.estimateTimer = window.setTimeout(
      () => updateEstimate(version),
      immediate ? 0 : 250,
    );
  }

  async function updateEstimate(version) {
    const session = state.session;
    if (!session || state.operation || version !== state.estimateVersion) return;
    const textarea = document.getElementById("chatComposer");
    const revision = session.revision;
    const draft = textarea?.value || "";
    try {
      const context = await state.api(
        `/admin/api/chat/sessions/${session.id}/estimate`,
        {
          method: "POST",
          body: JSON.stringify({ draft }),
        },
      );
      if (
        state.session?.id !== session.id ||
        state.session.revision !== revision ||
        state.draft !== draft ||
        state.operation ||
        version !== state.estimateVersion
      )
        return;
      state.context = context;
      state.contextError = "";
      updateContextMeter();
      refreshComposerState();
    } catch (error) {
      if (
        state.session?.id !== session.id ||
        state.session.revision !== revision ||
        state.draft !== draft ||
        state.operation ||
        version !== state.estimateVersion
      )
        return;
      state.context = null;
      state.contextError = error.message;
      updateContextMeter();
      refreshComposerState();
      setNotice(error.message, "error");
    }
  }

  function modelOption(modelRef) {
    return (state.bootstrap.models || []).find(
      (option) => option.model_ref === modelRef,
    );
  }

  function sendBlockReason() {
    if (!state.session) return "Chat unavailable";
    if (!mutationsReady()) return "Reconnecting…";
    if (state.operation) return "A chat operation is already running";
    const latest = state.turns[state.turns.length - 1];
    if (latest?.generation?.status === "running") return "This chat is running";
    const option = modelOption(state.session.model);
    if (!option) return "Choose an available model";
    if (option.supports_reasoning === false && state.session.reasoning !== "off") {
      return "This model requires Thinking Off";
    }
    if (state.contextError) return state.contextError;
    if (
      state.context?.usable_input_tokens !== null &&
      state.context?.estimated_input_tokens >
        state.context?.usable_input_tokens &&
      !state.context?.can_compact
    ) {
      return "This message exceeds the model context";
    }
    return "";
  }

  function refreshComposerState() {
    const textarea = document.getElementById("chatComposer");
    const send = document.getElementById("chatSend");
    const stop = document.getElementById("chatStop");
    const compact = document.getElementById("chatCompact");
    const status = document.getElementById("chatComposerStatus");
    if (!textarea || !send || !stop || !status) return;
    resizeComposer(textarea);
    const blocked = sendBlockReason();
    const latest = state.turns[state.turns.length - 1];
    const busy = Boolean(
      state.operation || latest?.generation?.status === "running",
    );
    send.disabled = Boolean(blocked) || !textarea.value.trim();
    send.hidden = Boolean(state.operation);
    stop.hidden = !state.operation;
    textarea.disabled = Boolean(state.operation && !state.operation.accepted);
    status.textContent =
      state.operation?.action === "compact"
        ? state.operation.status
        : state.operation
          ? ""
          : blocked;
    document
      .querySelectorAll(
        ".chat-controls button, .chat-controls input, .chat-controls select",
      )
      .forEach((control) => {
        control.disabled = busy || !mutationsReady();
      });
    document
      .querySelectorAll(".chat-header-row .danger-button, .chat-title")
      .forEach((control) => {
        control.disabled = !mutationsReady();
      });
    if (compact) {
      compact.disabled = busy || !state.context?.can_compact;
    }
  }

  async function sendMessage() {
    const textarea = document.getElementById("chatComposer");
    const text = textarea?.value || "";
    if (!text.trim() || !state.session) return;
    await runOperation("send", { text });
  }

  async function retryMessage() {
    await runOperation("retry", {});
  }

  async function regenerateMessage() {
    await runOperation("regenerate", {});
  }

  async function compactSession() {
    await runOperation("compact", {});
  }

  async function runOperation(action, extra) {
    if (!state.session || state.operation || !mutationsReady()) return;
    invalidateEstimate();
    const activeElementId = document.activeElement?.id;
    const operation = createOperation({
      id: crypto.randomUUID(),
      sessionId: state.session.id,
      action,
      phase: action === "compact" ? "compacting" : "generating",
      userText: extra.text || "",
      returnFocusToComposer:
        action === "send" &&
        (activeElementId === "chatComposer" || activeElementId === "chatSend"),
      commandPending: true,
    });
    if (action === "send") {
      state.draftOperationId = operation.id;
      state.draftSubmittedText = operation.userText;
      saveDraft();
    }
    state.pendingCommands.set(operation.sessionId, operation);
    state.operation = operation;
    renderSessionPreservingScroll();
    try {
      const acknowledgement = await state.api(
        `/admin/api/chat/sessions/${operation.sessionId}/${action}`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_revision: state.session.revision,
            operation_id: operation.id,
            ...extra,
          }),
        },
      );
      if (
        acknowledgement.session_id !== operation.sessionId ||
        acknowledgement.operation_id !== operation.id ||
        acknowledgement.kind !== action
      ) {
        throw new Error("The server returned an invalid operation acknowledgement.");
      }
      operation.commandPending = false;
      operation.serverObserved = true;
    } catch (error) {
      operation.commandPending = false;
      const active = state.activeOperations.get(operation.sessionId);
      const pending = state.pendingCommands.get(operation.sessionId);
      if (
        active?.id === operation.id ||
        (pending !== operation && operation.serverObserved)
      )
        return;
      if (error.status) {
        rejectCommand(operation, error.message);
        return;
      }
      operation.ambiguousError = error.message;
      reconcileAmbiguousCommand(operation);
    }
  }

  function commitPendingDeltas(operation) {
    const updates = [];
    operation.segments.forEach((segment, ordinal) => {
      if (!segment?.pending.length) return;
      const delta = segment.pending.join("");
      segment.pending.length = 0;
      segment.text += delta;
      updates.push({ ordinal, delta });
    });
    return updates;
  }

  function cancelOperationRender(operation) {
    if (operation.renderFrame === null) return;
    window.cancelAnimationFrame(operation.renderFrame);
    operation.renderFrame = null;
  }

  function scheduleLiveRender(operation) {
    if (operation.renderFrame !== null) return;
    operation.renderFrame = window.requestAnimationFrame(() => {
      operation.renderFrame = null;
      if (state.operation !== operation) return;
      const scroller = document.getElementById("chatTranscript");
      const shouldFollow = scroller ? nearBottom(scroller) : true;
      commitPendingDeltas(operation).forEach(({ ordinal, delta }) => {
        const content = scroller?.querySelector(
          `[data-live-segment="${ordinal}"]`,
        );
        const text = content?.firstChild;
        if (text?.nodeType === Node.TEXT_NODE) {
          text.appendData(delta);
        } else if (content) {
          content.textContent = operation.segments[ordinal].text;
        }
      });
      if (shouldFollow) scrollLatest(false);
    });
  }

  function renderOperationStructure(operation) {
    cancelOperationRender(operation);
    commitPendingDeltas(operation);
    const scroller = document.getElementById("chatTranscript");
    const shouldFollow = scroller ? nearBottom(scroller) : true;
    renderTranscript();
    refreshComposerState();
    if (shouldFollow) scrollLatest(false);
  }

  async function stopOperation() {
    const operation = state.operation;
    if (!operation) return;
    operation.status = "Stopping…";
    renderOperationStructure(operation);
    try {
      const result = await state.api(
        `/admin/api/chat/sessions/${operation.sessionId}/stop`,
        {
          method: "POST",
          body: JSON.stringify({ operation_id: operation.id }),
        },
      );
      if (!result.stopped) restartEventFeed();
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  async function reloadSession() {
    const id = state.session?.id;
    if (!id) return;
    const selection = captureComposerSelection();
    try {
      const detail = await state.api(`/admin/api/chat/sessions/${id}`);
      if (state.session?.id !== id) return;
      const titleEdit = captureTitleEdit();
      if (!applyDetail(detail)) return;
      renderSessionPreservingScroll(titleEdit);
      restoreComposerSelection(selection);
    } catch (error) {
      if (error.status === 404 && state.session?.id === id) {
        detachVisibleOperation();
        window.history.replaceState({}, "", "/admin/chat");
        await route(window.location.pathname);
        return;
      }
      setNotice(error.message, "error");
    }
  }

  function nearBottom(scroller) {
    return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  }

  function scrollLatest(smooth) {
    const scroller = document.getElementById("chatTranscript");
    if (!scroller) return;
    scroller.scrollTo({
      top: scroller.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }

  function relativeTime(timestamp) {
    const elapsed = Math.max(0, Date.now() - timestamp);
    if (elapsed < 60_000) return "just now";
    if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
    if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
    return new Date(timestamp).toLocaleDateString();
  }

  function formatCount(value) {
    return new Intl.NumberFormat(undefined, {
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(value);
  }

  function capitalize(value) {
    return value ? value[0].toUpperCase() + value.slice(1) : value;
  }

  window.ChatSessions = { initialize, activate, refresh };

  window.addEventListener("offline", () => {
    if (!state.initialized) return;
    state.feedSyncVersion += 1;
    state.feedStatus = "reconnecting";
    refreshFeedState();
  });

  window.addEventListener("online", () => {
    if (state.initialized && state.feedStatus !== "live") restartEventFeed();
  });

  document.addEventListener("pointerdown", (event) => {
    state.modelComboboxes.forEach((combobox) => {
      if (combobox.isOpen && !combobox.element.contains(event.target)) {
        combobox.close();
      }
    });
  });
})();
