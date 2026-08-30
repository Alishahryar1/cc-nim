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
    draft: "",
    draftSessionId: null,
    draftOperationId: null,
    routeVersion: 0,
    estimateTimer: null,
    estimateVersion: 0,
    foreignPollTimer: null,
    serverOperationActive: false,
    eventChannel: null,
    modelComboboxes: new Set(),
  };

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
    if (typeof window.BroadcastChannel === "function") {
      state.eventChannel = new BroadcastChannel("fcc-chat-sessions");
      state.eventChannel.addEventListener("message", handleCrossTabEvent);
    }
    state.initialized = true;
    if (chatIsVisible()) {
      await activate(window.location.pathname);
    } else {
      await refresh();
    }
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
    stopForeignOperationPoll();
    state.routeVersion += 1;
    renderLoading();
    await refresh(path);
  }

  function chatIsVisible() {
    const view = root()?.closest(".admin-view");
    return Boolean(view && !view.hidden);
  }

  async function route(path) {
    stopForeignOperationPoll();
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
    cancelLocalStream();
    state.session = null;
    state.serverOperationActive = false;
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
      state.libraryItems = reset
        ? page.sessions
        : [...state.libraryItems, ...page.sessions];
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
      list.appendChild(item);
    });
    const more = document.getElementById("chatLoadMore");
    if (more) {
      more.hidden = !state.libraryCursor;
      more.disabled = Boolean(state.libraryLoadMore);
    }
  }

  async function createSession() {
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

  async function showSession(id, version) {
    if (!state.bootstrap?.available) {
      renderUnavailable();
      return;
    }
    renderLoading();
    try {
      const detail = await state.api(`/admin/api/chat/sessions/${id}`);
      if (version !== state.routeVersion) return;
      applyDetail(detail);
      renderSession();
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
    invalidateEstimate();
    if (state.draftSessionId !== detail.session.id) {
      state.draft = "";
      state.draftSessionId = detail.session.id;
      state.draftOperationId = null;
    } else if (
      state.draftOperationId &&
      detail.turns.some((turn) => turn.operation_id === state.draftOperationId)
    ) {
      state.draft = "";
      state.draftOperationId = null;
    }
    state.session = detail.session;
    state.turns = detail.turns;
    state.nextBefore = detail.next_before;
    state.compaction = detail.compaction;
    state.context = detail.context;
    state.contextError = detail.context_error || "";
    state.serverOperationActive = Boolean(detail.active_operation);
    if (state.draft && !state.operation) scheduleEstimate(true);
  }

  function renderSession({ followLatest = true, scrollTop = 0 } = {}) {
    const session = state.session;
    if (!session) return;
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
    syncForeignOperationPoll();
  }

  function renderSessionPreservingScroll() {
    const scroller = document.getElementById("chatTranscript");
    renderSession({
      followLatest: !scroller || nearBottom(scroller),
      scrollTop: scroller?.scrollTop || 0,
    });
  }

  function renderSessionHeader(session) {
    const header = node("header", "chat-session-header");
    const top = node("div", "chat-header-row");
    top.appendChild(button("← Chats", "chat-back-button", goLibrary));
    const title = node("input", "chat-title");
    title.value = session.title;
    title.maxLength = 200;
    title.setAttribute("aria-label", "Chat title");
    title.addEventListener("change", () => updateSession({ title: title.value }));
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

  function renderTranscript() {
    const scroller = document.getElementById("chatTranscript");
    if (!scroller) return;
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
        index === state.turns.length - 1 &&
        ["retry", "regenerate"].includes(state.operation?.action);
      if (!replacingLatest) scroller.appendChild(renderAssistantMessage(turn));
      if (
        state.compaction &&
        turn.sequence === state.compaction.covered_through_sequence
      ) {
        scroller.appendChild(renderCompaction());
        dividerRendered = true;
      }
    });
    if (state.operation?.action === "send" && state.operation.userText) {
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
    if (!state.session) return;
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
        state.session.revision !== expectedRevision
      )
        return;
      state.session = session;
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
    if (!state.session) return;
    if (!window.confirm(`Permanently delete “${state.session.title}”?`)) return;
    const sessionId = state.session.id;
    try {
      await state.api(`/admin/api/chat/sessions/${sessionId}`, {
        method: "DELETE",
        body: JSON.stringify({ expected_revision: state.session.revision }),
      });
      state.eventChannel?.postMessage({ type: "session.deleted", sessionId });
      goLibrary();
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  function goLibrary() {
    cancelLocalStream();
    window.history.pushState({}, "", "/admin/chat");
    route(window.location.pathname);
  }

  function handleCrossTabEvent(event) {
    const message = event.data;
    if (
      !message ||
      message.type !== "session.deleted" ||
      typeof message.sessionId !== "string" ||
      (state.session?.id !== message.sessionId &&
        routedSessionId(window.location.pathname) !== message.sessionId)
    )
      return;
    cancelLocalStream();
    window.history.replaceState({}, "", "/admin/chat");
    route(window.location.pathname);
  }

  function editSystemPrompt() {
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
    if (state.operation) return "A chat operation is already running";
    const latest = state.turns[state.turns.length - 1];
    if (
      state.serverOperationActive ||
      latest?.generation?.status === "running"
    ) {
      return "This chat is running in another tab";
    }
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
      state.operation ||
        state.serverOperationActive ||
        latest?.generation?.status === "running",
    );
    send.disabled = Boolean(blocked) || !textarea.value.trim();
    send.hidden = Boolean(state.operation);
    stop.hidden = !state.operation;
    textarea.disabled = Boolean(state.operation);
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
        control.disabled = busy;
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
    if (!state.session || state.operation) return;
    invalidateEstimate();
    let failure = null;
    const operation = {
      id: crypto.randomUUID(),
      sessionId: state.session.id,
      action,
      controller: new AbortController(),
      segments: [],
      sequence: 0,
      status: action === "compact" ? "Compacting…" : "Thinking…",
      userText: extra.text || "",
      accepted: false,
      failureMessage: "",
      renderFrame: null,
    };
    if (action === "send") {
      state.draftOperationId = operation.id;
    }
    state.operation = operation;
    renderSessionPreservingScroll();
    try {
      const response = await fetch(
        `/admin/api/chat/sessions/${operation.sessionId}/${action}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            expected_revision: state.session.revision,
            operation_id: operation.id,
            ...extra,
          }),
          cache: "no-store",
          signal: operation.controller.signal,
        },
      );
      if (!response.ok) throw await responseError(response);
      if (!response.body) throw new Error("The browser could not open the chat stream.");
      await consumeEvents(response.body, operation);
      if (operation.failureMessage) {
        failure = new Error(operation.failureMessage);
      }
    } catch (error) {
      if (action === "send" && !operation.accepted) {
        state.draft = operation.userText;
        state.draftSessionId = operation.sessionId;
        state.draftOperationId = operation.id;
      }
      if (error.name !== "AbortError") failure = error;
    } finally {
      cancelOperationRender(operation);
      if (state.operation === operation) state.operation = null;
      if (state.session?.id === operation.sessionId) await reloadSession();
      if (failure) setNotice(failure.message, "error");
    }
  }

  async function consumeEvents(body, operation) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    const frameParts = [];
    let boundaryTail = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        const chunk = decoder.decode(value || new Uint8Array(), { stream: !done });
        const probe = boundaryTail + chunk;
        const boundary = /\r?\n\r?\n/g;
        const prefixLength = boundaryTail.length;
        let chunkStart = 0;
        for (const match of probe.matchAll(boundary)) {
          const chunkEnd = match.index + match[0].length - prefixLength;
          frameParts.push(chunk.slice(chunkStart, chunkEnd));
          applyStreamFrame(frameParts.join(""), operation);
          frameParts.length = 0;
          chunkStart = chunkEnd;
        }
        const remainder = chunk.slice(chunkStart);
        if (remainder) frameParts.push(remainder);
        boundaryTail = chunkStart
          ? remainder.slice(-3)
          : (boundaryTail + chunk).slice(-3);
        if (done) {
          const finalFrame = frameParts.join("");
          if (finalFrame.trim()) applyStreamFrame(finalFrame, operation);
          return;
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  function applyStreamFrame(frame, operation) {
    if (state.operation !== operation) return;
    let event = "message";
    let sequence = 0;
    let payload = null;
    frame.split(/\r?\n/).forEach((line) => {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("id:")) sequence = Number.parseInt(line.slice(3), 10);
      if (line.startsWith("data:")) payload = JSON.parse(line.slice(5).trim());
    });
    if (!payload || payload.operation_id !== operation.id) return;
    if (!Number.isFinite(sequence) || sequence <= operation.sequence) return;
    if (
      state.session?.id === operation.sessionId &&
      Number.isInteger(payload.revision)
    ) {
      state.session.revision = payload.revision;
    }
    let liveDelta = false;
    if (event === "turn.started") {
      operation.accepted = true;
      if (operation.action === "send") {
        state.draft = "";
        state.draftSessionId = operation.sessionId;
        state.draftOperationId = null;
        const textarea = document.getElementById("chatComposer");
        if (textarea) textarea.value = "";
      }
    } else if (event === "segment.started") {
      operation.segments[payload.ordinal] = {
        kind: payload.kind,
        text: "",
        pending: [],
      };
    } else if (event === "segment.delta") {
      const segment = operation.segments[payload.ordinal];
      if (segment) {
        segment.pending.push(payload.delta);
        liveDelta = true;
      }
    } else if (event === "compaction.completed") {
      operation.status = "Compacted";
    } else if (event === "compaction.failed") {
      operation.failureMessage = payload.message || "Compaction failed";
      operation.status = operation.failureMessage;
    } else if (event === "compaction.stopped") {
      operation.status = "Stopped";
    } else if (event === "turn.failed") {
      operation.failureMessage = payload.message || "Generation failed";
      operation.status = operation.failureMessage;
    } else if (event === "turn.stopped") {
      operation.status = "Stopped";
    }
    operation.sequence = sequence;
    if (liveDelta) {
      scheduleLiveRender(operation);
      return;
    }
    renderOperationStructure(operation);
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
      await state.api(`/admin/api/chat/sessions/${operation.sessionId}/stop`, {
        method: "POST",
        body: JSON.stringify({ operation_id: operation.id }),
      });
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  function cancelLocalStream() {
    if (!state.operation) return;
    cancelOperationRender(state.operation);
    state.operation.controller.abort();
    state.operation = null;
  }

  async function reloadSession() {
    const id = state.session?.id;
    if (!id) return;
    try {
      const detail = await state.api(`/admin/api/chat/sessions/${id}`);
      if (state.session?.id !== id) return;
      applyDetail(detail);
      renderSessionPreservingScroll();
    } catch (error) {
      if (error.status === 404 && state.session?.id === id) {
        cancelLocalStream();
        window.history.replaceState({}, "", "/admin/chat");
        await route(window.location.pathname);
        return;
      }
      setNotice(error.message, "error");
    }
  }

  function stopForeignOperationPoll() {
    window.clearTimeout(state.foreignPollTimer);
    state.foreignPollTimer = null;
  }

  function syncForeignOperationPoll() {
    stopForeignOperationPoll();
    const latest = state.turns[state.turns.length - 1];
    const chatView = root()?.closest(".admin-view");
    if (
      !state.session ||
      state.operation ||
      (!state.serverOperationActive &&
        latest?.generation?.status !== "running") ||
      chatView?.hidden
    ) {
      return;
    }
    const sessionId = state.session.id;
    state.foreignPollTimer = window.setTimeout(async () => {
      state.foreignPollTimer = null;
      if (state.session?.id !== sessionId || state.operation) return;
      await reloadSession();
      syncForeignOperationPoll();
    }, 750);
  }

  async function responseError(response) {
    try {
      const payload = await response.json();
      return new Error(payload.detail || `${response.status} ${response.statusText}`);
    } catch {
      return new Error(`${response.status} ${response.statusText}`);
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

  document.addEventListener("pointerdown", (event) => {
    state.modelComboboxes.forEach((combobox) => {
      if (combobox.isOpen && !combobox.element.contains(event.target)) {
        combobox.close();
      }
    });
  });
})();
