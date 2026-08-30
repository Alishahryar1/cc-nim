(() => {
  const state = {
    api: null,
    initialized: false,
    active: false,
    bootstrap: null,
    session: null,
    turns: [],
    compaction: null,
    nextBefore: null,
    context: null,
    contextError: "",
    libraryCursor: null,
    libraryItems: [],
    libraryQuery: "",
    operation: null,
    draft: "",
    draftSessionId: null,
    routeVersion: 0,
    estimateTimer: null,
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
    state.initialized = true;
    try {
      state.bootstrap = await api("/admin/api/chat/bootstrap");
    } catch (error) {
      state.bootstrap = {
        available: false,
        message: error.message,
        models: [],
        preferences: null,
      };
    }
    if (state.active) await route(window.location.pathname);
  }

  async function activate(path) {
    state.active = true;
    if (!state.initialized) {
      renderLoading();
      return;
    }
    await route(path);
  }

  async function route(path) {
    const version = ++state.routeVersion;
    const match = path.match(/^\/admin\/chat\/([0-9a-f-]+)$/i);
    if (!match) {
      await showLibrary(version);
      return;
    }
    await showSession(match[1], version);
  }

  function renderLoading() {
    const container = root();
    container.replaceChildren(node("div", "chat-empty", "Loading Chat Sessions…"));
  }

  function renderUnavailable() {
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
    if (!state.bootstrap?.available) {
      renderUnavailable();
      return;
    }
    renderLibraryShell();
    await loadLibrary(true, version);
  }

  function renderLibraryShell() {
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
    if (reset) {
      state.libraryItems = [];
      state.libraryCursor = null;
      list.replaceChildren(node("div", "chat-empty", "Loading chats…"));
    }
    const params = new URLSearchParams({ query: state.libraryQuery, limit: "25" });
    if (!reset && state.libraryCursor) params.set("cursor", state.libraryCursor);
    try {
      const page = await state.api(`/admin/api/chat/sessions?${params}`);
      if (version !== state.routeVersion || state.session) return;
      state.libraryItems = reset
        ? page.sessions
        : [...state.libraryItems, ...page.sessions];
      state.libraryCursor = page.next_cursor;
      renderLibraryItems();
    } catch (error) {
      setNotice(error.message, "error");
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
    if (more) more.hidden = !state.libraryCursor;
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
    if (state.draftSessionId !== detail.session.id) {
      state.draft = "";
      state.draftSessionId = detail.session.id;
    }
    state.session = detail.session;
    state.turns = detail.turns;
    state.nextBefore = detail.next_before;
    state.compaction = detail.compaction;
    state.context = detail.context;
    state.contextError = detail.context_error || "";
  }

  function renderSession() {
    const session = state.session;
    if (!session) return;
    const shell = node("div", "chat-session-shell");
    const header = renderSessionHeader(session);
    const notice = node("div", "chat-notice");
    notice.id = "chatNotice";
    notice.hidden = true;
    const scroller = node("div", "chat-transcript");
    scroller.id = "chatTranscript";
    scroller.setAttribute("aria-label", "Conversation");
    const jump = button("Jump to latest", "secondary-button chat-jump", () =>
      scrollLatest(true),
    );
    jump.id = "chatJumpLatest";
    jump.hidden = true;
    scroller.addEventListener("scroll", () => {
      jump.hidden = nearBottom(scroller);
    });
    const composer = renderComposer();
    shell.append(header, notice, scroller, jump, composer);
    root().replaceChildren(shell);
    renderTranscript();
    refreshComposerState();
    scrollLatest(false);
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
    const group = node("label", "chat-control chat-model-control");
    group.appendChild(node("span", "", "Model"));
    const filter = node("input", "chat-model-filter");
    filter.type = "search";
    filter.placeholder = "Filter models";
    filter.setAttribute("aria-label", "Filter models");
    const select = node("select");
    select.id = "chatModel";
    select.setAttribute("aria-label", "Selected model");
    fillModelOptions(select, "", session.model);
    filter.addEventListener("input", () =>
      fillModelOptions(select, filter.value, state.session.model),
    );
    select.addEventListener("change", () => updateSession({ model: select.value }));
    group.append(filter, select);
    return group;
  }

  function fillModelOptions(select, query, selected) {
    const folded = query.trim().toLowerCase();
    const matches = (state.bootstrap.models || []).filter(
      (option) => !folded || option.model_ref.toLowerCase().includes(folded),
    );
    const groups = new Map();
    matches.forEach((option) => {
      if (!groups.has(option.provider_id)) groups.set(option.provider_id, []);
      groups.get(option.provider_id).push(option);
    });
    select.replaceChildren();
    if (!(state.bootstrap.models || []).some((item) => item.model_ref === selected)) {
      const unavailable = node("option", "", `${selected} (Unavailable)`);
      unavailable.value = selected;
      select.appendChild(unavailable);
    }
    [...groups.keys()].sort().forEach((provider) => {
      const group = document.createElement("optgroup");
      group.label = provider;
      groups.get(provider).forEach((option) => {
        const item = node("option", "", option.model_id);
        item.value = option.model_ref;
        group.appendChild(item);
      });
      select.appendChild(group);
    });
    select.value = selected;
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
    element.textContent = `Context: ${percent}% · ${formatCount(context.estimated_input_tokens)} input + ${formatCount(context.completion_tokens)} output`;
    element.className = `chat-context-meter${percent >= 85 ? " warn" : ""}`;
  }

  function renderComposer() {
    const wrapper = node("div", "chat-composer");
    const textarea = node("textarea");
    textarea.id = "chatComposer";
    textarea.rows = 3;
    textarea.placeholder = "Message this model";
    textarea.value = state.draft;
    textarea.setAttribute("aria-label", "Message");
    textarea.addEventListener("input", () => {
      state.draft = textarea.value;
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
        index === state.turns.length - 1 && state.operation?.action === "retry";
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
    if (state.operation?.segments?.length) {
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
    operation.segments.forEach((segment) => {
      if (segment.kind === "thinking") {
        const details = document.createElement("details");
        details.className = "chat-thinking";
        details.open = true;
        details.appendChild(node("summary", "", "Thinking"));
        details.appendChild(node("div", "chat-message-plain", segment.text));
        message.appendChild(details);
      } else {
        message.appendChild(node("div", "chat-message-plain", segment.text));
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
    const oldHeight = scroller.scrollHeight;
    try {
      const page = await state.api(
        `/admin/api/chat/sessions/${state.session.id}/turns?before=${state.nextBefore}&limit=50`,
      );
      state.turns = [...page.turns, ...state.turns];
      state.nextBefore = page.next_before;
      state.compaction = page.compaction;
      renderTranscript();
      scroller.scrollTop += scroller.scrollHeight - oldHeight;
    } catch (error) {
      setNotice(error.message, "error");
    }
  }

  async function updateSession(changes) {
    if (!state.session) return;
    if (state.operation && (changes.model || changes.reasoning)) return;
    try {
      const session = await state.api(
        `/admin/api/chat/sessions/${state.session.id}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: state.session.revision,
            ...changes,
          }),
        },
      );
      state.session = session;
      renderSession();
      scheduleEstimate(true);
    } catch (error) {
      await reloadSession();
      setNotice(error.message, "error");
    }
  }

  async function deleteSession() {
    if (!state.session) return;
    if (!window.confirm(`Permanently delete “${state.session.title}”?`)) return;
    try {
      await state.api(`/admin/api/chat/sessions/${state.session.id}`, {
        method: "DELETE",
        body: JSON.stringify({ expected_revision: state.session.revision }),
      });
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

  function scheduleEstimate(immediate = false) {
    window.clearTimeout(state.estimateTimer);
    state.estimateTimer = window.setTimeout(updateEstimate, immediate ? 0 : 250);
  }

  async function updateEstimate() {
    const session = state.session;
    if (!session || state.operation) return;
    const textarea = document.getElementById("chatComposer");
    try {
      const context = await state.api(
        `/admin/api/chat/sessions/${session.id}/estimate`,
        {
          method: "POST",
          body: JSON.stringify({ draft: textarea?.value || "" }),
        },
      );
      if (state.session?.id !== session.id) return;
      state.context = context;
      state.contextError = "";
      updateContextMeter();
      refreshComposerState();
    } catch (error) {
      if (state.session?.id !== session.id) return;
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
    if (latest?.generation?.status === "running") {
      return "This chat is running in another tab";
    }
    const option = modelOption(state.session.model);
    if (!option) return "Choose an available model";
    if (option.supports_reasoning === false && state.session.reasoning !== "off") {
      return "This model requires Thinking Off";
    }
    if (state.contextError) return state.contextError;
    if (
      state.context?.usage_ratio !== null &&
      state.context?.usage_ratio > 1 &&
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
    const blocked = sendBlockReason();
    send.disabled = Boolean(blocked) || !textarea.value.trim();
    send.hidden = Boolean(state.operation);
    stop.hidden = !state.operation;
    textarea.disabled = Boolean(state.operation);
    status.textContent = state.operation ? state.operation.status : blocked;
    document.querySelectorAll(".chat-controls button, .chat-controls select").forEach(
      (control) => {
        control.disabled = Boolean(state.operation);
      },
    );
    if (compact) {
      compact.disabled = Boolean(state.operation) || !state.context?.can_compact;
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
    };
    state.operation = operation;
    renderSession();
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
    } catch (error) {
      if (action === "send" && !operation.accepted) {
        state.draft = operation.userText;
        state.draftSessionId = operation.sessionId;
      }
      if (error.name !== "AbortError") failure = error;
    } finally {
      if (state.operation === operation) state.operation = null;
      if (state.session?.id === operation.sessionId) await reloadSession();
      if (failure) setNotice(failure.message, "error");
    }
  }

  async function consumeEvents(body, operation) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || "";
        frames.forEach((frame) => applyStreamFrame(frame, operation));
        if (done) {
          if (buffer.trim()) applyStreamFrame(buffer, operation);
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
    if (event === "turn.started") {
      operation.accepted = true;
      if (operation.action === "send") {
        state.draft = "";
        state.draftSessionId = operation.sessionId;
        const textarea = document.getElementById("chatComposer");
        if (textarea) textarea.value = "";
      }
    } else if (event === "segment.started") {
      operation.segments[payload.ordinal] = { kind: payload.kind, text: "" };
    } else if (event === "segment.delta") {
      const segment = operation.segments[payload.ordinal];
      if (segment) segment.text += payload.delta;
    } else if (event.startsWith("compaction.")) {
      operation.status =
        event === "compaction.completed" ? "Compacted" : "Compacting…";
    } else if (event === "turn.failed") {
      operation.status = payload.message || "Generation failed";
    } else if (event === "turn.stopped") {
      operation.status = "Stopped";
    }
    operation.sequence = sequence;
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
    refreshComposerState();
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
      renderSession();
    } catch (error) {
      setNotice(error.message, "error");
    }
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
    return new Intl.NumberFormat(undefined, { notation: "compact" }).format(value);
  }

  function capitalize(value) {
    return value ? value[0].toUpperCase() + value.slice(1) : value;
  }

  window.ChatSessions = { initialize, activate };
})();
