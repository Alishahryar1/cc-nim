(() => {
  "use strict";
  const UI = window.SessionUI;
  const modelComboboxes = new Set();
  let modelControl,
    reasoningControl,
    catalog = [],
    catalogLoaded = false,
    settingsPending = false;
  let visibleIds = [],
    query = "",
    libraryToken = 0,
    listFlight = null,
    listAgain = false,
    retryTimer = null;
  let syncNotice = "",
    availabilityNotice = "";
  const base = "/admin/api/code";
  const activeStatuses = new Set(["preparing", "running", "stopping"]);
  const records = new Map();
  const deleted = new Set();
  const sends = new Set();
  let api,
    root,
    feed,
    epoch,
    connected = false,
    synchronized = false;
  let desiredPath = null,
    selected = null,
    rendered = null,
    syncToken = 0,
    viewToken = 0;
  let nextCursor = null,
    libraryLoading = false,
    notice = "",
    available = false;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function button(text, action, className = "secondary-button") {
    const node = element("button", text, className);
    node.type = "button";
    node.addEventListener("click", action);
    return node;
  }
  function saved(id) {
    try {
      return JSON.parse(sessionStorage.getItem(`fcc.code.${id}`)) || {};
    } catch {
      return {};
    }
  }
  function save(id, value) {
    try {
      sessionStorage.setItem(`fcc.code.${id}`, JSON.stringify(value));
    } catch {
      /* This tab still works when browser storage is unavailable. */
    }
  }
  function get(id) {
    if (!records.has(id))
      records.set(id, {
        id,
        session: null,
        run: null,
        version: -1,
        cursor: -1,
        items: new Map(),
        prompts: new Map(),
        activePrompts: new Map(),
        runs: new Map(),
        loaded: false,
        nextBefore: null,
      });
    return records.get(id);
  }
  function busy(record) {
    return activeStatuses.has(record?.run?.status);
  }
  function pending(record) {
    return [...(record?.activePrompts.values() || [])].some(
      (entry) => entry.active,
    );
  }

  function ready(record) {
    return synchronized && record?.loaded && record.session?.status === "ready";
  }
  function mergeEntries(target, entries, version) {
    for (const value of entries || []) {
      if ((target.get(value.id)?.version ?? -1) <= version)
        target.set(value.id, { value, version });
    }
  }
  function merge(data) {
    if (data.epoch !== epoch) return;
    const id = data.session_id || data.session?.id;
    if (!id || deleted.has(id)) return;
    const record = get(id),
      version = data.version;
    if (version >= record.version) {
      if (!record.session || data.session.revision >= record.session.revision)
        record.session = data.session;
      if (
        record.run?.id !== data.run?.id ||
        !activeStatuses.has(data.run?.status)
      )
        record.runNotice = "";
      record.run = data.run;
      record.version = version;
      record.cursor = Math.max(record.cursor, data.cursor || 0);
      if (record.run) accepted(id, record.run.id);
    }
    mergeEntries(
      record.runs,
      [...(data.runs || []), ...(data.run ? [data.run] : [])],
      version,
    );
    mergeEntries(record.items, data.item ? [data.item] : data.items, version);
    const prompts = data.prompt ? [data.prompt] : data.prompts || [];
    mergeEntries(record.prompts, prompts, version);
    for (const prompt of prompts) {
      if ((record.activePrompts.get(prompt.id)?.version ?? -1) <= version)
        record.activePrompts.set(prompt.id, {
          active: ["pending", "answering"].includes(prompt.status),
          version,
        });
    }
    if (data.active_prompt_ids) {
      const active = new Set(data.active_prompt_ids);
      for (const promptId of new Set([
        ...record.activePrompts.keys(),
        ...active,
      ])) {
        if ((record.activePrompts.get(promptId)?.version ?? -1) <= version)
          record.activePrompts.set(promptId, {
            active: active.has(promptId),
            version,
          });
      }
    }
  }

  function receive(type, event) {
    const data = JSON.parse(event.data);
    if (data.epoch !== epoch) return;
    const previousRevision = records.get(data.session_id)?.session?.revision;
    if (type === "session.deleted") {
      deleted.add(data.session_id);
      records.delete(data.session_id);
      visibleIds = visibleIds.filter((id) => id !== data.session_id);
      if (selected === data.session_id) {
        navigate(null);
        notice = "Session deleted.";
      }
    } else {
      merge(data);
      if (type === "session.notice" && data.session_id === selected)
        notice = data.message;
      if (type === "run.notice") {
        const record = records.get(data.session_id);
        if (record && busy(record) && data.version >= record.version)
          record.runNotice = data.will_retry
            ? `Retrying… ${data.message}`
            : data.message;
      }
    }
    if (
      !selected &&
      (type === "session.deleted" ||
        data.session?.revision !== previousRevision)
    )
      void refreshLibrary();
    render();
  }

  async function list(cursor = null) {
    const requestEpoch = epoch,
      token = syncToken,
      route = viewToken,
      search = query,
      request = ++libraryToken;
    const params = new URLSearchParams({ query: search });
    if (cursor) params.set("cursor", cursor);
    const data = await api(`${base}/sessions?${params}`);
    if (
      requestEpoch !== epoch ||
      data.epoch !== epoch ||
      token !== syncToken ||
      route !== viewToken ||
      search !== query ||
      request !== libraryToken
    )
      return;
    const ids = cursor ? [...visibleIds] : [];
    for (const session of data.sessions) {
      if (deleted.has(session.id)) continue;
      const record = get(session.id);
      if (
        (!record.session || session.revision >= record.session.revision) &&
        record.cursor <= data.cursor
      )
        record.session = session;
      if (!ids.includes(session.id)) ids.push(session.id);
    }
    visibleIds = ids;
    nextCursor = data.next_cursor;
    render();
  }
  async function refreshLibrary() {
    listAgain = true;
    if (listFlight) return listFlight;
    listFlight = (async () => {
      while (listAgain && !selected) {
        listAgain = false;
        try {
          await list();
        } catch (error) {
          restart(error.message);
          break;
        }
      }
    })();
    try {
      await listFlight;
    } finally {
      listFlight = null;
    }
  }

  async function detail(id, before = null) {
    const requestEpoch = epoch,
      token = viewToken,
      connection = syncToken;
    let data;
    try {
      data = await api(
        `${base}/sessions/${id}${before ? `/items?before=${encodeURIComponent(before)}` : ""}`,
      );
    } catch (error) {
      if (token !== viewToken || connection !== syncToken || selected !== id)
        return;
      if (error.status === 404) {
        deleted.add(id);
        records.delete(id);
        navigate(null);
        notice = "This session was deleted.";
        render();
        return;
      }
      throw error;
    }
    if (
      requestEpoch !== epoch ||
      data.epoch !== epoch ||
      token !== viewToken ||
      connection !== syncToken ||
      selected !== id ||
      deleted.has(id)
    )
      return;
    merge(data);
    const record = get(id);
    record.loaded = true;
    record.nextBefore = data.next_before;
    render();
  }
  async function bootstrap() {
    const token = syncToken;
    try {
      const data = await api(`${base}/bootstrap`);
      if (token !== syncToken || (epoch && data.epoch !== epoch)) return;
      available = data.available;
      catalog = data.models;
      catalogLoaded = true;
      availabilityNotice = data.message || "";
    } catch (error) {
      if (token === syncToken) catalogLoaded = false;
      throw error;
    }
  }

  async function synchronize(readyData) {
    const token = ++syncToken;
    synchronized = false;
    connected = true;
    clearTimeout(retryTimer);
    retryTimer = null;
    if (epoch !== readyData.epoch) {
      epoch = readyData.epoch;
      records.clear();
      deleted.clear();
      rendered = null;
    }
    const included = new Set(
      readyData.sessions.map((summary) => summary.session_id),
    );
    for (const record of records.values()) {
      if (
        !included.has(record.id) &&
        busy(record) &&
        record.cursor <= readyData.cursor
      ) {
        // The run is no longer active; detail supplies its actual outcome.
        record.run = null;
        record.runNotice = "";
        record.loaded = false;
        record.cursor = readyData.cursor;
      }
    }
    for (const summary of readyData.sessions)
      merge({ ...summary, cursor: readyData.cursor });
    render();
    try {
      const results = await Promise.allSettled([
        list(),
        selected ? detail(selected) : Promise.resolve(),
        bootstrap(),
      ]);
      if (token !== syncToken || !connected) return;
      for (const result of results)
        if (result.status === "rejected") throw result.reason;
      synchronized = true;
      syncNotice = "";
      for (const id of records.keys()) if (saved(id).pending) void deliver(id);
    } catch (error) {
      if (token === syncToken) restart(error.message);
    }
    render();
  }
  function restart(message = "Reconnecting…") {
    if (feed) feed.close();
    feed = null;
    connected = false;
    synchronized = false;
    ++syncToken;
    syncNotice = message;
    if (retryTimer === null)
      retryTimer = setTimeout(() => {
        retryTimer = null;
        connect();
      }, 1000);
    render();
  }

  function connect() {
    if (feed || retryTimer !== null) return;
    const source = new EventSource(`${base}/events`);
    feed = source;
    source.addEventListener("feed.ready", (event) => {
      if (feed === source) void synchronize(JSON.parse(event.data));
    });
    for (const type of [
      "session.updated",
      "session.deleted",
      "run.updated",
      "item.updated",
      "prompt.updated",
      "session.notice",
      "run.notice",
    ])
      source.addEventListener(type, (event) => {
        if (feed === source) receive(type, event);
      });
    const disconnected = async () => {
      if (feed !== source) return;
      restart();
      // A stopped store cannot open its event feed. Read its availability
      // so that the retry screen can explain why it is unavailable.
      try {
        await bootstrap();
        render();
      } catch {
        /* The scheduled connection retries the reads. */
      }
    };
    source.addEventListener("feed.resync_required", disconnected);
    source.onerror = disconnected;
  }

  function navigate(id) {
    const path = id ? `/admin/code/${id}` : "/admin/code";
    history.pushState({}, "", path);
    activate(path);
  }
  function activate(path) {
    desiredPath = path;
    if (!api) return;
    const id = /^\/admin\/code\/([0-9a-f-]+)$/.exec(path)?.[1] || null;
    if (id !== selected || !rendered) {
      selected = id;
      ++viewToken;
      rendered = null;
      settingsPending = false;
      notice = "";
      render();
      if (connected && epoch) {
        const reading = id ? detail(id) : refreshLibrary();
        const token = viewToken;
        void reading.catch((error) => {
          if (token === viewToken) restart(error.message);
        });
      }
    }
    connect();
  }

  function accepted(id, operationId) {
    const state = saved(id);
    if (state.pending?.operation_id !== operationId) return;
    const oldText = state.pending.text;
    delete state.pending;
    if (state.draft === oldText) state.draft = "";
    save(id, state);
    if (
      selected === id &&
      root?.querySelector("#codeComposer")?.value === oldText
    )
      root.querySelector("#codeComposer").value = "";
  }
  async function deliver(id) {
    const command = saved(id).pending;
    if (!command || sends.has(id) || !synchronized) return;
    sends.add(id);
    render();
    try {
      const receipt = await api(`${base}/sessions/${id}/turns`, {
        method: "POST",
        body: JSON.stringify(command),
      });
      accepted(id, receipt.id);
    } catch (error) {
      const state = saved(id);
      if (state.pending?.operation_id === command.operation_id) {
        if (error.status && error.status < 500) {
          delete state.pending;
          save(id, state);
        }
        if (selected === id) notice = error.message;
      }
      if (error.status === 400 || error.status === 409) {
        try {
          await bootstrap();
        } catch (failure) {
          restart(failure.message);
        }
      }
    } finally {
      sends.delete(id);
      render();
    }
  }
  function submit() {
    const record = records.get(selected);
    if (
      !ready(record) ||
      busy(record) ||
      pending(record) ||
      !available ||
      settingsPending ||
      selectionError(record)
    )
      return;
    const state = saved(selected),
      text = root.querySelector("#codeComposer").value;
    if (!state.pending && !text.trim()) return;
    state.draft = text;
    state.pending ||= {
      operation_id: crypto.randomUUID(),
      expected_revision: record.session.revision,
      expected_epoch: epoch,
      text,
    };
    save(selected, state);
    notice = "";
    void deliver(selected);
  }
  async function stop() {
    const record = records.get(selected);
    if (!busy(record)) return;
    const id = selected,
      run = record.run.id;
    try {
      await api(`${base}/sessions/${id}/stop`, {
        method: "POST",
        body: JSON.stringify({ operation_id: run }),
      });
    } catch (error) {
      if (selected === id) notice = error.message;
    }
    render();
  }
  function dialog(title, build, actionLabel, action) {
    const node = element("dialog", undefined, "code-dialog"),
      form = element("form");
    const heading = element("h3", title),
      error = element("p", "", "code-notice");
    heading.id = `code-dialog-${crypto.randomUUID()}`;
    node.setAttribute("aria-labelledby", heading.id);
    form.append(heading);
    const value = build(form);
    const actions = element("div", undefined, "code-actions");
    const cancel = button("Cancel", () => node.close());
    const confirm = button(actionLabel, () => {}, "primary-button");
    confirm.type = "submit";
    actions.append(cancel, confirm);
    form.append(error, actions);
    node.append(form);
    node.addEventListener("close", () => node.remove());
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      confirm.disabled = true;
      try {
        await action(value);
        node.close();
      } catch (failure) {
        error.textContent = failure.message;
        confirm.disabled = false;
      }
    });
    root.append(node);
    node.showModal();
  }
  function newSession() {
    const id = crypto.randomUUID();
    dialog(
      "New code session",
      (form) => {
        const harness = element("select");
        harness.append(new Option("Codex", "codex"));
        const folder = element("input");
        folder.required = true;
        folder.autocomplete = "off";
        label(form, "Harness", harness);
        label(form, "Folder", folder);
        form.append(
          element(
            "p",
            "Choose an existing folder on the computer running FCC.",
            "code-notice",
          ),
        );
        return folder;
      },
      "Create session",
      async (folder) => {
        await api(`${base}/sessions`, {
          method: "POST",
          body: JSON.stringify({
            session_id: id,
            harness: "codex",
            cwd: folder.value,
          }),
        });
        navigate(id);
      },
    );
  }
  async function updateSettings(changes) {
    const id = selected,
      record = records.get(id);
    if (!ready(record) || settingsPending) return;
    settingsPending = true;
    notice = "";
    renderControls();
    try {
      const session = await api(`${base}/sessions/${id}`, {
        method: "PATCH",
        body: JSON.stringify({
          expected_revision: record.session.revision,
          ...changes,
        }),
      });
      if (
        !deleted.has(id) &&
        session.revision >= (records.get(id)?.session?.revision || 0)
      )
        get(id).session = session;
    } catch (error) {
      if (selected === id) {
        notice = error.message;
        try {
          await detail(id);
        } catch (failure) {
          restart(failure.message);
        }
      }
    } finally {
      if (selected === id) {
        settingsPending = false;
        render();
      }
    }
  }
  function selectionError(record) {
    if (!record?.session) return "";
    if (!catalogLoaded) return "Model list unavailable. Reconnecting…";
    const model = catalog.find((model) => model.id === record.session.model);
    if (!model) return "Selected model is unavailable. Choose another model.";
    if (
      record.session.reasoning_effort &&
      !model.reasoning_efforts.includes(record.session.reasoning_effort)
    )
      return "Selected thinking effort is unavailable. Choose another effort or Model default.";
    return "";
  }

  function remove() {
    const record = records.get(selected),
      id = selected,
      revision = record.session.revision;
    if (record.session.status === "delete_uncertain") {
      void api(`${base}/sessions/${id}?expected_revision=${revision}`, {
        method: "DELETE",
      }).catch((error) => {
        notice = error.message;
        render();
      });
      return;
    }
    dialog(
      "Delete session?",
      (form) => {
        form.append(
          element(
            "p",
            "This deletes the conversation from FCC and Codex history. Your project files stay in place.",
          ),
        );
      },
      "Delete session",
      () =>
        api(`${base}/sessions/${id}?expected_revision=${revision}`, {
          method: "DELETE",
        }),
    );
  }
  function label(parent, title, input, description) {
    const node = element("label", undefined, "code-field");
    node.append(element("span", title), input);
    if (description) node.append(element("small", description));
    parent.append(node);
    return node;
  }
  function shell() {
    root.replaceChildren();
    modelComboboxes.clear();
    modelControl = reasoningControl = null;
    const message = element("div", "", "session-notice");
    message.id = "codeNotice";
    message.setAttribute("role", "status");
    if (selected) {
      const id = selected,
        record = get(id);
      const controls = element("div", undefined, "session-controls");
      modelControl = UI.modelControl(
        "codeModel",
        record.session?.model || "",
        () => catalog.map((model) => model.id),
        modelComboboxes,
        (model) => {
          void updateSettings({ model });
        },
      );
      reasoningControl = UI.reasoningControl(
        "codeReasoning",
        [],
        "",
        (value) => {
          void updateSettings({ reasoning_effort: value || null });
        },
      );
      controls.append(modelControl.group, reasoningControl.group);
      const deletion = button("Delete", remove, "danger-button");
      deletion.id = "codeDelete";
      const header = UI.header(
        "← Code sessions",
        () => navigate(null),
        record.session?.title || "Code session",
        "Code title",
        (title) => {
          if (
            selected === id &&
            title.value !== records.get(id)?.session?.title
          )
            void updateSettings({ title: title.value });
        },
        [deletion],
        controls,
      );
      header.append(element("p", "", "code-folder"));
      const transcript = element("div", undefined, "session-transcript");
      transcript.id = "codeTranscript";
      transcript.setAttribute("aria-label", "Conversation");
      const older = button(
        "Load older messages",
        async () => {
          const oldHeight = transcript.scrollHeight,
            top = transcript.scrollTop;
          older.disabled = true;
          try {
            await detail(id, records.get(id).nextBefore);
            if (selected === id)
              transcript.scrollTop = top + transcript.scrollHeight - oldHeight;
          } catch (error) {
            restart(error.message);
          }
          older.disabled = false;
          render();
        },
        "secondary-button session-older",
      );
      older.id = "codeOlder";
      transcript.append(
        older,
        element("div", undefined, "code-items"),
        element("div", undefined, "code-prompts"),
      );
      const composer = UI.composer(
        "code",
        saved(id).draft || "",
        "Ask Codex to work on this folder…",
        (value) => {
          save(id, { ...saved(id), draft: value });
          renderControls();
        },
        submit,
        () => {
          void stop();
        },
      );
      root.append(UI.shell(header, message, transcript, composer));
      UI.resizeComposer(composer.querySelector("textarea"));
    } else {
      const create = button("New code session", newSession, "primary-button");
      create.id = "codeNew";
      const header = UI.libraryHeader(
        "Code sessions",
        "Work with Codex in a project folder.",
        create,
      );
      const search = UI.search("Search titles and folders", query, (value) => {
        query = value;
        visibleIds = [];
        nextCursor = null;
        ++libraryToken;
        render();
        void refreshLibrary();
      });
      const more = button(
        "Load more sessions",
        async () => {
          libraryLoading = true;
          render();
          try {
            await list(nextCursor);
          } catch (error) {
            restart(error.message);
          }
          libraryLoading = false;
          render();
        },
        "secondary-button session-load-more",
      );
      more.id = "codeMore";
      const library = element("div", undefined, "session-list");
      library.id = "codeLibrary";
      const wrapper = element("div", undefined, "session-library");
      wrapper.append(header, search, message, library, more);
      root.append(wrapper);
    }
    rendered = selected || "library";
  }

  function render() {
    if (!root) return;
    if (rendered !== (selected || "library")) shell();
    const message = root.querySelector("#codeNotice");
    message.textContent =
      notice ||
      availabilityNotice ||
      syncNotice ||
      (!connected ? "Connecting…" : !synchronized ? "Synchronizing…" : "");
    message.hidden = !message.textContent;
    if (!selected) {
      const library = root.querySelector("#codeLibrary");
      for (const child of [...library.children])
        if (!child.dataset.id || !visibleIds.includes(child.dataset.id))
          child.remove();
      for (const record of visibleIds
        .map((id) => records.get(id))
        .filter((record) => record?.session)
        .sort(
          (a, b) =>
            b.session.updated_at - a.session.updated_at ||
            b.id.localeCompare(a.id),
        )) {
        let card = library.querySelector(
          `[data-id="${CSS.escape(record.id)}"]`,
        );
        if (!card) {
          card = UI.card("", "", "", () => navigate(record.id));
          card.dataset.id = record.id;
        }
        card.children[0].textContent = record.session.title;
        card.children[1].textContent = record.session.cwd;
        card.children[2].textContent = busy(record)
          ? "Running"
          : record.session.error || record.session.model;
        library.append(card);
      }
      if (!library.children.length)
        library.append(
          element(
            "div",
            synchronized
              ? query
                ? "No matching sessions."
                : "Start your first code session."
              : "Loading sessions…",
            "session-empty",
          ),
        );
      root.querySelector("#codeNew").disabled = !synchronized;
      const more = root.querySelector("#codeMore");
      more.hidden = !nextCursor;
      more.disabled = !synchronized || libraryLoading;
      return;
    }
    const record = records.get(selected),
      title = root.querySelector(".session-title");
    if (record?.session) {
      if (document.activeElement !== title) title.value = record.session.title;
      root.querySelector(".code-folder").textContent = record.session.cwd;
    }
    const transcript = root.querySelector("#codeTranscript"),
      bottom = UI.nearBottom(transcript),
      items = root.querySelector(".code-items");
    const runs = [...(record?.runs.values() || [])]
      .map((entry) => entry.value)
      .sort((a, b) => a.ordinal - b.ordinal);
    for (const run of runs) {
      const source = [...record.items.values()]
        .map((entry) => entry.value)
        .filter((item) => item.run_id === run.id)
        .sort((a, b) => a.sequence - b.sequence);
      if (!source.length) continue;
      let group = items.querySelector(`[data-run-id="${CSS.escape(run.id)}"]`);
      if (!group) {
        group = element("div", undefined, "code-run");
        group.dataset.runId = run.id;
        group.dataset.ordinal = run.ordinal;
        group.append(
          element("div", undefined, "code-run-items"),
          UI.message("assistant", "Codex"),
        );
        group.lastChild.classList.add("code-outcome");
        group.lastChild.append(element("div", "", "session-generation-status"));
        const next = [...items.children].find(
          (child) => Number(child.dataset.ordinal) > run.ordinal,
        );
        items.insertBefore(group, next || null);
      }
      for (const item of source) renderItem(group.firstChild, item);
      const outcome = group.querySelector(".code-outcome");
      outcome.hidden = !["failed", "interrupted"].includes(run.status);
      const status = outcome.querySelector(".session-generation-status");
      status.className = `session-generation-status ${run.status}`;
      status.textContent =
        run.error ||
        (run.status === "interrupted" ? "Turn stopped." : "This turn failed.");
    }
    const prompts = root.querySelector(".code-prompts");
    for (const { value: prompt } of record?.prompts.values() || [])
      renderPrompt(prompts, prompt);
    if (bottom) transcript.scrollTop = transcript.scrollHeight;
    root.querySelector("#codeOlder").hidden = !record?.nextBefore;
    renderControls();
  }

  function renderControls() {
    if (!selected || !root?.querySelector("#codeComposer")) return;
    const record = records.get(selected),
      isBusy = busy(record),
      input = root.querySelector("#codeComposer");
    const send = root.querySelector("#codeSend"),
      stop = root.querySelector("#codeStop");
    send.hidden = isBusy;
    stop.hidden = !isBusy;
    stop.disabled = !connected || record?.run?.stop_requested;
    send.textContent = saved(selected).pending ? "Retry Send" : "Send";
    send.disabled =
      !ready(record) ||
      !available ||
      pending(record) ||
      sends.has(selected) ||
      settingsPending ||
      !!selectionError(record) ||
      (!input.value.trim() && !saved(selected).pending);
    input.disabled = !record?.loaded || record?.session?.status !== "ready";
    root.querySelector(".session-title").disabled =
      !ready(record) || settingsPending;
    const deletion = root.querySelector("#codeDelete");
    deletion.textContent =
      record?.session?.status === "delete_uncertain"
        ? "Check deletion"
        : "Delete";
    deletion.disabled =
      !synchronized ||
      !record?.loaded ||
      record.session?.status === "deleting" ||
      isBusy ||
      pending(record);
    const disabled =
      !ready(record) ||
      isBusy ||
      pending(record) ||
      settingsPending ||
      !catalogLoaded;
    modelControl.update(record?.session?.model || "", disabled);
    const model = catalog.find((model) => model.id === record?.session?.model),
      effort = record?.session?.reasoning_effort;
    const options = [
      [
        "",
        model?.default_reasoning_effort
          ? `Model default (${model.default_reasoning_effort})`
          : "Model default",
      ],
      ...(model?.reasoning_efforts || []).map((value) => [value, value]),
    ];
    if (effort && !model?.reasoning_efforts.includes(effort))
      options.push([effort, `${effort} (unavailable)`]);
    reasoningControl.update(options, effort || "");
    reasoningControl.select.disabled =
      disabled || !model || (!model.reasoning_efforts.length && !effort);
    root.querySelector("#codeComposerStatus").textContent =
      record?.session?.error ||
      selectionError(record) ||
      (record?.session?.status !== "ready"
        ? "Deleting…"
        : record.run?.stop_requested && isBusy
          ? "Stopping…"
          : pending(record)
            ? "Waiting for input"
            : isBusy
              ? record.runNotice || "Working…"
              : "Codex");
    for (const node of root.querySelectorAll(".code-prompt")) {
      const prompt = record?.prompts.get(node.dataset.id)?.value;
      for (const control of node.querySelectorAll("input, select, button"))
        control.disabled =
          !ready(record) ||
          !record.activePrompts.get(node.dataset.id)?.active ||
          prompt?.status !== "pending" ||
          node.dataset.claiming === "true";
    }
    UI.resizeComposer(input);
  }

  function renderItem(parent, item) {
    let node = parent.querySelector(`[data-id="${CSS.escape(item.id)}"]`);
    if (!node) {
      node = UI.message(
        item.kind === "user" ? "user" : "assistant",
        item.kind === "user" ? "You" : "Codex",
      );
      node.classList.add("code-item");
      node.dataset.id = item.id;
      node.dataset.kind = item.kind;
      let content = node;
      if (item.kind === "reasoning") {
        content = UI.thinking();
        node.append(content);
      } else if (!["text", "user"].includes(item.kind)) {
        content = element("details", undefined, "code-tool");
        content.append(element("summary", item.title || "Tool"));
        node.append(content);
      }
      content.append(
        element(
          "div",
          "",
          `code-prose ${item.kind === "user" ? "session-message-plain" : "session-markdown"}`,
        ),
        element("pre", "", "code-item-detail"),
      );
      node.dataset.sequence = item.sequence;
      const next = [...parent.children].find(
        (child) => Number(child.dataset.sequence) > item.sequence,
      );
      parent.insertBefore(node, next || null);
    }
    const content = node.querySelector(".code-prose"),
      value = item.html ?? item.text;
    if (node.codeText !== value) {
      if (item.html != null) content.innerHTML = item.html;
      else content.textContent = item.text;
      node.codeText = value;
    }
    const detail = node.querySelector(".code-item-detail");
    detail.textContent = item.detail;
    detail.hidden = !item.detail;
  }

  function renderPrompt(parent, prompt) {
    let node = parent.querySelector(`[data-id="${CSS.escape(prompt.id)}"]`);
    if (!node) {
      node = element("form", undefined, "code-prompt");
      node.dataset.id = prompt.id;
      node.append(
        element("h3", prompt.form.title),
        element("pre", prompt.form.detail || ""),
      );
      buildPrompt(node, prompt);
      node.append(element("p", "", "code-prompt-state"));
      parent.append(node);
    }
    node.querySelector(".code-prompt-state").textContent =
      prompt.error ||
      (!records.get(selected)?.activePrompts.get(prompt.id)?.active &&
      ["pending", "answering"].includes(prompt.status)
        ? "No longer active"
        : "") ||
      {
        answering: "Answer sent…",
        resolved: "Resolved",
        expired: "No longer active",
      }[prompt.status] ||
      "";
  }
  function buildPrompt(form, prompt) {
    const sessionId = selected,
      actions = element("div", undefined, "code-actions");
    let responseId = null,
      submitting = false;
    const answer = async (value) => {
      if (
        submitting ||
        !ready(records.get(sessionId)) ||
        !records.get(sessionId).activePrompts.get(prompt.id)?.active
      )
        return;
      responseId ||= crypto.randomUUID();
      submitting = true;
      form.dataset.claiming = "true";
      renderControls();
      try {
        await api(
          `${base}/sessions/${sessionId}/prompts/${prompt.id}/responses`,
          {
            method: "POST",
            body: JSON.stringify({ response_id: responseId, answer: value }),
          },
        );
      } catch (error) {
        form.querySelector(".code-prompt-state").textContent = error.message;
      } finally {
        submitting = false;
        form.dataset.claiming = "false";
        renderControls();
      }
    };
    form.addEventListener("submit", (event) => event.preventDefault());
    const choice = (text, value) =>
      actions.append(
        button(text, () => {
          void answer(value);
        }),
      );
    if (prompt.kind === "approval") {
      for (const item of prompt.form.choices)
        choice(item.label, { choice: item.id });
    } else if (prompt.kind === "permissions") {
      const checks = prompt.form.choices.map((item) => {
        const input = element("input");
        input.type = "checkbox";
        input.value = item.id;
        const label = element("label", undefined, "code-choice");
        label.append(input, document.createTextNode(item.label));
        form.append(label);
        return input;
      });
      const scope = element("select");
      scope.append(
        new Option("This turn", "turn"),
        new Option("This session", "session"),
      );
      label(form, "Allow for", scope);
      actions.append(
        button("Allow selected", () => {
          void answer({
            selected: checks
              .filter((input) => input.checked)
              .map((input) => input.value),
            scope: scope.value,
          });
        }),
      );
      choice("Decline", { selected: [], scope: "turn" });
    } else if (prompt.kind === "questions") {
      const readers = prompt.form.questions.map((question) => {
        const field = element("fieldset");
        field.append(element("legend", question.label));
        form.append(field);
        if (question.options?.length) {
          const name = `question-${crypto.randomUUID()}`,
            radios = [];
          for (const option of question.options) {
            const radio = element("input");
            radio.type = "radio";
            radio.name = name;
            radio.value = option.label;
            radio.required = true;
            const optionLabel = element("label", undefined, "code-choice");
            optionLabel.append(
              radio,
              document.createTextNode(
                option.label +
                  (option.description ? ` — ${option.description}` : ""),
              ),
            );
            field.append(optionLabel);
            radios.push(radio);
          }
          let other;
          if (question.allow_other) {
            const radio = element("input");
            radio.type = "radio";
            radio.name = name;
            radio.value = "__other__";
            radios.push(radio);
            const otherLabel = element("label", undefined, "code-choice");
            otherLabel.append(radio, document.createTextNode("Other"));
            field.append(otherLabel);
            other = element("input");
            other.type = question.secret ? "password" : "text";
            other.autocomplete = "off";
            label(field, "Your answer", other);
            other.addEventListener("input", () => {
              radio.checked = true;
            });
          }
          return () => [
            question.id,
            [
              radios.find((radio) => radio.checked)?.value === "__other__"
                ? other.value
                : radios.find((radio) => radio.checked)?.value || "",
            ],
          ];
        }
        const input = element("input");
        input.type = question.secret ? "password" : "text";
        input.required = true;
        input.autocomplete = "off";
        label(field, question.header || "Your answer", input);
        return () => [question.id, [input.value]];
      });
      actions.append(
        button(
          "Submit answers",
          () => {
            if (form.reportValidity())
              void answer({
                answers: Object.fromEntries(readers.map((read) => read())),
              });
          },
          "primary-button",
        ),
      );
    } else {
      const readers = [];
      if (prompt.kind === "url") {
        try {
          const url = new URL(prompt.form.url);
          if (["https:", "http:"].includes(url.protocol)) {
            const link = element("a", "Open tool request");
            link.href = url.href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            form.append(link);
          } else throw new Error();
        } catch {
          form.append(element("p", "This tool supplied an unsupported link."));
        }
      } else if (prompt.form.unsupported)
        form.append(
          element(
            "p",
            "This tool's form cannot be displayed. Decline or cancel this request.",
          ),
        );
      else
        for (const field of prompt.form.fields || []) {
          const input = element(field.options?.length ? "select" : "input");
          if (field.options?.length) {
            input.multiple = field.type === "array";
            if (!input.multiple) input.append(new Option("Choose…", ""));
            field.options.forEach((option, index) =>
              input.append(new Option(option.label, String(index))),
            );
          } else
            input.type =
              field.type === "boolean"
                ? "checkbox"
                : ["integer", "number"].includes(field.type)
                  ? "number"
                  : "text";
          if (input.type === "number")
            input.step = field.type === "integer" ? "1" : "any";
          for (const [source, target] of [
            ["minimum", "min"],
            ["maximum", "max"],
            ["minLength", "minLength"],
            ["maxLength", "maxLength"],
          ])
            if (field[source] != null) input[target] = field[source];
          input.required = field.required && field.type !== "boolean";
          input.autocomplete = "off";
          if (field.default != null && !field.options?.length) {
            if (input.type === "checkbox") input.checked = field.default;
            else input.value = field.default;
          }
          label(form, field.label, input, field.description);
          readers.push(() => {
            if (input.type === "checkbox") return [field.id, input.checked];
            if (!input.value && !field.required) return null;
            if (field.options?.length)
              return [
                field.id,
                input.multiple
                  ? [...input.selectedOptions].map(
                      (option) => field.options[Number(option.value)].value,
                    )
                  : field.options[Number(input.value)].value,
              ];
            return [
              field.id,
              input.type === "number" ? Number(input.value) : input.value,
            ];
          });
        }
      if (!prompt.form.unsupported)
        actions.append(
          button(
            "Submit",
            () => {
              if (form.reportValidity())
                void answer({
                  action: "accept",
                  values: Object.fromEntries(
                    readers.map((read) => read()).filter(Boolean),
                  ),
                });
            },
            "primary-button",
          ),
        );
      choice("Decline", { action: "decline" });
      choice("Cancel request", { action: "cancel" });
    }
    form.append(actions);
  }
  window.CodeSessions = {
    initialize(client) {
      api = client;
      root = document.getElementById("codeRoot");
      if (desiredPath) activate(desiredPath);
    },
    activate,
    async refresh() {
      if (!api || !epoch) return;
      try {
        await bootstrap();
        render();
      } catch (error) {
        restart(error.message);
      }
    },
  };
  document.addEventListener("pointerdown", (event) => {
    for (const combobox of modelComboboxes)
      if (combobox.isOpen && !combobox.element.contains(event.target))
        combobox.close();
  });
})();
