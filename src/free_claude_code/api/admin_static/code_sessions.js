(() => {
  "use strict";
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
        loaded: false,
        nextBefore: null,
      });
    return records.get(id);
  }
  function busy(record) {
    return activeStatuses.has(record?.run?.status);
  }
  function pending(record) {
    return [...(record?.prompts.values() || [])].some(({ value }) =>
      ["pending", "answering"].includes(value.status),
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
      record.session = data.session;
      record.run = data.run;
      record.version = version;
      record.cursor = Math.max(record.cursor, data.cursor || 0);
      if (record.run) accepted(id, record.run.id);
    }
    mergeEntries(record.items, data.item ? [data.item] : data.items, version);
    mergeEntries(
      record.prompts,
      data.prompt ? [data.prompt] : data.prompts,
      version,
    );
  }
  function receive(type, event) {
    const data = JSON.parse(event.data);
    if (data.epoch !== epoch) return;
    if (type === "session.deleted") {
      deleted.add(data.session_id);
      records.delete(data.session_id);
      if (selected === data.session_id) {
        navigate(null);
        notice = "Session deleted.";
      }
    } else {
      merge(data);
      if (type === "session.notice" && data.session_id === selected)
        notice = data.message;
    }
    render();
  }
  async function list(cursor = null) {
    const requestEpoch = epoch,
      token = syncToken;
    const data = await api(
      `${base}/sessions${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ""}`,
    );
    if (requestEpoch !== epoch || data.epoch !== epoch || token !== syncToken)
      return;
    for (const session of data.sessions) {
      if (deleted.has(session.id)) continue;
      const record = get(session.id);
      if (record.cursor <= data.cursor) record.session = session;
    }
    nextCursor = data.next_cursor;
    render();
  }
  async function detail(id, before = null) {
    const requestEpoch = epoch,
      token = viewToken;
    const data = await api(
      `${base}/sessions/${id}${before ? `/items?before=${before}` : ""}`,
    );
    if (
      requestEpoch !== epoch ||
      data.epoch !== epoch ||
      token !== viewToken ||
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
  async function synchronize(readyData) {
    const token = ++syncToken;
    synchronized = false;
    connected = true;
    if (epoch !== readyData.epoch) {
      epoch = readyData.epoch;
      records.clear();
      deleted.clear();
      rendered = null;
    }
    for (const summary of readyData.sessions) merge(summary);
    render();
    try {
      const results = await Promise.allSettled([
        list(),
        selected ? detail(selected) : Promise.resolve(),
        api(`${base}/bootstrap`).then((data) => {
          if (token === syncToken) {
            available = data.available;
            if (data.message) notice = data.message;
          }
        }),
      ]);
      if (token !== syncToken || !connected) return;
      if (results[0].status === "rejected") throw results[0].reason;
      if (results[2].status === "rejected") throw results[2].reason;
      if (results[1].status === "rejected") notice = results[1].reason.message;
      synchronized = true;
      for (const id of records.keys()) {
        if (saved(id).pending) void deliver(id);
      }
    } catch (error) {
      if (token === syncToken) notice = error.message;
    }
    render();
  }
  function connect() {
    if (feed) return;
    feed = new EventSource(`${base}/events`);
    feed.addEventListener("feed.ready", (event) => {
      void synchronize(JSON.parse(event.data));
    });
    for (const type of [
      "session.updated",
      "session.deleted",
      "run.updated",
      "item.updated",
      "prompt.updated",
      "session.notice",
    ]) {
      feed.addEventListener(type, (event) => receive(type, event));
    }
    const disconnected = () => {
      connected = false;
      synchronized = false;
      ++syncToken;
      render();
    };
    feed.addEventListener("feed.resync_required", disconnected);
    feed.onerror = disconnected;
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
      notice = "";
      render();
      if (epoch && id)
        void detail(id).catch((error) => {
          if (selected === id) {
            notice = error.message;
            render();
          }
        });
      else if (epoch)
        void list().catch((error) => {
          notice = error.message;
          render();
        });
    }
    connect();
    void api(`${base}/bootstrap`)
      .then((data) => {
        available = data.available;
        if (data.message) notice = data.message;
        render();
      })
      .catch((error) => {
        notice = error.message;
        render();
      });
  }
  function accepted(id, operationId) {
    const state = saved(id);
    if (state.pending?.operation_id !== operationId) return;
    const oldText = state.pending.text;
    delete state.pending;
    if (state.draft === oldText) state.draft = "";
    save(id, state);
    if (selected === id && root?.querySelector("textarea")?.value === oldText)
      root.querySelector("textarea").value = "";
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
    } finally {
      sends.delete(id);
      render();
    }
  }
  function submit() {
    const record = records.get(selected);
    if (!ready(record) || busy(record) || pending(record) || !available) return;
    const state = saved(selected),
      text = root.querySelector("textarea").value;
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
  function rename() {
    const record = records.get(selected),
      id = selected,
      revision = record.session.revision;
    dialog(
      "Rename session",
      (form) => {
        const input = element("input");
        input.value = record.session.title;
        input.required = true;
        input.maxLength = 200;
        label(form, "Title", input);
        return input;
      },
      "Save title",
      (input) =>
        api(`${base}/sessions/${id}`, {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: revision,
            title: input.value,
          }),
        }),
    );
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
    const node = element("div", undefined, "code-shell");
    const header = element("header", undefined, "code-header"),
      titleGroup = element("div");
    titleGroup.append(
      element("h2", "Code sessions"),
      element("p", "", "code-folder"),
    );
    const actions = element("div", undefined, "code-actions");
    if (selected) {
      actions.append(
        button("All sessions", () => navigate(null)),
        button("Rename", rename),
        button("Delete", remove),
      );
    } else
      actions.append(button("New code session", newSession, "primary-button"));
    header.append(titleGroup, actions);
    const message = element("p", "", "code-notice");
    message.setAttribute("role", "status");
    node.append(header, message);
    if (selected) {
      const transcript = element("div", undefined, "code-transcript");
      const older = button("Load older messages", async () => {
        const id = selected,
          oldHeight = transcript.scrollHeight,
          top = transcript.scrollTop;
        older.disabled = true;
        try {
          await detail(id, records.get(id).nextBefore);
          if (selected === id)
            transcript.scrollTop = top + transcript.scrollHeight - oldHeight;
        } catch (error) {
          notice = error.message;
        }
        older.disabled = false;
        render();
      });
      older.classList.add("code-older");
      transcript.append(
        older,
        element("div", undefined, "code-items"),
        element("div", undefined, "code-prompts"),
      );
      const composer = element("form", undefined, "code-composer"),
        input = element("textarea");
      input.setAttribute("aria-label", "Message");
      input.placeholder = "Ask Codex to work on this folder…";
      input.value = saved(selected).draft || "";
      const id = selected;
      input.addEventListener("input", () => {
        save(id, { ...saved(id), draft: input.value });
        renderControls();
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          submit();
        }
      });
      composer.addEventListener("submit", (event) => {
        event.preventDefault();
        submit();
      });
      const footer = element("div", undefined, "code-actions");
      const send = button(
        "Send",
        () => {
          busy(records.get(selected)) ? void stop() : submit();
        },
        "primary-button",
      );
      send.classList.add("code-send");
      footer.append(element("span", "", "code-status"), send);
      composer.append(input, footer);
      node.append(transcript, composer);
    } else {
      node.append(element("div", undefined, "code-library"));
      const more = button("Load more sessions", async () => {
        libraryLoading = true;
        render();
        try {
          await list(nextCursor);
        } catch (error) {
          notice = error.message;
        }
        libraryLoading = false;
        render();
      });
      more.classList.add("code-more");
      node.append(more);
    }
    root.append(node);
    rendered = selected || "library";
  }
  function render() {
    if (!root) return;
    if (rendered !== (selected || "library")) shell();
    root.querySelector(".code-shell > .code-notice").textContent =
      notice ||
      (!connected ? "Connecting…" : !synchronized ? "Synchronizing…" : "");
    if (!selected) {
      const library = root.querySelector(".code-library");
      for (const child of [...library.children])
        if (!child.dataset.id || !records.has(child.dataset.id)) child.remove();
      for (const record of [...records.values()]
        .filter((record) => record.session)
        .sort(
          (a, b) =>
            b.session.updated_at - a.session.updated_at ||
            b.id.localeCompare(a.id),
        )) {
        let card = library.querySelector(
          `[data-id="${CSS.escape(record.id)}"]`,
        );
        if (!card) {
          card = button("", () => navigate(record.id), "code-session-card");
          card.dataset.id = record.id;
          card.append(element("strong"), element("span"), element("span"));
        }
        card.children[0].textContent = record.session.title;
        card.children[1].textContent = record.session.cwd;
        card.children[2].textContent = busy(record)
          ? "Running"
          : record.session.error || "Codex";
        library.append(card);
      }
      if (!library.children.length && synchronized)
        library.append(
          element(
            "p",
            "Start a code session to work with Codex in a project folder.",
            "code-notice",
          ),
        );
      root.querySelector(".code-header button").disabled = !synchronized;
      const more = root.querySelector(".code-more");
      more.hidden = !nextCursor;
      more.disabled = !synchronized || libraryLoading;
      return;
    }
    const record = records.get(selected);
    if (record?.session) {
      root.querySelector("h2").textContent = record.session.title;
      root.querySelector(".code-folder").textContent = record.session.cwd;
    }
    const transcript = root.querySelector(".code-transcript");
    const bottom =
      transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight <
      80;
    const items = root.querySelector(".code-items");
    for (const { value: item } of [...(record?.items.values() || [])].sort(
      (a, b) => a.value.sequence - b.value.sequence,
    ))
      renderItem(items, item);
    const prompts = root.querySelector(".code-prompts");
    for (const { value: prompt } of record?.prompts.values() || [])
      renderPrompt(prompts, prompt);
    if (bottom) transcript.scrollTop = transcript.scrollHeight;
    root.querySelector(".code-older").hidden = !record?.nextBefore;
    renderControls();
  }
  function renderControls() {
    if (!selected || !root?.querySelector("textarea")) return;
    const record = records.get(selected),
      isBusy = busy(record);
    const send = root.querySelector(".code-send");
    send.textContent = isBusy
      ? "Stop"
      : saved(selected).pending
        ? "Retry Send"
        : "Send";
    send.disabled = isBusy
      ? record.run.stop_requested
      : !ready(record) ||
        !available ||
        pending(record) ||
        sends.has(selected) ||
        (!root.querySelector("textarea").value.trim() &&
          !saved(selected).pending);
    root.querySelector("textarea").disabled =
      !record?.loaded || record?.session?.status !== "ready";
    const actions = root.querySelectorAll(".code-header button");
    actions[1].disabled = !ready(record);
    actions[2].textContent =
      record?.session?.status === "delete_uncertain"
        ? "Check deletion"
        : "Delete";
    actions[2].disabled =
      !synchronized ||
      !record?.loaded ||
      record.session?.status === "deleting" ||
      isBusy ||
      pending(record);
    root.querySelector(".code-status").textContent =
      record?.session?.error ||
      record?.run?.error ||
      (record?.session?.status !== "ready"
        ? "Deleting…"
        : record.run?.stop_requested && isBusy
          ? "Stopping…"
          : pending(record)
            ? "Waiting for input"
            : isBusy
              ? "Working…"
              : "Codex");
    for (const node of root.querySelectorAll(".code-prompt")) {
      const prompt = record?.prompts.get(node.dataset.id)?.value;
      for (const control of node.querySelectorAll("input, select, button"))
        control.disabled =
          !ready(record) ||
          prompt?.status !== "pending" ||
          node.dataset.claiming === "true";
    }
  }
  function renderItem(parent, item) {
    let node = parent.querySelector(`[data-id="${CSS.escape(item.id)}"]`);
    if (!node) {
      node = element("article", undefined, "code-item");
      node.dataset.id = item.id;
      node.dataset.kind = item.kind;
      node.append(element("strong", item.kind === "user" ? "You" : "Codex"));
      let content = node;
      if (!["text", "user"].includes(item.kind)) {
        content = element("details");
        content.append(
          element(
            "summary",
            item.title || (item.kind === "reasoning" ? "Thinking" : "Tool"),
          ),
        );
        node.append(content);
      }
      content.append(
        element(item.kind === "user" ? "pre" : "div", "", "code-prose"),
        element("pre", "", "code-item-detail"),
      );
      node.dataset.sequence = item.sequence;
      const next = [...parent.children].find(
        (child) => Number(child.dataset.sequence) > item.sequence,
      );
      parent.insertBefore(node, next || null);
    }
    const content = node.querySelector(".code-prose");
    const value = item.html ?? item.text;
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
      if (submitting || !ready(records.get(sessionId))) return;
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
  };
})();
