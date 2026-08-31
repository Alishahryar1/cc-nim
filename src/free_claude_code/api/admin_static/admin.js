const state = {
  config: null,
  fields: new Map(),
  modelOptions: [],
  modelComboboxes: new Set(),
  authPollers: new Map(),
  activeView: "providers",
  searchQuery: "",
  providerFilter: "all",
};

const MASKED_SECRET = "********";
const NULL_VALUE = "__FCC_NULL__";
const VIEW_GROUPS = [
  {
    id: "providers",
    label: "Providers",
    title: "Providers",
    subtitle: "Manage AI backends, credentials, and model orchestration",
    sections: ["providers", "runtime"],
    containerId: "providersSections",
  },
  {
    id: "model_config",
    label: "Model Config",
    title: "Model Config",
    subtitle: "Configure default models, reasoning effort, web tools, and fallback priorities",
    sections: ["models", "reasoning", "web_tools"],
    containerId: "modelConfigSections",
  },
  {
    id: "messaging",
    label: "Messaging",
    title: "Messaging",
    subtitle: "Manage Telegram, Discord, and Voice assistant gateways",
    sections: ["messaging", "voice"],
    containerId: "messagingSections",
  },
  {
    id: "chat",
    label: "Chat Sessions",
    title: "Chat Sessions",
    sections: [],
    containerId: "chatRoot",
  },
];

const byId = (id) => document.getElementById(id);

function sourceLabel(source) {
  const labels = {
    default: "default",
    managed_env: "",
    process: "process env",
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

function sourceText(field) {
  const parts = [];
  const label = sourceLabel(field.source);
  if (label) {
    parts.push(label);
  }
  if (field.locked) {
    parts.push("locked");
  }
  return parts.join(" ");
}

function statusClass(status) {
  if (["configured", "reachable", "running", "connected"].includes(status)) return "ok";
  if (["missing_key", "missing_config", "missing_url", "unknown", "connecting"].includes(status)) return "warn";
  if (["offline", "error"].includes(status)) return "error";
  return "neutral";
}

// ===== Task 3: Status dot helper for provider cards =====
function getProviderStatusDot(provider) {
  const status = provider.state || provider.status || "unknown";
  const cls = statusClass(status);
  const colors = {
    ok: "var(--ok)",
    warn: "var(--warn)",
    error: "var(--error)",
    neutral: "var(--neutral)",
  };
  const labels = {
    configured: "Configured",
    reachable: "Reachable",
    running: "running",
    connected: "Connected",
    missing_key: "Missing key",
    missing_config: "Missing configuration",
    missing_url: "Missing URL",
    unknown: "Unknown status",
    connecting: "Connecting",
    offline: "Offline",
    error: "Error",
    disconnected: "Not connected",
  };
  return {
    color: colors[cls] || colors.neutral,
    label: labels[status] || provider.label || "Unknown",
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : "";
    } catch {
      // The status remains useful when an upstream proxy returns a non-JSON page.
    }
    const error = new Error(detail || `${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function load() {
  showMessage("Loading admin config");
  const config = await api("/admin/api/config");
  state.config = config;
  state.fields = new Map(config.fields.map((field) => [field.key, field]));
  renderNav();
  renderProviders(config.provider_status);
  renderSections(config.sections, config.fields);
 const configPathEl = byId("configPath");
  if (configPathEl) {
    configPathEl.textContent = config.paths.managed;
  }
  await Promise.all([
    refreshConnectedAccounts(),
    hydrateModelOptions(),
    refreshLocalStatus(),
    window.ChatSessions ? window.ChatSessions.initialize(api) : Promise.resolve(),
  ]);

  updateDirtyState();
  updateStats();
  initSearch();
  initFilterTabs();
  initGlobalShortcuts();
  showMessage("");
}

function updateStats() {
  if (!state.config) return;
  const providers = state.config.provider_status || [];
  const regularProviders = providers.filter((p) => p.kind !== "connected_account");
  const configuredProviders = regularProviders.filter(
    (p) => !p.missing_configuration_keys || p.missing_configuration_keys.length === 0,
  );
  const connectedAccounts = providers.filter(
  (p) => p.kind === "connected_account" && p.connected === true
);
  const statProvCount = byId("statProvidersCount");
  const statProvSub = byId("statProvidersSub");
  if (statProvCount) {
    statProvCount.textContent = `${configuredProviders.length} / ${regularProviders.length}`;
  }
  if (statProvSub) {
    statProvSub.textContent = connectedAccounts.length
      ? `${connectedAccounts.length} subscription${connectedAccounts.length > 1 ? "s" : ""} connected`
      : "Standard API backends";
  }

  const statModelsCount = byId("statModelsCount");
  const statModelsSub = byId("statModelsSub");
  if (statModelsCount) {
    statModelsCount.textContent = state.modelOptions.length
      ? `${state.modelOptions.length}`
      : "Ready";
  }
  if (statModelsSub) {
    statModelsSub.textContent = state.modelOptions.length
      ? "Models discovered in catalog"
      : "Refresh to discover models";
  }

  const statConfigStatus = byId("statConfigStatus");
  const statConfigSub = byId("statConfigSub");
  const dirtyCount = Object.keys(changedValues()).length;
  if (statConfigStatus) {
    statConfigStatus.textContent = dirtyCount === 0 ? "Synced" : `${dirtyCount} Pending`;
  }
  if (statConfigSub) {
    statConfigSub.textContent = dirtyCount === 0 ? "No unsaved changes" : "Changes ready to apply";
  }
}

function initSearch() {
  const searchInput = byId("adminSearchInput");
  if (!searchInput || searchInput.dataset.initialized === "true") return;
  searchInput.dataset.initialized = "true";

  searchInput.addEventListener("input", () => {
    state.searchQuery = searchInput.value.trim().toLowerCase();
    filterViewsBySearch();
  });

  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      searchInput.focus();
      searchInput.select();
    } else if (event.key === "Escape" && document.activeElement === searchInput) {
      searchInput.value = "";
      state.searchQuery = "";
      filterViewsBySearch();
      searchInput.blur();
    }
  });
}

function initFilterTabs() {
  const container = byId("providerFilterTabs");
  if (!container || container.dataset.initialized === "true") return;
  container.dataset.initialized = "true";

  container.querySelectorAll(".filter-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      container.querySelectorAll(".filter-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      state.providerFilter = tab.dataset.filter || "all";
      filterViewsBySearch();
    });
  });
}

  function initGlobalShortcuts() {
  if (window.__fccAdminShortcutsInitialized) return;
  window.__fccAdminShortcutsInitialized = true;
  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      const applyBtn = byId("applyButton");
      if (applyBtn && !applyBtn.disabled) apply();
    }
  });
}

// ===== Task 3 (A1): filterViewsBySearch — toggles list-view mode + filters connected accounts =====
function filterViewsBySearch() {
  const q = state.searchQuery;
  const filter = state.providerFilter || "all";

  // Toggle list-view mode on the provider grid when searching
  const providerGrid = byId("providerGrid");
  const connectedAccountGrid = byId("connectedAccountGrid");
  const isSearching = !!q;

  if (providerGrid) {
    providerGrid.classList.toggle("list-view", isSearching);
  }
  if (connectedAccountGrid) {
    connectedAccountGrid.classList.toggle("list-view", isSearching);
  }

  // Filter provider cards
  document.querySelectorAll("#providerGrid .provider-card").forEach((card) => {
    const providerId = card.dataset.provider;
    const provider = state.config?.provider_status?.find((p) => p.provider_id === providerId);

    let filterMatch = true;
    if (filter === "configured") {
      filterMatch = !provider?.missing_configuration_keys || provider.missing_configuration_keys.length === 0;
    } else if (filter === "missing") {
      filterMatch = provider?.missing_configuration_keys && provider.missing_configuration_keys.length > 0;
    } else if (filter === "local") {
      filterMatch = provider?.kind === "local";
    }

    let searchMatch = true;
    if (q) {
      const text = (card.textContent || "").toLowerCase();
      searchMatch = text.includes(q);
    }

    // Completely hide non-matching cards
    card.hidden = !(filterMatch && searchMatch);
  });

  // Also filter connected account cards
  document.querySelectorAll("#connectedAccountGrid .provider-card").forEach((card) => {
    let searchMatch = true;
    if (q) {
      const text = (card.textContent || "").toLowerCase();
      searchMatch = text.includes(q);
    }
    card.hidden = !searchMatch;
  });

  // Filter fields in settings sections
  document.querySelectorAll(".settings-section").forEach((section) => {
    let hasVisibleField = false;
    const sectionText = (section.textContent || "").toLowerCase();
    const sectionMatches = q ? sectionText.includes(q) : true;

    section.querySelectorAll(".field").forEach((field) => {
      if (!q) {
        field.style.display = "";
        hasVisibleField = true;
        return;
      }
      const text = (field.textContent || "").toLowerCase();
      const match = text.includes(q);
      field.style.display = match ? "flex" : "none";
      if (match) hasVisibleField = true;
    });

    if (q) {
      section.hidden = !hasVisibleField && !sectionMatches;
    } else {
      section.hidden = false;
    }
  });

  // Show a "no results" message when searching and nothing matches
  showNoResultsMessage(isSearching);
}

// ===== Task 3 (A1): showNoResultsMessage helper =====
function showNoResultsMessage(isSearching) {
  let existing = byId("noResultsMessage");
  const providerGrid = byId("providerGrid");
  const connectedAccountGrid = byId("connectedAccountGrid");

  let anyVisible = false;
  if (providerGrid) {
    providerGrid.querySelectorAll(".provider-card:not([hidden])").forEach(() => {
      anyVisible = true;
    });
  }
  if (connectedAccountGrid && !anyVisible) {
    connectedAccountGrid.querySelectorAll(".provider-card:not([hidden])").forEach(() => {
      anyVisible = true;
    });
  }

  if (isSearching && !anyVisible) {
    if (!existing) {
      existing = document.createElement("div");
      existing.id = "noResultsMessage";
      existing.className = "no-results-message";
      existing.textContent = "No matching providers found.";
      if (providerGrid && providerGrid.parentNode) {
        providerGrid.parentNode.insertBefore(existing, providerGrid.nextSibling);
      }
    }
    existing.hidden = false;
  } else if (existing) {
    existing.hidden = true;
  }
}

function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `nav-link${index === 0 ? " active" : ""}`;
    button.dataset.view = view.id;
    button.textContent = view.label;
    if (index === 0) {
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => {
      navigateToView(view.id);
    });
    nav.appendChild(button);
  });
  setActiveView(state.activeView, { scroll: false });
}

function setActiveView(viewId, { scroll = false } = {}) {
  const activeView =
    VIEW_GROUPS.find((view) => view.id === viewId) || VIEW_GROUPS[0];
  state.activeView = activeView.id;
  byId("pageTitle").textContent = activeView.title;
  const subtitleEl = byId("pageSubtitle");
  if (subtitleEl && activeView.subtitle) {
    subtitleEl.textContent = activeView.subtitle;
  }

  document.querySelectorAll(".nav-link").forEach((link) => {
    const selected = link.dataset.view === activeView.id;
    link.classList.toggle("active", selected);
    if (selected) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  document.querySelectorAll(".admin-view").forEach((view) => {
    const selected = view.dataset.view === activeView.id;
    view.classList.toggle("active", selected);
    view.hidden = !selected;
  });

  if (scroll) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  if (chatActive && window.ChatSessions) {
    window.ChatSessions.activate(window.location.pathname);
  }
}

function navigateToView(viewId) {
  if (viewId === "chat") {
    if (window.location.pathname !== "/admin/chat") {
      window.history.pushState({}, "", "/admin/chat");
    }
  } else if (window.location.pathname.startsWith("/admin/chat")) {
    window.history.pushState({}, "", "/admin");
  }
  setActiveView(viewId, { scroll: true });
}

// ===== Task 3 (A2): renderProviders — now includes status dots =====
function renderProviders(providerStatus) {
  const grid = byId("providerGrid");
  const connectedGrid = byId("connectedAccountGrid");
  grid.innerHTML = "";
  connectedGrid.innerHTML = "";
  const connected = providerStatus.filter(
    (provider) => provider.kind === "connected_account",
  );
  byId("connectedAccountsSection").hidden = connected.length === 0;
  providerStatus.forEach((provider) => {
    if (provider.kind === "connected_account") {
      connectedGrid.appendChild(renderConnectedAccountCard(provider));
      return;
    }
    const card = document.createElement("article");
    card.className = "provider-card";
    card.dataset.provider = provider.provider_id;

    const title = document.createElement("div");
    title.className = "provider-title";

    // Status dot
    const dotInfo = getProviderStatusDot(provider);
    const dot = document.createElement("span");
    dot.className = "status-dot";
    dot.style.backgroundColor = dotInfo.color;
    dot.title = dotInfo.label;
    dot.setAttribute("aria-label", dotInfo.label);
    title.appendChild(dot);

    const name = document.createElement("strong");
    name.textContent = provider.display_name || provider.provider_id;

    const pill = document.createElement("span");
    pill.className = `status-pill ${statusClass(provider.status)}`;
    pill.textContent = provider.label;
    title.append(name, pill);

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    const configurationKeys = Array.isArray(provider.configuration_keys)
      ? provider.configuration_keys
      : [];
    const missingConfigurationKeys = Array.isArray(
      provider.missing_configuration_keys,
    )
      ? provider.missing_configuration_keys
      : [];
    meta.textContent = configurationKeys.join(" + ");

    const result = document.createElement("div");
    result.className = "provider-check-result";
    result.dataset.providerCheckResult = provider.provider_id;
    result.setAttribute("aria-live", "polite");
    result.hidden = true;

    const actions = document.createElement("div");
    actions.className = "provider-actions";
    if (configurationKeys.length) {
      const configuring = missingConfigurationKeys.length > 0;
      actions.appendChild(
        providerActionButton(configuring ? "Configure" : "Edit", () =>
          navigateToProviderConfiguration(provider, configuring),
        ),
      );
    }

    if (missingConfigurationKeys.length === 0) {
      const button = providerActionButton(
        provider.kind === "local" ? "Test" : "Refresh models",
        () => testProvider(provider.provider_id, button),
        "secondary-button",
      );
      actions.appendChild(button);
    }

    card.append(title, meta, result, actions);
    grid.appendChild(card);
  });
}

function providerActionButton(label, action, className = "test-button") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", action);
  return button;
}

function navigateToProviderConfiguration(provider, configuring) {
  const keys = configuring
    ? provider.missing_configuration_keys
    : provider.configuration_keys;
  const fieldKey = Array.isArray(keys) ? keys[0] : null;
  const input = fieldKey ? byId(`field-${fieldKey}`) : null;
  if (!input) {
    showMessage("Provider configuration field is unavailable.", "error");
    return;
  }
  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;
  input.scrollIntoView({
    behavior: reducedMotion ? "instant" : "smooth",
    block: "center",
  });
  input.focus({ preventScroll: true });
}

// ===== Task 3 (A3): renderConnectedAccountCard — now includes status dots =====
function renderConnectedAccountCard(provider, status = provider) {
  const card = document.createElement("article");
  card.className = "provider-card";
  card.dataset.provider = provider.provider_id;
  card.dataset.connectedAccount = "true";

  const title = document.createElement("div");
  title.className = "provider-title";

  // Status dot
  const dotInfo = getProviderStatusDot(status);
  const dot = document.createElement("span");
  dot.className = "status-dot";
  dot.style.backgroundColor = dotInfo.color;
  dot.title = dotInfo.label;
  dot.setAttribute("aria-label", dotInfo.label);
  title.appendChild(dot);

  const name = document.createElement("strong");
  name.textContent = provider.display_name || provider.provider_id;
  const pill = document.createElement("span");
  pill.className = `status-pill ${statusClass(status.state || status.status)}`;
  pill.textContent = connectedAccountLabel(status);
  title.append(name, pill);

  const meta = document.createElement("div");
  meta.className = "provider-meta";
  meta.textContent = connectedAccountMeta(status);

  const actions = document.createElement("div");
  actions.className = "provider-actions";
  populateConnectedAccountActions(provider, status, actions);
  card.append(title, meta, actions);
  return card;
}

function connectedAccountLabel(status) {
  const labels = {
    disconnected: "Not connected",
    connecting: "Connecting",
    connected: "Connected",
    error: "Needs attention",
  };
  return labels[status.state] || status.label || "Not connected";
}

function connectedAccountMeta(status) {
  if (status.connected) {
    const identity = status.email || "ChatGPT subscription connected";
    const models = Number.isInteger(status.model_count)
      ? `${status.model_count} model${status.model_count === 1 ? "" : "s"} available. `
      : "";
    const error = status.message ? `${status.message} ` : "";
    return `${identity}. ${models}${error}Restart your agent to refresh its model picker.`;
  }
  if (status.mode === "device" && status.user_code) {
    return `Enter code ${status.user_code} at ${status.verification_url}`;
  }
  if (status.state === "connecting") {
    return "Finish signing in, then return to this page.";
  }
  return status.message || "Connect a ChatGPT account to discover subscription models.";
}

function populateConnectedAccountActions(provider, status, actions) {
  const providerId = provider.provider_id;
  if (status.state === "connecting") {
    const target = status.authorization_url || status.verification_url;
    if (target) {
      actions.appendChild(authButton("Open sign-in", () => window.open(target, "_blank", "noopener")));
    }
    if (status.mode === "device" && status.user_code) {
      actions.appendChild(
        authButton(
          "Copy code",
          () => copyDeviceCode(status.user_code),
          "secondary-button",
        ),
      );
    }
    actions.appendChild(
      authButton("Cancel", () => cancelConnectedAccountLogin(providerId), "secondary-button"),
    );
    return;
  }
  if (status.connected) {
    actions.appendChild(
      authButton(
        "Reconnect",
        (button) => startConnectedAccountLogin(providerId, "browser", button),
      ),
    );
    actions.appendChild(
      authButton(
        "Disconnect",
        () => disconnectConnectedAccount(providerId),
        "secondary-button",
      ),
    );
    return;
  }
  actions.appendChild(
    authButton("Connect", (button) => startConnectedAccountLogin(providerId, "browser", button)),
    authButton(
      "Use device code",
      (button) => startConnectedAccountLogin(providerId, "device", button),
      "secondary-button",
    ),
  );
}

function authButton(label, action, className = "test-button") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", () => action(button));
  return button;
}

async function refreshConnectedAccounts() {
  const providers = (state.config?.provider_status || []).filter(
    (provider) => provider.kind === "connected_account",
  );
  await Promise.all(
    providers.map(async (provider) => {
      try {
        const status = await api(`/admin/api/providers/${provider.provider_id}/auth`);
        updateConnectedAccountCard(provider, status);
        if (status.state === "connecting") pollConnectedAccount(provider);
      } catch (error) {
        updateConnectedAccountCard(provider, {
          state: "error",
          connected: false,
          message: error.message,
        });
      }
    }),
  );
}

function updateConnectedAccountCard(provider, status) {
  const current = document.querySelector(
    `[data-provider="${provider.provider_id}"][data-connected-account="true"]`,
  );
  if (current) current.replaceWith(renderConnectedAccountCard(provider, status));
}

async function startConnectedAccountLogin(providerId, mode, button) {
  button.disabled = true;
  const popup = window.open("about:blank", "_blank");
  if (popup) popup.opener = null;
  try {
    const status = await api(`/admin/api/providers/${providerId}/auth/login`, {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    const provider = connectedAccountDescriptor(providerId);
    updateConnectedAccountCard(provider, status);
    const target = status.authorization_url || status.verification_url;
    if (target && popup) {
      popup.location.replace(target);
    } else if (target) {
      window.open(target, "_blank", "noopener");
    } else if (popup) {
      popup.close();
    }
    pollConnectedAccount(provider);
  } catch (error) {
    if (popup) popup.close();
    showMessage(error.message, true);
    button.disabled = false;
  }
}

async function cancelConnectedAccountLogin(providerId) {
  clearConnectedAccountPoll(providerId);
  const status = await api(`/admin/api/providers/${providerId}/auth/cancel`, {
    method: "POST",
  });
  updateConnectedAccountCard(connectedAccountDescriptor(providerId), status);
}

async function disconnectConnectedAccount(providerId) {
  if (!window.confirm("Disconnect this ChatGPT account from FCC?")) return;
  clearConnectedAccountPoll(providerId);
  const status = await api(`/admin/api/providers/${providerId}/auth`, {
    method: "DELETE",
  });
  updateConnectedAccountCard(connectedAccountDescriptor(providerId), status);
  await hydrateModelOptions();
  updateStats();
}

function pollConnectedAccount(provider) {
  clearConnectedAccountPoll(provider.provider_id);
  const poll = async () => {
    try {
      const status = await api(`/admin/api/providers/${provider.provider_id}/auth`);
      updateConnectedAccountCard(provider, status);
      if (status.state === "connecting") {
        state.authPollers.set(provider.provider_id, window.setTimeout(poll, 1000));
      } else {
        state.authPollers.delete(provider.provider_id);
        if (status.connected) {
          await hydrateModelOptions();
          updateStats();
        }
      }
    } catch (error) {
      state.authPollers.delete(provider.provider_id);
      showMessage(error.message, true);
    }
  };
  state.authPollers.set(provider.provider_id, window.setTimeout(poll, 1000));
}

function clearConnectedAccountPoll(providerId) {
  const timer = state.authPollers.get(providerId);
  if (timer) window.clearTimeout(timer);
  state.authPollers.delete(providerId);
}

function connectedAccountDescriptor(providerId) {
  return state.config.provider_status.find(
    (provider) => provider.provider_id === providerId,
  );
}

async function copyDeviceCode(code) {
  try {
    await navigator.clipboard.writeText(code);
    showMessage("Device code copied to clipboard.", "ok");
  } catch {
    showMessage(`Copy this device code: ${code}`);
  }
}

function updateProviderCheckResult(providerId, status, message) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const result = card.querySelector(".provider-check-result");
  result.className = `provider-check-result ${status}`;
  result.textContent = message;
  result.hidden = !message;
}

function renderSections(sections, fields) {
  state.modelComboboxes.clear();
  VIEW_GROUPS.forEach((view) => {
    byId(view.containerId).innerHTML = "";
  });

  const sectionById = new Map(sections.map((section) => [section.id, section]));
  const bySection = new Map();
  sections.forEach((section) => bySection.set(section.id, []));
  fields.forEach((field) => {
    if (!bySection.has(field.section)) bySection.set(field.section, []);
    bySection.get(field.section).push(field);
  });

  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    view.sections.forEach((sectionId) => {
      const section = sectionById.get(sectionId);
      const sectionFields = bySection.get(sectionId) || [];
      if (!section || sectionFields.length === 0) return;

      const sectionEl = document.createElement("section");
      sectionEl.className = "settings-section";
      sectionEl.id = `section-${section.id}`;

      const heading = document.createElement("div");
      heading.className = "section-heading";
      heading.innerHTML = `<div><h3>${section.label}</h3><p>${section.description}</p></div>`;
      if (section.id === "models") {
        const refreshButton = document.createElement("button");
        refreshButton.type = "button";
        refreshButton.className = "secondary-button";
        refreshButton.textContent = "Refresh models";
        refreshButton.addEventListener("click", () => refreshModelOptions(refreshButton));
        heading.appendChild(refreshButton);
      }
      sectionEl.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "field-grid";
      sectionFields.forEach((field) => {
        grid.appendChild(renderField(field));
      });
      sectionEl.appendChild(grid);

      if (sectionFields.some((field) => field.advanced)) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "ghost-button advanced-toggle";
        toggle.textContent = "Show advanced";
        toggle.addEventListener("click", () => {
          const showing = sectionEl.classList.toggle("show-advanced");
          toggle.textContent = showing ? "Hide advanced" : "Show advanced";
        });
        sectionEl.appendChild(toggle);
      }

      container.appendChild(sectionEl);
    });
  });
}

function renderField(field) {
  const wrapper = document.createElement("div");
  wrapper.className = `field${field.advanced ? " advanced-field" : ""}`;
  wrapper.dataset.key = field.key;

  const label = document.createElement("label");
  label.htmlFor = `field-${field.key}`;
  const labelText = document.createElement("span");
  labelText.textContent = field.label;
  label.appendChild(labelText);

  const source = sourceText(field);
  if (source) {
    const sourceEl = document.createElement("span");
    sourceEl.className = "field-source";
    sourceEl.textContent = source;
    label.appendChild(sourceEl);
  }

  const input = inputForField(field);
  input.id = `field-${field.key}`;
  input.dataset.key = field.key;
  input.dataset.original = comparableValue(field.value);
  input.dataset.secret = field.secret ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.dataset.nullable = field.nullable ? "true" : "false";
  input.dataset.remove = "false";
  input.dataset.fieldType = field.type;
  input.disabled = field.locked;
   input.addEventListener("input", updateDirtyState);
  input.addEventListener("change", updateDirtyState);
  input.addEventListener("input", () => {
    input.dataset.remove = "false";
  });
  input.addEventListener("input", () => {
    input.dataset.remove = "false";
  });
    if (field.type === "optional_model") {
    input.addEventListener("blur", () => {
      if (!input.value.trim() || input.value.trim().toLowerCase() === "none") {
        input.value = "None";
        updateDirtyState();
      }
    });
  }

  let control = input;
  if (field.type === "model" || field.type === "optional_model") {
    control = createModelCombobox(input, field).element;
  } else if (field.type === "model_list") {
    const editor = new ModelListEditor(input, field);
    label.htmlFor = editor.inputId;
    control = editor.element;
  }
  wrapper.append(label, control);
  if (field.secret && field.nullable && field.configured && !field.locked) {
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "ghost-button secret-remove";
    removeButton.textContent = "Remove";
     removeButton.addEventListener("click", () => {
      const removing = input.dataset.remove !== "true";
      input.dataset.remove = removing ? "true" : "false";
      input.readOnly = removing;
      removeButton.textContent = removing ? "Undo removal" : "Remove";
      updateDirtyState();
    });
    wrapper.appendChild(removeButton);
  }
  if (field.description) {
    const description = document.createElement("div");
    description.className = "field-description";
    description.textContent = field.description;
    wrapper.appendChild(description);
  }
  return wrapper;
}

function inputForField(field) {
  if (field.type === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = String(field.value).toLowerCase() === "true";
    input.dataset.original = input.checked ? "true" : "false";
    return input;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    field.options.forEach((item) =>
      select.appendChild(option(item.value, item.label)),
    );
    select.value = field.value || field.options[0]?.value || "";
    return select;
  }

  if (field.type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.value = field.value || "";
    return textarea;
  }

  if (field.type === "model" || field.type === "optional_model") {
    const input = document.createElement("input");
    input.type = "text";
    input.value = field.value || (field.type === "optional_model" ? "None" : "");
    input.autocomplete = "off";
    return input;
  }

  if (field.type === "model_list") {
    const input = document.createElement("input");
    input.type = "hidden";
    input.value = field.value || "";
    return input;
  }

  const input = document.createElement("input");
  input.type = field.type === "number" ? "number" : "text";
  if (field.type === "secret") {
    input.type = "password";
    input.placeholder = field.configured
      ? "Configured - enter a new value to replace"
      : "Not configured";
    input.value = "";
    input.autocomplete = "off";
  } else {
    input.value = field.value || "";
  }
  return input;
}

function createModelCombobox(input, field) {
  return new window.FccModelCombobox(input, {
    listboxId: `model-options-${field.key}`,
    label: field.label,
    values: () =>
      field.type === "optional_model"
        ? ["None", ...state.modelOptions]
        : state.modelOptions,
    emptyMessage: () =>
      state.modelOptions.length
        ? "No matching models. You can still enter a custom slug."
        : "No discovered models. Refresh models or enter a custom slug.",
    registry: state.modelComboboxes,
  });
}

class ModelListEditor {
  constructor(input, field) {
    this.input = input;
    this.field = field;
    this.values = input.value
      ? input.value.split(",").map((value) => value.trim()).filter(Boolean)
      : [];
    this.inputId = `field-${field.key}-add`;

    this.element = document.createElement("div");
    this.element.className = "model-list-editor";

    const addRow = document.createElement("div");
    addRow.className = "model-list-add";
    this.addInput = document.createElement("input");
    this.addInput.id = this.inputId;
    this.addInput.type = "text";
    this.addInput.autocomplete = "off";
    this.addInput.placeholder = "provider/model";
    this.addInput.disabled = field.locked;
    const addCombobox = createModelCombobox(this.addInput, {
      ...field,
      key: `${field.key}-add`,
      label: "fallback model",
      type: "model",
    });

    this.addButton = document.createElement("button");
    this.addButton.type = "button";
    this.addButton.className = "secondary-button";
    this.addButton.textContent = "Add";
    this.addButton.disabled = field.locked;
    this.addButton.addEventListener("click", () => this.add());
    addRow.append(addCombobox.element, this.addButton);

    this.rows = document.createElement("div");
    this.rows.className = "model-list-rows";
    this.element.append(input, addRow, this.rows);
    this.renderRows();
  }

  add() {
    const value = this.addInput.value.trim();
    if (!value) {
      showMessage("Enter a full provider/model fallback.", "error");
      return;
    }
    if (this.values.includes(value)) {
      showMessage("That fallback model is already in the list.", "error");
      return;
    }
    this.values.push(value);
    this.addInput.value = "";
    showMessage("");
    this.sync();
  }

  move(index, offset) {
    const destination = index + offset;
    if (destination < 0 || destination >= this.values.length) return;
    [this.values[index], this.values[destination]] = [
      this.values[destination],
      this.values[index],
    ];
    this.sync();
  }

  remove(index) {
    this.values.splice(index, 1);
    this.sync();
  }

    sync() {
    this.input.value = this.values.join(",");
    this.input.dataset.remove = "false";
    this.input.dispatchEvent(new Event("input", { bubbles: true }));
    this.renderRows();
  }

  renderRows() {
    this.rows.innerHTML = "";
    if (this.values.length === 0) {
      const empty = document.createElement("div");
      empty.className = "model-list-empty";
      empty.textContent = "No fallback models configured.";
      this.rows.appendChild(empty);
      return;
    }

    this.values.forEach((value, index) => {
      const row = document.createElement("div");
      row.className = "model-list-row";

      const model = document.createElement("span");
      model.className = "model-list-value";
      model.textContent = value;

      const up = this.actionButton("Move up", `Move ${value} up`, () =>
        this.move(index, -1),
      );
      up.disabled = this.field.locked || index === 0;
      const down = this.actionButton("Move down", `Move ${value} down`, () =>
        this.move(index, 1),
      );
      down.disabled = this.field.locked || index === this.values.length - 1;
      const remove = this.actionButton("Remove", `Remove ${value}`, () =>
        this.remove(index),
      );
      remove.disabled = this.field.locked;

      row.append(model, up, down, remove);
      this.rows.appendChild(row);
    });
  }

  actionButton(text, label, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost-button model-list-action";
    button.textContent = text;
    button.setAttribute("aria-label", label);
    button.addEventListener("click", action);
    return button;
  }
}

function option(value, label) {
  const optionEl = document.createElement("option");
  optionEl.value = value;
  optionEl.textContent = label;
  return optionEl;
}

function readFieldValue(input) {
  if (input.type === "checkbox") return input.checked ? "true" : "false";
  if (input.dataset.remove === "true") return null;
  if (
    input.dataset.fieldType === "optional_model" &&
    input.value.trim().toLowerCase() === "none"
  ) {
    return null;
  }
  if (input.dataset.secret === "true" && input.dataset.configured === "true") {
    return input.value ? input.value : MASKED_SECRET;
  }
  if (input.dataset.nullable === "true" && !input.value.trim()) return null;
  return input.value;
}

function comparableValue(value) {
  return value === null ? NULL_VALUE : String(value);
}

function changedValues() {
  const values = {};
  document.querySelectorAll("[data-key]").forEach((input) => {
    if (input.disabled || !input.matches("input, select, textarea")) return;
    const value = readFieldValue(input);
    if (comparableValue(value) !== input.dataset.original) {
      values[input.dataset.key] = value;
    }
  });
  return values;
}

function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  byId("dirtyState").textContent =
    count === 0 ? "No changes" : `${count} unsaved change${count === 1 ? "" : "s"}`;
  byId("applyButton").disabled = count === 0;
  updateStats();
}

async function apply() {
  const result = await api("/admin/api/config/apply", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (!result.applied) {
    showMessage(result.errors.join("; "), "error");
    return;
  }
  const restart = result.restart || {};
  if (restart.required && restart.automatic) {
    showMessage("Applied. Restarting server...", "ok");
    byId("applyButton").disabled = true;
    setTimeout(() => {
      window.location.href = restart.admin_url || "/admin";
    }, 1600);
    return;
  }
  const pending = restart.required ? restart.fields || [] : result.pending_fields || [];
  await load();
  showMessage(
    pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(", ")}`
      : "Applied",
    "ok",
  );
}

async function refreshLocalStatus() {
  const result = await api("/admin/api/providers/local-status");
  result.providers.forEach((provider) => {
    if (provider.status === "missing_url") return;
    if (provider.status === "reachable") {
      updateProviderCheckResult(
        provider.provider_id,
        "ok",
        `Reachable: ${provider.base_url}`,
      );
      return;
    }
    const detail = provider.message
      ? provider.message
      : provider.status_code
        ? `${provider.base_url} returned HTTP ${provider.status_code}`
        : "The local provider did not respond.";
    updateProviderCheckResult(
      provider.provider_id,
      "error",
      `Unavailable: ${detail}`,
    );
  });
  updateStats();
}

async function testProvider(providerId, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Checking...";
  updateProviderCheckResult(providerId, "checking", "Checking...");
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: "POST",
      body: "{}",
    });
    if (result.ok) {
      updateProviderCheckResult(
        providerId,
        "ok",
        `${result.models.length} models available`,
      );
      setModelOptions([
        ...state.modelOptions,
        ...result.models.map((model) => `${providerId}/${model}`),
      ]);
    } else {
      updateProviderCheckResult(
        providerId,
        "error",
        `Unavailable: ${result.message || "Provider check failed."}`,
      );
    }
  } catch {
    updateProviderCheckResult(
      providerId,
      "error",
      "Provider check could not be completed.",
    );
  } finally {
    button.disabled = false;
    button.textContent = original;
    updateStats();
  }
}

async function hydrateModelOptions() {
  try {
    await loadModelOptions();
  } catch {
    // Model fields remain editable when optional catalog hydration is unavailable.
  }
}

async function loadModelOptions(refresh = false) {
  const result = await api("/admin/api/models" + (refresh ? "/refresh" : ""), {
    method: refresh ? "POST" : "GET",
  });
  setModelOptions(result.models);
  if (refresh && window.ChatSessions) await window.ChatSessions.refresh();
  return result;
}

async function refreshModelOptions(button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Refreshing";
  try {
    const result = await loadModelOptions(true);
    const failedProviders = result.failed_providers || [];
    if (failedProviders.length) {
      const labels = failedProviders.map(providerDisplayName).join(", ");
      showMessage(
        `${state.modelOptions.length} models available; could not refresh ${labels}`,
        "warn",
      );
    } else {
      showMessage(`${state.modelOptions.length} models available`, "ok");
    }
  } catch (error) {
    showMessage(`Could not refresh models: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
    updateStats();
  }
}

function providerDisplayName(providerId) {
  const provider = state.config?.provider_status?.find(
    (candidate) => candidate.provider_id === providerId,
  );
  return provider?.display_name || providerId;
}

function setModelOptions(models) {
  state.modelOptions = Array.from(
    new Set(models.filter((model) => typeof model === "string" && model.trim())),
  ).sort((left, right) => left.localeCompare(right));
  state.modelComboboxes.forEach((combobox) => {
    if (combobox.isOpen) combobox.render(combobox.query);
  });
  updateStats();
}

function showMessage(message, kind = "") {
  const area = byId("messageArea");
  area.textContent = message;
  area.className = `message-area ${kind}`.trim();
}

byId("applyButton").addEventListener("click", apply);
document.addEventListener("pointerdown", (event) => {
  state.modelComboboxes.forEach((combobox) => {
    if (combobox.isOpen && !combobox.element.contains(event.target)) combobox.close();
  });
});

window.addEventListener("popstate", () => {
  const viewId = window.location.pathname.startsWith("/admin/chat")
    ? "chat"
    : "providers";
  setActiveView(viewId, { scroll: false });
});

load().catch((error) => {
  showMessage(error.message, "error");
});