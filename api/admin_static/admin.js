const state = {
  config: null,
  fields: new Map(),
  localStatus: new Map(),
  modelOptions: [],
  activeView: "providers",
};

const MASKED_SECRET = "********";

// Recognised Claude model ids that route through Settings.model_* overrides.
// These mirror config.settings.Settings.resolve_model / resolve_thinking on
// the backend; both sides must be updated together if the supported ids change.
const FABLE_IDS = ["claude-fable-5", "claude-fable-5-2026", "claude-fable-5-20250612"];
const FABLE_OPUS_IDS = ["claude-opus-4-8", "claude-opus-4-7"];
const FABLE_SONNET_IDS = ["claude-sonnet-4-6", "claude-sonnet-4-5"];
const FABLE_HAIKU_IDS = ["claude-haiku-4-5"];

const VIEW_GROUPS = [
  {
    id: "providers",
    label: "Providers",
    title: "Providers",
    sections: ["providers", "runtime"],
    containerId: "providersSections",
  },
  {
    id: "model_config",
    label: "Model Config",
    title: "Model Config",
    sections: ["models", "thinking", "web_tools"],
    containerId: "modelConfigSections",
  },
  {
    id: "messaging",
    label: "Messaging",
    title: "Messaging",
    sections: ["messaging", "voice"],
    containerId: "messagingSections",
  },
];

const THEME_STORAGE_KEY = "fcc-theme";

const byId = (id) => document.getElementById(id);

function sourceLabel(source) {
  const labels = {
    default: "default",
    template: "template",
    repo_env: "repo .env",
    managed_env: "",
    explicit_env_file: "FCC_ENV_FILE",
    process: "process",
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

function sourceText(field) {
  const parts = [];
  const label = sourceLabel(field.source);
  if (label) parts.push(label);
  if (field.locked) parts.push("locked");
  return parts.join(" / ");
}

function providerName(providerId) {
  const names = {
    nvidia_nim: "NVIDIA NIM",
    open_router: "OpenRouter",
    mistral_codestral: "Mistral Codestral",
    deepseek: "DeepSeek",
    lmstudio: "LM Studio",
    llamacpp: "llama.cpp",
    ollama: "Ollama",
    kimi: "Kimi",
    wafer: "Wafer",
    opencode: "OpenCode Zen",
    opencode_go: "OpenCode Go",
    zai: "Z.ai",
  };
  if (names[providerId]) return names[providerId];
  return providerId.split("_").map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(" ");
}

function statusClass(status) {
  if (["configured", "reachable", "running"].includes(status)) return "ok";
  if (["missing_key", "missing_url", "unknown"].includes(status)) return "warn";
  if (["offline", "error"].includes(status)) return "error";
  return "neutral";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function ensureFableFields(config) {
  const keys = new Set(config.fields.map((f) => f.key));

  if (!keys.has("MODEL_FABLE")) {
    const firstModelIdx = config.fields.findIndex((f) => f.section === "models" && f.key !== "MODEL" && f.key !== "MODEL_DEFAULT");
    const entry = {
      key: "MODEL_FABLE",
      label: "Fable Override",
      type: "text",
      value: "",
      section: "models",
      description: "Optional provider/model route for Fable requests.",
      source: "default",
      locked: false,
      secret: false,
      configured: false,
      advanced: false,
    };
    if (firstModelIdx >= 0) config.fields.splice(firstModelIdx, 0, entry);
    else config.fields.push(entry);
  }

  if (!keys.has("ENABLE_FABLE_THINKING")) {
    const firstThinkingIdx = config.fields.findIndex((f) => f.section === "thinking" && f.key !== "THINKING_ENABLED");
    const entry = {
      key: "ENABLE_FABLE_THINKING",
      label: "Fable Thinking",
      type: "tri_boolean",
      value: "",
      section: "thinking",
      description: "Blank inherits Enable Thinking.",
      source: "default",
      locked: false,
      secret: false,
      configured: false,
      advanced: false,
    };
    if (firstThinkingIdx >= 0) config.fields.splice(firstThinkingIdx, 0, entry);
    else config.fields.push(entry);
  }
}

async function load() {
  showMessage("Loading admin config");
  const config = await api("/admin/api/config");
  ensureFableFields(config);
  state.config = config;
  state.fields = new Map(config.fields.map((f) => [f.key, f]));
  renderNav();
  renderSections(config.sections, config.fields);
  const providersView = document.querySelector('section.admin-view[data-view="providers"]');
  renderProviders(config.provider_status, providersView);
  byId("configPath").textContent = config.paths.managed;
  await validate(false);
  await refreshLocalStatus();
  updateDirtyState();
  showMessage("");
}

/* ── Theme ───────────────────────────────────────────────────────────────── */
function getStoredTheme() {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return value === "dark" || value === "light" ? value : null;
  } catch {
    return null;
  }
}

function setStoredTheme(theme) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* localStorage unavailable — toggle still works for the session */
  }
}

function systemPrefersDark() {
  return typeof window !== "undefined"
    && window.matchMedia
    && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function resolveTheme() {
  return getStoredTheme() ?? (systemPrefersDark() ? "dark" : "light");
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
}

function initTheme() {
  applyTheme(resolveTheme());

  const toggle = byId("themeToggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const next =
        document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      setStoredTheme(next);
    });
  }

  // Auto-follow system changes only when user has not picked explicitly.
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      if (getStoredTheme() === null) applyTheme(mq.matches ? "dark" : "light");
    };
    if (mq.addEventListener) mq.addEventListener("change", handler);
    else if (mq.addListener) mq.addListener(handler); // older Safari
  }
}

/* ── Navigation ─────────────────────────────────────────────────────────── */
function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "nav-link";
    btn.dataset.view = view.id;
    btn.textContent = view.label;
    btn.addEventListener("click", () => setActiveView(view.id));
    nav.appendChild(btn);
  });

  setActiveView(state.activeView);
}

function setActiveView(viewId) {
  const active = VIEW_GROUPS.find((v) => v.id === viewId) || VIEW_GROUPS[0];
  state.activeView = active.id;
  byId("pageTitle").textContent = active.title;

  document.querySelectorAll(".nav-link").forEach((l) => {
    l.classList.toggle("active", l.dataset.view === active.id);
  });

  document.querySelectorAll(".admin-view").forEach((v) => {
    const show = v.dataset.view === active.id;
    v.classList.toggle("active", show);
    v.hidden = !show;
  });
}

/* ── Provider cards ──────────────────────────────────────────────────────── */
function renderProviders(providerStatus, container) {
  const grid = document.createElement("div");
  grid.className = "provider-grid";
  container.prepend(grid);
  providerStatus.forEach((p) => {
    const card = document.createElement("div");
    card.className = "provider-card";
    card.dataset.provider = p.provider_id;

    const top = document.createElement("div");
    top.className = "provider-top";

    const name = document.createElement("span");
    name.className = "provider-name";
    name.textContent = providerName(p.provider_id);

    const pill = document.createElement("span");
    pill.className = `status-pill ${statusClass(p.status)}`;
    pill.textContent = p.label;

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    meta.textContent = p.kind === "local" ? (p.base_url || "No local URL") : p.credential_env;

    const btn = document.createElement("button");
    btn.className = "test-button";
    btn.type = "button";
    btn.textContent = p.kind === "local" ? "test connection →" : "refresh →";
    btn.addEventListener("click", () => testProvider(p.provider_id, btn));

    top.append(name, pill);
    card.append(top, meta, btn);
    grid.appendChild(card);
  });
}

function updateProviderCard(providerId, status, label, metaText) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const pill = card.querySelector(".status-pill");
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = label;
  if (metaText) card.querySelector(".provider-meta").textContent = metaText;
}

/* ── Settings sections ──────────────────────────────────────────────────── */
function renderSections(sections, fields) {
  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    container.innerHTML = "";
    view.sections.forEach((sectionId) => {
      renderSection(sections, fields, sectionId, container);
    });
  });
}

function renderSection(allSections, allFields, sectionId, container) {
  const section = allSections.find((s) => s.id === sectionId);
  const sectionFields = allFields.filter((f) => f.section === sectionId);
  if (!section) return;

  const el = document.createElement("div");
  el.className = "settings-section";
  el.id = `section-${section.id}`;

  el.innerHTML = `
    <div class="section-head">
      <h3>${section.label}</h3>
      <p>${section.description || ""}</p>
    </div>
    <div class="section-body"></div>`;

  const body = el.querySelector(".section-body");

  if (sectionFields.length === 0) {
    container.appendChild(el);
    return;
  }

  // Highlight routed Claude ids for the models section so operators can see
  // — even before scrolling — which model names MODEL_FABLE / MODEL_OPUS /
  // MODEL_SONNET / MODEL_HAIKU already cover.
  if (sectionId === "models") {
    body.appendChild(buildModelRoutingBanner(sectionFields));
  }

  const grid = document.createElement("div");
  grid.className = "field-grid";
  body.appendChild(grid);

  sectionFields.forEach((field) => grid.appendChild(renderField(field)));

  if (sectionFields.some((f) => f.advanced)) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "advanced-toggle";
    toggle.textContent = "show advanced ▾";
    toggle.addEventListener("click", () => {
      const showing = el.classList.toggle("show-advanced");
      toggle.textContent = showing ? "hide advanced ▴" : "show advanced ▾";
    });
    body.appendChild(toggle);
  }

  container.appendChild(el);
}

function buildModelRoutingBanner(fields) {
  const banner = document.createElement("div");
  banner.className = "routing-banner";

  const title = document.createElement("div");
  title.className = "routing-banner-title";
  title.textContent = "Recognised Claude model routes";
  banner.appendChild(title);

  // Always list the four routing tiers the resolver understands. Keys come
  // from config.settings (MODEL_FABLE / MODEL_OPUS / MODEL_SONNET / MODEL_HAIKU).
  const rows = [
    { label: "Fable", ids: FABLE_IDS, hint: "MODEL_FABLE" },
    { label: "Opus", ids: FABLE_OPUS_IDS, hint: "MODEL_OPUS" },
    { label: "Sonnet", ids: FABLE_SONNET_IDS, hint: "MODEL_SONNET" },
    { label: "Haiku", ids: FABLE_HAIKU_IDS, hint: "MODEL_HAIKU" },
  ];

  rows.forEach((row) => {
    const rowEl = document.createElement("div");
    rowEl.className = "routing-row";
    const tag = document.createElement("span");
    tag.className = "routing-tag";
    tag.textContent = row.label;
    const ids = document.createElement("span");
    ids.className = "routing-ids";
    ids.textContent = row.ids.join("  ·  ");
    const target = document.createElement("span");
    target.className = "routing-target";
    target.textContent = `→ ${row.hint}`;
    rowEl.append(tag, ids, target);
    banner.appendChild(rowEl);
  });

  return banner;
}

/* ── Field rendering ────────────────────────────────────────────────────── */
function renderField(field) {
  const wrapper = document.createElement("div");
  wrapper.className = `field${field.advanced ? " advanced-field" : ""}`;
  wrapper.dataset.key = field.key;

  const label = document.createElement("label");
  const labelText = document.createElement("span");
  labelText.className = "field-label";
  labelText.textContent = field.label;
  label.appendChild(labelText);

  const src = sourceText(field);
  if (src) {
    const badge = document.createElement("span");
    badge.className = `field-source${field.locked ? " locked" : ""}`;
    badge.textContent = src;
    label.appendChild(badge);
  }

  const input = inputForField(field);
  input.id = `field-${field.key}`;
  input.dataset.key = field.key;
  input.dataset.original = field.value || "";
  input.dataset.secret = field.secret ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.disabled = field.locked;
  input.addEventListener("input", updateDirtyState);
  input.addEventListener("change", updateDirtyState);

  wrapper.append(label, input);
  appendFieldExtras(wrapper, field);
  return wrapper;
}

function appendFieldExtras(wrapper, field) {
  if (field.description) {
    const desc = document.createElement("div");
    desc.className = "field-description";
    desc.textContent = field.description;
    wrapper.appendChild(desc);
  }
}

function inputForField(field) {
  if (field.type === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = String(field.value).toLowerCase() === "true";
    input.dataset.original = input.checked ? "true" : "false";
    return input;
  }

  if (field.type === "tri_boolean") {
    const select = document.createElement("select");
    [
      ["", "Inherit"],
      ["true", "Enabled"],
      ["false", "Disabled"],
    ].forEach(([val, lbl]) => {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = lbl;
      select.appendChild(opt);
    });
    select.value = field.value || "";
    return select;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    field.options.forEach((opt) => {
      const el = document.createElement("option");
      el.value = opt;
      el.textContent = opt;
      select.appendChild(el);
    });
    select.value = field.value || field.options[0] || "";
    return select;
  }

  if (field.type === "textarea") {
    const ta = document.createElement("textarea");
    ta.value = field.value || "";
    return ta;
  }

  const input = document.createElement("input");
  if (field.type === "secret") {
    input.type = "password";
    input.placeholder = field.configured
      ? "Configured — enter a new value to replace"
      : "Not configured";
    input.value = "";
    input.autocomplete = "off";
  } else {
    input.type = field.type === "number" ? "number" : "text";
    input.value = field.value || "";
    if (field.key.startsWith("MODEL")) {
      input.setAttribute("list", "model-options");
    }
  }
  return input;
}

function readFieldValue(input) {
  if (input.type === "checkbox") return input.checked ? "true" : "false";
  if (input.dataset.secret === "true" && input.dataset.configured === "true") {
    return input.value ? input.value : MASKED_SECRET;
  }
  return input.value;
}

function changedValues() {
  const values = {};
  document.querySelectorAll("[data-key]").forEach((input) => {
    if (input.disabled || !input.matches("input, select, textarea")) return;
    const val = readFieldValue(input);
    if (val !== input.dataset.original) values[input.dataset.key] = val;
  });
  return values;
}

/* ── Dirty state ────────────────────────────────────────────────────────── */
function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  byId("dirtyState").textContent =
    count === 0 ? "No changes" : `${count} unsaved change${count === 1 ? "" : "s"}`;
  byId("applyButton").disabled = count === 0;
}

/* ── Validation ─────────────────────────────────────────────────────────── */
async function validate(showResult = true) {
  const result = await api("/admin/api/config/validate", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (showResult) showValidationResult(result);
  return result;
}

function showValidationResult(result) {
  if (result.valid) {
    showMessage("Config shape is valid", "ok");
  } else {
    showMessage(result.errors.join("; "), "error");
  }
}

/* ── Apply ──────────────────────────────────────────────────────────────── */
async function apply() {
  const result = await api("/admin/api/config/apply", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (!result.applied) {
    showValidationResult(result);
    return;
  }
  const restart = result.restart || {};
  if (restart.required && restart.automatic) {
    showMessage("Applied. Restarting server…", "ok");
    byId("applyButton").disabled = true;
    setTimeout(() => { window.location.href = restart.admin_url || "/admin"; }, 1600);
    return;
  }
  const pending = restart.required ? (restart.fields || []) : (result.pending_fields || []);
  await load();
  showMessage(
    pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(", ")}`
      : "Applied",
    "ok",
  );
}

/* ── Provider testing ───────────────────────────────────────────────────── */
async function refreshLocalStatus() {
  const result = await api("/admin/api/providers/local-status");
  result.providers.forEach((p) => {
    state.localStatus.set(p.provider_id, p);
    updateProviderCard(
      p.provider_id,
      p.status,
      p.label,
      p.status_code ? `${p.base_url} · HTTP ${p.status_code}` : p.base_url,
    );
  });
}

async function testProvider(providerId, button) {
  const label = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="animation:spin 1s linear infinite"><path d="M10 2v4M10 14v4M2 10h4M14 10h4"/></svg>Testing`;
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, { method: "POST", body: "{}" });
    if (result.ok) {
      const summary = result.models.length
        ? `${result.models.length} models · ${result.models.slice(0, 2).join(", ")}`
        : "No models returned";
      updateProviderCard(providerId, "reachable", "Reachable", summary);
      state.modelOptions = Array.from(
        new Set([
          ...state.modelOptions,
          ...result.models.map((m) => `${providerId}/${m}`),
        ]),
      ).sort();
      syncModelDatalist();
    } else {
      updateProviderCard(providerId, "offline", result.error_type, result.error_type);
    }
  } finally {
    button.disabled = false;
    button.innerHTML = label;
  }
}

function syncModelDatalist() {
  let dl = byId("model-options");
  if (!dl) {
    dl = document.createElement("datalist");
    dl.id = "model-options";
    document.body.appendChild(dl);
  }
  dl.innerHTML = "";
  state.modelOptions.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    dl.appendChild(opt);
  });
}

/* ── Status bar messages ────────────────────────────────────────────────── */
function showMessage(msg, kind = "") {
  const area = byId("messageArea");
  area.textContent = msg;
  area.className = `message-area ${kind}`.trim();
}

/* ── Init ───────────────────────────────────────────────────────────────── */
initTheme();
byId("validateButton").addEventListener("click", () => validate(true));
byId("applyButton").addEventListener("click", apply);

load().catch((err) => showMessage(err.message, "error"));