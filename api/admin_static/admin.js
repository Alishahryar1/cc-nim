/* ================================================================
   FREE CLAUDE CODE — ADMIN UI JS  ·  Production Build
   ================================================================ */

"use strict";

// ── Constants ─────────────────────────────────────────────────
const MASKED_SECRET     = "********";
const AUTO_REFRESH_MS   = 30_000;
const TOAST_DURATION_MS = 4_200;
const TOAST_ANIM_MS     = 320;

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
  {
    id: "metrics",
    label: "Metrics",
    title: "Request Metrics",
  },
];

// ── App State ─────────────────────────────────────────────────
const state = {
  config:            null,
  fields:            new Map(),
  localStatus:       new Map(),
  modelOptions:      [],
  activeView:        "providers",
  autoRefreshTimer:  null,
};

// ── DOM Helpers ───────────────────────────────────────────────
const byId = (id) => document.getElementById(id);

// ================================================================
//  TOAST SYSTEM
// ================================================================
const Toast = (() => {
  function show(message, kind = "info") {
    const container = byId("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${kind}`;
    toast.setAttribute("role", "alert");

    const dot = document.createElement("span");
    dot.className = "toast-dot";
    dot.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.textContent = message;

    toast.append(dot, text);
    container.appendChild(toast);

    let dismissed = false;

    const dismiss = () => {
      if (dismissed) return;
      dismissed = true;
      toast.classList.add("dismissing");
      setTimeout(() => toast.remove(), TOAST_ANIM_MS);
    };

    const timer = setTimeout(dismiss, TOAST_DURATION_MS);

    toast.addEventListener("click", () => {
      clearTimeout(timer);
      dismiss();
    });
  }

  return { show };
})();

// ================================================================
//  API LAYER
// ================================================================
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let msg = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) msg = String(body.detail);
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  return response.json();
}

// ================================================================
//  BOOTSTRAP / LOAD
// ================================================================
async function load() {
  renderSkeletons();
  showMessage("Loading…");

  try {
    const config = await api("/admin/api/config");
    state.config = config;
    state.fields = new Map(config.fields.map((f) => [f.key, f]));

    renderNav();
    renderProviders(config.provider_status);
    renderSections(config.sections, config.fields);
    byId("configPath").textContent = config.paths.managed;

    await validate(/* showResult */ false);
    await refreshLocalStatus();
    updateDirtyState();
    showMessage("");
    startAutoRefresh();
  } catch (err) {
    showMessage(err.message, "error");
    Toast.show(`Failed to load config: ${err.message}`, "error");
  }
}

// ── Skeletons shown while loading ─────────────────────────────
function renderSkeletons() {
  const grid = byId("providerGrid");
  if (!grid) return;
  grid.innerHTML = "";
  for (let i = 0; i < 8; i++) {
    const sk = document.createElement("div");
    sk.className = "skeleton skeleton-card";
    grid.appendChild(sk);
  }
}

// ================================================================
//  NAVIGATION
// ================================================================
function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `nav-link${index === 0 ? " active" : ""}`;
    btn.dataset.view = view.id;
    btn.textContent = view.label;
    if (index === 0) btn.setAttribute("aria-current", "page");
    btn.addEventListener("click", () => setActiveView(view.id, { scroll: true }));
    nav.appendChild(btn);
  });
  setActiveView(state.activeView, { scroll: false });
}

function setActiveView(viewId, { scroll = false } = {}) {
  const activeView =
    VIEW_GROUPS.find((v) => v.id === viewId) || VIEW_GROUPS[0];
  state.activeView = activeView.id;

  byId("pageTitle").textContent = activeView.title;

  document.querySelectorAll(".nav-link").forEach((link) => {
    const selected = link.dataset.view === activeView.id;
    link.classList.toggle("active", selected);
    selected
      ? link.setAttribute("aria-current", "page")
      : link.removeAttribute("aria-current");
  });

  document.querySelectorAll(".admin-view").forEach((view) => {
    const selected = view.dataset.view === activeView.id;
    view.classList.toggle("active", selected);
    view.hidden = !selected;
  });

  if (activeView.id === "metrics") loadMetrics();
  if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
}

// ================================================================
//  PROVIDERS
// ================================================================
const PROVIDER_NAMES = {
  nvidia_nim:        "NVIDIA NIM",
  open_router:       "OpenRouter",
  mistral_codestral: "Mistral Codestral",
  mistral:           "Mistral",
  deepseek:          "DeepSeek",
  lmstudio:          "LM Studio",
  llamacpp:          "llama.cpp",
  ollama:            "Ollama",
  kimi:              "Kimi",
  wafer:             "Wafer",
  opencode:          "OpenCode Zen",
  opencode_go:       "OpenCode Go",
  zai:               "Z.ai",
  gemini:            "Google Gemini",
  groq:              "Groq",
  cerebras:          "Cerebras",
  openai:            "OpenAI",
  fireworks:         "Fireworks AI",
};

function providerName(id) {
  if (PROVIDER_NAMES[id]) return PROVIDER_NAMES[id];
  return id
    .split("_")
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function statusClass(status) {
  if (["configured", "reachable", "running"].includes(status)) return "ok";
  if (["missing_key", "missing_url", "unknown"].includes(status))  return "warn";
  if (["offline", "error"].includes(status))                       return "error";
  return "";
}

function renderProviders(providerStatus) {
  const grid = byId("providerGrid");
  grid.innerHTML = "";
  providerStatus.forEach((provider) => {
    const card = document.createElement("article");
    card.className = "provider-card";
    card.dataset.provider = provider.provider_id;

    const title = document.createElement("div");
    title.className = "provider-title";

    const name = document.createElement("strong");
    name.textContent = providerName(provider.provider_id);

    const pill = document.createElement("span");
    pill.className = `status-pill ${statusClass(provider.status)}`;
    pill.textContent = provider.label;

    title.append(name, pill);

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    meta.textContent =
      provider.kind === "local"
        ? provider.base_url || "No local URL configured"
        : provider.credential_env;

    const spark = document.createElement("div");
    spark.className = "provider-spark";
    spark.dataset.providerSpark = provider.provider_id;
    spark.title = "Health history — newest right";

    const testBtn = document.createElement("button");
    testBtn.type = "button";
    testBtn.className = "test-button";
    testBtn.textContent =
      provider.kind === "local" ? "Test" : "Refresh models";
    testBtn.addEventListener("click", () =>
      testProvider(provider.provider_id, testBtn)
    );

    card.append(title, meta, spark, testBtn);
    grid.appendChild(card);
  });
  loadHealthHistory();
}

async function loadHealthHistory() {
  try {
    const data = await api("/admin/api/health-history");
    renderHealthHistory(data.providers || {});
  } catch (_err) {
    // Endpoint is loopback-only; remote contexts get a 403 — fail silently.
  }
}

function renderHealthHistory(providers) {
  document.querySelectorAll("[data-provider-spark]").forEach((el) => {
    const pid = el.dataset.providerSpark;
    const entries = providers[pid] || [];
    el.innerHTML = renderSparkBars(entries);
  });
}

function renderSparkBars(entries) {
  if (entries.length === 0) {
    return '<span class="spark-empty">No history yet</span>';
  }
  const recent = entries.slice(-30);
  const maxLat = Math.max(...recent.map((e) => e.latency_ms || 0), 1);
  return recent
    .map((e) => {
      const ok = e.status === "ok";
      const pct = Math.max(8, Math.round(((e.latency_ms || 0) / maxLat) * 100));
      const cls = !ok
        ? "spark-bar error-bar"
        : e.latency_ms > 1500
        ? "spark-bar slow"
        : "spark-bar";
      const label = ok
        ? `${fmtLatency(e.latency_ms)}`
        : `${e.error_type || "error"}`;
      return `<div class="${cls}" style="height:${pct}%" title="${escapeHtml(label)}"></div>`;
    })
    .join("");
}

function updateProviderCard(providerId, status, label, metaText) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const pill = card.querySelector(".status-pill");
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = label;
  if (metaText) card.querySelector(".provider-meta").textContent = metaText;
}

// ================================================================
//  SECTIONS & FIELDS
// ================================================================
function renderSections(sections, fields) {
  VIEW_GROUPS.forEach((view) => {
    if (!view.containerId) return;
    byId(view.containerId).innerHTML = "";
  });

  const sectionById = new Map(sections.map((s) => [s.id, s]));
  const bySection   = new Map();
  sections.forEach((s) => bySection.set(s.id, []));
  fields.forEach((f) => {
    if (!bySection.has(f.section)) bySection.set(f.section, []);
    bySection.get(f.section).push(f);
  });

  VIEW_GROUPS.forEach((view) => {
    if (!view.sections || !view.containerId) return;
    const container = byId(view.containerId);
    view.sections.forEach((sectionId) => {
      const section       = sectionById.get(sectionId);
      const sectionFields = bySection.get(sectionId) || [];
      if (!section || sectionFields.length === 0) return;

      const sectionEl = document.createElement("section");
      sectionEl.className = "settings-section";
      sectionEl.id = `section-${section.id}`;

      const heading = document.createElement("div");
      heading.className = "section-heading";

      const headingInner = document.createElement("div");
      const h3 = document.createElement("h3");
      h3.textContent = section.label;
      const p = document.createElement("p");
      p.textContent = section.description;
      headingInner.append(h3, p);
      heading.appendChild(headingInner);
      sectionEl.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "field-grid";
      sectionFields.forEach((field) => grid.appendChild(renderField(field)));
      sectionEl.appendChild(grid);

      if (sectionFields.some((f) => f.advanced)) {
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

  // Label
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

  // Input
  const input = inputForField(field);
  input.id             = `field-${field.key}`;
  input.dataset.key    = field.key;
  input.dataset.original = field.value || "";
  input.dataset.secret    = field.secret    ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.disabled       = field.locked;
  input.addEventListener("input",  updateDirtyState);
  input.addEventListener("change", updateDirtyState);

  wrapper.appendChild(label);

  // Wrap plain text fields with copy button
  const isCopyable =
    !field.secret &&
    !field.locked &&
    field.type !== "boolean" &&
    field.type !== "tri_boolean" &&
    field.type !== "select" &&
    field.type !== "textarea";

  if (isCopyable) {
    const wrap = document.createElement("div");
    wrap.className = "field-input-wrap";
    wrap.appendChild(input);
    wrap.appendChild(makeCopyButton(input));
    wrapper.appendChild(wrap);
  } else {
    wrapper.appendChild(input);
  }

  if (field.description) {
    const desc = document.createElement("div");
    desc.className = "field-description";
    desc.textContent = field.description;
    wrapper.appendChild(desc);
  }

  return wrapper;
}

// SVG icons as strings
const ICON_COPY = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const ICON_CHECK = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>`;

function makeCopyButton(input) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "copy-btn";
  btn.setAttribute("aria-label", "Copy value");
  btn.innerHTML = ICON_COPY;

  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    const val = input.value.trim();
    if (!val) return;
    try {
      await navigator.clipboard.writeText(val);
      btn.classList.add("copied");
      btn.innerHTML = ICON_CHECK;
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.innerHTML = ICON_COPY;
      }, 1_500);
    } catch {
      Toast.show("Copy failed — requires a secure context (HTTPS/localhost)", "warn");
    }
  });

  return btn;
}

function inputForField(field) {
  if (field.type === "boolean") {
    const input = document.createElement("input");
    input.type    = "checkbox";
    input.checked = String(field.value).toLowerCase() === "true";
    input.dataset.original = input.checked ? "true" : "false";
    return input;
  }

  if (field.type === "tri_boolean") {
    const select = document.createElement("select");
    for (const [val, lbl] of [["", "Inherit"], ["true", "Enabled"], ["false", "Disabled"]]) {
      select.appendChild(option(val, lbl));
    }
    select.value = field.value || "";
    return select;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    for (const v of field.options) select.appendChild(option(v, v));
    select.value = field.value || field.options[0] || "";
    return select;
  }

  if (field.type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.value = field.value || "";
    return textarea;
  }

  const input = document.createElement("input");
  input.type = field.type === "number" ? "number" : "text";

  if (field.type === "secret") {
    input.type        = "password";
    input.placeholder = field.configured
      ? "Configured — enter a new value to replace"
      : "Not configured";
    input.value       = "";
    input.autocomplete = "off";
  } else {
    input.value = field.value || "";
  }

  if (field.key.startsWith("MODEL")) input.setAttribute("list", "model-options");
  return input;
}

function option(value, label) {
  const opt = document.createElement("option");
  opt.value      = value;
  opt.textContent = label;
  return opt;
}

// ── Field source labels ────────────────────────────────────────
function sourceLabel(source) {
  const labels = {
    default: "default",
    template: "template",
    repo_env: "repo .env",
    managed_env: "",
    explicit_env_file: "FCC_ENV_FILE",
    process: "process env",
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

function sourceText(field) {
  const parts = [];
  const lbl = sourceLabel(field.source);
  if (lbl) parts.push(lbl);
  if (field.locked) parts.push("locked");
  return parts.join(" ");
}

// ================================================================
//  DIRTY STATE
// ================================================================
function readFieldValue(input) {
  if (input.type === "checkbox") return input.checked ? "true" : "false";
  if (input.dataset.secret === "true" && input.dataset.configured === "true") {
    return input.value ? input.value : MASKED_SECRET;
  }
  return input.value;
}

function changedValues() {
  const values = {};
  document.querySelectorAll("[data-key]").forEach((el) => {
    if (el.disabled || !el.matches("input, select, textarea")) return;
    const val = readFieldValue(el);
    if (val !== el.dataset.original) values[el.dataset.key] = val;
  });
  return values;
}

function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  const label = byId("dirtyState");

  if (count === 0) {
    label.textContent = "No changes";
    label.classList.remove("has-changes");
  } else {
    label.textContent = `${count} unsaved change${count === 1 ? "" : "s"}`;
    label.classList.add("has-changes");
  }

  byId("applyButton").disabled = count === 0;
}

// ================================================================
//  VALIDATE / APPLY
// ================================================================
async function validate(showResult = true) {
  try {
    const result = await api("/admin/api/config/validate", {
      method: "POST",
      body: JSON.stringify({ values: changedValues() }),
    });
    if (showResult) showValidationResult(result);
    return result;
  } catch (err) {
    showMessage(err.message, "error");
    if (showResult) Toast.show(err.message, "error");
  }
}

function showValidationResult(result) {
  if (result.valid) {
    showMessage("Config shape is valid ✓", "ok");
  } else {
    const msg = result.errors.join("; ");
    showMessage(msg, "error");
    Toast.show(result.errors[0], "error");
  }
}

async function apply() {
  const applyBtn = byId("applyButton");
  const prevText = applyBtn.textContent;
  applyBtn.disabled    = true;
  applyBtn.textContent = "Applying…";

  try {
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
      showMessage("Applied — restarting server…", "ok");
      Toast.show("Config applied — server is restarting", "ok");
      setTimeout(() => {
        window.location.href = restart.admin_url || "/admin";
      }, 1_800);
      return;
    }

    const pending = restart.required
      ? restart.fields || []
      : result.pending_fields || [];

    await load();

    const msg = pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(", ")}`
      : "Config applied successfully";
    showMessage(msg, "ok");
    Toast.show(msg, "ok");
  } catch (err) {
    showMessage(err.message, "error");
    Toast.show(`Apply failed: ${err.message}`, "error");
  } finally {
    applyBtn.disabled    = false;
    applyBtn.textContent = prevText;
  }
}

// ================================================================
//  PROVIDER STATUS
// ================================================================
async function refreshLocalStatus() {
  try {
    const result = await api("/admin/api/providers/local-status");
    result.providers.forEach((p) => {
      state.localStatus.set(p.provider_id, p);
      const meta = p.status_code
        ? `${p.base_url} — HTTP ${p.status_code}`
        : p.base_url;
      updateProviderCard(p.provider_id, p.status, p.label, meta);
    });
  } catch { /* non-critical — silent fail */ }
}

async function testProvider(providerId, button) {
  const original    = button.textContent;
  button.disabled   = true;
  button.textContent = "Testing…";

  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: "POST",
      body: "{}",
    });

    if (result.ok) {
      const n = result.models.length;
      updateProviderCard(
        providerId,
        "reachable",
        `${n} model${n !== 1 ? "s" : ""}`,
        result.models.slice(0, 3).join(", ") || "No models returned",
      );
      state.modelOptions = Array.from(
        new Set([
          ...state.modelOptions,
          ...result.models.map((m) => `${providerId}/${m}`),
        ])
      ).sort();
      syncModelDatalist();
      Toast.show(`${providerName(providerId)}: ${n} model${n !== 1 ? "s" : ""} found`, "ok");
    } else {
      updateProviderCard(providerId, "offline", result.error_type, result.error_type);
      Toast.show(`${providerName(providerId)}: ${result.error_type}`, "error");
    }
  } catch (err) {
    Toast.show(`${providerName(providerId)}: ${err.message}`, "error");
  } finally {
    button.disabled    = false;
    button.textContent = original;
    loadHealthHistory();
  }
}

// ================================================================
//  MODEL DATALIST
// ================================================================
function syncModelDatalist() {
  let datalist = byId("model-options");
  if (!datalist) {
    datalist = document.createElement("datalist");
    datalist.id = "model-options";
    document.body.appendChild(datalist);
  }
  datalist.innerHTML = "";
  state.modelOptions.forEach((m) => datalist.appendChild(option(m, m)));
}

// ================================================================
//  STATUS BAR
// ================================================================
function showMessage(message, kind = "") {
  const area = byId("messageArea");
  area.textContent = message;
  area.className   = `message-area ${kind}`.trim();
}

// ================================================================
//  AUTO-REFRESH
// ================================================================
function startAutoRefresh() {
  stopAutoRefresh();
  const dot = byId("autoRefreshIndicator");
  if (dot) dot.classList.add("active");

  state.autoRefreshTimer = setInterval(() => {
    refreshLocalStatus();
  }, AUTO_REFRESH_MS);
}

function stopAutoRefresh() {
  if (state.autoRefreshTimer !== null) {
    clearInterval(state.autoRefreshTimer);
    state.autoRefreshTimer = null;
  }
  const dot = byId("autoRefreshIndicator");
  if (dot) dot.classList.remove("active");
}

// ================================================================
//  KEYBOARD SHORTCUTS
// ================================================================
document.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (!mod) return;

  if (e.key === "s") {
    e.preventDefault();
    validate(true);
  }

  if (e.key === "Enter") {
    e.preventDefault();
    const applyBtn = byId("applyButton");
    if (!applyBtn.disabled) apply();
  }
});

// ================================================================
//  UNSAVED CHANGES GUARD
// ================================================================
window.addEventListener("beforeunload", (e) => {
  if (Object.keys(changedValues()).length > 0) {
    e.preventDefault();
    e.returnValue = "";
  }
});

// ================================================================
//  REFRESH BUTTON
// ================================================================
const refreshBtn = byId("refreshProvidersBtn");
if (refreshBtn) {
  refreshBtn.addEventListener("click", async () => {
    refreshBtn.classList.add("spinning");
    try {
      await refreshLocalStatus();
    } finally {
      refreshBtn.classList.remove("spinning");
    }
  });
}

// ================================================================
//  EVENT LISTENERS
// ================================================================
byId("validateButton").addEventListener("click", () => validate(true));
byId("applyButton").addEventListener("click", apply);

// ================================================================
//  THEME TOGGLE
// ================================================================
const THEME_KEY = "fcc-theme";

const ICON_SUN = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
const ICON_MOON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

function applyTheme(theme) {
  const isLight = theme === "light";
  document.documentElement.dataset.theme = isLight ? "light" : "";
  const btn = byId("themeToggle");
  if (!btn) return;
  btn.innerHTML       = isLight ? ICON_MOON : ICON_SUN;
  const label         = isLight ? "Switch to dark theme" : "Switch to light theme";
  btn.title           = label;
  btn.setAttribute("aria-label", label);
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  applyTheme(saved === "light" ? "light" : "dark");
}

const themeBtn = byId("themeToggle");
if (themeBtn) {
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(next);
    localStorage.setItem(THEME_KEY, next);
  });
}

// ================================================================
//  SEARCH / FILTER
// ================================================================
function filterFields(query) {
  const needle = query.trim().toLowerCase();

  document.querySelectorAll(".settings-section").forEach((section) => {
    let visible = 0;
    section.querySelectorAll(".field").forEach((field) => {
      // Match against the env-var key and the human label text
      const key   = (field.dataset.key  || "").toLowerCase();
      const label = (field.querySelector("label span")?.textContent || "").toLowerCase();
      const match = !needle || key.includes(needle) || label.includes(needle);
      field.classList.toggle("search-hidden", !match);
      if (match) visible++;
    });
    // Hide the whole section card when nothing matches (preserves layout)
    section.classList.toggle("search-empty", visible === 0);
  });
}

function initSearch() {
  const input = byId("searchInput");
  if (!input) return;

  input.addEventListener("input", () => filterFields(input.value));

  // Clear filter when user switches views
  document.querySelectorAll(".nav-link").forEach((btn) => {
    btn.addEventListener("click", () => {
      input.value = "";
      filterFields("");
    });
  });
}

// ================================================================
//  METRICS
// ================================================================
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadMetrics() {
  const container = byId("metricsContent");
  if (!container) return;
  container.innerHTML = '<p class="metrics-empty">Loading…</p>';
  try {
    const data = await api("/admin/api/metrics");
    renderMetrics(container, data);
  } catch (err) {
    container.innerHTML = `<p class="metrics-empty">${escapeHtml(err.message)}</p>`;
  }
}

function fmtLatency(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`;
}

function renderMetrics(container, data) {
  const { requests, summary } = data;

  const summaryHtml = `
    <div class="metrics-summary">
      <div class="metric-card">
        <div class="metric-value">${summary.total}</div>
        <div class="metric-label">Requests</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${fmtLatency(summary.avg_latency_ms)}</div>
        <div class="metric-label">Avg latency</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${fmtLatency(summary.p95_latency_ms)}</div>
        <div class="metric-label">P95 latency</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${(summary.total_input_tokens + summary.total_output_tokens).toLocaleString()}</div>
        <div class="metric-label">Total tokens</div>
      </div>
    </div>`;

  if (requests.length === 0) {
    container.innerHTML = summaryHtml +
      '<p class="metrics-empty">No requests yet — run a query through the proxy to see latency and token data here.</p>';
    return;
  }

  const maxLat = Math.max(...requests.slice(0, 60).map((r) => r.latency_ms), 1);

  const sparkBars = requests.slice(0, 60).reverse().map((r) => {
    const pct = Math.max(4, Math.round((r.latency_ms / maxLat) * 100));
    const cls = r.status !== "ok"
      ? "spark-bar error-bar"
      : r.latency_ms > 5000
      ? "spark-bar very-slow"
      : r.latency_ms > 2000
      ? "spark-bar slow"
      : "spark-bar";
    return `<div class="${cls}" style="height:${pct}%" title="${escapeHtml(r.provider_id)} ${fmtLatency(r.latency_ms)}"></div>`;
  }).join("");

  const sparkHtml = `<div class="sparkline" title="Latency sparkline — newest right">${sparkBars}</div>`;

  const rows = requests.map((r) => {
    const dt = new Date(r.ts * 1000);
    const timeStr = dt.toLocaleTimeString();
    const barPct  = Math.max(4, Math.round((r.latency_ms / maxLat) * 80));
    const barCls  = r.status !== "ok"
      ? "latency-bar error-bar"
      : r.latency_ms > 5000
      ? "latency-bar very-slow"
      : r.latency_ms > 2000
      ? "latency-bar slow"
      : "latency-bar";
    const stCls = r.status === "ok" ? "ok" : "error";
    return `<tr>
      <td>${escapeHtml(timeStr)}</td>
      <td>${escapeHtml(r.provider_id)}</td>
      <td class="model-cell" title="${escapeHtml(r.model)}">${escapeHtml(r.model)}</td>
      <td>${r.input_tokens.toLocaleString()}</td>
      <td>${r.output_tokens.toLocaleString()}</td>
      <td><div class="latency-bar-wrap"><div class="${barCls}" style="width:${barPct}px"></div><span>${fmtLatency(r.latency_ms)}</span></div></td>
      <td><span class="status-pill ${stCls}">${escapeHtml(r.status)}</span></td>
    </tr>`;
  }).join("");

  container.innerHTML = summaryHtml + sparkHtml + `
    <div class="metrics-table-wrap">
      <table class="metrics-table" aria-label="Recent requests">
        <thead>
          <tr>
            <th>Time</th><th>Provider</th><th>Model</th>
            <th>In</th><th>Out</th><th>Latency</th><th>Status</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ================================================================
//  BOOT
// ================================================================
load().catch((err) => {
  showMessage(err.message, "error");
  Toast.show(`Failed to load: ${err.message}`, "error");
});

initTheme();
initSearch();
