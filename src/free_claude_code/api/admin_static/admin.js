const state = {
  config: null,
  fields: new Map(),
  localStatus: new Map(),
  modelOptions: [],
  activeView: "providers",
};

const MASKED_SECRET = "********";
const VIEW_GROUPS = [
  {
    id: "providers",
    label: "Providers",
    title: "Providers",
    sections: ["providers", "runtime"],
    containerId: "providersSections",
    icon: `<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="nav-icon"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>`,
  },
  {
    id: "model_config",
    label: "Model Config",
    title: "Model Config",
    sections: ["models", "thinking", "web_tools"],
    containerId: "modelConfigSections",
    icon: `<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="nav-icon"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`,
  },
  {
    id: "messaging",
    label: "Messaging",
    title: "Messaging",
    sections: ["messaging", "voice"],
    containerId: "messagingSections",
    icon: `<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="nav-icon"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>`,
  },
  {
    id: "model_validator",
    label: "Model Validator",
    title: "Model Validator",
    sections: [],
    containerId: "modelValidatorSections",
    icon: `<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="nav-icon"><polyline points="22 11.08 22 12 12 22 2 12 12 2"></polyline><path d="M22 4L12 14.01l-3-3"></path></svg>`,
  },
];

const byId = (id) => document.getElementById(id);

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
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
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
  byId("configPath").textContent = config.paths.managed;
  await validate(false);
  await refreshLocalStatus();
  updateDirtyState();
  await initModelValidator();
  showMessage("");
}

function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `nav-link${index === 0 ? " active" : ""}`;
    button.dataset.view = view.id;
    
    // Add icon span
    const iconSpan = document.createElement("span");
    iconSpan.className = "nav-link-icon";
    iconSpan.innerHTML = view.icon;
    
    // Add text span
    const textSpan = document.createElement("span");
    textSpan.className = "nav-link-text";
    textSpan.textContent = view.label;
    
    button.append(iconSpan, textSpan);
    
    if (index === 0) {
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => {
      setActiveView(view.id, { scroll: true });
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
    title.innerHTML = `<strong>${provider.display_name || provider.provider_id}</strong>`;

    const pill = document.createElement("span");
    pill.className = `status-pill ${statusClass(provider.status)}`;
    pill.textContent = provider.label;
    title.appendChild(pill);

    const meta = document.createElement("div");
    meta.className = "provider-meta";
    meta.textContent =
      provider.kind === "local"
        ? provider.base_url || "No local URL configured"
        : provider.credential_env;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "test-button";
    button.textContent = provider.kind === "local" ? "Test" : "Refresh models";
    button.addEventListener("click", () => testProvider(provider.provider_id, button));

    card.append(title, meta, button);
    grid.appendChild(card);
  });
}

function updateProviderCard(providerId, status, label, metaText) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const pill = card.querySelector(".status-pill");
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = label;
  if (metaText) {
    card.querySelector(".provider-meta").textContent = metaText;
  }
}

function renderSections(sections, fields) {
  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    if (container) {
      container.innerHTML = "";
    }
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
    if (!container) return;
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
  input.dataset.original = field.value || "";
  input.dataset.secret = field.secret ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.disabled = field.locked;
  input.addEventListener("input", updateDirtyState);
  input.addEventListener("change", updateDirtyState);

  wrapper.append(label, input);
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

  if (field.type === "tri_boolean") {
    const select = document.createElement("select");
    [
      ["", "Inherit"],
      ["true", "Enabled"],
      ["false", "Disabled"],
    ].forEach(([value, label]) => select.appendChild(option(value, label)));
    select.value = field.value || "";
    return select;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    field.options.forEach((value) => select.appendChild(option(value, value)));
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
    input.type = "password";
    input.placeholder = field.configured
      ? "Configured - enter a new value to replace"
      : "Not configured";
    input.value = "";
    input.autocomplete = "off";
  } else {
    input.value = field.value || "";
  }
  if (field.key.startsWith("MODEL")) {
    input.setAttribute("list", "model-options");
  }
  return input;
}

function option(value, label) {
  const optionEl = document.createElement("option");
  optionEl.value = value;
  optionEl.textContent = label;
  return optionEl;
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
    const value = readFieldValue(input);
    if (value !== input.dataset.original) {
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
}

async function validate(showResult = true) {
  const result = await api("/admin/api/config/validate", {
    method: "POST",
    body: JSON.stringify({ values: changedValues() }),
  });
  if (showResult) {
    showValidationResult(result);
  }
  return result;
}

function showValidationResult(result) {
  if (result.valid) {
    showMessage("Config shape is valid", "ok");
  } else {
    showMessage(result.errors.join("; "), "error");
  }
}

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
    state.localStatus.set(provider.provider_id, provider);
    const meta = provider.status_code
      ? `${provider.base_url} returned HTTP ${provider.status_code}`
      : provider.base_url;
    updateProviderCard(provider.provider_id, provider.status, provider.label, meta);
  });
}

async function testProvider(providerId, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Testing";
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: "POST",
      body: "{}",
    });
    if (result.ok) {
      updateProviderCard(
        providerId,
        "reachable",
        `${result.models.length} models`,
        result.models.slice(0, 3).join(", ") || "No models returned",
      );
      state.modelOptions = Array.from(
        new Set([
          ...state.modelOptions,
          ...result.models.map((model) => `${providerId}/${model}`),
        ]),
      ).sort();
      syncModelDatalist();
    } else {
      updateProviderCard(providerId, "offline", result.error_type, result.error_type);
    }
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function syncModelDatalist() {
  let datalist = byId("model-options");
  if (!datalist) {
    datalist = document.createElement("datalist");
    datalist.id = "model-options";
    document.body.appendChild(datalist);
  }
  datalist.innerHTML = "";
  state.modelOptions.forEach((model) => datalist.appendChild(option(model, model)));
}

function showMessage(message, kind = "") {
  const area = byId("messageArea");
  area.textContent = message;
  area.className = `message-area ${kind}`.trim();
}

let validatorPollInterval = null;

async function initModelValidator() {
  try {
    const data = await api("/admin/api/test-models/models");
    const container = byId("modelChecklist");
    container.innerHTML = "";
    
    const providers = Object.keys(data.grouped).sort();
    providers.forEach(provider => {
      const group = document.createElement("div");
      group.className = "provider-group";
      
      const header = document.createElement("div");
      header.className = "provider-group-header";
      
      const selectAllCheckbox = document.createElement("input");
      selectAllCheckbox.type = "checkbox";
      selectAllCheckbox.id = `select-provider-${provider}`;
      selectAllCheckbox.checked = true;
      
      const label = document.createElement("label");
      label.htmlFor = `select-provider-${provider}`;
      label.innerHTML = `<strong>${provider}</strong>`;
      
      header.append(selectAllCheckbox, label);
      
      const list = document.createElement("div");
      list.className = "provider-models-list";
      
      data.grouped[provider].forEach(model => {
        const item = document.createElement("label");
        item.className = "model-checkbox-item";
        
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = model;
        checkbox.checked = true;
        checkbox.className = "model-checkbox";
        
        const modelSpan = document.createElement("span");
        let displayName = model;
        if (displayName.startsWith("anthropic/")) {
          displayName = displayName.substring("anthropic/".length);
        } else if (displayName.startsWith("claude-3-freecc-no-thinking/")) {
          displayName = displayName.substring("claude-3-freecc-no-thinking/".length);
        }
        if (displayName.startsWith(provider + "/")) {
          displayName = displayName.substring(provider.length + 1);
        }
        modelSpan.textContent = displayName;
        modelSpan.title = model;
        
        item.append(checkbox, modelSpan);
        list.appendChild(item);
        
        checkbox.addEventListener("change", () => {
          const allCheckboxes = list.querySelectorAll(".model-checkbox");
          const checkedCheckboxes = list.querySelectorAll(".model-checkbox:checked");
          selectAllCheckbox.checked = allCheckboxes.length === checkedCheckboxes.length;
          selectAllCheckbox.indeterminate = checkedCheckboxes.length > 0 && checkedCheckboxes.length < allCheckboxes.length;
        });
      });
      
      selectAllCheckbox.addEventListener("change", () => {
        list.querySelectorAll(".model-checkbox").forEach(cb => {
          cb.checked = selectAllCheckbox.checked;
        });
      });
      
      group.append(header, list);
      container.appendChild(group);
    });
    
    const searchInput = byId("modelSearchInput");
    if (searchInput) {
      // Clear any search term from previous loads
      searchInput.value = "";
      searchInput.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase().trim();
        const groups = container.querySelectorAll(".provider-group");
        
        groups.forEach(group => {
          const providerLabel = group.querySelector(".provider-group-header label strong").textContent.toLowerCase();
          const items = group.querySelectorAll(".model-checkbox-item");
          let visibleCount = 0;
          
          items.forEach(item => {
            const modelVal = item.querySelector("input").value.toLowerCase();
            const modelText = item.querySelector("span").textContent.toLowerCase();
            
            const match = modelVal.includes(query) || modelText.includes(query) || providerLabel.includes(query);
            if (match) {
              item.style.display = "flex";
              visibleCount++;
            } else {
              item.style.display = "none";
            }
          });
          
          if (visibleCount > 0 || providerLabel.includes(query)) {
            group.style.display = "block";
            if (providerLabel.includes(query) && query !== "") {
              items.forEach(item => {
                item.style.display = "flex";
              });
            }
          } else {
            group.style.display = "none";
          }
        });
      });
    }
    
    await pollValidatorStatus();
  } catch (err) {
    console.error("Failed to load testable models:", err);
  }
}

byId("btnSelectAllModels").addEventListener("click", () => {
  document.querySelectorAll("#modelChecklist .provider-group").forEach(group => {
    if (group.style.display === "none") return;
    group.querySelectorAll(".model-checkbox-item").forEach(item => {
      if (item.style.display === "none") return;
      const cb = item.querySelector("input[type='checkbox']");
      if (cb) cb.checked = true;
    });
    const selectAllCb = group.querySelector(".provider-group-header input[type='checkbox']");
    if (selectAllCb) {
      selectAllCb.checked = true;
      selectAllCb.indeterminate = false;
    }
  });
});

byId("btnDeselectAllModels").addEventListener("click", () => {
  document.querySelectorAll("#modelChecklist .provider-group").forEach(group => {
    if (group.style.display === "none") return;
    group.querySelectorAll(".model-checkbox-item").forEach(item => {
      if (item.style.display === "none") return;
      const cb = item.querySelector("input[type='checkbox']");
      if (cb) cb.checked = false;
    });
    const selectAllCb = group.querySelector(".provider-group-header input[type='checkbox']");
    if (selectAllCb) {
      selectAllCb.checked = false;
      selectAllCb.indeterminate = false;
    }
  });
});

byId("btnStartValidation").addEventListener("click", async () => {
  const selectedModels = Array.from(document.querySelectorAll(".model-checkbox:checked")).map(cb => cb.value);
  if (selectedModels.length === 0) {
    alert("Please select at least one model to validate.");
    return;
  }
  
  const startBtn = byId("btnStartValidation");
  startBtn.disabled = true;
  startBtn.textContent = "⌛ Starting...";
  
  try {
    await api("/admin/api/test-models/run", {
      method: "POST",
      body: JSON.stringify({ models: selectedModels })
    });
    
    const tbody = byId("resultsTableBody");
    tbody.innerHTML = "";
    selectedModels.forEach(model => {
      const row = document.createElement("tr");
      row.id = `row-${model.replace(/\//g, '_')}`;

      const modelCell = document.createElement("td");
      const code = document.createElement("code");
      code.textContent = model;
      modelCell.appendChild(code);

      const statusCell = document.createElement("td");
      const statusSpan = document.createElement("span");
      statusSpan.className = "status-cell pending";
      statusSpan.textContent = "Pending";
      statusCell.appendChild(statusSpan);

      const httpCell = document.createElement("td");
      httpCell.textContent = "-";

      const latencyCell = document.createElement("td");
      latencyCell.textContent = "-";

      const detailCell = document.createElement("td");
      const detailSpan = document.createElement("span");
      detailSpan.className = "response-preview";
      detailSpan.textContent = "-";
      detailCell.appendChild(detailSpan);

      row.append(modelCell, statusCell, httpCell, latencyCell, detailCell);
      tbody.appendChild(row);
    });
    
    byId("summaryTotal").textContent = `0 / ${selectedModels.length}`;
    byId("summaryPassed").textContent = "0";
    byId("summaryFailed").textContent = "0";
    byId("summaryAvgLatency").textContent = "0 ms";
    
    byId("progressBarStatus").textContent = 'Run' + 'ning';
    byId("progressBarPercent").textContent = "0%";
    byId("progressBarFill").style.width = "0%";
    
    if (validatorPollInterval) clearInterval(validatorPollInterval);
    validatorPollInterval = setInterval(pollValidatorStatus, 1000);
  } catch (err) {
    alert("Failed to start model validator: " + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "🚀 Start Validation";
  }
});

async function pollValidatorStatus() {
  try {
    const status = await api("/admin/api/test-models/status");
    const startBtn = byId("btnStartValidation");
    
    if (status.is_running) {
      startBtn.disabled = true;
      startBtn.textContent = "⌛ Validating...";
      if (!validatorPollInterval) {
        validatorPollInterval = setInterval(pollValidatorStatus, 1000);
      }
    } else {
      startBtn.disabled = false;
      startBtn.textContent = "🚀 Start Validation";
      if (validatorPollInterval) {
        clearInterval(validatorPollInterval);
        validatorPollInterval = null;
      }
    }
    
    const percent = status.total > 0 ? Math.round((status.tested / status.total) * 100) : 0;
    byId("progressBarStatus").textContent = status.is_running ? "Running tests..." : "Idle";
    byId("progressBarPercent").textContent = `${percent}%`;
    byId("progressBarFill").style.width = `${percent}%`;
    
    let passed = 0;
    let failed = 0;
    let totalLatency = 0;
    let passedCount = 0;
    
    const tbody = byId("resultsTableBody");
    if (Object.keys(status.results).length > 0) {
      tbody.innerHTML = "";
      
      const sortedKeys = Object.keys(status.results).sort();
      sortedKeys.forEach(model => {
        const res = status.results[model];
        
        if (res.status === "passed") {
          passed++;
          totalLatency += res.latency_ms;
          passedCount++;
        } else if (["failed", "timeout", "error"].includes(res.status)) {
          failed++;
        }
        
        const row = document.createElement("tr");
        row.id = `row-${model.replace(/\//g, '_')}`;
        
        const modelCell = document.createElement("td");
        const code = document.createElement("code");
        code.textContent = model;
        modelCell.appendChild(code);
        
        const statusCell = document.createElement("td");
        const statusSpan = document.createElement("span");
        statusSpan.className = `status-cell ${res.status}`;
        statusSpan.textContent = res.status;
        statusCell.appendChild(statusSpan);
        
        const httpCell = document.createElement("td");
        httpCell.textContent = res.http_status || "-";
        
        const latencyCell = document.createElement("td");
        latencyCell.textContent = res.latency_ms ? `${res.latency_ms} ms` : "-";
        
        const detailCell = document.createElement("td");
        const detailSpan = document.createElement("span");
        detailSpan.className = "response-preview";
        
        if (res.status === "passed") {
          detailSpan.textContent = res.response ? res.response.substring(0, 100) : "(no body)";
        } else if (res.status === "running") {
          detailSpan.textContent = "Testing...";
        } else if (res.status === "pending") {
          detailSpan.textContent = "Queued";
        } else {
          detailSpan.textContent = res.error_message || res.error_type || "Unknown error";
          detailSpan.style.color = "var(--error)";
        }
        detailCell.appendChild(detailSpan);
        
        row.append(modelCell, statusCell, httpCell, latencyCell, detailCell);
        tbody.appendChild(row);
      });
    }
    
    const avgLatency = passedCount > 0 ? Math.round(totalLatency / passedCount) : 0;
    byId("summaryTotal").textContent = `${status.tested} / ${status.total}`;
    byId("summaryPassed").textContent = passed;
    byId("summaryFailed").textContent = failed;
    byId("summaryAvgLatency").textContent = `${avgLatency} ms`;
    
  } catch (err) {
    console.error("Error polling validator status:", err);
  }
}

byId("validateButton").addEventListener("click", () => validate(true));
byId("applyButton").addEventListener("click", apply);

// Sidebar Toggle Logic
const toggleBtn = byId("sidebarToggle");
if (toggleBtn) {
  toggleBtn.addEventListener("click", () => {
    const shell = document.querySelector(".app-shell");
    const isCollapsed = shell.classList.toggle("collapsed");
    localStorage.setItem("sidebar-collapsed", isCollapsed ? "true" : "false");
  });
}

// Restore collapsed state on load
if (localStorage.getItem("sidebar-collapsed") === "true") {
  const shell = document.querySelector(".app-shell");
  if (shell) shell.classList.add("collapsed");
}

load().catch((error) => {
  showMessage(error.message, "error");
});
