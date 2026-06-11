const state = {
  config: null,
  fields: new Map(),
  localStatus: new Map(),
  modelOptions: [],
  activeView: 'providers',
  sidebarCollapsed: false,
  collapsedGroups: {},
};

const MASKED_SECRET = '********';
const VIEW_GROUPS = [
  {
    id: 'providers',
    label: 'Providers',
    title: 'Providers',
    sections: ['providers', 'runtime'],
    containerId: 'providersSections',
  },
  {
    id: 'model_config',
    label: 'Model Config',
    title: 'Model Config',
    sections: ['models', 'thinking', 'web_tools'],
    containerId: 'modelConfigSections',
  },
  {
    id: 'messaging',
    label: 'Messaging',
    title: 'Messaging',
    sections: ['messaging', 'voice'],
    containerId: 'messagingSections',
  },
];

const byId = (id) => document.getElementById(id);

function sourceLabel(source) {
  const labels = {
    default: 'default',
    template: 'template',
    repo_env: 'repo .env',
    managed_env: "",
    explicit_env_file: 'FCC_ENV_FILE',
    process: 'process env',
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
  return parts.join(' ');
}

function providerName(providerId) {
  const names = {
    nvidia_nim: 'NVIDIA NIM',
    open_router: 'OpenRouter',
    mistral_codestral: 'Mistral Codestral',
    deepseek: 'DeepSeek',
    lmstudio: 'LM Studio',
    llamacpp: 'llama.cpp',
    ollama: 'Ollama',
    kimi: 'Kimi',
    wafer: 'Wafer',
    opencode: 'OpenCode Zen',
    opencode_go: 'OpenCode Go',
    zai: 'Z.ai',
  };
  if (names[providerId]) return names[providerId];
  return providerId
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function statusClass(status) {
  if (['configured', 'reachable', 'running'].includes(status)) return 'ok';
  if (['missing_key', 'missing_url', 'unknown'].includes(status)) return 'warn';
  if (['offline', 'error'].includes(status)) return 'error';
  return 'neutral';
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function load() {
  showMessage('Loading admin config');
  const config = await api('/admin/api/config');
  state.config = config;
  state.fields = new Map(config.fields.map((field) => [field.key, field]));
  renderNav();
  renderProviders(config.provider_status);
  renderProviderCharts(config.provider_status);
  renderSections(config.sections, config.fields);
  byId('configPath').textContent = config.paths.managed;
  await validate(false);
  await refreshLocalStatus();
  updateDirtyState();
  showMessage('');
}

function renderNav() {
  VIEW_GROUPS.forEach((view) => {
    const button = document.querySelector(`[data-view="${view.id}"]`);
    if (button) {
      button.addEventListener('click', () => {
        setActiveView(view.id, { scroll: true });
      });
    }
  });
  setActiveView(state.activeView, { scroll: false });
}

function setActiveView(viewId, { scroll = false } = {}) {
  const activeView = VIEW_GROUPS.find((view) => view.id === viewId) || VIEW_GROUPS[0];
  state.activeView = activeView.id;
  byId('pageTitle').textContent = activeView.title;

  document.querySelectorAll('.nav-item').forEach((item) => {
    const selected = item.dataset.view === activeView.id;
    item.classList.toggle('active', selected);
  });

  document.querySelectorAll('.admin-view').forEach((view) => {
    const selected = view.dataset.view === activeView.id;
    view.classList.toggle('active', selected);
    view.hidden = !selected;
  });

  if (scroll) {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

function renderProviders(providerStatus) {
  const grid = byId('providerGrid');
  grid.innerHTML = '';
  providerStatus.forEach((provider) => {
    const card = document.createElement('article');
    card.className = 'provider-card';
    card.dataset.provider = provider.provider_id;
    card.dataset.status = statusClass(provider.status);

    const title = document.createElement('div');
    title.className = 'provider-title';
    title.innerHTML = `<strong>${providerName(provider.provider_id)}</strong>`;

    const pill = document.createElement('span');
    pill.className = `status-pill ${statusClass(provider.status)}`;
    pill.textContent = provider.label;
    title.appendChild(pill);

    const meta = document.createElement('div');
    meta.className = 'provider-meta';
    meta.textContent =
      provider.kind === 'local'
        ? provider.base_url || 'No local URL configured'
        : provider.credential_env;

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'test-button';
    button.textContent = provider.kind === 'local' ? 'Test' : 'Refresh models';
    button.addEventListener('click', () => testProvider(provider.provider_id, button));

    card.append(title, meta, button);
    grid.appendChild(card);
  });
}

function updateProviderCard(providerId, status, label, metaText) {
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const pill = card.querySelector('.status-pill');
  pill.className = `status-pill ${statusClass(status)}`;
  pill.textContent = label;
  card.dataset.status = statusClass(status);
  if (metaText) {
    card.querySelector('.provider-meta').textContent = metaText;
  }
}

function renderSections(sections, fields) {
  VIEW_GROUPS.forEach((view) => {
    byId(view.containerId).innerHTML = '';
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

      const sectionEl = document.createElement('section');
      sectionEl.className = 'settings-section';
      sectionEl.id = `section-${section.id}`;

      const heading = document.createElement('div');
      heading.className = 'section-heading';
      heading.innerHTML = `<div><h3>${section.label}</h3><p>${section.description}</p></div>`;
      sectionEl.appendChild(heading);

      const grid = document.createElement('div');
      grid.className = 'field-grid';
      sectionFields.forEach((field) => {
        grid.appendChild(renderField(field));
      });
      sectionEl.appendChild(grid);

      if (sectionFields.some((field) => field.advanced)) {
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'ghost-button advanced-toggle';
        toggle.textContent = 'Show advanced';
        toggle.addEventListener('click', () => {
          const showing = sectionEl.classList.toggle('show-advanced');
          toggle.textContent = showing ? 'Hide advanced' : 'Show advanced';
        });
        sectionEl.appendChild(toggle);
      }

      container.appendChild(sectionEl);
    });
  });
}

function renderField(field) {
  const wrapper = document.createElement('div');
  wrapper.className = `field${field.advanced ? ' advanced-field' : ''}`;
  wrapper.dataset.key = field.key;

  const label = document.createElement('label');
  label.htmlFor = `field-${field.key}`;
  const labelText = document.createElement('span');
  labelText.textContent = field.label;
  label.appendChild(labelText);

  const source = sourceText(field);
  if (source) {
    const sourceEl = document.createElement('span');
    sourceEl.className = 'field-source';
    sourceEl.textContent = source;
    label.appendChild(sourceEl);
  }

  const input = inputForField(field);
  input.id = `field-${field.key}`;
  input.dataset.key = field.key;
  input.dataset.original = field.value || '';
  input.dataset.secret = field.secret ? 'true' : 'false';
  input.dataset.configured = field.configured ? 'true' : 'false';
  input.disabled = field.locked;
  input.addEventListener('input', updateDirtyState);
  input.addEventListener('change', updateDirtyState);

  wrapper.append(label, input);
  if (field.description) {
    const description = document.createElement('div');
    description.className = 'field-description';
    description.textContent = field.description;
    wrapper.appendChild(description);
  }
  return wrapper;
}

function inputForField(field) {
  if (field.type === 'boolean') {
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = String(field.value).toLowerCase() === 'true';
    input.dataset.original = input.checked ? 'true' : 'false';
    return input;
  }

  if (field.type === 'tri_boolean') {
    const select = document.createElement('select');
    [
      ['', 'Inherit'],
      ['true', 'Enabled'],
      ['false', 'Disabled'],
    ].forEach(([value, label]) => select.appendChild(option(value, label)));
    select.value = field.value || '';
    return select;
  }

  if (field.type === 'select') {
    const select = document.createElement('select');
    field.options.forEach((value) => select.appendChild(option(value, value)));
    select.value = field.value || field.options[0] || '';
    return select;
  }

  if (field.type === 'textarea') {
    const textarea = document.createElement('textarea');
    textarea.value = field.value || '';
    return textarea;
  }

  const input = document.createElement('input');
  input.type = field.type === 'number' ? 'number' : 'text';
  if (field.type === 'secret') {
    input.type = 'password';
    input.placeholder = field.configured
      ? 'Configured - enter a new value to replace'
      : 'Not configured';
    input.value = '';
    input.autocomplete = 'off';
  } else {
    input.value = field.value || '';
  }
  if (field.key.startsWith('MODEL')) {
    input.setAttribute('list', 'model-options');
  }
  return input;
}

function option(value, label) {
  const optionEl = document.createElement('option');
  optionEl.value = value;
  optionEl.textContent = label;
  return optionEl;
}

function readFieldValue(input) {
  if (input.type === 'checkbox') return input.checked ? 'true' : 'false';
  if (input.dataset.secret === 'true' && input.dataset.configured === 'true') {
    return input.value ? input.value : MASKED_SECRET;
  }
  return input.value;
}

function changedValues() {
  const values = {};
  document.querySelectorAll('[data-key]').forEach((input) => {
    if (input.disabled || !input.matches('input, select, textarea')) return;
    const value = readFieldValue(input);
    if (value !== input.dataset.original) {
      values[input.dataset.key] = value;
    }
  });
  return values;
}

function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  byId('dirtyState').textContent =
    count === 0 ? 'No changes' : `${count} unsaved change${count === 1 ? '' : 's'}`;
  byId('dirtyState').className = `status-pill${count === 0 ? '' : ' warn'}`;
  byId('applyButton').disabled = count === 0;
}

async function validate(showResult = true) {
  const result = await api('/admin/api/config/validate', {
    method: 'POST',
    body: JSON.stringify({ values: changedValues() }),
  });
  if (showResult) {
    showValidationResult(result);
  }
  return result;
}

function showValidationResult(result) {
  if (result.valid) {
    showMessage('Config shape is valid', 'ok');
  } else {
    showMessage(result.errors.join('; '), 'error');
  }
}

async function apply() {
  const result = await api('/admin/api/config/apply', {
    method: 'POST',
    body: JSON.stringify({ values: changedValues() }),
  });
  if (!result.applied) {
    showValidationResult(result);
    return;
  }
  const restart = result.restart || {};
  if (restart.required && restart.automatic) {
    showMessage('Applied. Restarting server...', 'ok');
    byId('applyButton').disabled = true;
    setTimeout(() => {
      window.location.href = restart.admin_url || '/admin';
    }, 1600);
    return;
  }
  const pending = restart.required ? restart.fields || [] : result.pending_fields || [];
  await load();
  showMessage(
    pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(', ')}`
      : 'Applied',
    'ok',
  );
}

async function refreshLocalStatus() {
  const result = await api('/admin/api/providers/local-status');
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
  button.textContent = 'Testing';
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: 'POST',
      body: '{}',
    });
    if (result.ok) {
      updateProviderCard(
        providerId,
        'reachable',
        `${result.models.length} models`,
        result.models.slice(0, 3).join(', ') || 'No models returned',
      );
      state.modelOptions = Array.from(
        new Set([
          ...state.modelOptions,
          ...result.models.map((model) => `${providerId}/${model}`),
        ]),
      ).sort();
      syncModelDatalist();
    } else {
      updateProviderCard(providerId, 'offline', result.error_type, result.error_type);
    }
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function syncModelDatalist() {
  let datalist = byId('model-options');
  if (!datalist) {
    datalist = document.createElement('datalist');
    datalist.id = 'model-options';
    document.body.appendChild(datalist);
  }
  datalist.innerHTML = '';
  state.modelOptions.forEach((model) => datalist.appendChild(option(model, model)));
}

function showMessage(message, kind = '') {
  const area = byId('messageArea');
  area.textContent = message;
  area.className = `message-area ${kind}`.trim();
}

function toggleSidebar() {
  const sidebar = byId('sidebar');
  state.sidebarCollapsed = !state.sidebarCollapsed;
  sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
  localStorage.setItem('sidebarCollapsed', state.sidebarCollapsed);
}

function toggleGroup(groupName) {
  const arrow = document.querySelector(`[data-group="${groupName}"] .nav-group-arrow`);
  const items = document.getElementById(
    groupName === 'management' ? 'navManagement' : 'navMessaging',
  );

  if (arrow && items) {
    state.collapsedGroups[groupName] = !state.collapsedGroups[groupName];
    arrow.classList.toggle('collapsed', state.collapsedGroups[groupName]);
    items.style.display = state.collapsedGroups[groupName] ? 'none' : 'flex';
    localStorage.setItem('collapsedGroups', JSON.stringify(state.collapsedGroups));
  }
}

function renderProviderCharts(providerStatus) {
  const statusCounts = {
    ok: 0,
    warn: 0,
    error: 0,
    neutral: 0,
  };

  providerStatus.forEach((provider) => {
    const status = statusClass(provider.status);
    if (statusCounts[status] !== undefined) {
      statusCounts[status]++;
    }
  });

  const total = providerStatus.length;
  if (total === 0) return;

  const container = document.createElement('section');
  container.className = 'provider-charts';
  container.innerHTML = `
    <div class="charts-grid">
      <div class="chart-card">
        <h4>Provider Status Distribution</h4>
        <div class="pie-chart" id="statusPieChart"></div>
      </div>
      <div class="chart-card">
        <h4>Provider Types</h4>
        <div class="bar-chart" id="providerTypesChart"></div>
      </div>
    </div>
  `;

  const chartsSection = document.createElement('section');
  chartsSection.className = 'provider-strip';
  chartsSection.innerHTML = `
    <div class="strip-header">
      <h3>Provider Analytics</h3>
    </div>
  `;
  chartsSection.appendChild(container);

  const providerStrip = byId('providerGrid').parentElement;
  providerStrip.insertBefore(chartsSection, byId('providerGrid'));

  renderPieChart('statusPieChart', statusCounts, total);
  renderBarChart('providerTypesChart', providerStatus);
}

function renderPieChart(id, data, total) {
  const container = document.getElementById(id);
  if (!container) return;

  const colors = {
    ok: 'var(--success)',
    warn: 'var(--warning)',
    error: 'var(--error)',
    neutral: 'var(--text-muted)',
  };

  let angle = 0;
  const radius = 50;
  const centerX = 100;
  const centerY = 100;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '200');
  svg.setAttribute('height', '200');

  for (const [status, count] of Object.entries(data)) {
    if (count === 0) continue;

    const percentage = (count / total) * 100;
    const strokeDasharray = `${(count / total) * 628.318} 628.318`;
    const strokeDashoffset = 628.318 - (count / total) * 628.318;

    const slice = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    slice.setAttribute('cx', centerX.toString());
    slice.setAttribute('cy', centerY.toString());
    slice.setAttribute('r', radius.toString());
    slice.setAttribute('fill', 'none');
    slice.setAttribute('stroke', colors[status] || '#ccc');
    slice.setAttribute('stroke-width', '20');
    slice.setAttribute('stroke-dasharray', strokeDasharray);
    slice.setAttribute('stroke-dashoffset', strokeDashoffset.toString());
    slice.setAttribute('transform', `rotate(${angle} ${centerX} ${centerY})`);

    container.appendChild(slice);

    angle += (count / total) * 360;
  }

  const legend = document.createElement('div');
  legend.className = 'pie-legend';

  for (const [status, count] of Object.entries(data)) {
    if (count === 0) continue;

    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `
      <span class="legend-color" style="background: ${colors[status]}"></span>
      <span class="legend-label">${status.toUpperCase()}</span>
      <span class="legend-count">${count}</span>
    `;
    legend.appendChild(item);
  }

  container.appendChild(legend);
}

function renderBarChart(id, providers) {
  const container = document.getElementById(id);
  if (!container) return;

  const typeCounts = {};
  providers.forEach((provider) => {
    const type = provider.kind === 'local' ? 'Local' : 'Remote';
    typeCounts[type] = (typeCounts[type] || 0) + 1;
  });

  const maxCount = Math.max(...Object.values(typeCounts), 1);
  const barWidth = 60;
  const barSpacing = 20;
  const chartWidth = Object.keys(typeCounts).length * (barWidth + barSpacing);

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', chartWidth.toString());
  svg.setAttribute('height', '200');

  let x = 0;

  for (const [type, count] of Object.entries(typeCounts)) {
    const barHeight = (count / maxCount) * 150;
    const y = 150 - barHeight;

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x.toString());
    rect.setAttribute('y', y.toString());
    rect.setAttribute('width', barWidth.toString());
    rect.setAttribute('height', barHeight.toString());
    rect.setAttribute('fill', type === 'Local' ? 'var(--accent)' : 'var(--success)');
    rect.setAttribute('rx', '4');

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', (x + barWidth / 2).toString());
    label.setAttribute('y', (y - 5).toString());
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-family', 'Inter');
    label.setAttribute('font-size', '12');
    label.setAttribute('fill', 'var(--text-primary)');
    label.textContent = type;

    const value = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    value.setAttribute('x', (x + barWidth / 2).toString());
    value.setAttribute('y', (y + barHeight + 15).toString());
    value.setAttribute('text-anchor', 'middle');
    value.setAttribute('font-family', 'Inter');
    value.setAttribute('font-size', '14');
    value.setAttribute('font-weight', '600');
    value.setAttribute('fill', 'var(--text-primary)');
    value.textContent = count.toString();

    svg.appendChild(rect);
    svg.appendChild(label);
    svg.appendChild(value);

    x += barWidth + barSpacing;
  }

  container.appendChild(svg);
}

function initEventListeners() {
  byId('validateButton').addEventListener('click', () => validate(true));
  byId('applyButton').addEventListener('click', apply);
  byId('collapseBtn').addEventListener('click', toggleSidebar);

  document.querySelectorAll('.nav-group-label').forEach((label) => {
    label.addEventListener('click', () => {
      toggleGroup(label.dataset.group);
    });
  });

  const savedCollapsed = localStorage.getItem('sidebarCollapsed');
  if (savedCollapsed === 'true') {
    state.sidebarCollapsed = true;
    byId('sidebar').classList.add('collapsed');
  }

  const savedGroups = localStorage.getItem('collapsedGroups');
  if (savedGroups) {
    try {
      state.collapsedGroups = JSON.parse(savedGroups);
      Object.entries(state.collapsedGroups).forEach(([group, collapsed]) => {
        if (collapsed) {
          const arrow = document.querySelector(`[data-group="${group}"] .nav-group-arrow`);
          const items = document.getElementById(
            group === 'management' ? 'navManagement' : 'navMessaging',
          );
          if (arrow) arrow.classList.add('collapsed');
          if (items) items.style.display = 'none';
        }
      });
    } catch {
      // ignore parse errors
    }
  }
}

initEventListeners();

load().catch((error) => {
  showMessage(error.message, 'error');
});
