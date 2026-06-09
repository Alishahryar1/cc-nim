<script lang="ts">
  import { sourceText, MASKED_SECRET, type Field, type ViewId } from '../lib/config';

  export let fields: Field[] = [];
  export let sections: { id: string; label: string; description: string }[] = [];
  export let viewSections: string[] = [];

  let showAdvanced: Record<string, boolean> = {};
  let modelOptions: string[] = [];

  function toggleAdvanced(sectionId: string) {
    showAdvanced[sectionId] = !showAdvanced[sectionId];
  }

  function getSection(sectionId: string) {
    return sections.find(s => s.id === sectionId);
  }

  function getFields(sectionId: string) {
    return fields.filter(f => f.section === sectionId);
  }

  export function getChangedValues(): Record<string, string> {
    const values: Record<string, string> = {};
    const inputs = document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('[data-key]');
    inputs.forEach(input => {
      if (input.disabled) return;
      const key = input.dataset.key!;
      let value: string;
      if (input.type === 'checkbox') {
        value = (input as HTMLInputElement).checked ? 'true' : 'false';
      } else if (input.dataset.secret === 'true' && input.dataset.configured === 'true') {
        value = input.value || MASKED_SECRET;
      } else {
        value = input.value;
      }
      if (value !== input.dataset.original) {
        values[key] = value;
      }
    });
    return values;
  }

  export function updateField(key: string, value: string) {
    const input = document.querySelector<HTMLInputElement>(`[data-key="${key}"]`);
    if (input) {
      input.value = value;
      input.dataset.original = value;
    }
  }

  function inputForField(field: Field): string {
    if (field.type === 'boolean') {
      return `<input type="checkbox" ${String(field.value).toLowerCase() === 'true' ? 'checked' : ''} />`;
    }
    if (field.type === 'tri_boolean') {
      const selected = field.value || '';
      return `<select><option value=""${selected === '' ? ' selected' : ''}>Inherit</option><option value="true"${selected === 'true' ? ' selected' : ''}>Enabled</option><option value="false"${selected === 'false' ? ' selected' : ''}>Disabled</option></select>`;
    }
    if (field.type === 'select') {
      const opts = (field.options || []).map(o => `<option value="${o}"${field.value === o ? ' selected' : ''}>${o}</option>`).join('');
      return `<select>${opts}</select>`;
    }
    if (field.type === 'textarea') {
      return `<textarea>${field.value || ''}</textarea>`;
    }
    const isSecret = field.type === 'secret';
    const inputType = field.type === 'number' ? 'number' : (isSecret ? 'password' : 'text');
    const placeholder = isSecret ? (field.configured ? 'Configured - enter a new value to replace' : 'Not configured') : '';
    const val = isSecret ? '' : (field.value || '');
    return `<input type="${inputType}" value="${val.replace(/"/g, '&quot;')}" placeholder="${placeholder}" autocomplete="off" />`;
  }
</script>

{#each viewSections as sectionId}
  {@const section = getSection(sectionId)}
  {@const sectionFields = getFields(sectionId)}
  {#if section && sectionFields.length > 0}
    <section class="settings-section" id="section-{section.id}" class:show-advanced={showAdvanced[section.id]}>
      <div class="section-heading">
        <div>
          <h3>{section.label}</h3>
          <p>{section.description}</p>
        </div>
      </div>
      <div class="field-grid">
        {#each sectionFields as field}
          <div class="field" class:advanced-field={field.advanced} data-key={field.key}>
            <label for="field-{field.key}">
              <span>{field.label}</span>
              {#if sourceText(field)}
                <span class="field-source">{sourceText(field)}</span>
              {/if}
            </label>
            <!-- svelte-ignore a11y-no-autofocus -->
            <div data-key={field.key} data-original={field.value || ''} data-secret={field.secret ? 'true' : 'false'} data-configured={field.configured ? 'true' : 'false'} data-disabled={field.locked}>
              {#if field.type === 'boolean'}
                <input type="checkbox" id="field-{field.key}" {field} checked={String(field.value).toLowerCase() === 'true'} disabled={field.locked} on:change on:input />
              {:else if field.type === 'tri_boolean'}
                <select id="field-{field.key}" disabled={field.locked} on:change>
                  <option value="" selected={!field.value}>Inherit</option>
                  <option value="true" selected={field.value === 'true'}>Enabled</option>
                  <option value="false" selected={field.value === 'false'}>Disabled</option>
                </select>
              {:else if field.type === 'select'}
                <select id="field-{field.key}" disabled={field.locked} on:change>
                  {#each field.options || [] as opt}
                    <option value={opt} selected={field.value === opt}>{opt}</option>
                  {/each}
                </select>
              {:else if field.type === 'textarea'}
                <textarea id="field-{field.key}" disabled={field.locked} on:input>{field.value || ''}</textarea>
              {:else}
                <input
                  type={field.type === 'number' ? 'number' : (field.type === 'secret' ? 'password' : 'text')}
                  id="field-{field.key}"
                  disabled={field.locked}
                  placeholder={field.type === 'secret' ? (field.configured ? 'Configured - enter a new value to replace' : 'Not configured') : ''}
                  value={field.type === 'secret' ? '' : field.value || ''}
                  autocomplete="off"
                  on:input
                  on:change
                />
              {/if}
            </div>
            {#if field.description}
              <div class="field-description">{field.description}</div>
            {/if}
          </div>
        {/each}
      </div>
      {#if sectionFields.some(f => f.advanced)}
        <button class="advanced-toggle ghost-button" on:click={() => toggleAdvanced(section.id)}>
          {showAdvanced[section.id] ? 'Hide advanced' : 'Show advanced'}
        </button>
      {/if}
    </section>
  {/if}
{/each}

<style>
  .settings-section {
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    background: #141414;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    padding: 20px;
    scroll-margin-top: 20px;
    transition: border-color 0.15s ease;
  }

  .settings-section:hover { border-color: #252525; }

  .section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
  }

  .section-heading h3 { font-size: 16px; font-weight: 600; margin: 0; }
  .section-heading p { color: #666; font-size: 12px; margin: 2px 0 0; }

  .field-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
  }

  .field { display: grid; gap: 6px; align-content: start; }

  .field label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
  }

  .field-source { color: #666; font-size: 11px; font-weight: 500; }

  .field :global(input),
  .field :global(select),
  .field :global(textarea) {
    width: 100%;
    min-height: 38px;
    border: 1px solid #1f1f1f;
    border-radius: 8px;
    background: #1a1a1a;
    color: #f0f0f0;
    padding: 8px 12px;
    transition: border-color 0.15s ease;
  }

  .field :global(input:focus),
  .field :global(select:focus),
  .field :global(textarea:focus) {
    outline: none;
    border-color: #ffffff;
  }

  .field :global(input:hover),
  .field :global(select:hover),
  .field :global(textarea:hover) {
    border-color: #252525;
  }

  .field :global(input:disabled),
  .field :global(select:disabled),
  .field :global(textarea:disabled) {
    background: #111111;
    color: #666;
    cursor: not-allowed;
  }

  .field-description { color: #666; font-size: 12px; line-height: 1.4; }

  .advanced-field { display: none; }
  .show-advanced .advanced-field { display: grid; }

  .advanced-toggle { justify-self: start; margin-top: 12px; }

  .ghost-button {
    min-height: 34px;
    border-radius: 8px;
    border: 1px solid #1f1f1f;
    padding: 7px 14px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    transition: all 0.15s ease;
    background: #141414;
    color: #f0f0f0;
  }

  .ghost-button:hover { border-color: #ffffff; background: #1a1a1a; }
</style>
