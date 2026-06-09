<script lang="ts">
  import { statusClass, providerName, type ProviderStatus, type TestResult } from '../lib/config';
  import { testProvider as apiTestProvider } from '../lib/api';

  export let providers: ProviderStatus[] = [];

  let testing: Record<string, boolean> = {};

  async function handleTest(providerId: string) {
    testing[providerId] = true;
    try {
      const result: TestResult = await apiTestProvider(providerId);
      if (result.ok) {
        updateCard(providerId, 'reachable', `${result.models?.length ?? 0} models`, result.models?.slice(0, 3).join(', ') || 'No models returned');
      } else {
        updateCard(providerId, 'offline', result.error_type || 'Error', result.error_type || 'Error');
      }
    } finally {
      testing[providerId] = false;
    }
  }

  function updateCard(providerId: string, status: string, label: string, metaText: string) {
    const cards = document.querySelectorAll(`[data-provider="${providerId}"]`);
    cards.forEach(card => {
      const pill = card.querySelector('.status-pill');
      if (pill) {
        pill.className = `status-pill ${statusClass(status)}`;
        pill.textContent = label;
      }
      const meta = card.querySelector('.provider-meta');
      if (meta) meta.textContent = metaText;
    });
  }
</script>

<section class="provider-strip">
  <div class="strip-header">
    <h3>Provider Status</h3>
  </div>
  <div class="provider-grid">
    {#each providers as provider (provider.provider_id)}
      <article class="provider-card" data-provider={provider.provider_id}>
        <div class="provider-title">
          <strong>{providerName(provider.provider_id)}</strong>
          <span class="status-pill {statusClass(provider.status)}">{provider.label}</span>
        </div>
        <div class="provider-meta">
          {provider.kind === 'local' ? (provider.base_url || 'No local URL configured') : provider.credential_env}
        </div>
        <button class="test-button" on:click={() => handleTest(provider.provider_id)} disabled={testing[provider.provider_id]}>
          {testing[provider.provider_id] ? 'Testing...' : (provider.kind === 'local' ? 'Test' : 'Refresh models')}
        </button>
      </article>
    {/each}
  </div>
</section>

<style>
  .provider-strip {
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    background: #141414;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    padding: 16px;
    transition: border-color 0.15s ease;
  }

  .strip-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
  }

  .strip-header h3 {
    font-size: 16px;
    font-weight: 600;
    margin: 0;
  }

  .provider-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
  }

  .provider-card {
    display: grid;
    gap: 8px;
    min-height: 108px;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 14px;
    background: #111111;
    transition: all 0.15s ease;
  }

  .provider-card:hover {
    background: #1a1a1a;
    border-color: #252525;
  }

  .provider-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .provider-title strong { font-size: 14px; font-weight: 600; }
  .provider-meta { color: #666; font-size: 12px; word-break: break-word; }

  .status-pill {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    border: 1px solid #1f1f1f;
    border-radius: 999px;
    padding: 4px 12px;
    background: #141414;
    color: #666;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
  }

  :global(.status-pill.ok) {
    color: #22c55e;
    background: rgba(34,197,94,0.1);
    border-color: rgba(34,197,94,0.3);
  }

  :global(.status-pill.warn) {
    color: #f59e0b;
    background: rgba(245,158,11,0.1);
    border-color: rgba(245,158,11,0.3);
  }

  :global(.status-pill.error) {
    color: #ef4444;
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.3);
  }

  .test-button {
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

  .test-button:hover:not(:disabled) {
    border-color: #ffffff;
    background: #1a1a1a;
  }

  .test-button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
