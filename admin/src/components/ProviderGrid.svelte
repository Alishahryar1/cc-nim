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
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    background: rgba(20, 20, 30, 0.5);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    padding: 16px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }

  .provider-strip:hover {
    border-color: rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(129, 140, 248, 0.06);
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
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 14px;
    background: rgba(17, 17, 24, 0.6);
    backdrop-filter: blur(8px);
    transition: all 0.15s ease;
  }

  .provider-card:hover {
    background: rgba(20, 20, 30, 0.7);
    border-color: rgba(129, 140, 248, 0.2);
  }

  .provider-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .provider-title strong { font-size: 14px; font-weight: 600; }
  .provider-meta { color: #606070; font-size: 12px; word-break: break-word; }

  .status-pill {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 999px;
    padding: 4px 12px;
    background: rgba(20, 20, 30, 0.5);
    color: #606070;
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
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 7px 14px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    transition: all 0.15s ease;
    background: rgba(20, 20, 30, 0.5);
    color: #f0f0f5;
  }

  .test-button:hover:not(:disabled) {
    border-color: rgba(129, 140, 248, 0.3);
    background: rgba(129, 140, 248, 0.1);
    color: #818cf8;
  }

  .test-button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
