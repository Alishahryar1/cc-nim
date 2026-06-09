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
  <div class="provider-grid" id="providerGrid">
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
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    background: #333333;
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
    border: 1px solid #3a3a3a;
    border-radius: 10px;
    padding: 14px;
    background: #202020;
    transition: all 0.15s ease;
  }

  .provider-card:hover {
    background: #252525;
    border-color: #888888;
  }

  .provider-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .provider-title strong {
    font-size: 14px;
    font-weight: 600;
  }

  .provider-meta {
    color: #888888;
    font-size: 12px;
    word-break: break-word;
  }

  :global(.status-pill.ok) {
    color: #66bb6a;
    background: rgba(102, 187, 106, 0.1);
    border-color: rgba(102, 187, 106, 0.3);
  }

  :global(.status-pill.warn) {
    color: #ffb74d;
    background: rgba(255, 183, 77, 0.1);
    border-color: rgba(255, 183, 77, 0.3);
  }

  :global(.status-pill.error) {
    color: #ef5350;
    background: rgba(239, 83, 80, 0.1);
    border-color: rgba(239, 83, 80, 0.3);
  }

  .test-button {
    min-height: 34px;
    border-radius: 6px;
    border: 1px solid #3a3a3a;
    padding: 7px 14px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    transition: all 0.15s ease;
    background: #333333;
    color: #f0f0f0;
  }

  .test-button:hover:not(:disabled) {
    background-color: rgba(240, 240, 240, 0.06);
    border-color: #e0e0e0;
  }

  .test-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
