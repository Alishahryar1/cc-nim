<script lang="ts">
  import { VIEW_GROUPS, type ViewId } from '../lib/config';

  export let activeView: ViewId = 'providers';
  export let collapsed = false;

  let collapsedGroups: Record<string, boolean> = {};

  function toggleGroup(key: string) {
    collapsedGroups[key] = !collapsedGroups[key];
  }

  function isGroupCollapsed(key: string): boolean {
    return !!collapsedGroups[key];
  }

  $: activeViewClass = (id: ViewId) => activeView === id ? 'active' : '';

  function handleCollapse() {
    collapsed = !collapsed;
  }
</script>

<aside class="sidebar" class:collapsed>
  <div class="sidebar-logo">
    <svg class="logo-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/>
      <line x1="12" y1="22" x2="12" y2="15.5"/>
      <polyline points="22 8.5 12 15.5 2 8.5"/>
      <polyline points="2 15.5 12 8.5 22 15.5"/>
      <line x1="12" y1="2" x2="12" y2="8.5"/>
    </svg>
    {#if !collapsed}
      <span class="logo-text">Free Claude Code</span>
    {/if}
  </div>

  <button class="collapse-btn" on:click={handleCollapse} title="Toggle sidebar">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points={collapsed ? "9 18 15 12 9 6" : "15 18 9 12 15 6"}/>
    </svg>
  </button>

  <nav class="sidebar-nav">
    <div class="nav-group">
      <div class="nav-group-label" on:click={() => toggleGroup('management')}>
        <span>MANAGEMENT</span>
        <svg class="nav-group-arrow" class:collapsed={isGroupCollapsed('management')} width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </div>
      {#if !isGroupCollapsed('management')}
        <div class="nav-group-items">
          {#each VIEW_GROUPS as view}
            <button class="nav-item {activeViewClass(view.id)}" on:click={() => activeView = view.id}>
              {#if view.id === 'providers'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="3"/><path d="M12 1v4"/><path d="M12 19v4"/><path d="M1 12h4"/><path d="M19 12h4"/><path d="M4.22 4.22l2.83 2.83"/><path d="M16.95 16.95l2.83 2.83"/><path d="M4.22 19.78l2.83-2.83"/><path d="M16.95 7.05l2.83-2.83"/>
                </svg>
              {:else if view.id === 'model_config'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/>
                </svg>
              {:else if view.id === 'messaging'}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
              {/if}
              {#if !collapsed}
                <span>{view.label}</span>
              {/if}
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </nav>

  <div class="sidebar-footer">
    <div class="status-row">
      <div class="status-indicator connected">
        <span class="status-dot"></span>
        {#if !collapsed}
          <span class="status-text">Server Online</span>
        {/if}
      </div>
    </div>
  </div>
</aside>

<style>
  .sidebar {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 240px;
    background: #0d0d0d;
    border-right: 1px solid #1f1f1f;
    display: flex;
    flex-direction: column;
    padding: 0 12px 20px;
    z-index: 100;
    transition: width 0.25s ease;
    overflow: hidden;
  }

  .sidebar.collapsed {
    width: 64px;
    padding: 0 8px 12px;
  }

  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 20px 12px;
    margin: 0 -12px;
    color: #f0f0f0;
    cursor: pointer;
    background: #141414;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    border-bottom: 1px solid #1f1f1f;
    position: relative;
    overflow: hidden;
  }

  .logo-icon {
    width: 28px;
    height: 28px;
    flex-shrink: 0;
    color: #f0f0f0;
  }

  .logo-text {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
  }

  .sidebar.collapsed .sidebar-logo {
    padding: 16px 8px;
    justify-content: center;
  }

  .collapse-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: none;
    background: none;
    color: #666;
    border-radius: 6px;
    cursor: pointer;
    flex-shrink: 0;
    margin: 8px auto;
    transition: all 0.15s ease;
  }

  .collapse-btn:hover {
    color: #f0f0f0;
    background: rgba(255,255,255,0.08);
  }

  .sidebar-nav {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
    overflow-y: auto;
    overflow-x: hidden;
    padding-top: 12px;
    scrollbar-width: none;
  }

  .sidebar-nav::-webkit-scrollbar { display: none; }

  .nav-group { display: flex; flex-direction: column; gap: 2px; }
  .nav-group-items { display: flex; flex-direction: column; gap: 2px; }

  .nav-group-label {
    font-size: 10px;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 8px 12px 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    user-select: none;
    border-radius: 6px;
    transition: color 0.15s ease;
  }
  .nav-group-label:hover { color: #a0a0a0; }

  .nav-group-arrow { transition: transform 0.15s ease; flex-shrink: 0; }
  .nav-group-arrow.collapsed { transform: rotate(-90deg); }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border: none;
    background: none;
    color: #a0a0a0;
    font-size: 14px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    text-align: left;
  }
  .nav-item:hover { background: rgba(255,255,255,0.06); color: #f0f0f0; }
  .nav-item.active { background: rgba(255,255,255,0.1); color: #ffffff; }

  .sidebar.collapsed .nav-item { justify-content: center; padding: 10px 4px; }
  .sidebar.collapsed .nav-item span { display: none; }

  .sidebar-footer { padding-top: 8px; border-top: 1px solid #1f1f1f; margin-top: auto; }

  .status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
  }

  .status-indicator { display: flex; align-items: center; gap: 8px; font-size: 12px; }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .status-indicator.connected .status-dot {
    background-color: #22c55e;
    box-shadow: 0 0 6px rgba(34,197,94,0.5);
    animation: pulse 2s infinite;
  }

  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }

  .status-text { color: #a0a0a0; }
</style>
