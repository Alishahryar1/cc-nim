<script lang="ts">
  import { VIEW_GROUPS, type ViewId } from '../lib/config';

  export let activeView: ViewId = 'providers';
  export let collapsed = false;
  export let onViewChange: ((viewId: ViewId) => void) | undefined = undefined;

  let collapsedGroups: Record<string, boolean> = {};

  function toggleGroup(key: string) {
    collapsedGroups[key] = !collapsedGroups[key];
  }

  function isGroupCollapsed(key: string): boolean {
    return !!collapsedGroups[key];
  }

  function handleNavClick(viewId: ViewId) {
    activeView = viewId;
    onViewChange?.(viewId);
  }
</script>

<aside class="sidebar flex flex-col shrink-0 overflow-hidden transition-all duration-300 ease-in-out {collapsed ? 'w-16' : 'w-60'}">
  <!-- Logo -->
  <div class="sidebar-logo flex items-center {collapsed ? 'justify-center px-2 py-4' : 'gap-3 px-4 py-5'}">
    <svg class="logo-icon {collapsed ? 'w-7 h-7' : 'w-6 h-6'}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
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

  <!-- Collapse toggle -->
  <button
    class="collapse-btn absolute top-[16px] right-[12px] z-10 {collapsed ? '!relative !top-auto !right-auto mx-auto mt-1 mb-2' : ''}"
    on:click={() => collapsed = !collapsed}
    title="Toggle sidebar"
  >
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points={collapsed ? "9 18 15 12 9 6" : "15 18 9 12 15 6"}/>
    </svg>
  </button>

  <!-- Navigation -->
  <nav class="sidebar-nav flex-1 flex flex-col gap-0.5 overflow-y-auto min-h-0 {collapsed ? 'px-2 pt-10' : 'px-3 pt-10'}">
    <!-- Management group -->
    <div class="nav-group-label" on:click={() => toggleGroup('management')}>
      {#if !collapsed}
        <span>MANAGEMENT</span>
        <svg class="nav-arrow transition-transform duration-150 {isGroupCollapsed('management') ? '-rotate-90' : ''}" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      {:else}
        <div class="nav-divider"></div>
      {/if}
    </div>

    {#if !isGroupCollapsed('management')}
      <div class="nav-group-items flex flex-col gap-0.5">
        {#each VIEW_GROUPS as view}
          <button
            class="nav-item {activeView === view.id ? 'active' : ''}"
            on:click={() => handleNavClick(view.id)}
          >
            {#if view.id === 'providers'}
              <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/><path d="M12 1v4"/><path d="M12 19v4"/><path d="M1 12h4"/><path d="M19 12h4"/><path d="M4.22 4.22l2.83 2.83"/><path d="M16.95 16.95l2.83 2.83"/><path d="M4.22 19.78l2.83-2.83"/><path d="M16.95 7.05l2.83-2.83"/>
              </svg>
            {:else if view.id === 'model_config'}
              <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/>
              </svg>
            {:else if view.id === 'messaging'}
              <svg class="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
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
  </nav>

  <!-- Footer -->
  <div class="sidebar-footer {collapsed ? 'px-2' : 'px-3'}">
    <div class="sidebar-footer-inner">
      <div class="status-indicator {collapsed ? 'justify-center' : ''}">
        <span class="status-dot connected"></span>
        {#if !collapsed}
          <span class="status-text">Connected</span>
        {/if}
      </div>
    </div>
  </div>
</aside>

<style>
  .sidebar {
    background: var(--color-bg-surface);
    border-right: 1px solid var(--color-border-subtle);
    position: relative;
  }

  .sidebar-logo {
    border-bottom: 1px solid var(--color-border-subtle);
    background: var(--color-bg-elevated);
  }

  .logo-icon {
    flex-shrink: 0;
    color: var(--color-accent);
  }

  .logo-text {
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
    letter-spacing: -0.01em;
    white-space: nowrap;
  }

  .collapse-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    border: none;
    background: transparent;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .collapse-btn:hover {
    color: var(--color-text-primary);
    background: var(--color-bg-hover);
  }

  .sidebar-nav {
    scrollbar-width: none;
  }

  .sidebar-nav::-webkit-scrollbar {
    display: none;
  }

  .nav-group-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px 6px;
    font-size: 10px;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    cursor: pointer;
    user-select: none;
    transition: color 0.15s ease;
  }

  .nav-group-label:hover {
    color: var(--color-text-secondary);
  }

  .nav-arrow {
    color: var(--color-text-muted);
    flex-shrink: 0;
  }

  .nav-divider {
    width: 100%;
    height: 1px;
    background: var(--color-border-subtle);
    margin: 4px 0;
  }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border: none;
    background: transparent;
    color: var(--color-text-secondary);
    font-size: 13px;
    font-weight: 400;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    text-align: left;
    position: relative;
  }

  .nav-item:hover {
    background: rgba(255, 255, 255, 0.04);
    color: var(--color-text-primary);
  }

  .nav-item.active {
    background: var(--color-accent-subtle);
    color: var(--color-accent);
    font-weight: 500;
  }

  .nav-item.active::before {
    content: '';
    position: absolute;
    left: -12px;
    top: 50%;
    transform: translateY(-50%);
    width: 3px;
    height: 16px;
    border-radius: 0 2px 2px 0;
    background: var(--color-accent);
  }

  .nav-icon {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }

  .sidebar-footer {
    border-top: 1px solid var(--color-border-subtle);
    padding-top: 8px;
  }

  .sidebar-footer-inner {
    padding: 8px 12px;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-text {
    font-size: 12px;
    color: var(--color-text-muted);
  }

  /* Collapsed state */
  .sidebar.collapsed .nav-item {
    justify-content: center;
    padding: 8px;
  }

  .sidebar.collapsed .nav-item.active::before {
    left: -8px;
  }

  .sidebar.collapsed .nav-group-label {
    justify-content: center;
    padding: 8px 0;
  }

  .sidebar.collapsed .nav-group-label span {
    max-width: 36px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
