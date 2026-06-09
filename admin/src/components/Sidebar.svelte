<script lang="ts">
  import { VIEW_GROUPS, type ViewId } from '../lib/config';

  export let activeView: ViewId = 'providers';
  export let collapsed = false;
  export let onViewChange: ((viewId: ViewId) => void) | undefined = undefined;

  // Track accordion states with proper Svelte 3/4 reactivity
  let collapsedGroups: Record<string, boolean> = {};

  function toggleGroup(key: string) {
    collapsedGroups = {
      ...collapsedGroups,
      [key]: !collapsedGroups[key]
    };
  }

  function handleNavClick(viewId: ViewId) {
    activeView = viewId;
    onViewChange?.(viewId);
  }

  function handleGroupKeyDown(event: KeyboardEvent, key: string) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleGroup(key);
    }
  }
</script>

<aside 
  class="sidebar flex flex-col shrink-0 overflow-hidden transition-all duration-300 ease-in-out"
  class:collapsed
  aria-label="Main Navigation"
>
  <div class="sidebar-logo flex items-center transition-all duration-300">
    <svg class="logo-icon transition-transform duration-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/>
      <line x1="12" y1="22" x2="12" y2="15.5"/>
      <polyline points="22 8.5 12 15.5 2 8.5"/>
      <polyline points="2 15.5 12 8.5 22 15.5"/>
      <line x1="12" y1="2" x2="12" y2="8.5"/>
    </svg>
    <span class="logo-text transition-opacity duration-200">Free Claude Code</span>
  </div>

  <button
    type="button"
    class="collapse-btn"
    on:click={() => collapsed = !collapsed}
    aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
    title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
  >
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points={collapsed ? "9 18 15 12 9 6" : "15 18 9 12 15 6"}/>
    </svg>
  </button>

  <nav class="sidebar-nav flex-1 flex flex-col gap-0.5 overflow-y-auto min-h-0">
    <button 
      type="button"
      class="nav-group-label" 
      on:click={() => toggleGroup('management')}
      on:keydown={(e) => handleGroupKeyDown(e, 'management')}
      aria-expanded={!collapsedGroups['management']}
    >
      <span class="group-label-text">Management</span>
      <svg class="nav-arrow transition-transform duration-200" class:rotated={collapsedGroups['management']} width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
      <div class="nav-divider" aria-hidden="true"></div>
    </button>

    <div 
      class="nav-group-items flex flex-col gap-0.5 transition-all duration-300 overflow-hidden"
      class:collapsed-group={collapsedGroups['management'] && !collapsed}
    >
      {#each VIEW_GROUPS as view}
        <button
          type="button"
          class="nav-item"
          class:active={activeView === view.id}
          on:click={() => handleNavClick(view.id)}
          title={collapsed ? view.label : undefined}
          aria-current={activeView === view.id ? 'page' : undefined}
        >
          <div class="nav-icon-wrapper">
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
          </div>
          <span class="nav-text">{view.label}</span>
        </button>
      {/each}
    </div>
  </nav>

  <div class="sidebar-footer">
    <div class="sidebar-footer-inner">
      <div class="status-indicator">
        <span class="status-dot connected"></span>
        <span class="status-text">Connected</span>
      </div>
    </div>
  </div>
</aside>

<style>
  /* Base Layout Contexts */
  .sidebar {
    width: 240px;
    background: var(--color-bg-surface);
    border-right: 1px solid var(--color-border-subtle);
    position: relative;
    height: 100%;
  }

  .sidebar.collapsed {
    width: 64px;
  }

  /* Structural Components */
  .sidebar-logo {
    height: 56px;
    padding: 0 16px;
    gap: 12px;
    border-bottom: 1px solid var(--color-border-subtle);
    background: var(--color-bg-elevated);
  }

  .sidebar.collapsed .sidebar-logo {
    justify-content: center;
    padding: 0;
    gap: 0;
  }

  .logo-icon {
    width: 24px;
    height: 24px;
    flex-shrink: 0;
    color: var(--color-accent);
  }

  .logo-text {
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
    letter-spacing: -0.01em;
    white-space: nowrap;
    opacity: 1;
  }

  .sidebar.collapsed .logo-text {
    opacity: 0;
    width: 0;
    pointer-events: none;
    overflow: hidden;
  }

  /* Collapse Button Setup */
  .collapse-btn {
    position: absolute;
    top: 16px;
    right: 12px;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border-subtle);
    background: var(--color-bg-surface);
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .collapse-btn:hover {
    color: var(--color-text-primary);
    background: var(--color-bg-hover);
    transform: scale(1.05);
  }

  .collapse-btn:focus-visible {
    outline: 2px solid var(--color-accent);
    outline-offset: 2px;
  }

  .sidebar.collapsed .collapse-btn {
    position: relative;
    top: auto;
    right: auto;
    margin: 8px auto 4px auto;
  }

  /* Navigation Layout */
  .sidebar-nav {
    padding: 16px 12px;
    scrollbar-width: none;
  }
  .sidebar-nav::-webkit-scrollbar {
    display: none;
  }

  .sidebar.collapsed .sidebar-nav {
    padding: 8px;
  }

  /* Accordion Headers */
  .nav-group-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 10px 8px 6px;
    background: transparent;
    border: none;
    font-size: 11px;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    cursor: pointer;
    user-select: none;
    text-align: left;
  }

  .nav-group-label:focus-visible {
    outline: 2px solid var(--color-accent);
    border-radius: var(--radius-sm);
  }

  .group-label-text {
    opacity: 1;
    transition: opacity 0.2s ease;
    white-space: nowrap;
  }

  .nav-arrow {
    color: var(--color-text-muted);
    flex-shrink: 0;
    display: block;
  }
  .nav-arrow.rotated {
    transform: rotate(-90deg);
  }

  .nav-divider {
    display: none;
    width: 100%;
    height: 1px;
    background: var(--color-border-subtle);
  }

  /* Toggle Accordion Display and Collapse Smoothness */
  .sidebar.collapsed .group-label-text,
  .sidebar.collapsed .nav-arrow {
    display: none;
  }

  .sidebar.collapsed .nav-divider {
    display: block;
    margin: 8px 0;
  }

  .nav-group-items {
    max-height: 500px;
    opacity: 1;
    transition: max-height 0.25s ease-in-out, opacity 0.2s ease;
  }

  .nav-group-items.collapsed-group {
    max-height: 0;
    opacity: 0;
    pointer-events: none;
  }

  /* Individual Navigation Links */
  .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border: none;
    background: transparent;
    color: var(--color-text-secondary);
    font-size: 13px;
    font-weight: 400;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
    width: 100%;
    text-align: left;
    position: relative;
  }

  .nav-item:hover {
    background: var(--color-bg-hover);
    color: var(--color-text-primary);
  }

  .nav-item:focus-visible {
    outline: 2px solid var(--color-accent);
    outline-offset: -2px;
  }

  .nav-icon-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }

  .nav-text {
    opacity: 1;
    white-space: nowrap;
    transition: opacity 0.15s ease;
  }

  /* Collapsed Sidebar Navigation Item Variations */
  .sidebar.collapsed .nav-item {
    justify-content: center;
    padding: 10px;
  }

  .sidebar.collapsed .nav-text {
    opacity: 0;
    width: 0;
    overflow: hidden;
    pointer-events: none;
    position: absolute;
  }

  /* Active State Indicators */
  .nav-item.active {
    background: var(--color-accent-subtle);
    color: var(--color-accent);
    font-weight: 500;
  }

  .nav-item.active::before {
    content: '';
    position: absolute;
    left: 0;
    top: 25%;
    height: 50%;
    width: 3px;
    border-radius: 0 4px 4px 0;
    background: var(--color-accent);
  }

  /* Footer & Status Element Layout */
  .sidebar-footer {
    border-top: 1px solid var(--color-border-subtle);
    background: var(--color-bg-surface);
  }

  .sidebar-footer-inner {
    padding: 12px;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    height: 24px;
  }

  .sidebar.collapsed .status-indicator {
    justify-content: center;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--color-status-offline);
    flex-shrink: 0;
  }

  .status-dot.connected {
    background: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
  }

  .status-text {
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-muted);
    transition: opacity 0.15s ease;
  }

  .sidebar.collapsed .status-text {
    opacity: 0;
    width: 0;
    overflow: hidden;
    pointer-events: none;
    position: absolute;
  }
</style>