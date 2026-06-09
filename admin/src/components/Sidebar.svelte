<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { VIEW_GROUPS, type ViewId } from '../lib/config';

  export let activeView: ViewId = 'providers';
  export let collapsed = false;

  const dispatch = createEventDispatcher();

  let collapsedGroups: Record<string, boolean> = {};

  function toggleGroup(key: string) {
    collapsedGroups[key] = !collapsedGroups[key];
  }

  function isGroupCollapsed(key: string): boolean {
    return !!collapsedGroups[key];
  }

  function handleNavClick(viewId: ViewId) {
    activeView = viewId;
    dispatch('viewChange', { viewId });
  }
</script>

<aside class="sidebar flex flex-col border-r border-border-default bg-bg-surface shrink-0 overflow-hidden {collapsed ? 'w-16' : 'w-60'}" class:collapsed>
  <!-- Logo -->
  <div class="flex items-center gap-2.5 px-4 py-5 border-b border-border-default bg-bg-elevated {collapsed ? 'justify-center px-2' : ''}">
    <svg class="w-7 h-7 shrink-0 text-accent" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/>
      <line x1="12" y1="22" x2="12" y2="15.5"/>
      <polyline points="22 8.5 12 15.5 2 8.5"/>
      <polyline points="2 15.5 12 8.5 22 15.5"/>
      <line x1="12" y1="2" x2="12" y2="8.5"/>
    </svg>
    {#if !collapsed}
      <span class="text-sm font-semibold text-text-primary tracking-wide whitespace-nowrap">Free Claude Code</span>
    {/if}
  </div>

  <!-- Collapse toggle -->
  <button
    class="flex items-center justify-center w-7 h-7 rounded-md text-text-muted hover:text-text-primary hover:bg-bg-hover transition-all duration-150 cursor-pointer border-none {collapsed ? 'mx-auto my-2' : 'absolute top-[18px] right-3'}"
    on:click={() => collapsed = !collapsed}
    title="Toggle sidebar"
  >
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points={collapsed ? "9 18 15 12 9 6" : "15 18 9 12 15 6"}/>
    </svg>
  </button>

  <!-- Navigation -->
  <nav class="flex-1 flex flex-col gap-1.5 overflow-y-auto min-h-0 pt-3 px-3 scrollbar-none">
    <div class="flex flex-col gap-0.5">
      <div
        class="flex items-center justify-between px-3 py-1.5 text-2xs font-semibold text-text-muted uppercase tracking-wider cursor-pointer select-none rounded-md hover:text-text-secondary transition-colors duration-150"
        on:click={() => toggleGroup('management')}
      >
        <span>Management</span>
        <svg class="w-3 h-3 transition-transform duration-150 shrink-0 {isGroupCollapsed('management') ? '-rotate-90' : ''}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </div>
      {#if !isGroupCollapsed('management')}
        <div class="flex flex-col gap-0.5">
          {#each VIEW_GROUPS as view}
            <button
              class="flex items-center gap-2.5 px-3 py-2.5 text-sm rounded-md transition-all duration-150 cursor-pointer w-full text-left border-none {activeView === view.id ? 'bg-accent-subtle text-accent font-medium' : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'}"
              on:click={() => handleNavClick(view.id)}
            >
              {#if view.id === 'providers'}
                <svg class="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                  <circle cx="12" cy="12" r="3"/><path d="M12 1v4"/><path d="M12 19v4"/><path d="M1 12h4"/><path d="M19 12h4"/><path d="M4.22 4.22l2.83 2.83"/><path d="M16.95 16.95l2.83 2.83"/><path d="M4.22 19.78l2.83-2.83"/><path d="M16.95 7.05l2.83-2.83"/>
                </svg>
              {:else if view.id === 'model_config'}
                <svg class="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                  <rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/>
                </svg>
              {:else if view.id === 'messaging'}
                <svg class="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
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

  <!-- Footer -->
  <div class="pt-2 border-t border-border-default px-3">
    <div class="flex items-center justify-between py-2 px-3">
      <div class="flex items-center gap-2 text-xs">
        <span class="w-2 h-2 rounded-full bg-success shadow-[0_0_6px_rgba(34,197,94,0.5)]"></span>
        {#if !collapsed}
          <span class="text-text-secondary">Server Online</span>
        {/if}
      </div>
    </div>
  </div>
</aside>
