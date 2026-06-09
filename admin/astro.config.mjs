// @ts-check
import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';

// https://astro.build/config
export default defineConfig({
  integrations: [svelte()],
  outDir: '../api/admin_static',
  base: '/admin',
  build: {
    format: 'file',
  },
  server: {
    port: 4321,
  },
});
