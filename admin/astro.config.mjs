// @ts-check
import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  integrations: [svelte()],
  vite: {
    plugins: [tailwindcss()],
  },
  outDir: '../api/admin_static',
  base: '/admin',
  build: {
    format: 'file',
  },
  server: {
    port: 4321,
  },
});
