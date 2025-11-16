import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/api': {
				target: 'http://localhost:8020',
				changeOrigin: true,
				timeout: 0,
				proxyTimeout: 0
			}
		}
	}
});
