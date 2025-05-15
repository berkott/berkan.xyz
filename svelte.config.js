import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

import remarkMath from 'remark-math';
import rehypeKatexSvelte from 'rehype-katex-svelte';

const config = {
	preprocess: [
		vitePreprocess(),
		mdsvex({
			remarkPlugins: [remarkMath],
			rehypePlugins: [[rehypeKatexSvelte, { output: 'mathml' }]],
			extensions: ['.md']
		})
	],
	kit: { adapter: adapter() },
	extensions: ['.svelte', '.md']
};

export default config;