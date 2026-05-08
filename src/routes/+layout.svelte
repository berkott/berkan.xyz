<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<div class="max-w-[100ch] mx-auto px-4 mb-10">
	<nav class="flex flex-col md:flex-row md:items-center md:justify-between py-4 border-b border-olive mb-3">
		<a href="/" class="text-olive font-bold text-2xl md:text-2xl no-underline hover:underline mb-4 md:mb-0">Berkan</a>
		<div class="flex flex-col md:flex-row md:space-x-6 space-y-4 md:space-y-0">
			<a href="/blog" class="text-olive no-underline hover:underline">Blog</a>
			<a href="/projects" class="text-olive no-underline hover:underline">Projects</a>
			<a href="/publications" class="text-olive no-underline hover:underline">Publications</a>
			<a href="/digitalGarden" class="text-olive no-underline hover:underline">Digital Garden</a>
			<!-- <a href="/resources" class="text-olive no-underline hover:underline">Resources</a> -->
		</div>
	</nav>
	<div>
		{@render children()}
	</div>
</div>