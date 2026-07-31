<script>
	// Wrapper exists only so this post's table styling can be scoped to it,
	// leaving every other page on the site untouched.
	let { children } = $props();
</script>

<div class="dashes-post">
	{@render children()}
</div>
