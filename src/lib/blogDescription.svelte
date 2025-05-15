<script lang="ts">
  export let title: string;
  export let link: string;
  export let date: string;
  export let excerpt: string;
</script>

<div class="blog mb-6">
  <div>
    <h2 class="mt-0">
      <a href={`/blog/${link}`} class="hover:underline">{title}</a>
    </h2>
    <p class="mb-0"><strong>Published:</strong> {date}</p>
  </div>
  <p class="mt-0">{excerpt}</p>
</div>
