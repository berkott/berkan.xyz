<script lang="ts">
  export let title: string;
  export let link: string;
  export let date: string;
  export let description: string;
  export let imageSrc: string;
</script>

<div class="project">
  <div class="flex items-center gap-4">
    <img src={imageSrc} alt="Project Picture" class="w-24 h-24 object-cover">
    <div>
      <h2 class="mt-0">
        <a href={link} class="hover:underline">{title}</a>
      </h2>
      <p class="mb-0"><strong>Created:</strong> {date}</p>
    </div>
  </div>
  <p class="mt-0">{@html description}</p>
</div>
