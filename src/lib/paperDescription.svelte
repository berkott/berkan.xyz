<script lang="ts">
  export let title: string;
  export let link: string;
  export let authors: string;
  export let year: string;
  export let venue: string = "";
  export let rawBibtex: string;
  
  let isDescriptionExpanded = false;
  let isBibtexExpanded = false;
  
  import ChevronDown from '$lib/chevronDown.svelte';
  import { Math } from "svelte-math";
  
  function toggleDescription() {
    isDescriptionExpanded = !isDescriptionExpanded;
  }
  
  function toggleBibtex() {
    isBibtexExpanded = !isBibtexExpanded;
  }
</script>

<div class="paper mb-4">
  <a href={link} class="font-bold">{title}</a>
  <p class="mt-0 mb-0 ml-4">{authors}.</p>
  <p class="mt-0 mb-0 ml-4">{venue}, {year}.</p>
  
  <button
    on:click={toggleBibtex}
    class="text-olive underline bg-transparent border-none p-0 m-0 ml-4 cursor-pointer flex items-center"
    type="button"
  >
    <span>{isBibtexExpanded ? 'Hide' : 'BibTeX'}</span>
    <ChevronDown rotated={isBibtexExpanded} />
  </button>
  
  {#if isBibtexExpanded}
    <div class="mt-2 pl-2 pr-2 border-l-2 border-olive ml-4">
      <pre class="mt-2 bg-gray-100 p-2 rounded text-xs overflow-x-auto">{rawBibtex}</pre>
    </div>
  {/if}
  
  <button
    on:click={toggleDescription}
    class="text-olive underline bg-transparent border-none p-0 m-0 ml-4 cursor-pointer flex items-center"
    type="button"
  >
    <span>{isDescriptionExpanded ? 'Hide' : 'Description and my thoughts'}</span>
    <ChevronDown rotated={isDescriptionExpanded} />
  </button>
  
  {#if isDescriptionExpanded}
  <!-- TODO: Make this a blockquote!!!! -->
    <div class="mt-2 pl-2 pr-2 border-l-2 border-olive ml-4 bg-light-olive">
      <slot></slot>
    </div>
  {/if}
</div>