<script lang="ts">
  export let rotated = false;
  export let className = "";
</script>

<svg
  class="w-4 h-4 ml-1 transition-transform {rotated ? 'rotate-180' : ''} {className}"
  fill="none"
  stroke="currentColor"
  viewBox="0 0 24 24"
  xmlns="http://www.w3.org/2000/svg"
>
  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
</svg>
