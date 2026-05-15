<template>
  <section class="bg-surface-container border border-outline-variant rounded-xl ">
    <div class="flex gap-4">
      <div class="flex-1 relative">
        <input
          v-model="query"
          class="w-full bg-background border border-outline-variant rounded-lg px-4 py-2 text-on-surface focus:ring-2 focus:outline-none transition-all placeholder:text-on-surface-variant"
          style="font-family: 'Geist'; font-size: 14px; line-height: 1.5"
          placeholder="Enter orchestration prompt..."
          type="text"
        />
        <div class="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
          <span class="px-2 py-1 bg-surface-container-high rounded text-on-surface-variant border border-outline-variant" style="font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 500">
            ⌘ K
          </span>
        </div>
      </div>
      <button
        @click="executeQuery"
        class="px-3 bg-primary text-on-primary font-bold rounded-lg flex items-center gap-2 hover:brightness-110 active:scale-95 transition-all shadow-lg"
        style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 600"
      >
        <span class="material-symbols-outlined" style="font-size: 15px">bolt</span>
        <span>Execute</span>
      </button>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'

const orchestrationStore = useOrchestrationStore()
const query = computed({
  get: () => orchestrationStore.query,
  set: (val) => orchestrationStore.updateQuery(val)
})

const executeQuery = () => {
  orchestrationStore.startOrchestration()
}
</script>

<style scoped>
input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(78, 222, 163, 0.2);
}
</style>
