<template>
  <section class="bg-surface-container-low border border-outline-variant rounded-xl p-3">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-on-surface flex items-center gap-2" style="font-family: 'Geist'; font-size: 15px; font-weight: 500; line-height: 1.4">
        <span class="material-symbols-outlined text-primary" style="font-size: 20px">account_tree</span>
        Active Workflow Stream
      </h2>
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 bg-primary rounded-full" :class="{ 'animate-pulse': isOrchestrating }"></span>
        <span class="text-primary" style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 500">
          {{ statusLabel }}
        </span>
      </div>
    </div>

    <div class="relative flex items-center px-2 py-4">
      <!-- Connecting track (gray) — sits between first and last circle centers.
           top offset = parent_center - (label_h + gap)/2 so the line passes through circle centers,
           not the labels below them. -->
      <div
        class="absolute h-[2px] bg-outline-variant z-0"
        style="left: 26px; right: 26px; top: calc(50% - 17px); transform: translateY(-50%);"
      >
        <!-- Progress fill (primary) -->
        <div
          class="h-full bg-primary transition-all duration-700"
          :style="{ width: progress + '%' }"
        ></div>
      </div>

      <!-- Steps distributed evenly -->
      <div class="flex justify-between w-full relative z-10">
        <WorkflowStep v-for="step in steps" :key="step.id" :step="step" />
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'
import WorkflowStep from './WorkflowStep.vue'

const store = useOrchestrationStore()
const steps = computed(() => store.steps)
const isOrchestrating = computed(() => store.isOrchestrating)

const statusLabel = computed(() => {
  if (isOrchestrating.value) return 'Orchestrating...'
  if (store.status === 'completed') return 'Completed'
  if (store.status === 'failed') return 'Failed'
  return 'Ready'
})

const progress = computed(() => {
  const list = steps.value
  if (!list || list.length < 2) return 0
  let lastIdx = -1
  list.forEach((s, i) => {
    if (['completed', 'skipped', 'active'].includes(s.status)) lastIdx = i
  })
  if (lastIdx < 0) return 0
  return Math.round((lastIdx / (list.length - 1)) * 100)
})
</script>

<style scoped>
.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
