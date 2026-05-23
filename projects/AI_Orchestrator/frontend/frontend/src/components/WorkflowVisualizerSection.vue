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

    <!-- Scroll wrapper: enables internal horizontal scroll on small viewports
         so the step strip stays readable instead of compressing. -->
    <div class="workflow-scroll-wrapper">
      <div class="workflow-track relative flex items-center px-2 py-4">
        <!-- Connecting track (gray) — sits between first and last circle centers. -->
        <div
          class="absolute h-[2px] bg-outline-variant z-0"
          style="left: 26px; right: 26px; top: calc(50% - 17px); transform: translateY(-50%);"
        >
          <!-- Progress fill (primary) -->
          <div
            class="full-height bg-primary transition-all duration-700"
            :style="{ width: progress + '%' }"
          ></div>
        </div>

        <!-- Steps distributed evenly -->
        <div class="flex justify-between full-width relative z-10">
          <WorkflowStep v-for="step in steps" :key="step.id" :step="step" />
        </div>
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

.workflow-scroll-wrapper {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

/* Mobile (<md): give the step strip a min-width so circles + labels stay readable
   and the wrapper scrolls instead. Desktop (≥md): natural width, no scroll. */
@media (max-width: 1023px) {
  .workflow-track {
    min-width: 560px;
  }
}
</style>
