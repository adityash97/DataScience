<template>
  <section class="bg-surface-container-low border border-outline-variant rounded-xl p-3">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-on-surface flex items-center gap-2" style="font-family: 'Geist'; font-size: 15px; font-weight: 500; line-height: 1.4">
        <span class="material-symbols-outlined text-primary" style="font-size: 20px">account_tree</span>
        Active Workflow Stream
      </h2>
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 bg-primary rounded-full animate-pulse"></span>
        <span class="text-primary" style="font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 500">{{ isOrchestrating ? 'Orchestrating...' : 'Ready' }}</span>
      </div>
    </div>

    <div class="relative flex justify-between items-center px-4">
      <!-- Connecting Lines -->
      <div class="absolute top-1/2 left-0 w-full h-[2px] bg-outline-variant -translate-y-1/2 z-0">
        <div class="h-full bg-primary transition-all duration-1000" :style="{ width: progress + '%' }"></div>
      </div>

      <!-- Workflow Steps -->
      <WorkflowStep v-for="step in steps" :key="step.id" :step="step" />
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'
import WorkflowStep from './WorkflowStep.vue'

const orchestrationStore = useOrchestrationStore()
const steps = computed(() => orchestrationStore.steps)
const isOrchestrating = computed(() => orchestrationStore.isOrchestrating)
const progress = computed(() => orchestrationStore.workflowProgress)
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
