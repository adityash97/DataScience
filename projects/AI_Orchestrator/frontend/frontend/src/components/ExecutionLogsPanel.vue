<template>
  <div class="bg-[#0A0A0B] border border-outline-variant rounded-xl overflow-hidden column full-width min-h-[280px]" style="min-width: 0">
    <!-- Header -->
    <div class="bg-surface-container-high px-4 py-2 border-b border-outline-variant flex items-center justify-between">
      <div class="flex items-center gap-2">
        <div class="flex gap-1.5">
          <div class="w-3 h-3 rounded-full bg-error/30 border border-error/50"></div>
          <div class="w-3 h-3 rounded-full bg-tertiary-container/30 border border-tertiary-container/50"></div>
          <div class="w-3 h-3 rounded-full bg-primary/30 border border-primary/50"></div>
        </div>
        <span class="ml-4 text-on-surface-variant uppercase" style="font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 500; letter-spacing: 0.1em">
          Execution Logs
        </span>
      </div>
      <span class="text-primary" style="font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 500; opacity: 0.6">
        session_id: {{ sessionId }}
      </span>
    </div>

    <!-- Logs Content — fills panel; capped internal scroll only for very long log lists -->
    <div class="col-grow p-3 space-y-2 overflow-y-auto" style="font-family: 'JetBrains Mono'; font-size: 12px; line-height: 1.5; font-weight: 400; max-height: 500px; overflow-x: auto;">
      <div v-for="(log, index) in logs" :key="log.id" class="flex gap-4" style="min-width: max-content">
        <span class="text-on-surface-variant/30 select-none" style="width: 40px; flex-shrink: 0">{{ String(index + 1).padStart(3, '0') }}</span>
        <p v-if="log.level === 'CURRENT'" class="text-primary animate-pulse">{{ log.message }}</p>
        <p v-else :class="getLevelColor(log.level)">
          <span class="font-bold">[{{ log.level }}]</span>
          <span class="text-on-surface">{{ log.message }}</span>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'

const orchestrationStore = useOrchestrationStore()
const logs = computed(() => orchestrationStore.logs)
const sessionId = computed(() => orchestrationStore.finalResponse.sessionId)

const getLevelColor = (level) => {
  if (level === 'INFO' || level === 'PLANNER') {
    return 'text-primary-container'
  }
  if (level === 'RETRIEVER') {
    return 'text-secondary-container'
  }
  return 'text-on-surface'
}
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
