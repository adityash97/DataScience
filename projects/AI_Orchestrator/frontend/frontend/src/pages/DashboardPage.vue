<template>
  <div class="space-y-4 w-full pb-4">
    <!-- Section 1: Query Input -->
    <QueryInputSection />

    <!-- Error banner -->
    <div
      v-if="errorMessage"
      class="bg-error/10 border border-error/40 rounded-lg px-3 py-2 text-error"
      style="font-family: 'JetBrains Mono'; font-size: 12px"
    >
      {{ errorMessage }}
    </div>

    <!-- Section 2: Workflow Visualizer -->
    <WorkflowVisualizerSection />

    <!-- Section 3: Tool Usage -->
    <ToolUsagePanel />

    <!-- Section 4: Bottom Two-Column Content — grid stretches both cells to same height -->
    <div class="grid grid-cols-10 gap-4">
      <div class="col-span-6 flex">
        <ExecutionLogsPanel />
      </div>
      <div class="col-span-4 flex">
        <FinalResponsePanel />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'

import QueryInputSection from 'components/QueryInputSection.vue'
import WorkflowVisualizerSection from 'components/WorkflowVisualizerSection.vue'
import ToolUsagePanel from 'components/ToolUsagePanel.vue'
import ExecutionLogsPanel from 'components/ExecutionLogsPanel.vue'
import FinalResponsePanel from 'components/FinalResponsePanel.vue'

const store = useOrchestrationStore()
const errorMessage = computed(() => store.errorMessage)

onMounted(() => {
  store.loadWorkflows()
})
</script>
