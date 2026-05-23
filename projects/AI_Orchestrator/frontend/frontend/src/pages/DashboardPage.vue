<template>
  <div class="full-width q-pb-md q-gutter-y-md">
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

    <!-- Section 4: Bottom Two-Column Content — Quasar responsive grid.
         Desktop (≥md): 7/5 side-by-side. Mobile/tablet (<md): stacked full-width. -->
    <div class="row q-col-gutter-md items-stretch">
      <div class="col-12 col-md-7" style="display: flex; min-width: 0">
        <ExecutionLogsPanel />
      </div>
      <div class="col-12 col-md-5" style="display: flex; min-width: 0">
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
