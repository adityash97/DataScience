<template>
  <div class="space-y-4 w-full">
    <!-- Section 1: Query Input -->
    <QueryInputSection />

    <!-- Section 2: Workflow Visualizer -->
    <WorkflowVisualizerSection />

    <!-- Section 3: Bottom Two-Column Content -->
    <div class="grid grid-cols-10 gap-4">
      <!-- Left (60%): Execution Logs Panel -->
      <div class="col-span-6 h-full">
        <ExecutionLogsPanel />
      </div>

      <!-- Right (40%): Response Panel -->
      <div class="col-span-4 h-full">
        <FinalResponsePanel />
      </div>
    </div>
  </div>
</template>

<script setup>
import QueryInputSection from 'components/QueryInputSection.vue'
import WorkflowVisualizerSection from 'components/WorkflowVisualizerSection.vue'
import ExecutionLogsPanel from 'components/ExecutionLogsPanel.vue'
import FinalResponsePanel from 'components/FinalResponsePanel.vue'
</script>
