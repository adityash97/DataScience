<template>
  <div class="bg-surface-container border border-outline-variant rounded-xl flex flex-col shadow-xl min-h-[100px] h-full">
    <div class="p-3 border-b border-outline-variant flex items-center justify-between">
      <h3 class="text-on-surface flex items-center gap-2" style="font-family: 'Geist'; font-size: 15px; font-weight: 500; line-height: 1.4">
        <span class="material-symbols-outlined text-primary-container" style="font-size: 20px">article</span>
        Final Response
      </h3>
      <button class="text-on-surface-variant hover:text-on-surface transition-colors duration-200">
        <span class="material-symbols-outlined" style="font-size: 20px">open_in_full</span>
      </button>
    </div>

    <!-- Content -->
    <div class="flex-1 p-3 overflow-y-auto">
      <div class="bg-background rounded-lg border border-outline-variant p-3 space-y-3">
        <!-- Title with accent bar -->
        <div class="flex items-center gap-2 mb-2">
          <span class="w-1 h-4 bg-primary rounded-full"></span>
          <h4 class="text-on-surface font-bold" style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700">{{ response.title }}</h4>
        </div>

        <!-- Summary -->
        <p class="text-on-surface-variant leading-relaxed" style="font-family: 'Geist'; font-size: 14px; line-height: 1.5">
          {{ response.content }}
        </p>

        <!-- Sentiment and Confidence Grid -->
        <div class="grid grid-cols-2 gap-3">
          <div class="p-2 bg-surface rounded border border-outline-variant">
            <p class="text-on-surface-variant" style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500">Sentiment</p>
            <p class="text-primary" style="font-family: 'Geist'; font-size: 14px; font-weight: 500; line-height: 1.4">{{ response.sentiment }}</p>
          </div>
          <div class="p-2 bg-surface rounded border border-outline-variant">
            <p class="text-on-surface-variant" style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500">Confidence</p>
            <p class="text-primary" style="font-family: 'Geist'; font-size: 14px; font-weight: 500; line-height: 1.4">{{ response.confidence }}</p>
          </div>
        </div>

        <!-- Risks -->
        <p class="text-on-surface-variant leading-relaxed" style="font-family: 'Geist'; font-size: 14px; line-height: 1.5">
          {{ response.risks }}
        </p>

        <!-- Action Buttons -->
        <div class="pt-2 border-t border-outline-variant flex justify-end gap-2">
          <button
            class="px-3 py-1.5 rounded bg-surface border border-outline-variant text-on-surface-variant hover:bg-surface-container transition-colors duration-200"
            style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 500"
          >
            Export PDF
          </button>
          <button
            class="px-3 py-1.5 rounded bg-primary text-on-primary font-bold hover:opacity-90 transition-opacity duration-200"
            style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700"
          >
            Copy to Clipboard
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'

const orchestrationStore = useOrchestrationStore()
const response = computed(() => orchestrationStore.finalResponse)
</script>

<style scoped>
button:hover {
  opacity: 0.9;
}
</style>
