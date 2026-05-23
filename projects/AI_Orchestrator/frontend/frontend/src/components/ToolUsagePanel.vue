<template>
  <section class="bg-surface-container border border-outline-variant rounded-xl p-3" style="min-width: 0">
    <div class="flex items-center justify-between mb-3">
      <h3
        class="text-on-surface flex items-center gap-2"
        style="font-family: 'Geist'; font-size: 15px; font-weight: 500; line-height: 1.4"
      >
        <span class="material-symbols-outlined text-primary" style="font-size: 20px">build</span>
        Tool Usage
      </h3>
      <span
        v-if="toolUsed"
        class="px-2 py-1 bg-primary-container text-on-primary-container rounded"
        style="font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 600"
      >
        {{ toolUsed }}
      </span>
      <span
        v-else
        class="text-on-surface-variant"
        style="font-family: 'JetBrains Mono'; font-size: 11px"
      >
        no tool
      </span>
    </div>

    <div v-if="!toolUsed" class="text-on-surface-variant text-sm">
      No tool was required for this workflow run.
    </div>

    <div v-else class="space-y-2">
      <div class="row q-col-gutter-sm">
        <div class="col-6">
          <div class="p-2 bg-surface rounded border border-outline-variant full-height">
            <p class="text-on-surface-variant" style="font-family: 'JetBrains Mono'; font-size: 10px">Called by</p>
            <p class="text-on-surface" style="font-family: 'Geist'; font-size: 13px; font-weight: 500">{{ calledBy }}</p>
          </div>
        </div>
        <div class="col-6">
          <div class="p-2 bg-surface rounded border border-outline-variant full-height">
            <p class="text-on-surface-variant" style="font-family: 'JetBrains Mono'; font-size: 10px">Status</p>
            <p
              :class="success ? 'text-primary' : 'text-error'"
              style="font-family: 'Geist'; font-size: 13px; font-weight: 500"
            >
              {{ success ? 'Success' : 'Failed' }}
            </p>
          </div>
        </div>
      </div>

      <div class="p-2 bg-background rounded border border-outline-variant">
        <p class="text-on-surface-variant mb-1" style="font-family: 'JetBrains Mono'; font-size: 10px">Result summary</p>
        <p class="text-on-surface" style="font-family: 'Geist'; font-size: 13px; line-height: 1.4">
          {{ summary }}
        </p>
      </div>

      <details class="bg-background rounded border border-outline-variant">
        <summary
          class="px-2 py-1 cursor-pointer text-on-surface-variant"
          style="font-family: 'JetBrains Mono'; font-size: 11px"
        >
          Raw tool_result
        </summary>
        <pre
          class="px-2 py-1 text-on-surface-variant overflow-x-auto"
          style="font-family: 'JetBrains Mono'; font-size: 11px; line-height: 1.4"
        >{{ rawResult }}</pre>
      </details>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useOrchestrationStore } from 'stores/orchestration'

const store = useOrchestrationStore()

const toolUsed = computed(() => store.toolUsed)
const toolResult = computed(() => store.toolResult || {})
const success = computed(() => toolResult.value?.success === true)

const calledBy = computed(() => {
  const step = (store.executionSteps || []).find((s) => s.tool === toolUsed.value)
  return step?.agent || 'tool_runner'
})

const summary = computed(() => {
  const r = toolResult.value
  if (!r || !toolUsed.value) return ''
  if (r.success === false) return `Error: ${r.error || 'unknown'}`
  if (toolUsed.value === 'calculator') return `${r.a} ${r.op} ${r.b} = ${r.result}`
  if (toolUsed.value === 'web_search') {
    const hits = r.results || []
    return `Found ${hits.length} result(s) for "${r.query}". First: ${hits[0]?.title || 'n/a'}`
  }
  if (toolUsed.value === 'database') return `Fetched ${r.resource} (count: ${r.count ?? 'n/a'})`
  return 'Tool completed.'
})

const rawResult = computed(() => JSON.stringify(toolResult.value, null, 2))
</script>
