<template>
  <div
    class="bg-surface-container border border-outline-variant rounded-xl column shadow-xl full-width min-h-[280px]"
    style="min-width: 0"
  >
    <div class="p-3 border-b border-outline-variant flex items-center justify-between">
      <h3
        class="text-on-surface flex items-center gap-2"
        style="font-family: 'Geist'; font-size: 15px; font-weight: 500; line-height: 1.4"
      >
        <span class="material-symbols-outlined text-primary-container" style="font-size: 20px">article</span>
        Final Response
      </h3>
      <span
        class="text-on-surface-variant"
        style="font-family: 'JetBrains Mono'; font-size: 11px"
      >
        {{ response.sessionId }}
      </span>
    </div>

    <div class="col-grow p-3">
      <div class="bg-background rounded-lg border border-outline-variant p-3 space-y-3">
        <div class="flex items-center gap-2 mb-2">
          <span class="w-1 h-4 bg-primary rounded-full"></span>
          <h4
            class="text-on-surface font-bold"
            style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700"
          >
            {{ response.title }}
          </h4>
        </div>

        <p
          class="text-on-surface-variant leading-relaxed"
          style="font-family: 'Geist'; font-size: 14px; line-height: 1.5"
        >
          {{ response.content }}
        </p>

        <div class="row q-col-gutter-sm">
          <div class="col-6">
            <div class="p-2 bg-surface rounded border border-outline-variant full-height">
              <p
                class="text-on-surface-variant"
                style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500"
              >
                Agents executed
              </p>
              <p
                class="text-primary"
                style="font-family: 'Geist'; font-size: 13px; font-weight: 500; line-height: 1.4"
              >
                {{ response.agents.length ? response.agents.join(' → ') : '—' }}
              </p>
            </div>
          </div>
          <div class="col-6">
            <div class="p-2 bg-surface rounded border border-outline-variant full-height">
              <p
                class="text-on-surface-variant"
                style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500"
              >
                Retries
              </p>
              <p
                class="text-primary"
                style="font-family: 'Geist'; font-size: 14px; font-weight: 500; line-height: 1.4"
              >
                {{ response.retryCount }}
              </p>
            </div>
          </div>
        </div>

        <div
          v-if="response.tool"
          class="p-2 bg-surface rounded border border-outline-variant"
        >
          <p
            class="text-on-surface-variant"
            style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500"
          >
            Tool used
          </p>
          <p
            class="text-on-surface"
            style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 600"
          >
            {{ response.tool }}
          </p>
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
