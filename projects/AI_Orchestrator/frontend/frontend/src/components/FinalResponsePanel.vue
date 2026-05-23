<template>
  <div
    class="bg-surface-container border border-outline-variant rounded-xl column shadow-xl full-width"
    style="min-width: 0"
  >
    <!-- Header -->
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

    <!-- Body -->
    <div class="col-grow p-3">
      <div class="bg-background rounded-lg border border-outline-variant p-3 space-y-3">

        <!-- Title -->
        <div class="flex items-center gap-2">
          <span class="w-1 shrink-0 h-4 bg-primary rounded-full"></span>
          <h4
            class="text-on-surface font-bold"
            style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700"
          >
            {{ response.title }}
          </h4>
        </div>

        <!-- Structured web search results -->
        <div v-if="response.toolResults.length" class="space-y-2">
          <p
            class="text-on-surface-variant"
            style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em"
          >
            Search Results
          </p>
          <div
            v-for="(result, i) in response.toolResults"
            :key="i"
            class="p-2 bg-surface rounded border border-outline-variant"
          >
            <p
              class="text-on-surface"
              style="font-family: 'Geist'; font-size: 13px; font-weight: 600; line-height: 1.4; margin-bottom: 4px"
            >
              {{ result.title || '—' }}
            </p>
            <p
              v-if="result.snippet"
              class="text-on-surface-variant"
              style="font-family: 'Geist'; font-size: 12px; line-height: 1.5; margin-bottom: 4px"
            >
              {{ result.snippet }}
            </p>
            <a
              v-if="result.url"
              :href="result.url"
              target="_blank"
              rel="noopener noreferrer"
              class="text-primary"
              style="font-family: 'JetBrains Mono'; font-size: 11px; overflow-wrap: break-word; display: block"
            >
              {{ result.url }}
            </a>
          </div>
        </div>

        <!-- Plain text response (non-tool or fallback) -->
        <p
          v-else
          class="text-on-surface-variant"
          style="font-family: 'Geist'; font-size: 14px; line-height: 1.6; white-space: pre-line"
        >
          {{ response.content }}
        </p>

        <!-- Meta row: agents + retries -->
        <div class="row q-col-gutter-sm">
          <div class="col-6">
            <div class="p-2 bg-surface rounded border border-outline-variant full-height">
              <p
                class="text-on-surface-variant"
                style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500; margin-bottom: 4px"
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
                style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500; margin-bottom: 4px"
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

        <!-- Tool used -->
        <div
          v-if="response.tool"
          class="p-2 bg-surface rounded border border-outline-variant"
        >
          <p
            class="text-on-surface-variant"
            style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 500; margin-bottom: 4px"
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
