<template>
  <div class="relative z-10 flex flex-col items-center gap-2">
    <div
      :class="[
        'rounded-full flex items-center justify-center border-2 transition-all duration-200',
        statusClasses
      ]"
      :style="sizeStyle"
    >
      <span class="material-symbols-outlined" :style="iconStyle">
        {{ step.icon }}
      </span>
    </div>

    <div class="text-center">
      <p
        :class="[
          step.status === 'completed'
            ? 'text-on-surface font-bold'
            : 'text-on-surface-variant'
        ]"
        :style="
          step.status === 'active'
            ? {
                fontFamily: 'JetBrains Mono',
                fontSize: '12px',
                fontWeight: 600,
                color: 'var(--color-primary)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }
            : {
                fontFamily: 'JetBrains Mono',
                fontSize: '12px',
                fontWeight: 500
              }
        "
      >
        {{ step.name }}
      </p>

      <p
        :class="[
          step.status === 'completed'
            ? 'text-on-surface-variant'
            : 'text-on-surface-variant/50'
        ]"
        :style="{
          fontFamily: 'JetBrains Mono',
          fontSize: '10px',
          fontWeight: 500,
          lineHeight: 1.2
        }"
      >
        {{ step.subtitle }}
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  step: {
    type: Object,
    required: true
  }
})

const sizeStyle = computed(() => {
  if (props.step.status === 'active') {
    return {
      width: '44px',
      height: '44px'
    }
  }

  return {
    width: '36px',
    height: '36px'
  }
})

const iconStyle = computed(() => {
  return {
    fontSize: props.step.status === 'active' ? '22px' : '18px'
  }
})

const statusClasses = computed(() => {
  const step = props.step

  if (step.status === 'completed') {
    return 'bg-primary-container text-on-primary-container border-primary shadow-lg'
  }

  if (step.status === 'active') {
    return 'bg-surface-container-highest text-primary border-primary animate-pulse'
  }

  return 'bg-surface-container text-on-surface-variant border-outline-variant'
})
</script>

<style scoped>
/* Glow effect for active step */
.animate-pulse {
  animation: pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  box-shadow: 0 0 20px rgba(78, 222, 163, 0.3);
}

@keyframes pulse-glow {
  0%,
  100% {
    opacity: 1;
  }

  50% {
    opacity: 0.7;
  }
}
</style>