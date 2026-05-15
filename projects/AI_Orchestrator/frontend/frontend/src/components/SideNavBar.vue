<template>
  <aside
    class="flex flex-col h-screen w-64 min-w-64 flex-shrink-0 overflow-y-auto bg-surface-container-low border-r border-outline-variant"
  >
    <!-- Brand Section -->
    <div class="p-4">
      <div class="flex items-center gap-2 text-primary font-black">
        <span
          class="material-symbols-outlined"
          style="font-size: 24px; font-variation-settings: 'FILL' 1"
        >
          hub
        </span>

        <span
          style="
            font-size: 18px;
            line-height: 1.4;
            font-weight: 600;
          "
        >
          AI Orchestrator
        </span>
      </div>

      <!-- Profile -->
      <div
        class="mt-4 flex items-center gap-3 p-2 rounded-lg bg-surface-container transition-all duration-200 cursor-pointer hover:opacity-90"
      >
        <img
          alt="Orchestrator Admin"
          src="https://lh3.googleusercontent.com/aida-public/AB6AXuDZSSmNKSzy6N9q_a6DVsjOsm_DBCFXvpm1Bdf9_qcGJsZc5hsvt4Bywi0fJ11GPzn0efOsjMMSVp-A-iPhWkEMUEsJitKbWON6p4_ApbpdSUOJRU3K-ZAO5ukUogz45AwknyNfiFL4r4NO68yoSOS4Axk7Xqw9U0QAb_mjYBElLjySo-gov5xvO0HqHH2PPqLlV6C01tabmg1SGEYxMpnEvGX9CcnezsQ0o_luijOK8vu4eMpvuaNAbcKN-SvwxX-vIW2o13Hu2YI"
          style="
            width: 32px;
            height: 32px;
            min-width: 32px;
            min-height: 32px;
            border-radius: 9999px;
            object-fit: cover;
            border: 1px solid var(--color-outline-variant);
            display: block;
          "
        />

        <div class="overflow-hidden flex-1">
          <p
            class="truncate"
            style="
              font-size: 13px;
              line-height: 1.4;
              font-weight: 500;
              font-family: 'JetBrains Mono';
              color: var(--color-on-surface);
            "
          >
            Admin
          </p>

          <p
            style="
              font-size: 11px;
              line-height: 1.2;
              font-weight: 500;
              font-family: 'JetBrains Mono';
              color: var(--color-primary);
              opacity: 0.8;
            "
          >
            Active Session
          </p>
        </div>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 px-4 space-y-1">
      <router-link
        to="/dashboard"
        class="flex items-center gap-3 px-3 py-2 rounded-lg text-primary bg-surface-container-high border-r-4 border-primary transition-all duration-200 active-nav"
        style="
          font-size: 13px;
          font-weight: 500;
          font-family: 'JetBrains Mono';
        "
      >
        <span
          class="material-symbols-outlined"
          style="font-size: 20px"
        >
          dashboard
        </span>

        <span>Dashboard</span>
      </router-link>
    </nav>

    <!-- Bottom Section -->
    <div class="mt-auto">
      <!-- CTA -->
      <div class="px-4 mb-4">
        <button
          class="w-full py-3 bg-primary text-on-primary font-bold rounded-lg flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 transition-all"
          style="
            font-size: 13px;
            font-weight: 700;
            font-family: 'JetBrains Mono';
          "
        >
          <span
            class="material-symbols-outlined"
            style="font-size: 20px"
          >
            add
          </span>

          <span>New Workflow</span>
        </button>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-outline-variant space-y-1">
        <a
          href="#"
          class="flex items-center gap-3 px-4 py-2 rounded-lg text-on-surface-variant hover:bg-surface-container transition-all duration-200"
          style="
            font-size: 13px;
            font-weight: 500;
            font-family: 'JetBrains Mono';
          "
        >
          <span
            class="material-symbols-outlined"
            style="font-size: 20px"
          >
            description
          </span>

          <span>Docs</span>
        </a>

        <a
          href="#"
          class="flex items-center gap-3 px-4 py-2 rounded-lg text-on-surface-variant hover:bg-surface-container transition-all duration-200"
          style="
            font-size: 13px;
            font-weight: 500;
            font-family: 'JetBrains Mono';
          "
        >
          <span
            class="material-symbols-outlined"
            style="font-size: 20px"
          >
            logout
          </span>

          <span>Logout</span>
        </a>
      </div>
    </div>
  </aside>
</template>

<script setup>
// Presentational sidebar component
</script>

<style scoped>
.active-nav {
  color: var(--color-primary);
  background-color: var(--color-surface-container-high);
  border-right-color: var(--color-primary);
}

:deep(.router-link-active) {
  color: var(--color-primary);
  background-color: var(--color-surface-container-high);
  border-right-color: var(--color-primary);
}
</style>