<template>
  <div class="flex min-h-screen bg-background text-on-surface">
    <!-- Sidebar -->
    <SideNavBar />

    <!-- Main Content — no inner scroll container; document body scrolls naturally -->
    <main class="flex-1">
      <div class="p-3 max-w-[1600px] mx-auto w-full">
        <router-view />
      </div>
    </main>
  </div>
</template>

<script setup>
import SideNavBar from 'components/SideNavBar.vue'
</script>