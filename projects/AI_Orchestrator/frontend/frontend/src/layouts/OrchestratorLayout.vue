<template>
  <div class="flex h-screen bg-background text-on-surface">
    <!-- Sidebar -->
    <SideNavBar />

    <!-- Main Content -->
    <main class="flex-1 overflow-y-auto">
      <div class="p-3 max-w-[1600px] mx-auto w-full">
        <router-view />
      </div>
    </main>
  </div>
</template>

<script setup>
import SideNavBar from 'components/SideNavBar.vue'
</script>