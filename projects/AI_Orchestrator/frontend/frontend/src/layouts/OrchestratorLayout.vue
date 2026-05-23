<template>
  <q-layout view="hHh Lpr fFf" class="bg-background text-on-surface">
    <q-header
      v-if="$q.screen.lt.md"
      elevated
      class="bg-surface-container-low text-on-surface"
    >
      <q-toolbar>
        <q-btn
          flat
          dense
          round
          icon="menu"
          aria-label="Menu"
          @click="leftDrawerOpen = !leftDrawerOpen"
        />
        <q-toolbar-title
          class="text-primary"
          style="font-family: 'JetBrains Mono'; font-size: 14px; font-weight: 600"
        >
          AI Orchestrator
        </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-drawer
      v-model="leftDrawerOpen"
      show-if-above
      bordered
      :width="256"
      :breakpoint="1023"
      class="bg-surface-container-low"
    >
      <SideNavBar />
    </q-drawer>

    <q-page-container>
      <q-page :class="$q.screen.lt.md ? 'q-pa-sm' : 'q-pa-md'">
        <div style="max-width: 1600px; margin: 0 auto; width: 100%">
          <router-view />
        </div>
      </q-page>
    </q-page-container>
  </q-layout>
</template>

<script setup>
import { ref } from 'vue'
import SideNavBar from 'components/SideNavBar.vue'

const leftDrawerOpen = ref(false)
</script>
