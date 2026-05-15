# PROJECT_CONTEXT.md

## Overview

**AI_Orchestrator** is a portfolio-quality AI workflow platform demonstrating multi-agent orchestration, LangGraph-driven workflows, AI routing, shared execution state, tool calling, and workflow monitoring.

**Current branch:** `agents/html-to-quasar-migration` — the original static HTML prototype has been migrated into a full Quasar/Vue 3 component architecture. The frontend dashboard UI is complete with mock data. The Django backend and LangGraph AI engine remain unimplemented.

---

## High-Level Architecture

Three loosely-coupled domains:

1. **Backend API** — Django 6 + (planned) Django REST Framework
2. **Frontend** — Quasar 2 + Vue 3 (Composition API, `<script setup>`) — **dashboard UI done**
3. **AI Workflow Engine** — (planned) LangGraph + LangChain for multi-agent orchestration

---

## Repository Layout

```
AI_Orchestrator/
├── AI_Orchestrator/              # Django project package
│   ├── settings.py               # Django 6, SQLite, DEBUG=True
│   ├── urls.py                   # Only /admin/ wired up
│   └── ...
├── frontend/
│   └── frontend/                 # Quasar app root
│       ├── index.html            # Google Fonts: Material Symbols, Geist, JetBrains Mono
│       ├── src/
│       │   ├── App.vue
│       │   ├── layouts/
│       │   │   └── OrchestratorLayout.vue   # App shell: sidebar + scrollable main
│       │   ├── pages/
│       │   │   └── DashboardPage.vue        # Root page, composes all sections
│       │   ├── components/
│       │   │   ├── SideNavBar.vue           # Left sidebar, brand, nav, footer
│       │   │   ├── QueryInputSection.vue    # Prompt input + Execute button
│       │   │   ├── WorkflowVisualizerSection.vue  # Step progress bar
│       │   │   ├── WorkflowStep.vue         # Individual step node (icon + label)
│       │   │   ├── ExecutionLogsPanel.vue   # Terminal-style log viewer (60% width)
│       │   │   └── FinalResponsePanel.vue   # AI response card (40% width)
│       │   ├── stores/
│       │   │   └── orchestration.js         # Pinia store — all dashboard state + mock data
│       │   ├── router/
│       │   │   └── routes.js                # / → /dashboard via OrchestratorLayout
│       │   └── css/
│       │       ├── app.scss                 # Imports theme.css + globals.css
│       │       ├── theme.css                # MD3 dark theme CSS variables + typography
│       │       └── globals.css             # Custom Tailwind-style utility classes
│       ├── quasar.config.js
│       └── package.json
├── manage.py
├── CLAUDE.md                     # Engineering guidelines
├── PROJECT_CONTEXT.md            # This file
└── test.html                     # Original HTML prototype (reference only)
```

---

## Frontend (Quasar + Vue 3)

### Stack
- **Framework:** Quasar `^2.16.0` on Vue `^3.5.22`
- **Build tool:** `@quasar/app-vite` `^2.5.1`
- **State:** Pinia `^3.0.1`
- **Routing:** `vue-router` `^5.0.3`
- **Icons:** Material Symbols Outlined (Google Fonts CDN)
- **Fonts:** Geist (body), JetBrains Mono (labels/code)

### Design System
- **Theme:** Material Design 3 dark, defined in `theme.css` as CSS custom properties
- **Primary:** `#4edea3` (green), **Background:** `#131317` (near-black)
- **Utility classes:** Custom Tailwind-like classes in `globals.css` (no Tailwind dependency)
- **Typography scale:** `text-headline-lg/md/sm`, `text-body-lg/md`, `text-label-md/sm`, `text-code-block`

### Layout
`OrchestratorLayout.vue` implements an app-shell pattern:
- `html { height: 100% }` + `body { height: 100%; overflow: hidden }` + `#q-app { height: 100%; overflow: hidden }` — constrains scroll to `<main>` only
- `SideNavBar` — `h-screen flex-col`, fixed left, 256px wide
- `<main>` — `flex-1 overflow-y-auto`, only scrollable region

### Dashboard Page (`DashboardPage.vue`)
Four stacked sections (`space-y-4`):

| Section | Component | Description |
|---|---|---|
| Query input | `QueryInputSection` | Text input + Execute button; wired to store |
| Workflow visualizer | `WorkflowVisualizerSection` | Progress bar + `WorkflowStep` nodes |
| Execution logs | `ExecutionLogsPanel` | Terminal-style log list, 60% width (col-span-6) |
| Final response | `FinalResponsePanel` | AI result card with sentiment/confidence, 40% width (col-span-4) |

Panels use `h-full` inside `grid grid-cols-10` so bottoms align regardless of content height.

### Orchestration Store (`stores/orchestration.js`)
Pinia store, all mock data (no API calls yet):

| State | Type | Description |
|---|---|---|
| `query` | `ref<string>` | Current prompt text |
| `isOrchestrating` | `ref<bool>` | Running state |
| `workflowProgress` | `ref<number>` | 0–100 percent |
| `steps` | `ref<Step[]>` | 5 agents: Planner, Retriever, Analyst, Critic, Final Response |
| `logs` | `ref<Log[]>` | Execution log entries with level (INFO/PLANNER/RETRIEVER/CURRENT) |
| `finalResponse` | `ref<Response>` | Title, content, sentiment, confidence, risks, sessionId |

`startOrchestration()` simulates progress with `setInterval` — no backend call.

### Scripts
| Script | Command |
|---|---|
| `dev` | `quasar dev` |
| `build` | `quasar build` |
| `lint` | `eslint -c ./eslint.config.js "./src*/**/*.{js,cjs,mjs,vue}"` |

---

## Backend (Django)

- **Framework:** Django 6.0.5, SQLite (dev), DEBUG=True
- **URLs:** `/admin/` only; no custom routes
- **DRF:** Not installed
- **Custom apps / models / migrations:** None
- **SECRET_KEY:** Auto-generated dev key — rotate before any shared use

---

## AI Workflow Engine

Not yet implemented. Planned:
- LangGraph multi-agent orchestration (Planner → Retriever → Analyst → Critic → Final Response)
- AI routing layer
- Shared workflow state
- Tool calling subsystem
- Django API endpoints to drive frontend store

No `requirements.txt` or `pyproject.toml` committed yet.

---

## Current Status

| Area | Status |
|---|---|
| Django bootstrap | Done |
| DRF / API endpoints | Not started |
| Custom Django apps | Not started |
| Models / migrations | Not started |
| Quasar bootstrap | Done |
| OrchestratorLayout (app shell) | Done |
| SideNavBar | Done |
| QueryInputSection | Done |
| WorkflowVisualizerSection + WorkflowStep | Done |
| ExecutionLogsPanel | Done |
| FinalResponsePanel | Done |
| Pinia orchestration store (mock) | Done |
| MD3 dark theme + CSS utilities | Done |
| Backend API integration | Not started |
| LangGraph workflows | Not started |
| Agent definitions | Not started |
| Real-time execution updates (WebSocket/SSE) | Not started |
| Auth (beyond Django default) | Not started |
| Test suite | Not started |
| CI | Not configured |

---

## Suggested Next Steps

1. Pin Python deps: `requirements.txt` with Django, DRF, LangGraph, LangChain.
2. Create `workflows` Django app — `Workflow`, `WorkflowRun`, `AgentNode` models.
3. Wire DRF into `INSTALLED_APPS` + `urls.py`; expose run/status endpoints.
4. Replace mock data in `orchestration.js` with real API calls.
5. Add WebSocket or SSE for real-time log streaming to `ExecutionLogsPanel`.
6. Move `SECRET_KEY` and `DEBUG` to env vars before any shared deployment.

---

## References

- Engineering rules: [CLAUDE.md](CLAUDE.md)
- Frontend README: [frontend/frontend/README.md](frontend/frontend/README.md)
- Original HTML prototype: [test.html](test.html)
