# PROJECT_CONTEXT.md

## 1. Project Overview

**Project:** AI_Orchestrator
**Purpose:** Portfolio-quality AI workflow platform demonstrating multi-agent orchestration, LangGraph workflows, AI routing, shared execution state, tool calling, and workflow monitoring.

**Backend Goals:**
- Clean REST API foundation (Django + DRF)
- Environment-driven configuration
- Reusable response utilities
- Health check endpoint
- Lightweight logging
- Extendable for LangGraph integration

**Frontend Stack:** Quasar 2 + Vue 3 (Composition API, `<script setup>`), Pinia, Vue Router

**Backend Stack:** Django 6.0.5, Django REST Framework, python-dotenv, django-cors-headers, SQLite (dev)

---

## 2. Current Folder Structure

```
AI_Orchestrator/
├── AI_Orchestrator/              # Django project package
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── frontend/
│   └── frontend/                 # Quasar app root
│       ├── src/
│       │   ├── App.vue
│       │   ├── layouts/OrchestratorLayout.vue
│       │   ├── pages/DashboardPage.vue
│       │   ├── components/
│       │   ├── stores/orchestration.js
│       │   └── router/routes.js
│       └── quasar.config.js
├── .venv/                        # Python 3.14 virtualenv
├── health/                       # Health check app (M1 Step 6)
│   ├── __init__.py
│   ├── views.py
│   └── urls.py
├── workflows/                    # Workflow app (M2)
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   └── urls.py
├── utils/                        # Shared utilities (M1 Steps 4–5)
│   ├── __init__.py
│   ├── response.py               # success_response(), error_response()
│   └── exceptions.py             # custom_exception_handler
├── .venv/
├── .env
├── .env.example
├── manage.py
├── requirements.txt
├── CLAUDE.md
└── PROJECT_CONTEXT.md
```

---

## 3. Backend Architecture Rules

- Keep architecture simple — no enterprise patterns
- Avoid unnecessary abstractions or layers
- Keep files readable (no giant files)
- Use reusable utilities (api_response, exception handler)
- Preserve existing project structure — only add where needed
- No overengineering — this is a portfolio demo
- Follow PEP8, meaningful naming, clean imports

---

## 4. Milestone Tracking

### Milestone 1 — Backend Foundation Setup ✅ COMPLETE

| Step | Description | Status |
|------|-------------|--------|
| Step 1 | Install and configure DRF | ✅ Done |
| Step 2 | Configure CORS | ✅ Done |
| Step 3 | Setup environment configuration (.env → settings.py) | ✅ Done |
| Step 4 | Create reusable API response utility | ✅ Done |
| Step 5 | Create centralized exception handler | ✅ Done |
| Step 6 | Create health check app + endpoint | ✅ Done |
| Step 7 | Add logging configuration | ✅ Done |
| Step 8 | Final cleanup and validation | ✅ Done |

---

### Milestone 2 — Workflow Models & Execution APIs

| Step | Description | Status |
|------|-------------|--------|
| Step 1 | Create `workflows` Django app + register in settings | ✅ Done |
| Step 2 | Implement Workflow + WorkflowExecution models + migrations | ✅ Done |
| Step 3 | Implement WorkflowSerializer + WorkflowExecutionSerializer | ✅ Done |
| Step 4 | Implement Workflow APIs (list, create, detail) | ✅ Done |
| Step 5 | Implement Execution APIs (run/mock, history, detail) | ✅ Done |
| Step 6 | Wire URLs (`/api/workflows/`, `/api/executions/`) | ✅ Done |
| Step 7 | Admin configuration | ✅ Done |
| Step 8 | Final validation | ✅ Done |

---

## 5. CURRENT_AGENT_STATUS

```
==================================================
CURRENT_STEP:
Milestone 2 — COMPLETE

STATUS:
DONE

LAST_COMPLETED_STEP:
Step 8 — Final validation

NEXT_MILESTONE:
Milestone 3 — Frontend API Integration (replace mock store data with real API calls)

NOTES:
- Workflow model: name, description, workflow_type, configuration (JSONField), is_active, timestamps
- WorkflowExecution model: FK to Workflow, status choices (pending/running/completed/failed), input/output JSONFields, timestamps
- Migration 0001_initial applied cleanly
- WorkflowSerializer + WorkflowExecutionSerializer (ModelSerializer, workflow_name read-only field)
- APIs verified live:
    POST /api/workflows/create/      → 201 + workflow data
    GET  /api/workflows/             → list active workflows
    GET  /api/workflows/<pk>/        → single workflow
    POST /api/workflows/<pk>/run/    → mock execution → 201 + execution data
    GET  /api/workflows/executions/  → execution history
    GET  /api/workflows/executions/<pk>/ → execution detail
- Admin: WorkflowAdmin + WorkflowExecutionAdmin registered with list_display, filters, search
- All responses use success_response/error_response from utils.response
- _mock_execution_output() is placeholder — LangGraph replaces in future milestone
==================================================
```

---

## Frontend (Quasar + Vue 3)

### Stack
- **Framework:** Quasar `^2.16.0` on Vue `^3.5.22`
- **State:** Pinia `^3.0.1`
- **Routing:** `vue-router` `^5.0.3`
- **Icons:** Material Symbols Outlined (Google Fonts CDN)
- **Fonts:** Geist (body), JetBrains Mono (labels/code)

### Design System
- **Theme:** Material Design 3 dark — CSS custom properties in `theme.css`
- **Primary:** `#4edea3` (green), **Background:** `#131317` (near-black)
- **Typography:** `text-headline-lg/md/sm`, `text-body-lg/md`, `text-label-md/sm`, `text-code-block`

### Dashboard Sections

| Section | Component |
|---------|-----------|
| Query input | `QueryInputSection.vue` |
| Workflow visualizer | `WorkflowVisualizerSection.vue` + `WorkflowStep.vue` |
| Execution logs | `ExecutionLogsPanel.vue` |
| Final response | `FinalResponsePanel.vue` |

### Orchestration Store (`stores/orchestration.js`)
Pinia store with mock data — no live API calls yet.
`startOrchestration()` simulates progress with `setInterval`.

---

## Backend (Django)

- **Framework:** Django 6.0.5, SQLite (dev), DEBUG=True
- **DRF:** Installed and configured (Step 1 ✅)
- **CORS:** Not yet configured (Step 2)
- **Custom apps:** None yet
- **SECRET_KEY:** Hardcoded dev key — move to .env in Step 3

---

## AI Workflow Engine

Not yet implemented. Planned:
- LangGraph multi-agent orchestration (Planner → Retriever → Analyst → Critic → Final Response)
- AI routing layer
- Shared workflow state
- Tool calling subsystem
- Django API endpoints to drive frontend store

---

## Current Status

| Area | Status |
|------|--------|
| Django bootstrap | Done |
| DRF installed + configured | Done ✅ |
| CORS setup | Done ✅ |
| Environment config (.env) | Done ✅ |
| API response utilities | Done ✅ |
| Exception handler | Done ✅ |
| Health check endpoint (`GET /api/health/`) | Done ✅ |
| Logging config | Done ✅ |
| Quasar frontend (mock data) | Done |
| Backend API integration | Not started |
| LangGraph workflows | Not started |

---

## References

- Engineering rules: [CLAUDE.md](CLAUDE.md)
- Original HTML prototype: [test.html](test.html)
