## Live Demo
(https://ai-orchestrator-sigma-wine.vercel.app/#/dashboard)

# AI Orchestrator

AI Orchestrator is a workflow orchestration project that combines a Django REST backend with a Quasar/Vue frontend. It is designed to manage AI workflows, execute agent-driven tasks, and visualize orchestration steps in a modern web UI.

## Features

- Django backend with REST API
- Vue 3 + Quasar frontend application
- AI workflow orchestration and execution engine
- Local SQLite database for quick setup
- CORS-enabled for frontend development
- Health and workflow management endpoints



## Project Structure

- `AI_Orchestrator/` — Django project settings and server entry points
- `workflows/` — workflow models, serializers, views, and API routes
- `ai/` — AI orchestration core with agents, planners, runners, tools, prompts, and state management
- `frontend/frontend/` — Quasar Vue frontend application
- `health/` — health check endpoints
- `requirements.txt` — Python backend dependencies
- `db.sqlite3` — local development database

## Getting Started

### Backend

```bash
cd /Users/adityaanand/Desktop/DataScience/projects/AI_Orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend

```bash
cd frontend/frontend
pnpm install
pnpm dev
```

## Build for Production

### Backend

```bash
python manage.py collectstatic --noinput
gunicorn AI_Orchestrator.wsgi:application
```

### Frontend

```bash
cd frontend/frontend
pnpm build
```

## Environment Variables

The backend uses a `.env` file if present, with the following optional values:

- `SECRET_KEY`
- `DEBUG`
- `FRONTEND_ORIGIN`

## Notes

- Backend is built with Django 6.0 and Django REST Framework
- Frontend uses Vue 3, Quasar, Pinia, and Vue Router
- The backend is configured to allow cross-origin requests from local dev servers

---
