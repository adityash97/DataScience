const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api'

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = body?.message || `Request failed (${response.status})`
    const err = new Error(message)
    err.status = response.status
    err.body = body
    throw err
  }
  return body
}

export async function listWorkflows() {
  const res = await request('/workflows/')
  return res.data || []
}

export async function startWorkflow(workflowId, query) {
  const res = await request(`/workflows/${workflowId}/run/`, {
    method: 'POST',
    body: JSON.stringify({ input_payload: { query } })
  })
  return res.data || {}
}

export async function getExecution(executionId) {
  const res = await request(`/workflows/executions/${executionId}/`)
  return res.data || {}
}

export async function listExecutions() {
  const res = await request('/workflows/executions/')
  return res.data || []
}
