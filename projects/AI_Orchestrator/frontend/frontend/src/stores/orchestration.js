import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  startWorkflow,
  getExecution,
  listWorkflows
} from 'src/services/workflowService'

const POLL_INTERVAL_MS = 2000
const TERMINAL_STATUSES = new Set(['completed', 'failed'])

const AGENT_TEMPLATE = [
  { id: 'planner', name: 'Planner', icon: 'psychology', status: 'pending', subtitle: 'Pending' },
  { id: 'tool', name: 'Tool', icon: 'build', status: 'pending', subtitle: 'Pending' },
  { id: 'executor', name: 'Executor', icon: 'memory', status: 'pending', subtitle: 'Pending' },
  { id: 'response', name: 'Response', icon: 'auto_awesome', status: 'pending', subtitle: 'Pending' }
]

const AGENT_TO_STEP = {
  planner: 'planner',
  tool_runner: 'tool',
  executor: 'executor',
  response: 'response'
}

function buildSteps(executionSteps, status, toolUsed) {
  const reached = new Set()
  for (const step of executionSteps || []) {
    const id = AGENT_TO_STEP[step.agent]
    if (id) reached.add(id)
  }

  return AGENT_TEMPLATE.map((agent) => {
    if (agent.id === 'tool' && !toolUsed) {
      return { ...agent, status: 'skipped', subtitle: 'Not used' }
    }

    if (reached.has(agent.id)) {
      const isLast =
        status === 'running' &&
        agent.id === [...reached].pop()
      return {
        ...agent,
        status: isLast ? 'active' : 'completed',
        subtitle: isLast ? 'Processing...' : 'Completed'
      }
    }
    if (status === 'running') return { ...agent, status: 'pending', subtitle: 'Pending' }
    if (status === 'failed') return { ...agent, status: 'pending', subtitle: 'Halted' }
    return { ...agent, status: 'pending', subtitle: 'Pending' }
  })
}

function buildLogs(messages, status) {
  const logs = (messages || []).map((msg, index) => {
    const match = msg.match(/^\[([A-Za-z]+)\]\s*(.*)$/)
    const level = match ? match[1].toUpperCase() : 'INFO'
    const message = match ? match[2] : msg
    return { id: index + 1, level, message, colored: true }
  })
  if (status === 'running') {
    logs.push({ id: logs.length + 1, level: 'CURRENT', message: '_', colored: false })
  }
  return logs
}

function progressFromSteps(steps) {
  const total = steps.length
  if (!total) return 0
  const completed = steps.filter((s) => s.status === 'completed' || s.status === 'skipped').length
  const active = steps.some((s) => s.status === 'active') ? 0.5 : 0
  return Math.min(100, Math.round(((completed + active) / total) * 100))
}

export const useOrchestrationStore = defineStore('orchestration', () => {
  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------
  const query = ref('What is 12 * 7?')
  const workflows = ref([])
  const selectedWorkflowId = ref(null)
  const isOrchestrating = ref(false)
  const errorMessage = ref('')

  const execution = ref(null)            // raw execution payload from API
  const steps = ref(AGENT_TEMPLATE.map((a) => ({ ...a })))
  const logs = ref([])
  const workflowProgress = ref(0)

  let pollHandle = null

  // -------------------------------------------------------------------------
  // Computed
  // -------------------------------------------------------------------------
  const status = computed(() => execution.value?.status || 'idle')
  const toolUsed = computed(() => execution.value?.tool_used || '')
  const toolResult = computed(() => execution.value?.tool_result || null)
  const executionSteps = computed(() => execution.value?.execution_steps || [])
  const isActive = computed(() => isOrchestrating.value)

  const finalResponse = computed(() => {
    const output = execution.value?.output_payload || {}
    const rawToolResult = output.tool_result || {}
    return {
      title: output.result ? `Response — ${execution.value?.workflow_name || ''}` : 'Awaiting response',
      content: output.result || 'No response yet. Start a workflow to see output here.',
      agents: output.agents_executed || [],
      tool: output.tool_used || null,
      retryCount: output.retry_count ?? 0,
      sessionId: execution.value ? `exec-${execution.value.id}` : '—',
      toolResults: rawToolResult.results || [],
      toolSource: rawToolResult.tool || null,
    }
  })

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------
  function applyExecution(payload) {
    execution.value = payload
    const execStatus = payload?.status || 'idle'
    steps.value = buildSteps(payload?.execution_steps, execStatus, payload?.tool_used)
    logs.value = buildLogs(payload?.execution_log, execStatus)
    workflowProgress.value = progressFromSteps(steps.value)
  }

  function stopPolling() {
    if (pollHandle) {
      clearInterval(pollHandle)
      pollHandle = null
    }
  }

  async function pollOnce(executionId) {
    try {
      const data = await getExecution(executionId)
      applyExecution(data)
      if (TERMINAL_STATUSES.has(data.status)) {
        stopPolling()
        isOrchestrating.value = false
      }
    } catch (err) {
      errorMessage.value = err.message || 'Polling failed'
      stopPolling()
      isOrchestrating.value = false
    }
  }

  // -------------------------------------------------------------------------
  // Public actions
  // -------------------------------------------------------------------------
  const updateQuery = (newQuery) => {
    query.value = newQuery
  }

  async function loadWorkflows() {
    try {
      const result = await listWorkflows()
      workflows.value = result
      if (!selectedWorkflowId.value && result.length) {
        selectedWorkflowId.value = result[0].id
      }
    } catch (err) {
      errorMessage.value = err.message || 'Failed to load workflows'
    }
  }

  async function startOrchestration() {
    if (!selectedWorkflowId.value) {
      await loadWorkflows()
    }
    if (!selectedWorkflowId.value) {
      errorMessage.value = 'No workflow available. Create one in the admin first.'
      return
    }

    errorMessage.value = ''
    isOrchestrating.value = true
    stopPolling()
    applyExecution({ status: 'running', execution_steps: [], execution_log: [], tool_used: '' })

    try {
      const initial = await startWorkflow(selectedWorkflowId.value, query.value)
      applyExecution(initial)

      if (!TERMINAL_STATUSES.has(initial.status) && initial.id) {
        pollHandle = setInterval(() => pollOnce(initial.id), POLL_INTERVAL_MS)
      } else {
        isOrchestrating.value = false
      }
    } catch (err) {
      errorMessage.value = err.message || 'Workflow execution failed'
      isOrchestrating.value = false
    }
  }

  return {
    // state
    query,
    workflows,
    selectedWorkflowId,
    isOrchestrating,
    errorMessage,
    execution,
    steps,
    logs,
    workflowProgress,
    // computed
    status,
    toolUsed,
    toolResult,
    executionSteps,
    isActive,
    finalResponse,
    // actions
    updateQuery,
    loadWorkflows,
    startOrchestration
  }
})
