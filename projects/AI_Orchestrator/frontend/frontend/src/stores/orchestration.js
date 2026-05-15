import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useOrchestrationStore = defineStore('orchestration', () => {
  // State
  const query = ref('Analyze Tesla stock and generate investment summary')
  const isOrchestrating = ref(false)
  const workflowProgress = ref(35) // 0-100 percentage

  // Workflow steps state
  const steps = ref([
    {
      id: 1,
      name: 'Planner',
      icon: 'check',
      status: 'completed',
      subtitle: 'Completed'
    },
    {
      id: 2,
      name: 'Retriever',
      icon: 'cloud_download',
      status: 'active',
      subtitle: 'Processing...'
    },
    {
      id: 3,
      name: 'Analyst',
      icon: 'monitoring',
      status: 'pending',
      subtitle: 'Pending'
    },
    {
      id: 4,
      name: 'Critic',
      icon: 'gavel',
      status: 'pending',
      subtitle: 'Pending'
    },
    {
      id: 5,
      name: 'Final Response',
      icon: 'auto_awesome',
      status: 'pending',
      subtitle: 'Pending'
    }
  ])

  // Execution logs
  const logs = ref([
    { id: 1, level: 'INFO', message: 'Initializing Multi-Agent Session...', colored: true },
    { id: 2, level: 'PLANNER', message: 'Decomposing query: "Analyze Tesla stock..."', colored: true },
    {
      id: 3,
      level: 'PLANNER',
      message: 'Task 1: Fetch SEC 10-K filings. Task 2: Search news (7d). Task 3: Quantitative analysis.',
      colored: true
    },
    { id: 4, level: 'RETRIEVER', message: 'Fetching data from SEC filings...', colored: true },
    {
      id: 5,
      level: 'RETRIEVER',
      message: 'Querying internal knowledge graph for market context.',
      colored: true
    },
    { id: 6, level: 'RETRIEVER', message: 'Searching latest news via Perplexity API...', colored: true },
    { id: 7, level: 'CURRENT', message: '_', colored: false }
  ])

  // Response data
  const finalResponse = ref({
    title: 'Executive Summary: TSLA',
    content:
      'Tesla, Inc. (TSLA) shows strong fundamentals in its latest quarterly filings with a 23% increase in solar deployment and record-breaking delivery numbers in APAC regions.',
    sentiment: 'Bullish',
    confidence: '94%',
    risks:
      'Key risks identified by the Critic agent include potential supply chain volatility in lithium sourcing and increased competitive pressure from European EV manufacturers...',
    sessionId: 'flow-8812'
  })

  // Computed
  const isActive = computed(() => isOrchestrating.value)

  // Methods
  const updateQuery = (newQuery) => {
    query.value = newQuery
  }

  const startOrchestration = () => {
    isOrchestrating.value = true
    workflowProgress.value = 0
    // Simulate progress
    const interval = setInterval(() => {
      if (workflowProgress.value < 100) {
        workflowProgress.value += Math.random() * 15
      } else {
        clearInterval(interval)
        isOrchestrating.value = false
      }
    }, 1500)
  }

  const updateStepStatus = (stepId, newStatus) => {
    const step = steps.value.find((s) => s.id === stepId)
    if (step) {
      step.status = newStatus
    }
  }

  const addLog = (level, message) => {
    const newLog = {
      id: logs.value.length + 1,
      level,
      message,
      colored: true
    }
    logs.value.push(newLog)
  }

  const updateResponse = (newResponse) => {
    finalResponse.value = { ...finalResponse.value, ...newResponse }
  }

  return {
    query,
    isOrchestrating,
    workflowProgress,
    steps,
    logs,
    finalResponse,
    isActive,
    updateQuery,
    startOrchestration,
    updateStepStatus,
    addLog,
    updateResponse
  }
})
