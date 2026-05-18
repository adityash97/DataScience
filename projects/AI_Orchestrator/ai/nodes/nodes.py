import logging

from ai.state.workflow_state import WorkflowState

logger = logging.getLogger(__name__)


def input_node(state: WorkflowState) -> dict:
    logger.info('[input_node] Received input: %.60s', state['user_input'])
    return {
        'current_step': 'input',
        'status': 'running',
        'messages': [f"Input received: {state['user_input']}"],
    }


def processing_node(state: WorkflowState) -> dict:
    logger.info('[processing_node] Analyzing input')
    return {
        'current_step': 'processing',
        'messages': ['Routing to agents: Planner → Retriever → Analyst → Critic'],
    }


def output_node(state: WorkflowState) -> dict:
    logger.info('[output_node] Formatting final output')
    return {
        'current_step': 'output',
        'status': 'completed',
        'output': {
            'result': f"Processed query: {state['user_input']}",
            'agents_executed': ['Planner', 'Retriever', 'Analyst', 'Critic', 'Final Response'],
            'execution_log': state['messages'],
            'note': 'LangGraph mock execution — LLM integration pending.',
        },
        'messages': ['Workflow completed successfully.'],
    }
