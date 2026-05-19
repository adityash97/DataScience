from langgraph.graph import END, START, StateGraph

from ai.nodes.nodes import (
    error_handler,
    executor_node,
    input_node,
    output_node,
    planner_node,
    response_node,
    retry_handler,
)
from ai.routing.router import route_after_executor
from ai.state.workflow_state import WorkflowState


def build_workflow_graph():
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node('input', input_node)
    graph.add_node('planner', planner_node)
    graph.add_node('executor', executor_node)
    graph.add_node('retry_handler', retry_handler)
    graph.add_node('error_handler', error_handler)
    graph.add_node('response', response_node)
    graph.add_node('output', output_node)

    # Linear edges
    graph.add_edge(START, 'input')
    graph.add_edge('input', 'planner')
    graph.add_edge('planner', 'executor')

    # Conditional routing after executor: success → response, fail → retry → executor loop, exhaust → error
    graph.add_conditional_edges(
        'executor',
        route_after_executor,
        {'response': 'response', 'retry': 'retry_handler', 'error': 'error_handler'},
    )
    graph.add_edge('retry_handler', 'executor')  # retry loop

    graph.add_edge('response', 'output')
    graph.add_edge('output', END)
    graph.add_edge('error_handler', END)

    return graph.compile()
