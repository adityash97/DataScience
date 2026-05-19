from langgraph.graph import END, START, StateGraph

from ai.nodes.nodes import (
    error_handler,
    executor_node,
    input_node,
    output_node,
    planner_node,
    response_node,
    retry_handler,
    tool_node,
)
from ai.routing.router import route_after_executor, route_after_planner
from ai.state.workflow_state import WorkflowState


def build_workflow_graph():
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node('input', input_node)
    graph.add_node('planner', planner_node)
    graph.add_node('tool', tool_node)
    graph.add_node('executor', executor_node)
    graph.add_node('retry_handler', retry_handler)
    graph.add_node('error_handler', error_handler)
    graph.add_node('response', response_node)
    graph.add_node('output', output_node)

    # Linear edges
    graph.add_edge(START, 'input')
    graph.add_edge('input', 'planner')

    # Conditional routing after planner: tool needed → tool_node, else → executor
    graph.add_conditional_edges(
        'planner',
        route_after_planner,
        {'tool': 'tool', 'executor': 'executor'},
    )
    graph.add_edge('tool', 'executor')

    # Conditional routing after executor: success → response, fail → retry/error
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
