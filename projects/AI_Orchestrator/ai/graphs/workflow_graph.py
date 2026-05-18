from langgraph.graph import END, START, StateGraph

from ai.nodes.nodes import input_node, output_node, processing_node
from ai.state.workflow_state import WorkflowState


def build_workflow_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node('input', input_node)
    graph.add_node('processing', processing_node)
    graph.add_node('output', output_node)

    graph.add_edge(START, 'input')
    graph.add_edge('input', 'processing')
    graph.add_edge('processing', 'output')
    graph.add_edge('output', END)

    return graph.compile()
