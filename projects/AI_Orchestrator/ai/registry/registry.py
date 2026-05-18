from ai.agents.executor import ExecutorAgent
from ai.agents.planner import PlannerAgent
from ai.agents.response import ResponseAgent

_REGISTRY: dict = {
    'planner': PlannerAgent(),
    'executor': ExecutorAgent(),
    'response': ResponseAgent(),
}


def get_agent(name: str):
    agent = _REGISTRY.get(name)
    if agent is None:
        raise ValueError(f"Agent '{name}' not found. Available: {list_agents()}")
    return agent


def list_agents() -> list[str]:
    return list(_REGISTRY.keys())
