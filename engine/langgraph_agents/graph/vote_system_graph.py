import os
import pandas as pd
import yaml

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from engine.util import load_config, get_logger
from ..nodes import (
    cpu_checker,
    mem_checker,
    pending_checker,
    decision_agent,
    explainer_agent,
)

logger = get_logger("vote_system_graph")

# Maps the short checker names used in the graph state to the keys used in
# the `agents:` block of the config file.
_AGENT_CONFIG_KEYS = {
    "cpu": "cpu_checker",
    "mem": "memory_checker",
    "pending": "pending_checker",
}


def load_votes_config() -> tuple[dict, str]:
    """Load checker enabled/weight settings and the aggregation strategy.

    Reads `ai.multi_agent_votes.agents`, falling back to `ai.multi_agent.agents`
    (where the block historically lived). Missing agents default to
    enabled=True, weight=1.0.

    Returns:
        Tuple of (agents settings keyed by short name, aggregation strategy:
        "weighted_sum" or "llm_judge").
    """
    ai_cfg = load_config().get("ai", {})
    votes_cfg = ai_cfg.get("multi_agent_votes", {}) or {}
    agents_cfg = votes_cfg.get("agents") or ai_cfg.get("multi_agent", {}).get("agents") or {}

    agents = {}
    for short, cfg_key in _AGENT_CONFIG_KEYS.items():
        agent = agents_cfg.get(cfg_key, {}) or {}
        agents[short] = {
            "enabled": bool(agent.get("enabled", True)),
            "weight": float(agent.get("weight", 1.0)),
        }

    aggregation = votes_cfg.get("aggregation", "weighted_sum")
    return agents, aggregation


# -----------------------
# Functions for each node
# -----------------------
def cpu_node(state: dict) -> dict:
    """Collect CPU votes, or skip the checker (votes=None) when disabled."""
    agents_cfg, _ = load_votes_config()
    if not agents_cfg["cpu"]["enabled"]:
        logger.info("cpu_checker disabled in config, skipping")
        state["cpu_votes"] = None
        return state
    df = pd.DataFrame(state["workloads"])
    state["cpu_votes"] = cpu_checker(df, state.get("cluster_info"))
    return state


def mem_node(state: dict) -> dict:
    """Collect Memory votes, or skip the checker (votes=None) when disabled."""
    agents_cfg, _ = load_votes_config()
    if not agents_cfg["mem"]["enabled"]:
        logger.info("memory_checker disabled in config, skipping")
        state["mem_votes"] = None
        return state
    df = pd.DataFrame(state["workloads"])
    state["mem_votes"] = mem_checker(df, state.get("cluster_info"))
    return state


def pending_node(state: dict) -> dict:
    """Collect Pending votes, or skip the checker (votes=None) when disabled."""
    agents_cfg, _ = load_votes_config()
    if not agents_cfg["pending"]["enabled"]:
        logger.info("pending_checker disabled in config, skipping")
        state["pending_votes"] = None
        return state
    df = pd.DataFrame(state["workloads"])
    state["pending_votes"] = pending_checker(df, state.get("cluster_info"))
    return state


def decision_node(state: dict) -> dict:
    """Aggregate the checkers' votes into final decisions.

    Strategy comes from config (`ai.multi_agent_votes.aggregation`):
    - "weighted_sum" (default): deterministic weighted majority — migrate (1)
      when the weights of the migrate votes exceed half of the total active
      weight; ties stay (0).
    - "llm_judge": delegate to the decision_agent LLM, injecting the weights
      into its prompt.
    """
    agents_cfg, aggregation = load_votes_config()
    n = len(state["workloads"])

    votes = {
        "cpu": state.get("cpu_votes"),
        "mem": state.get("mem_votes"),
        "pending": state.get("pending_votes"),
    }
    active = {name: v for name, v in votes.items() if v is not None}

    if not active:
        logger.warning("All vote checkers disabled; defaulting all decisions to 0")
        state["final_decisions"] = [0] * n
        return state

    if aggregation == "llm_judge":
        weights = {name: agents_cfg[name]["weight"] for name in active}
        state["final_decisions"] = decision_agent(active, weights, n)
        return state

    total_weight = sum(agents_cfg[name]["weight"] for name in active)
    decisions = []
    for i in range(n):
        score = sum(agents_cfg[name]["weight"] * active[name][i] for name in active)
        decisions.append(1 if score > total_weight / 2 else 0)
    state["final_decisions"] = decisions
    return state


def explainer_node(state: dict) -> dict:
    """Generate explanations for the final decisions made.

    Args:
        state (dict): Current state containing workloads and final decisions.

    Returns:
        Dict: Updated state with explanations.
    """
    df = pd.DataFrame(state["workloads"])
    state["explanations"] = explainer_agent(
        df,
        {
            "cpu": state.get("cpu_votes") if state.get("cpu_votes") is not None else "disabled",
            "mem": state.get("mem_votes") if state.get("mem_votes") is not None else "disabled",
            "pending": state.get("pending_votes") if state.get("pending_votes") is not None else "disabled",
        },
        state["final_decisions"],
    )
    return state


# -----------------------
# Graph Creation
# -----------------------

_node_function_map = {
    "cpu": cpu_node,
    "mem": mem_node,
    "pending": pending_node,
    "decision": decision_node,
    "explainer": explainer_node,
}


def create_vote_system_migration_graph():
    """Create a migration graph based on configuration.

    Returns:
        StateGraph: Configured state graph for migration analysis.
    """
    graph = StateGraph(dict)

    graph.add_node("cpu", cpu_node)
    graph.add_node("mem", mem_node)
    graph.add_node("pending", pending_node)
    graph.add_node("decision", decision_node)
    graph.add_node("explainer", explainer_node)

    graph.set_entry_point("cpu")

    graph.add_edge("cpu", "mem")
    graph.add_edge("mem", "pending")
    graph.add_edge("pending", "decision")

    graph.add_edge("decision", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile(checkpointer=MemorySaver())


def load_config_nodes() -> tuple[bool, bool, bool]:
    """Load node execution configuration from YAML file.
    Returns:
        Tuple[bool, bool, bool]: Flags indicating whether to execute pending, cpu, and mem nodes.
    """
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    nodes = config.get("nodes", {})
    cpu = nodes.get("cpu", {}).get("execute", False)
    mem = nodes.get("memory", {}).get("execute", False)
    pending = nodes.get("pending", {}).get("execute", False)

    return pending, cpu, mem
