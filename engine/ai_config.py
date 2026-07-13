import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .util import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


class WorkloadLabelOutput(BaseModel):
    """Structured output for workload labeling"""

    decisions: List[int] = Field(
        description="Array of decisions where 0=private, 1=public for each workload"
    )
    explanations: List[str] = Field(
        description="Array of explanations for each decision"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadLabelOutput":
        """Create instance from dictionary"""
        return cls(**data)

    def validate_output(self) -> bool:
        """Validate that decisions and explanations have matching lengths"""
        return len(self.decisions) == len(self.explanations)


class WorkloadRecommendation(BaseModel):
    """Structured output for workload recommendations"""

    batch_id: int | None = None
    workload_id: str
    kind: str
    origin_cluster: int
    destination_cluster: int
    reason: str  # Explanation for the decision


MODEL_CONFIGS: Dict[str, Any] = {}  # Deprecated placeholder

PROMPTS = {
    "label_workloads": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are a Kubernetes orchestrator. For each workload, decide if it should run in the 'private' cluster (0) or 'public' cluster (1).
Rules:
- If private is overloaded or workload needs high resources, prefer public (1).
- If workload is in public and private has capacity, allow migrating back to private (0).
- Use percent_pending and cluster_load to guide decisions.

Respond with JSON:
{{
  "decisions": [0, 1, ...],
  "explanations": [
    "Short explanation for workload 1",
    "Short explanation for workload 2",
    ...
  ]
}}

Workloads: {workloads_json}
""",
    },
    "label_workloads_tools": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are a Kubernetes orchestrator. For each workload, decide if it should run in the 'private' cluster (0) or 'public' cluster (1).
Rules:
- If private is overloaded or workload needs high resources, prefer public (1).
- If workload is in public and private has capacity, allow migrating back to private (0).
- Use percent_pending, the cluster state and the analysis results below to guide decisions.

Respond with JSON:
{{
  "decisions": [0, 1, ...],
  "explanations": [
    "Short explanation for workload 1",
    "Short explanation for workload 2",
    ...
  ]
}}

Workloads: {workloads_json}

Clusters: {clusters_json}

Pending pods per workload: {pending_json}

Analysis results (capacity, pending and pricing tools): {analysis_json}
""",
    },
    "cpu_checker": {
        "version": "1.0",
        "template": """You are a CPU usage specialist for Kubernetes clusters.

Your task is to analyze each workload and decide whether it should migrate to the public cluster (1) or stay in the private cluster (0), based **only on CPU usage**.

Guidelines:
- Use the cluster state below to assess the CPU load of the cluster each workload currently runs on.
- If the workload is in a cluster where CPU usage is high (above 80%), suggest migration (1).
- Otherwise, recommend staying (0).
- Ignore memory and pending pods.

Respond with a JSON list of 0s and 1s only.
Example: [0, 1, 1, 0]

Workloads: {workloads_json}

Clusters: {clusters_json}
""",
    },
    "mem_checker": {
        "version": "1.0",
        "template": """You are a memory usage specialist for Kubernetes workloads.

Your task is to analyze each workload and decide whether it should migrate to the public cluster (1) or stay in the private cluster (0), based **only on memory usage**.

Guidelines:
- Use the cluster state below to assess the memory load of the cluster each workload currently runs on.
- If the workload demands high memory and the current cluster is overloaded, suggest migration (1).
- Otherwise, recommend staying (0).
- Ignore CPU and pending pods.

Respond with a JSON list of 0s and 1s only.
Example: [1, 0, 1, 0]

Workloads: {workloads_json}

Clusters: {clusters_json}
""",
    },
    "pending_checker": {
        "version": "1.0",
        "template": """You are a pending pod specialist in Kubernetes.

Your task is to analyze each workload and decide whether it should migrate to the public cluster 
(1) or stay in the private cluster (0), based **only on the percentage of pending pods**.

Guidelines:
- If the workload has 50% or more of the pods pending, suggest migration (1).
- The cluster state below shows pending pods per cluster; use it as supporting context.
- Otherwise, recommend staying (0).
- Ignore CPU and memory.

Respond with a JSON list of 0s and 1s only.
Example: [0, 1, 1, 0]

Workloads: {workloads_json}

Clusters: {clusters_json}
""",
    },
    "decision": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are the final decision judge for Kubernetes workload migration.

You receive the votes of specialist agents (cpu, mem, pending) as JSON lists of 0s and 1s,
where 1 means migrate to the public cluster and 0 means remain in the private cluster,
along with the weight of each agent. For each workload, weigh the agents' votes according
to their weights and decide 1 (migrate) only when the weighted support for migration
exceeds half of the total weight; otherwise decide 0.

Respond with a JSON list of 0s and 1s only.
Example: [0, 1, 1, 0]
Votes and weights: {workload_json}
""",
    },
}


def get_model_config(model_key="gemini"):
    """
    Get model configuration by key.

    Args:
        model_key: The key for the model configuration

    Returns:
        Dictionary with model configuration
    """
    cfg = load_config()
    return cfg.get("ai", {}).get("models", {}).get(model_key, {})


def get_prompt(prompt_key, **kwargs):
    """
    Get a prompt by key and format it with provided kwargs.

    Args:
        prompt_key: The key for the prompt template
        **kwargs: Format arguments for the prompt template

    Returns:
        Formatted prompt string
    """
    prompt_data = PROMPTS.get(prompt_key, {})
    template = prompt_data.get("template", "")
    return template.format(**kwargs)


def get_agent_mode():
    """
    Get the current agent mode from configuration.
    
    Returns:
        str: 'single_agent' or 'multi_agent'
    """
    cfg = load_config()
    ai_config = cfg.get("ai", {})
    
    # Check for new mode field
    mode = ai_config.get("mode")
    if mode:
        return mode
    
    # Legacy support for multi_agent boolean
    if "multi_agent" in ai_config and isinstance(ai_config["multi_agent"], bool):
        return "multi_agent" if ai_config["multi_agent"] else "single_agent"
    
    # Default to single_agent
    return "single_agent"


def get_agent_config(mode=None):
    """
    Get the configuration for a specific agent mode.
    
    Args:
        mode (str, optional): Agent mode. If None, uses current mode from config.
    
    Returns:
        dict: Configuration for the specified agent mode
    """
    cfg = load_config()
    ai_config = cfg.get("ai", {})
    
    if mode is None:
        mode = get_agent_mode()
    
    # Get mode-specific config, falling back to default_config
    mode_config = ai_config.get(mode, {})
    default_config = ai_config.get("default_config", {})
    
    # Merge with defaults
    config = {**default_config, **mode_config}
    
    return config


def build_system_prompt_from_config(mode=None):
    """
    Build the system prompt by loading it from a file based on the agent mode.
    
    Args:
        mode (str, optional): Agent mode ('single_agent' or 'multi_agent'). 
                             If None, reads from config.
    
    Returns:
        str: The complete system prompt from the file
    """
    import os
    
    cfg = load_config()
    ai_config = cfg.get("ai", {})
    
    # Determine the mode
    if mode is None:
        mode = ai_config.get("mode", "single_agent")
    
    # Support legacy multi_agent boolean field for backward compatibility
    if "multi_agent" in ai_config and isinstance(ai_config["multi_agent"], bool):
        mode = "multi_agent" if ai_config["multi_agent"] else "single_agent"
        logger.warning("Using legacy 'multi_agent' boolean field. Please update to use 'mode' field.")
    
    # Get the mode-specific configuration
    mode_config = ai_config.get(mode, {})
    
    # Get the selected prompt for this mode
    selected_prompt = mode_config.get("selected_prompt")
    
    # Fallback to legacy selected_prompt field if mode-specific one doesn't exist
    if not selected_prompt:
        selected_prompt = ai_config.get("selected_prompt", "single_agent_v1")
        logger.info(f"Using legacy selected_prompt field: {selected_prompt}")
    
    # Build the prompt file path
    prompt_file = f"prompts/{selected_prompt}.txt"
    
    # Try to read the prompt from file
    system_prompt = None
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            logger.info(f"Loaded {mode} prompt from file: {prompt_file}")
        except Exception as e:
            logger.warning(f"Failed to load prompt from {prompt_file}: {e}")
    
    # Fall back to default if file reading failed
    if not system_prompt:
        default_prompts = {
            "single_agent": "You are an expert Kubernetes workload migration advisor. Analyze the provided workloads and make migration decisions.",
            "multi_agent": "You are part of a multi-agent system for Kubernetes workload management. Collaborate with other agents to make optimal migration decisions."
        }
        system_prompt = default_prompts.get(mode, default_prompts["single_agent"])
        logger.warning(f"Using default {mode} prompt - prompt file not found")
    
    return system_prompt

def build_cluster_selection_from_config():
    """
    Considers the configuration and builds the list of clusters to be used. Ensure that the clusters
    are strings and strips any leading/trailing whitespace.
    
    Returns:
        List of cluster labels as strings
    """
    cfg = load_config()
    cluster_selection = cfg.get("ai", {}).get("cluster_selection", {})
    listed_clusters = cluster_selection.get("listed_clusters", [])

    # Ensure clusters are strings and strip whitespace
    normalized_clusters = []
    for cluster in listed_clusters:
        new_cluster = str(cluster).strip()
        
        if new_cluster:
            normalized_clusters.append(new_cluster)
            
    return normalized_clusters