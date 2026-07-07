from typing import List, Union, Dict, Any, Tuple
import pandas as pd
import re
import json
import uuid
from .ai_config import get_model_config, get_prompt, PROMPTS, build_system_prompt_from_config, get_agent_mode, get_agent_config
from .util import get_logger, load_config, log_token_usage

from .langgraph_agents.graph.tool_system_graph import create_tool_system_migration_graph
from .client import OpenRouterClient, OllamaClient, get_client

logger = get_logger("agents")

# Initialize AI client (Ollama, OpenRouter, etc.)
try:
    ai_client = get_client()
    HAS_CLIENT = True
except Exception as e:
    logger.warning(f"Failed to initialize AI client: {e}")
    ai_client = None
    HAS_CLIENT = False

# Keep legacy imports for fallback
try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False


# global variables
REQUEST_COUNTER = 0
TOKEN_TOTALS = {"input": 0, "output": 0, "total": 0}


def _normalize_workloads_to_dataframe(
    workloads: Union[list, "pd.DataFrame"],
) -> "pd.DataFrame":
    """Convert workloads input to DataFrame format."""
    return pd.DataFrame(workloads) if isinstance(workloads, list) else workloads


def _extract_json_from_response(text_response) -> Dict[str, Any]:
    """Extract and parse JSON from model response."""
    # The client already parses to a dict when response_format={"type": "json_object"}
    # (see OllamaClient.chat); in that case there is nothing left to extract.
    if isinstance(text_response, dict):
        return text_response

    json_match = re.search(r"\{[\s\S]*\}", text_response)
    if not json_match:
        raise ValueError("No JSON object found in model response")

    json_str = json_match.group(0)
    return json.loads(json_str)


def _validate_and_extract_decisions(
    response_data: Dict[str, Any],
) -> Tuple[List[int], List[str]]:
    """Validate response using schema and extract decisions/explanations."""
    prompt_config = PROMPTS.get("label_workloads", {})
    output_schema = prompt_config.get("output_schema")

    if output_schema:
        try:
            output = output_schema.from_dict(response_data)
            if not output.validate_output():
                logger.warning(
                    "Output validation failed: decisions and explanations have different lengths"
                )
            return output.decisions, output.explanations
        except Exception as e:
            logger.error(f"Failed to validate response with schema: {e}")

    # Fallback to direct extraction
    decisions = response_data.get("decisions", [])
    explanations = response_data.get("explanations", [])
    return decisions, explanations


def _create_explanation_output(
    labels: List[int], explanations: List[str], df: "pd.DataFrame"
) -> Dict[str, Any]:
    """Create structured explanation output and log decisions."""
    explanation_output = {
        "workload_explanations": [],
    }

    for idx, (label, workload) in enumerate(zip(labels, df.iterrows())):
        workload_id = workload[1].get("workload_id", f"workload-{idx}")
        kind = workload[1].get("kind", "unknown")
        destination = "public" if label == 1 else "private"

        # Log the decision
        logger.info(f"Decision for {workload_id} ({kind}): Cluster {destination}")

        # Add explanation for this workload
        explanation = (
            explanations[idx]
            if idx < len(explanations)
            else f"Workload {workload_id} recommended for {destination} cluster based on resource requirements"
        )
        explanation_output["workload_explanations"].append(explanation)

    return explanation_output


def label_workloads_with_llm(
    workloads: Union[list, "pd.DataFrame"],
    model: str = "qwen3:8b",
    client = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Uses LLM API to decide workload labels with explanations.
    Each label: 0 = private, 1 = public.
    Args:
        workloads: list of dicts or DataFrame with workload fields.
        model: Model to use via Provider (default: google/gemini-2.0-flash-001)
    Returns:
        Tuple containing:
        - List of labels (0 or 1) in the same order
        - Dictionary with explanations for each workload
    """
    global REQUEST_COUNTER, TOKEN_TOTALS

    logger.info("Starting workload analysis for migration decision using AI client")

    # Dependency validation - early return if not available
    if not HAS_CLIENT or not ai_client:
        logger.error("AI client not available, skipping this cycle")
        return [], {}

    # Data preparation
    df = _normalize_workloads_to_dataframe(workloads)
    user_prompt = get_prompt(
        "label_workloads", workloads_json=df.to_json(orient="records", indent=2)
    )


    # Initialize to avoid UnboundLocalError if exception raised before assignment
    text_response = ""

    try:
        config = load_config()
        model = config.get("ai", {}).get("selected_model", "qwen3:8b")

        model_config = config.get("ai", {}).get("default_config", {})
        system_prompt = build_system_prompt_from_config()
        generation_config = model_config.get("generation_config", {})

        logger.info(f"CONFIG: Using model: {model}")
        logger.info(f"CONFIG: Using generation config: {generation_config}")

        text_response = ai_client.chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
            temperature=generation_config.get("temperature", 0.1),
            max_tokens=generation_config.get("max_output_tokens", 8000),
        )

        REQUEST_COUNTER += 1
        logger.info(f"AI API request count: {REQUEST_COUNTER}")

        # Parse and validate response
        response_data = _extract_json_from_response(text_response)
        decisions, explanations = _validate_and_extract_decisions(response_data)

        # Validate decision count - handle mismatches gracefully
        if len(decisions) != len(df):
            logger.warning(
                f"Decision count mismatch: expected {len(df)}, got {len(decisions)}"
            )

            # Handle mismatch by adjusting decisions list
            if len(decisions) > len(df):
                # Too many decisions - truncate
                decisions = decisions[: len(df)]
                explanations = (
                    explanations[: len(df)]
                    if len(explanations) > len(df)
                    else explanations
                )
                logger.info(f"Truncated decisions to match {len(df)} workloads")
            else:
                # Too few decisions - pad with original cluster labels (no migration)
                missing_count = len(df) - len(decisions)

                # Get original cluster labels for missing decisions
                for i in range(len(decisions), len(df)):
                    workload_row = df.iloc[i]
                    original_cluster = workload_row.get('cluster_label', 'private')
                    # Convert cluster label to decision: private=0, public=1
                    original_decision = 0 if original_cluster == 'private' else 1
                    decisions.append(original_decision)
                    explanations.append(
                        f"Maintaining original cluster ({original_cluster}) due to missing AI decision"
                    )

                logger.info(
                    f"Padded {missing_count} missing decisions with original cluster assignments (no migration)"
                )

        # Convert to integers and create output
        labels = [int(decision) for decision in decisions]
        logger.info("Migration decisions extracted from JSON response")

        explanation_output = _create_explanation_output(labels, explanations, df)
        return labels, explanation_output

    except Exception as e:
        logger.error(f"Error parsing JSON response: {e}")
        try:
            logger.error(f"Raw LLM response (first 500 chars): {text_response[:500]}")
        except Exception:
            logger.error("No LLM response available (error occurred before API call)")
        logger.error("LLM failed to generate recommendations, skipping this cycle")
        return [], {}
    except Exception as e:
        logger.error(f"Error using {client} API: {e}")
        logger.error("LLM failed to generate recommendations, skipping this cycle")
        return [], {}


# ---------------------------------------------------------------------------
# MultiAgent implementation
# ---------------------------------------------------------------------------
def label_workloads_multiagent(
    workloads: List[dict], cluster_info: List[dict], interval_duration: str = None
) -> Tuple[List[int], Dict[str, Any]]:
    df = _normalize_workloads_to_dataframe(workloads)
    
    # Build initial state WITHOUT pre-populating 'decisions' or 'explanations'
    # so they only appear in the graph output (not in the input trace).
    state = {
        "workloads": df.to_dict(orient="records"),
        "cluster_info": cluster_info,
        "interval_duration": interval_duration
    }

    graph = create_tool_system_migration_graph()
    logger.info("Using LangGraph for workload recommendations")

    thread_id = str(uuid.uuid4())

    final_state = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    

    recommendations_dict = final_state.get("explanations", {})
    final_decisions = final_state.get("decisions", [])
    workload_explanations = recommendations_dict.get("workload_explanations", [])

    
    if not final_decisions or len(final_decisions) != len(workloads):
        logger.warning(
            f"Number of decisions ({len(final_decisions)}) does not match number of workloads ({len(workloads)}). "
            "Filling missing recommendations with -1."
        )

        
        corrected_decisions = []
        missing_workloads = []

        for i, wl in enumerate(workloads):
            if i < len(final_decisions):
                corrected_decisions.append(final_decisions[i])
            else:
                corrected_decisions.append(-1)
                missing_workloads.append(wl.get("workload_id", f"workload_{i}"))

        if missing_workloads:
            logger.warning(
                f"No response from LLM for workloads: {', '.join(missing_workloads)}"
            )

        labels = corrected_decisions
    else:
        labels = final_decisions

    final_explanations = {
        "workload_explanations": workload_explanations,
    }

    return labels, final_explanations


def label_workloads_multiagent_votes(workloads, provider="langgraph"):
    """
    Label workloads using a multi-agent with voting system.
    """
    df = _normalize_workloads_to_dataframe(workloads)
    # Do not pre-populate 'final_decisions' or 'explanations' here either;
    # let the graph/nodes produce them as output so they don't show up in input traces.
    state = {
        "workloads": df.to_dict(orient="records"),
        "cpu_votes": [],
        "mem_votes": [],
        "pending_votes": [],
        "final_decisions": [],
        "explanations": {},
    }

    graph = create_migration_graph()

    thread_id = str(uuid.uuid4())

    final_state = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

    labels = final_state.get("final_decisions", [])

    explanations = final_state.get("explanations", {})

    return labels, explanations


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------
def label_workloads(
    workloads: Union[list, "pd.DataFrame"],
    cluster_info: List[dict] = None,
    interval_duration: str = None,
    provider: str | None = None,
    multiagent: bool | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Public API to label workloads with the configured AI provider.
    
    Args:
        workloads: List of workloads or DataFrame
        cluster_info: List of cluster information dictionaries (optional)
        provider: Override the configured provider (optional)
        multiagent: Override the configured mode (optional, legacy parameter)
    
    Returns:
        Tuple of (labels, explanations)
    """
    cfg = load_config()

    # Determine the agent mode
    if multiagent is not None:
        # Legacy parameter support
        mode = "multi_agent" if multiagent else "single_agent"
        logger.info(f"Using legacy multiagent parameter: mode={mode}")
    else:
        mode = get_agent_mode()
    
    # Get mode-specific configuration
    agent_config = get_agent_config(mode)
    
    # Determine provider
    if provider is None:
        provider = agent_config.get("provider", "ollama")
    
    provider = provider.lower()

    try:
        model = cfg.get("ai", {}).get("selected_model", "qwen3:8b")
        generation_config = agent_config.get("generation_config", {})

        logger.info(f"CONFIG: Using agent mode: {mode}")
        logger.info(f"CONFIG: Using provider: {provider}")
        logger.info(f"CONFIG: Using model: {model}")
        logger.info(f"CONFIG: Using generation config: {generation_config}")
    except Exception as e:
        logger.warning(f"Could not log model configuration: {e}")

    labels = []
    explanations = {}
    # Route to appropriate labeling function based on mode
    if mode == "multi_agent":
        # Use provided cluster_info or default to empty list
        if cluster_info is None:
            cluster_info = []
            logger.warning("No cluster_info provided to label_workloads, using empty list")
        labels, explanations = label_workloads_multiagent(workloads, cluster_info, interval_duration)
    elif provider in {"gemini", "google", "openrouter", "ollama"}:
        labels, explanations = label_workloads_with_llm(workloads)
    else:
        logger.warning(f"Unknown provider '{provider}'. skipping this cycle.")
        explanations = {
            "workload_explanations": [],
        }

    # Normalize explanations format
    if isinstance(explanations, list):
        workload_explanations = explanations
        explanations = {
            "workload_explanations": workload_explanations,
        }

    return labels, explanations

# Get metrics of token usage and requests
def get_usage_metrics() -> Dict[str, Any]:
    return {"total_requests": REQUEST_COUNTER, "total_tokens": TOKEN_TOTALS}
