import json
import pandas as pd
import re
import logging

from typing import List, Dict, Union
from dotenv import load_dotenv
from engine.ai_config import get_prompt
from engine.util import load_config, get_logger
from openai import OpenAI
from engine.ai_config import get_prompt
from engine.util import load_config, log_token_usage
from engine.client import get_client
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI

logger = get_logger("agents")
logger = logging.getLogger(__name__)
logger = get_logger("langgraph_agents")

load_dotenv()


class LLMInvokeModel:
    """
    Lightweight wrapper exposing a LangChain-like `.invoke(input)` interface,
    backed by our AI Client (OpenAI compatible — works with Ollama, OpenRouter, etc.).
    """

    def __init__(
        self, client, model_name: str, system_prompt: str, **gen_cfg
    ):
        self._client = client
        self._model_name = model_name
        self._system_prompt = system_prompt
        # Normalize generation config keys to OpenAI chat params
        self._gen_cfg = {
            "temperature": gen_cfg.get("temperature", 0.1),
            # In config we may store as max_output_tokens; OpenAI param is max_tokens
            "max_tokens": gen_cfg.get("max_output_tokens", 2048),
        }

    def invoke(self, user_prompt: str) -> str:
        return self._client.chat(
            model=self._model_name,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
            **self._gen_cfg,
        )


def get_llm():
    """Initialize and return a model wrapper with `.invoke` and its config."""
    try:
        client = get_client()
        config = load_config()
        selected_model = config["ai"].get("selected_model", "qwen3:8b")
        model_cfg = config["ai"]["models"].get(selected_model, {})
        if model_cfg is None:
            raise ValueError(f"Model '{selected_model}' not found in configuration.")

        model_name = selected_model
        # Work on a mutable copy to avoid accidental global mutation
        generation_config = dict(model_cfg.get("generation_config", {}))

        # Prefer config-provided system prompt if available, else fall back to default
        system_prompt = generation_config.get("system_prompt", "You are an expert Kubernetes workload migration advisor. Analyze the provided workloads and make migration decisions.")

        model = LLMInvokeModel(
            client, model_name, system_prompt, **generation_config
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize AI client/model: {e}")
        raise ValueError(f"Failed to initialize AI client/model: {e}")


model = get_llm()


# -----------------------
# Helper Functions
# -----------------------
def normalize_votes(votes: List[int], expected_length: int) -> List[int]:
    """Normalize a list of votes to ensure it matches the expected length.
    Args:
        votes (List[int]): List of votes (0 or 1).
        expected_length (int): The expected number of votes.

    Returns:
        List[int]: Normalized list of votes with the expected length.
    """
    if len(votes) < expected_length:
        votes.extend([0] * (expected_length - len(votes)))
    elif len(votes) > expected_length:
        votes = votes[:expected_length]
    return votes


def _parse_llm_response(response: str, expected_length: int) -> List[int]:
    """Parse LLM response to extract votes as a list of integers (0 or 1).

    Args:
        response (str): The raw response from the LLM.
        expected_length (int): The expected number of votes.

    Returns:
        List[int]: A list of votes (0 or 1) normalized to the expected length
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "decisions" in parsed:
            votes = parsed["decisions"]
        elif isinstance(parsed, list):
            votes = parsed
        else:
            votes = [int(c) for c in str(parsed) if c in "01"]
    except Exception:
        votes = [int(c) for c in response if c in "01"]

    return normalize_votes(votes, expected_length)


def _invoke_model(prompt: str) -> str:
    """Centraliza chamada ao modelo AI para ter consistência"""
    try:
        # Use the LangChain-like interface with `.invoke`
        response_text = model.invoke(prompt)

        # Token accounting (estimate) for observability
        token_counts = log_token_usage(prompt, response_text, model_type="llm")
        logger.info(
            f"Token usage (estimate): input={token_counts['input_tokens']}, output={token_counts['output_tokens']}, total={token_counts['total_tokens']}"
        )

        logger.debug(f"LLM response: {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise


# -----------------------
# Langraph agents
# -----------------------
def _checker_prompt(prompt_key: str, workloads, cluster_info=None) -> tuple[str, int]:
    """Build a checker prompt with workloads and cluster state; returns (prompt, n)."""
    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)
    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt(
        prompt_key,
        workloads_json=json.dumps(workloads_list, indent=2),
        clusters_json=json.dumps(cluster_info or [], indent=2),
    )
    return prompt, len(workloads_list)


@traceable(name="cpu_checker")
def cpu_checker(
    workloads: Union[list, "pd.DataFrame"],
    cluster_info: List[dict] | None = None,
) -> List[int]:
    """Check CPU usage of workloads and return migration votes.
    Args:
        workloads (Union[list, pd.DataFrame]): List or DataFrame of workload items.
        cluster_info (List[dict], optional): Current state of each cluster.
    Returns:
        List[int]: List of votes (0 or 1) for each workload.
    """
    prompt, n = _checker_prompt("cpu_checker", workloads, cluster_info)
    text = _invoke_model(prompt)
    return _parse_llm_response(text, n)


@traceable(name="mem_checker")
def mem_checker(
    workloads: Union[list, "pd.DataFrame"],
    cluster_info: List[dict] | None = None,
) -> List[int]:
    """Check Memory usage of workloads and return migration votes.

    Args:
        workloads (Union[list, pd.DataFrame]): List or DataFrame of workload items.
        cluster_info (List[dict], optional): Current state of each cluster.

    Returns:
        List[int]: List of votes (0 or 1) for each workload.
    """
    prompt, n = _checker_prompt("mem_checker", workloads, cluster_info)
    text = _invoke_model(prompt)
    return _parse_llm_response(text, n)


@traceable(name="pending_checker")
def pending_checker(
    workloads: Union[list, "pd.DataFrame"],
    cluster_info: List[dict] | None = None,
) -> List[int]:
    """Check Pending status of workloads and return migration votes.

    Args:
        workloads (Union[list, pd.DataFrame]): List or DataFrame of workload items.
        cluster_info (List[dict], optional): Current state of each cluster.

    Returns:
        List[int]: List of votes (0 or 1) for each workload.
    """
    prompt, n = _checker_prompt("pending_checker", workloads, cluster_info)
    text = _invoke_model(prompt)
    return _parse_llm_response(text, n)


@traceable(name="decision_agent")
def decision_agent(
    votes: Dict[str, List[int]],
    weights: Dict[str, float],
    expected_length: int,
) -> List[int]:
    """Make final migration decisions based on the checkers' votes and weights.

    Args:
        votes (Dict[str, List[int]]): Votes per active checker (cpu/mem/pending).
        weights (Dict[str, float]): Weight of each active checker.
        expected_length (int): Number of workloads (and expected decisions).

    Returns:
        List[int]: Final migration decisions (0 or 1) for each workload.
    """
    payload = {"votes": votes, "weights": weights}
    prompt = get_prompt("decision", workload_json=json.dumps(payload, indent=2))
    text = _invoke_model(prompt)
    return _parse_llm_response(text, expected_length)


@traceable(name="explainer_agent")
def explainer_agent(
    workloads: pd.DataFrame, votes: Dict[str, List[int]], final_decisions: List[int]
) -> Dict:
    """Generate explanations for migration decisions based on votes and workload characteristics.

    Args:
        workloads (pd.DataFrame): DataFrame of workload items.
        votes (Dict[str, List[int]]): Dictionary of votes from different agents.
        final_decisions (List[int]): Final migration decisions for each workload.

    Returns:
        Dict: Explanation of decisions including per-workload explanations.
    """
    workloads_list = workloads.to_dict(orient="records")
    prompt = f"""
You are a Kubernetes systems expert.

Briefly explain (in one sentence per workload) the decision to migrate (1) or maintain (0),
based on the experts' votes (CPU, Memory, Pending) and the characteristics of the workload.

Answer in JSON format:
{{
  "explanation": "General decision strategy",
  "workload_explanations": [
    {{
      "workload_id": "...",
      "decision": 1,
      "explanation": "..."
    }},
    ...
  ]
}}
Workloads:
{json.dumps(workloads_list, indent=2)}

Votes:
CPU: {votes["cpu"]}
Memory: {votes["mem"]}
Pending: {votes["pending"]}

Final decisions: {final_decisions}
"""
    try:
        text = _invoke_model(prompt)
        if not text:
            raise ValueError("Empty response from LLM")

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return parsed
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:

        logger.error("❌ Error processing explanation: %s", e)
        return {
            "explanation": f"Error generating explanation: {str(e)}",
            "workload_explanations": [
                {
                    "workload_id": w.get("workload_id", f"#{i}"),
                    "decision": (
                        final_decisions[i] if i < len(final_decisions) else "unknown"
                    ),
                    "explanation": "No explanation.",
                }
                for i, w in enumerate(workloads_list)
            ],
        }
