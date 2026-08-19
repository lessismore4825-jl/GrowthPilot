import os
import time

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()


# =========================================================
# Model Configuration
# =========================================================

MODELS = {
    "step": "stepfun-ai/Step-3.5-Flash",
    "qwen": "Qwen/Qwen3.5-35B-A3B",
}


DEFAULT_MODEL_KEY = os.getenv(
    "GENERATOR_MODEL",
    "step",
).lower()


JUDGE_MODEL_KEY = os.getenv(
    "JUDGE_MODEL",
    "step",
).lower()


# =========================================================
# API Configuration
# =========================================================

api_key = os.getenv(
    "MODELSCOPE_API_KEY"
)


if not api_key:

    raise ValueError(
        "MODELSCOPE_API_KEY was not found in .env"
    )


# =========================================================
# ModelScope Client
# =========================================================
#
# No timeout:
# We allow ModelScope to finish naturally.
#
# max_retries=0:
# GrowthPilot handles important Judge retries itself.
# This prevents hidden SDK retries from distorting
# benchmark latency.
# =========================================================

client = OpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key=api_key,
    max_retries=0,
)


# =========================================================
# Model Helper
# =========================================================

def get_model_id(
    model_key: str,
) -> str:
    """
    Convert a short model key into
    the full ModelScope model ID.

    Example:

    step
        ->
    stepfun-ai/Step-3.5-Flash
    """

    model_key = model_key.lower()


    if model_key not in MODELS:

        raise ValueError(
            f"Unsupported model: {model_key}. "
            f"Available models: {list(MODELS.keys())}"
        )


    return MODELS[
        model_key
    ]


# =========================================================
# Unified Streaming LLM Call
# =========================================================

def generate_text(
    prompt: str,
    model_key: str | None = None,
    temperature: float = 0.3,
) -> str:
    """
    Send a prompt to ModelScope using streaming mode.

    The streaming chunks are accumulated internally
    and returned as one final string.

    GrowthPilot modules therefore still receive a
    normal string and do not need to know that the
    API transport uses streaming.
    """

    selected_model_key = (
        model_key
        or DEFAULT_MODEL_KEY
    ).lower()


    model_id = get_model_id(
        selected_model_key
    )


    # =====================================================
    # Start Streaming Request
    # =====================================================

    stream = client.chat.completions.create(
        model=model_id,

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],

        temperature=temperature,

        stream=True,
    )


    # =====================================================
    # Accumulate Generated Content
    # =====================================================

    content_parts = []


    for chunk in stream:

        # Some providers may send metadata chunks
        # without choices.
        choices = getattr(
            chunk,
            "choices",
            None,
        )


        if not choices:
            continue


        delta = getattr(
            choices[0],
            "delta",
            None,
        )


        if delta is None:
            continue


        content = getattr(
            delta,
            "content",
            None,
        )


        if content:

            content_parts.append(
                content
            )


    # =====================================================
    # Build Final Text
    # =====================================================

    final_content = "".join(
        content_parts
    ).strip()


    if not final_content:

        raise RuntimeError(
            f"{model_id} returned empty content."
        )


    return final_content


# =========================================================
# Timed LLM Call
# =========================================================

def generate_text_with_metrics(
    prompt: str,
    model_key: str,
    temperature: float = 0.3,
) -> dict:
    """
    Generate text and record total API latency.

    Used by GrowthPilot experiments and benchmarks.
    """

    start_time = (
        time.perf_counter()
    )


    content = generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=temperature,
    )


    total_latency = (
        time.perf_counter()
        - start_time
    )


    return {
        "content":
            content,

        "model_key":
            model_key,

        "model_id":
            get_model_id(
                model_key
            ),

        "total_latency":
            round(
                total_latency,
                2,
            ),
    }