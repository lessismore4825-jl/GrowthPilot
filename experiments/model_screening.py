import os
import time

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Environment
# =========================================================

load_dotenv()

api_key = os.getenv("MODELSCOPE_API_KEY")

if not api_key:
    raise ValueError(
        "MODELSCOPE_API_KEY was not found in .env"
    )


# =========================================================
# ModelScope Client
# =========================================================

client = OpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key=api_key,
    max_retries=0,
)


# =========================================================
# Screening Candidates
# =========================================================

MODELS = {
    "Step-3.5-Flash":
        "stepfun-ai/Step-3.5-Flash",

    "Qwen3.5-35B-A3B":
        "Qwen/Qwen3.5-35B-A3B",

    "GLM-4.7-Flash":
        "ZhipuAI/GLM-4.7-Flash",

    "Kimi-K2.5":
        "moonshotai/Kimi-K2.5",
}


# =========================================================
# Shared Screening Prompt
# =========================================================

PROMPT = """
You are an AI marketing assistant.

Brand:
BeanTrail

Product:
Cold Brew Coffee Concentrate

Verified facts:
- 100% Arabica coffee
- No added sugar
- Approximately 8 servings per bottle
- Can be mixed with water or milk

Restrictions:
- Do not invent product facts
- Do not invent prices
- Do not make health claims

Task:
Write a short Xiaohongshu marketing post
for university students.

Return only the final marketing content.
"""


# =========================================================
# Model Screening
# =========================================================

def screen_model(
    model_name: str,
    model_id: str,
) -> dict:

    print("\n" + "=" * 70)
    print(f"Testing: {model_name}")
    print(f"Model ID: {model_id}")
    print("Sending request...")

    start_time = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                }
            ],
            temperature=0.3,
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        choices = getattr(
            response,
            "choices",
            None,
        )

        if not choices:

            print("\n❌ Failed")
            print(
                "Reason: API returned no choices."
            )

            return {
                "model": model_name,
                "model_id": model_id,
                "status": "FAILED",
                "latency": round(latency, 2),
                "reason": "No choices returned",
            }

        message = choices[0].message

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:

            print("\n❌ Failed")
            print(
                "Reason: API returned empty content."
            )

            return {
                "model": model_name,
                "model_id": model_id,
                "status": "FAILED",
                "latency": round(latency, 2),
                "reason": "Empty content",
            }


        print("\n✅ Success")

        print(
            f"Latency: "
            f"{latency:.2f}s"
        )

        print("\nOutput:\n")

        print(
            content.strip()
        )


        return {
            "model": model_name,
            "model_id": model_id,
            "status": "SUCCESS",
            "latency": round(latency, 2),
            "reason": "",
        }


    except Exception as e:

        latency = (
            time.perf_counter()
            - start_time
        )

        print("\n❌ Failed")

        print(
            f"Latency before failure: "
            f"{latency:.2f}s"
        )

        print(
            f"Error: {e}"
        )


        return {
            "model": model_name,
            "model_id": model_id,
            "status": "FAILED",
            "latency": round(latency, 2),
            "reason": str(e),
        }


# =========================================================
# Run Screening
# =========================================================

if __name__ == "__main__":

    print(
        "\nGrowthPilot Model Screening"
    )

    print(
        "=" * 70
    )

    results = []

    for model_name, model_id in MODELS.items():

        result = screen_model(
            model_name=model_name,
            model_id=model_id,
        )

        results.append(
            result
        )


    # =====================================================
    # Summary
    # =====================================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "SCREENING SUMMARY"
    )

    print(
        "=" * 70
    )


    for result in results:

        print(
            f"\n{result['model']}"
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Latency: "
            f"{result['latency']:.2f}s"
        )

        if result["reason"]:

            print(
                f"Reason: "
                f"{result['reason']}"
            )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "Model screening finished."
    )