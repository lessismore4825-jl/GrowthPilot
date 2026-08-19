import json
import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def evaluate_content(
    brand_info: str,
    campaign_brief: str,
    generated_content: str
) -> dict:

    prompt = f"""
You are an AI marketing content evaluator.

Evaluate the generated marketing content strictly based on the provided
brand information and campaign brief.

BRAND INFORMATION:
{brand_info}

CAMPAIGN BRIEF:
{campaign_brief}

GENERATED CONTENT:
{generated_content}

Evaluate the content using the following dimensions.

1. brand_alignment
How well the content matches the brand positioning and identity.

2. tone_match
How well the writing style matches the requested tone and platform.

3. selling_point_coverage
How well the important product selling points are included.

4. factual_consistency
Whether every factual statement is supported by the brand information.

5. unsupported_claim_risk
Risk that the content contains invented, exaggerated, medical,
or unsupported claims.

Scoring rules:

For dimensions 1-4:
10 = excellent
1 = very poor

For unsupported_claim_risk:
1 = very low risk
10 = very high risk

Be strict and evidence-based.
Do not give high scores simply because the content is fluent or well-written.

Return ONLY valid JSON using exactly this structure:

{{
    "brand_alignment": 0,
    "tone_match": 0,
    "selling_point_coverage": 0,
    "factual_consistency": 0,
    "unsupported_claim_risk": 0,
    "issues": [
        "issue 1",
        "issue 2"
    ],
    "suggestions": [
        "suggestion 1",
        "suggestion 2"
    ]
}}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    response_text = interaction.output_text.strip()

    response_text = response_text.replace("```json", "")
    response_text = response_text.replace("```", "").strip()

    evaluation = json.loads(response_text)

    # Convert claim risk into a positive safety score.
    # Risk 1/10 -> Safety 10/10
    # Risk 10/10 -> Safety 1/10
    claim_safety = 11 - evaluation["unsupported_claim_risk"]

    # Overall score is calculated by product logic,
    # rather than allowing the LLM to decide the final score.
    overall_score = (
        evaluation["brand_alignment"] * 0.20
        + evaluation["tone_match"] * 0.15
        + evaluation["selling_point_coverage"] * 0.20
        + evaluation["factual_consistency"] * 0.30
        + claim_safety * 0.15
    )

    evaluation["overall_score"] = round(overall_score, 1)

    return evaluation