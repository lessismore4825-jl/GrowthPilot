import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def revise_content(
    brand_info: str,
    campaign_brief: str,
    original_content: str,
    evaluation: dict
) -> str:

    issues = "\n".join(
        f"- {issue}"
        for issue in evaluation["issues"]
    )

    suggestions = "\n".join(
        f"- {suggestion}"
        for suggestion in evaluation["suggestions"]
    )

    prompt = f"""
You are an AI marketing content editor.

Your task is to revise the original marketing content based on the
evaluation results.

BRAND INFORMATION:
{brand_info}

CAMPAIGN BRIEF:
{campaign_brief}

ORIGINAL CONTENT:
{original_content}

IDENTIFIED ISSUES:
{issues}

IMPROVEMENT SUGGESTIONS:
{suggestions}

Requirements:

1. Fix all factual inconsistencies.
2. Remove unsupported or exaggerated claims.
3. Improve alignment with the brand identity.
4. Improve tone for the requested platform.
5. Include important selling points that are missing.
6. Preserve useful parts of the original content.
7. Do not invent any new product facts.

Return ONLY the revised marketing content.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text.strip()