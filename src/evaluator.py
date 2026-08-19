import ast
import json
import re

from src.llm_client import (
    JUDGE_MODEL_KEY,
    generate_text,
)


# =========================================================
# Required Evaluation Fields
# =========================================================

REQUIRED_SCORE_FIELDS = [
    "brand_alignment",
    "tone_match",
    "selling_point_coverage",
    "factual_consistency",
    "unsupported_claim_risk",
]


# =========================================================
# Response Cleaning
# =========================================================

def clean_response_text(
    response_text: str,
) -> str:
    """
    Clean common formatting artifacts from an LLM response.

    Handles:
    - UTF-8 BOM
    - zero-width characters
    - Markdown code fences
    - <think>...</think> reasoning blocks
    - surrounding whitespace
    """

    if not isinstance(
        response_text,
        str,
    ):
        raise ValueError(
            "Evaluator response must be a string."
        )

    cleaned = response_text

    # -----------------------------------------------------
    # Remove BOM and invisible characters
    # -----------------------------------------------------

    cleaned = (
        cleaned
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

    # -----------------------------------------------------
    # Remove reasoning / think blocks
    # -----------------------------------------------------

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # -----------------------------------------------------
    # Remove Markdown fences
    # -----------------------------------------------------

    cleaned = re.sub(
        r"```(?:json|JSON|python|Python)?",
        "",
        cleaned,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    )

    return cleaned.strip()


# =========================================================
# Extract JSON Object
# =========================================================

def extract_json_object(
    text: str,
) -> str:
    """
    Extract the first outer JSON-like object.

    Example:

    Here is the result:
    {
        "brand_alignment": 9
    }

    becomes:

    {
        "brand_alignment": 9
    }
    """

    start_index = text.find(
        "{"
    )

    end_index = text.rfind(
        "}"
    )

    if (
        start_index == -1
        or end_index == -1
        or end_index <= start_index
    ):
        raise ValueError(
            "No JSON object was found "
            "in evaluator response."
        )

    return text[
        start_index:end_index + 1
    ].strip()


# =========================================================
# Repair Common JSON Formatting Problems
# =========================================================

def repair_json_text(
    text: str,
) -> str:
    """
    Repair a small set of common LLM JSON mistakes.

    We intentionally keep this conservative.
    """

    repaired = text

    # -----------------------------------------------------
    # Normalize smart quotation marks
    # -----------------------------------------------------

    repaired = (
        repaired
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # -----------------------------------------------------
    # Remove trailing commas
    #
    # Example:
    #
    # {
    #   "score": 9,
    # }
    #
    # or:
    #
    # [
    #   "issue",
    # ]
    # -----------------------------------------------------

    repaired = re.sub(
        r",\s*([}\]])",
        r"\1",
        repaired,
    )

    return repaired.strip()


# =========================================================
# JSON Parser
# =========================================================

def parse_evaluation_json(
    response_text: str,
) -> dict:
    """
    Parse evaluator output robustly.

    Parsing order:

    1. Standard JSON
    2. Extract JSON object
    3. Repair common JSON issues
    4. Python dict fallback via ast.literal_eval

    If parsing still fails, include a preview of
    the raw model response in the error message.
    """

    cleaned_text = clean_response_text(
        response_text
    )

    if not cleaned_text:

        raise ValueError(
            "Evaluator returned an empty response."
        )


    # =====================================================
    # Attempt 1:
    # Standard JSON on complete cleaned response
    # =====================================================

    try:

        parsed = json.loads(
            cleaned_text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass


    # =====================================================
    # Extract JSON-looking object
    # =====================================================

    try:

        json_text = extract_json_object(
            cleaned_text
        )

    except ValueError as error:

        preview = repr(
            cleaned_text[:500]
        )

        raise ValueError(
            "Evaluator did not return a JSON object. "
            f"Raw response preview: {preview}"
        ) from error


    # =====================================================
    # Attempt 2:
    # Standard JSON on extracted object
    # =====================================================

    try:

        parsed = json.loads(
            json_text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass


    # =====================================================
    # Repair common formatting problems
    # =====================================================

    repaired_text = repair_json_text(
        json_text
    )


    # =====================================================
    # Attempt 3:
    # JSON after repair
    # =====================================================

    try:

        parsed = json.loads(
            repaired_text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass


    # =====================================================
    # Attempt 4:
    # Python dictionary fallback
    #
    # Handles cases such as:
    #
    # {
    #   'brand_alignment': 9,
    #   'issues': []
    # }
    #
    # This is NOT valid JSON but is still
    # structurally understandable.
    # =====================================================

    try:

        parsed = ast.literal_eval(
            repaired_text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except (
        ValueError,
        SyntaxError,
    ):
        pass


    # =====================================================
    # Final Failure
    # =====================================================

    preview = repr(
        cleaned_text[:500]
    )

    raise ValueError(
        "Evaluator returned malformed JSON "
        "that could not be repaired. "
        f"Raw response preview: {preview}"
    )


# =========================================================
# Score Normalization
# =========================================================

def normalize_score(
    value,
    field_name: str,
):
    """
    Normalize score values.

    Accepts:
    - 9
    - 9.0
    - "9"
    - "9.0"

    Rejects values outside 1-10.
    """

    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{field_name} must be numeric, "
            "not boolean."
        )


    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        try:

            value = float(
                value
            )

        except ValueError as error:

            raise ValueError(
                f"{field_name} must be numeric. "
                f"Received: {value}"
            ) from error


    if not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{field_name} must be numeric."
        )


    if not 1 <= value <= 10:
        raise ValueError(
            f"{field_name} must be "
            f"between 1 and 10. "
            f"Received: {value}"
        )


    # Convert 9.0 → 9
    if float(
        value
    ).is_integer():

        return int(
            value
        )


    return round(
        float(value),
        2,
    )


# =========================================================
# List Normalization
# =========================================================

def normalize_text_list(
    value,
) -> list:
    """
    Normalize issues / suggestions into
    clean lists of strings.
    """

    if value is None:
        return []


    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if not value:
            return []

        return [
            value
        ]


    if not isinstance(
        value,
        list,
    ):

        return [
            str(value)
        ]


    result = []


    for item in value:

        if item is None:
            continue

        text = str(
            item
        ).strip()

        if text:
            result.append(
                text
            )


    return result


# =========================================================
# Evaluation Validation
# =========================================================

def validate_evaluation(
    evaluation: dict,
) -> dict:
    """
    Validate and normalize evaluator output
    before calculating the product score.
    """

    if not isinstance(
        evaluation,
        dict,
    ):
        raise ValueError(
            "Evaluator output must be a dictionary."
        )


    # =====================================================
    # Validate Scores
    # =====================================================

    for field in REQUIRED_SCORE_FIELDS:

        if field not in evaluation:

            raise ValueError(
                f"Missing evaluation field: "
                f"{field}"
            )


        evaluation[
            field
        ] = normalize_score(
            evaluation[field],
            field,
        )


    # =====================================================
    # Normalize Issues
    # =====================================================

    evaluation[
        "issues"
    ] = normalize_text_list(
        evaluation.get(
            "issues",
            [],
        )
    )


    # =====================================================
    # Normalize Suggestions
    # =====================================================

    evaluation[
        "suggestions"
    ] = normalize_text_list(
        evaluation.get(
            "suggestions",
            [],
        )
    )


    return evaluation


# =========================================================
# Product-defined Overall Score
# =========================================================

def calculate_overall_score(
    evaluation: dict,
) -> float:
    """
    Calculate GrowthPilot's product-defined
    Overall Score.

    Weights:

    Brand Alignment            20%
    Tone Match                 15%
    Selling Point Coverage     20%
    Factual Consistency        30%
    Claim Safety               15%

    Claim Safety =
    11 - Unsupported Claim Risk
    """

    claim_safety = (
        11
        - evaluation[
            "unsupported_claim_risk"
        ]
    )


    overall_score = (
        evaluation[
            "brand_alignment"
        ] * 0.20

        + evaluation[
            "tone_match"
        ] * 0.15

        + evaluation[
            "selling_point_coverage"
        ] * 0.20

        + evaluation[
            "factual_consistency"
        ] * 0.30

        + claim_safety * 0.15
    )


    return round(
        overall_score,
        1,
    )


# =========================================================
# Main Evaluator
# =========================================================

def evaluate_content(
    brand_info: str,
    campaign_brief: str,
    generated_content: str,
    judge_model_key: str | None = None,
) -> dict:
    """
    Evaluate generated marketing content.

    Online Demo:
        Uses the default Judge from .env.

    Benchmark:
        judge_model_key can explicitly select
        Step or Qwen for Cross-Judge Evaluation.
    """

    selected_judge = (
        judge_model_key
        or JUDGE_MODEL_KEY
    )


    # =====================================================
    # Evaluation Prompt
    # =====================================================

    prompt = f"""
You are a strict AI marketing content evaluator.

Your task is to evaluate GENERATED CONTENT
using ONLY the evidence contained in:

1. BRAND INFORMATION
2. CAMPAIGN BRIEF

Do not reward fluent, creative, persuasive,
or attractive writing when factual, safety,
brand, or campaign requirements are violated.


BRAND INFORMATION:

{brand_info}


CAMPAIGN BRIEF:

{campaign_brief}


GENERATED CONTENT:

{generated_content}


EVALUATION DIMENSIONS:


1. brand_alignment

Evaluate how well the content follows:
- brand positioning
- brand identity
- brand tone
- brand restrictions


2. tone_match

Evaluate how well the content follows:
- requested platform
- target audience
- requested tone
- requested content style


3. selling_point_coverage

Evaluate whether the important selling points
explicitly provided in BRAND INFORMATION
are appropriately included.

Do not reward invented selling points.


4. factual_consistency

Evaluate whether factual statements are
fully supported by BRAND INFORMATION.

Penalize:
- invented facts
- altered numbers
- altered qualifiers
- invented ingredients
- invented specifications
- invented features
- invented user experiences
- unsupported product functions

Important:

Do not allow the model to strengthen factual claims.

Examples:

"up to 24 hours"
must NOT become
"24 hours guaranteed"

"approximately 8 servings"
must NOT become
"exactly 8 servings"

"may help"
must NOT become
"will help"


5. unsupported_claim_risk

Evaluate the risk of:
- invented claims
- exaggerated claims
- medical claims
- health claims
- guaranteed results
- unsupported comparisons
- unsupported performance claims
- invented personal experiences
- invented clinical evidence


SCORING RULES:


For:

brand_alignment
tone_match
selling_point_coverage
factual_consistency

use:

10 = excellent
9 = very strong
8 = good with minor issues
7 = acceptable but noticeable issues
5-6 = significant problems
3-4 = poor
1-2 = severe failure


For unsupported_claim_risk:

1 = very low risk
2 = low risk
3-4 = moderate risk
5-6 = meaningful risk
7-8 = high risk
9-10 = severe risk


IMPORTANT EVALUATION RULES:

- Be strict.
- Be evidence-based.
- Do not assume facts that were not provided.
- Brand restrictions override conflicting
  campaign requests.
- Fluency does not compensate for factual errors.
- Creativity does not justify invented facts.
- Platform style does not justify inventing
  personal experiences.
- Identify concrete issues rather than
  vague criticism.
- Do not invent problems that are not present.


OUTPUT FORMAT RULES:

Your response MUST contain exactly ONE JSON object.

The FIRST character of your response MUST be:

{{

The LAST character of your response MUST be:

}}

Do NOT output:
- Markdown
- ```json
- code fences
- explanations
- introductions
- conclusions
- reasoning
- comments
- XML
- <think> tags
- text before the JSON
- text after the JSON

Use ONLY valid JSON syntax.

JSON requirements:

- Use double quotes for ALL keys.
- Use double quotes for ALL string values.
- Do not use single quotes.
- Do not use trailing commas.
- Do not use NaN.
- Do not use Infinity.
- Do not include comments.
- All five scores must be numbers from 1 to 10.
- "issues" must be a JSON array of strings.
- "suggestions" must be a JSON array of strings.

Return EXACTLY this structure:

{{
    "brand_alignment": 1,
    "tone_match": 1,
    "selling_point_coverage": 1,
    "factual_consistency": 1,
    "unsupported_claim_risk": 1,
    "issues": [
        "specific issue"
    ],
    "suggestions": [
        "specific improvement suggestion"
    ]
}}
"""


    # =====================================================
    # Call Judge Model
    # =====================================================

    response_text = generate_text(
        prompt=prompt,
        model_key=selected_judge,
        temperature=0.0,
    )


    # =====================================================
    # Parse Model Response
    # =====================================================

    evaluation = parse_evaluation_json(
        response_text
    )


    # =====================================================
    # Validate + Normalize
    # =====================================================

    evaluation = validate_evaluation(
        evaluation
    )


    # =====================================================
    # Product-defined Score
    # =====================================================

    evaluation[
        "overall_score"
    ] = calculate_overall_score(
        evaluation
    )


    # =====================================================
    # Record Actual Judge
    # =====================================================

    evaluation[
        "judge_model"
    ] = selected_judge


    return evaluation