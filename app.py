import time

import pandas as pd
import streamlit as st

from src.generator import generate_content
from src.evaluator import evaluate_content
from src.reviser import revise_content
from src.llm_client import JUDGE_MODEL_KEY


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="GrowthPilot",
    page_icon="🚀",
    layout="wide",
)


# =========================================================
# Product Configuration
# =========================================================

MODEL_OPTIONS = {
    "Step-3.5-Flash": "step",
    "Qwen3.5-35B-A3B": "qwen",
}

MODEL_DISPLAY_NAMES = {
    "step": "Step-3.5-Flash",
    "qwen": "Qwen3.5-35B-A3B",
}

QUALITY_THRESHOLD = 9.0
CLAIM_RISK_THRESHOLD = 2


# =========================================================
# Session State
# =========================================================

DEFAULT_SESSION_STATE = {
    "v1_content": None,
    "v1_evaluation": None,
    "v1_generation_latency": None,
    "v1_evaluation_latency": None,

    "v2_content": None,
    "v2_evaluation": None,
    "v2_revision_latency": None,
    "v2_evaluation_latency": None,

    "saved_brand_info": None,
    "saved_campaign_brief": None,

    "saved_model_key": None,
    "saved_model_name": None,
}


for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# Helper Functions
# =========================================================

def reset_results():
    """
    Clear previous generation and evaluation results
    before starting a new experiment.
    """

    st.session_state.v1_content = None
    st.session_state.v1_evaluation = None
    st.session_state.v1_generation_latency = None
    st.session_state.v1_evaluation_latency = None

    st.session_state.v2_content = None
    st.session_state.v2_evaluation = None
    st.session_state.v2_revision_latency = None
    st.session_state.v2_evaluation_latency = None


def show_evaluation(
    evaluation: dict,
    title: str,
):
    """
    Display evaluation scores, issues, and suggestions.
    """

    st.subheader(title)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Score",
        f"{evaluation['overall_score']:.1f}/10",
    )

    col2.metric(
        "Factual Consistency",
        f"{evaluation['factual_consistency']}/10",
    )

    col3.metric(
        "Unsupported Claim Risk",
        f"{evaluation['unsupported_claim_risk']}/10",
        help="Lower is better.",
    )

    score_df = pd.DataFrame(
        {
            "Dimension": [
                "Brand Alignment",
                "Tone Match",
                "Selling Point Coverage",
                "Factual Consistency",
                "Unsupported Claim Risk",
            ],
            "Score": [
                evaluation["brand_alignment"],
                evaluation["tone_match"],
                evaluation["selling_point_coverage"],
                evaluation["factual_consistency"],
                evaluation["unsupported_claim_risk"],
            ],
        }
    )

    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True,
    )

    issues = evaluation.get(
        "issues",
        [],
    )

    suggestions = evaluation.get(
        "suggestions",
        [],
    )

    if issues:
        st.markdown("**Issues Identified**")

        for issue in issues:
            st.write(f"• {issue}")

    else:
        st.success(
            "No major issues were identified."
        )

    if suggestions:
        st.markdown("**Improvement Suggestions**")

        for suggestion in suggestions:
            st.write(f"• {suggestion}")


def needs_revision(
    evaluation: dict,
) -> bool:
    """
    Product-defined quality gate.

    Revision is triggered when:
    - overall score is below threshold, OR
    - unsupported claim risk is too high, OR
    - evaluator identifies concrete issues.
    """

    return (
        evaluation["overall_score"]
        < QUALITY_THRESHOLD

        or evaluation[
            "unsupported_claim_risk"
        ] > CLAIM_RISK_THRESHOLD

        or len(
            evaluation.get(
                "issues",
                [],
            )
        ) > 0
    )


def build_comparison_dataframe(
    v1: dict,
    v2: dict,
) -> pd.DataFrame:
    """
    Build V1 vs V2 evaluation comparison.

    For normal quality metrics:
    positive improvement means V2 > V1.

    For Unsupported Claim Risk:
    positive improvement means risk decreased.
    """

    rows = []

    normal_metrics = [
        (
            "Overall Score",
            "overall_score",
        ),
        (
            "Brand Alignment",
            "brand_alignment",
        ),
        (
            "Tone Match",
            "tone_match",
        ),
        (
            "Selling Point Coverage",
            "selling_point_coverage",
        ),
        (
            "Factual Consistency",
            "factual_consistency",
        ),
    ]

    for display_name, key in normal_metrics:

        v1_value = v1[key]
        v2_value = v2[key]

        rows.append(
            {
                "Metric": display_name,
                "V1": v1_value,
                "V2": v2_value,
                "Improvement": round(
                    v2_value - v1_value,
                    1,
                ),
            }
        )

    v1_risk = v1[
        "unsupported_claim_risk"
    ]

    v2_risk = v2[
        "unsupported_claim_risk"
    ]

    rows.append(
        {
            "Metric":
                "Unsupported Claim Risk",

            "V1":
                v1_risk,

            "V2":
                v2_risk,

            "Improvement":
                round(
                    v1_risk - v2_risk,
                    1,
                ),
        }
    )

    return pd.DataFrame(rows)


# =========================================================
# Header
# =========================================================

st.title("🚀 GrowthPilot")

st.caption(
    "AI Marketing Content Generation, "
    "Evaluation and Revision System"
)

st.markdown(
    """
GrowthPilot generates marketing content,
evaluates quality and factual safety,
then automatically supports revision
when the content does not meet the
product quality threshold.
"""
)


# =========================================================
# Model Selection
# =========================================================

st.header("1. Model Configuration")

selected_model_name = st.selectbox(
    "Candidate Model",
    options=list(
        MODEL_OPTIONS.keys()
    ),
)

selected_model_key = MODEL_OPTIONS[
    selected_model_name
]


judge_display_name = MODEL_DISPLAY_NAMES.get(
    JUDGE_MODEL_KEY,
    JUDGE_MODEL_KEY,
)


st.info(
    f"Generator / Reviser: "
    f"{selected_model_name}  |  "
    f"Fixed Judge: "
    f"{judge_display_name}"
)


# =========================================================
# Inputs
# =========================================================

st.header("2. Campaign Input")


default_brand_info = """
Brand: LumiSkin

Product:
Barrier Repair Moisturizer

Key selling points:
- Contains ceramides
- Contains hyaluronic acid
- Designed for dry and sensitive skin
- Lightweight texture
- Non-greasy
- Provides up to 24 hours of hydration

Brand tone:
Professional, calm, trustworthy and modern.

Restrictions:
- Do not claim to treat or cure diseases
- Do not invent clinical studies
- Do not guarantee medical results
"""


default_campaign_brief = """
Platform:
Xiaohongshu

Target audience:
Women aged 20-30

Objective:
Introduce the moisturizer for daily hydration.

Style:
Natural lifestyle recommendation.

Length:
Approximately 150 Chinese characters.
"""


brand_info = st.text_area(
    "Brand Information",
    value=default_brand_info.strip(),
    height=300,
)


campaign_brief = st.text_area(
    "Campaign Brief",
    value=default_campaign_brief.strip(),
    height=220,
)


# =========================================================
# Generate V1
# =========================================================

st.header("3. Generate & Evaluate V1")


if st.button(
    "Generate V1",
    type="primary",
    use_container_width=True,
):

    if not brand_info.strip():
        st.error(
            "Please provide Brand Information."
        )

    elif not campaign_brief.strip():
        st.error(
            "Please provide a Campaign Brief."
        )

    else:

        reset_results()

        st.session_state.saved_brand_info = (
            brand_info
        )

        st.session_state.saved_campaign_brief = (
            campaign_brief
        )

        st.session_state.saved_model_key = (
            selected_model_key
        )

        st.session_state.saved_model_name = (
            selected_model_name
        )


        # -------------------------------------------------
        # Generate V1
        # -------------------------------------------------

        try:

            with st.spinner(
                f"Generating V1 with "
                f"{selected_model_name}..."
            ):

                start_time = (
                    time.perf_counter()
                )

                v1_content = generate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    model_key=selected_model_key,
                )

                generation_latency = (
                    time.perf_counter()
                    - start_time
                )


            st.session_state.v1_content = (
                v1_content
            )

            st.session_state.v1_generation_latency = (
                round(
                    generation_latency,
                    2,
                )
            )


            # -------------------------------------------------
            # Evaluate V1
            # -------------------------------------------------

            with st.spinner(
                f"Evaluating V1 with fixed judge "
                f"{judge_display_name}..."
            ):

                start_time = (
                    time.perf_counter()
                )

                v1_evaluation = evaluate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    generated_content=v1_content,
                )

                evaluation_latency = (
                    time.perf_counter()
                    - start_time
                )


            st.session_state.v1_evaluation = (
                v1_evaluation
            )

            st.session_state.v1_evaluation_latency = (
                round(
                    evaluation_latency,
                    2,
                )
            )


        except Exception as e:

            st.error(
                f"Generation or evaluation failed: {e}"
            )


# =========================================================
# Display V1
# =========================================================

if st.session_state.v1_content:

    st.subheader("V1 Generated Content")

    st.write(
        st.session_state.v1_content
    )


    latency_col1, latency_col2 = (
        st.columns(2)
    )


    latency_col1.metric(
        "Generation Latency",
        f"{st.session_state.v1_generation_latency:.2f}s",
    )

    latency_col2.metric(
        "Evaluation Latency",
        f"{st.session_state.v1_evaluation_latency:.2f}s",
    )


if st.session_state.v1_evaluation:

    show_evaluation(
        st.session_state.v1_evaluation,
        "V1 Evaluation",
    )


# =========================================================
# Quality Gate
# =========================================================

if st.session_state.v1_evaluation:

    revision_required = needs_revision(
        st.session_state.v1_evaluation
    )


    st.header("4. Quality Gate")


    if revision_required:

        st.warning(
            "V1 does not fully meet the "
            "GrowthPilot quality threshold. "
            "AI revision is recommended."
        )


        if st.button(
            "Revise with AI",
            use_container_width=True,
        ):

            try:

                # -----------------------------------------
                # Revise V1 → V2
                # -----------------------------------------

                with st.spinner(
                    f"Revising with "
                    f"{st.session_state.saved_model_name}..."
                ):

                    start_time = (
                        time.perf_counter()
                    )

                    v2_content = revise_content(
                        brand_info=(
                            st.session_state
                            .saved_brand_info
                        ),

                        campaign_brief=(
                            st.session_state
                            .saved_campaign_brief
                        ),

                        original_content=(
                            st.session_state
                            .v1_content
                        ),

                        evaluation=(
                            st.session_state
                            .v1_evaluation
                        ),

                        model_key=(
                            st.session_state
                            .saved_model_key
                        ),
                    )

                    revision_latency = (
                        time.perf_counter()
                        - start_time
                    )


                st.session_state.v2_content = (
                    v2_content
                )

                st.session_state.v2_revision_latency = (
                    round(
                        revision_latency,
                        2,
                    )
                )


                # -----------------------------------------
                # Evaluate V2
                # -----------------------------------------

                with st.spinner(
                    f"Evaluating V2 with fixed judge "
                    f"{judge_display_name}..."
                ):

                    start_time = (
                        time.perf_counter()
                    )

                    v2_evaluation = evaluate_content(
                        brand_info=(
                            st.session_state
                            .saved_brand_info
                        ),

                        campaign_brief=(
                            st.session_state
                            .saved_campaign_brief
                        ),

                        generated_content=(
                            v2_content
                        ),
                    )

                    v2_evaluation_latency = (
                        time.perf_counter()
                        - start_time
                    )


                st.session_state.v2_evaluation = (
                    v2_evaluation
                )

                st.session_state.v2_evaluation_latency = (
                    round(
                        v2_evaluation_latency,
                        2,
                    )
                )


            except Exception as e:

                st.error(
                    f"Revision or evaluation failed: {e}"
                )


    else:

        st.success(
            "V1 meets the GrowthPilot quality threshold. "
            "Revision is not required."
        )


# =========================================================
# Display V2
# =========================================================

if st.session_state.v2_content:

    st.header("5. Revised Content")

    st.subheader("V2 Generated Content")

    st.write(
        st.session_state.v2_content
    )


    latency_col1, latency_col2 = (
        st.columns(2)
    )


    latency_col1.metric(
        "Revision Latency",
        f"{st.session_state.v2_revision_latency:.2f}s",
    )

    latency_col2.metric(
        "V2 Evaluation Latency",
        f"{st.session_state.v2_evaluation_latency:.2f}s",
    )


if st.session_state.v2_evaluation:

    show_evaluation(
        st.session_state.v2_evaluation,
        "V2 Evaluation",
    )


# =========================================================
# V1 vs V2 Comparison
# =========================================================

if (
    st.session_state.v1_evaluation
    and st.session_state.v2_evaluation
):

    st.header("6. V1 vs V2 Comparison")


    comparison_df = (
        build_comparison_dataframe(
            st.session_state.v1_evaluation,
            st.session_state.v2_evaluation,
        )
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )


    v1_score = (
        st.session_state
        .v1_evaluation[
            "overall_score"
        ]
    )

    v2_score = (
        st.session_state
        .v2_evaluation[
            "overall_score"
        ]
    )


    v1_risk = (
        st.session_state
        .v1_evaluation[
            "unsupported_claim_risk"
        ]
    )

    v2_risk = (
        st.session_state
        .v2_evaluation[
            "unsupported_claim_risk"
        ]
    )


    score_change = round(
        v2_score - v1_score,
        1,
    )

    risk_reduction = (
        v1_risk - v2_risk
    )


    result_col1, result_col2 = (
        st.columns(2)
    )


    result_col1.metric(
        "Overall Score Change",
        f"{score_change:+.1f}",
    )


    result_col2.metric(
        "Claim Risk Reduction",
        f"{risk_reduction:+.1f}",
        help=(
            "Positive means unsupported "
            "claim risk decreased."
        ),
    )


    if (
        score_change > 0
        and risk_reduction >= 0
    ):

        st.success(
            "Revision improved the overall quality "
            "without increasing unsupported claim risk."
        )


    elif (
        score_change > 0
        and risk_reduction < 0
    ):

        st.warning(
            "Revision improved the overall score, "
            "but unsupported claim risk increased."
        )


    elif score_change == 0:

        st.info(
            "Revision did not materially change "
            "the overall score."
        )


    else:

        st.warning(
            "Revision reduced the overall score. "
            "The revision strategy may require improvement."
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "GrowthPilot MVP · "
    "Model Selection + AI Evaluation + "
    "Automated Revision"
)