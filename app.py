import time

import pandas as pd
import streamlit as st

from src.generator import generate_content
from src.evaluator import evaluate_content
from src.reviser import (
    fix_compliance_issues,
    optimize_quality,
)
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
# Model Configuration
# =========================================================

MODEL_OPTIONS = {
    "Step-3.5-Flash": "step",
    "Qwen3.5-35B-A3B": "qwen",
}

MODEL_DISPLAY_NAMES = {
    "step": "Step-3.5-Flash",
    "qwen": "Qwen3.5-35B-A3B",
}


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
    "v2_mode": None,

    "saved_brand_info": None,
    "saved_campaign_brief": None,
    "saved_policy_context": None,
    "saved_model_key": None,
    "saved_model_name": None,
}


for key, value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# =========================================================
# Helpers
# =========================================================

def reset_results():

    for key in [
        "v1_content",
        "v1_evaluation",
        "v1_generation_latency",
        "v1_evaluation_latency",
        "v2_content",
        "v2_evaluation",
        "v2_revision_latency",
        "v2_evaluation_latency",
        "v2_mode",
    ]:

        st.session_state[
            key
        ] = None


def show_evaluation(
    evaluation: dict,
    title: str,
):

    st.subheader(
        title
    )

    col1, col2, col3 = st.columns(
        3
    )

    col1.metric(
        "Heuristic Composite Score",
        f"{evaluation['heuristic_composite_score']:.1f}/10",
        help=(
            "Diagnostic comparison signal only. "
            "This is not a calibrated pass/fail threshold."
        ),
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
                evaluation[
                    "brand_alignment"
                ],

                evaluation[
                    "tone_match"
                ],

                evaluation[
                    "selling_point_coverage"
                ],

                evaluation[
                    "factual_consistency"
                ],

                evaluation[
                    "unsupported_claim_risk"
                ],
            ],
        }
    )


    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # Compliance Findings
    # =====================================================

    compliance_findings = evaluation.get(
        "compliance_findings",
        [],
    )


    st.markdown(
        "### Compliance Review"
    )


    if compliance_findings:

        st.error(
            f"{len(compliance_findings)} blocking "
            "compliance issue(s) detected."
        )


        for index, finding in enumerate(
            compliance_findings,
            start=1,
        ):

            with st.expander(
                f"Blocking Issue {index}",
                expanded=True,
            ):

                st.markdown(
                    "**Problematic Content**"
                )

                st.write(
                    finding[
                        "evidence"
                    ]
                )


                st.markdown(
                    "**Policy Source**"
                )

                st.write(
                    finding[
                        "policy_source"
                    ]
                )


                st.markdown(
                    "**Policy Basis**"
                )

                st.write(
                    finding[
                        "policy_basis"
                    ]
                )


                st.markdown(
                    "**Why It Conflicts**"
                )

                st.write(
                    finding[
                        "reason"
                    ]
                )


                st.markdown(
                    "**Required Action**"
                )

                st.write(
                    finding[
                        "required_action"
                    ]
                )


    else:

        st.success(
            "No blocking compliance issue was "
            "detected by the current evaluation system."
        )

        st.caption(
            "This does not constitute legal approval "
            "or a guarantee that the content is ready "
            "for publication."
        )


    # =====================================================
    # Advisory Findings
    # =====================================================

    advisory_findings = evaluation.get(
        "advisory_findings",
        [],
    )


    st.markdown(
        "### Quality Advisory"
    )


    if advisory_findings:

        st.info(
            f"{len(advisory_findings)} non-blocking "
            "improvement suggestion(s) found."
        )


        for finding in advisory_findings:

            area = finding.get(
                "area",
                "General",
            )

            reason = finding.get(
                "reason",
                "",
            )

            suggestion = finding.get(
                "suggestion",
                "",
            )

            st.markdown(
                f"**{area}**"
            )

            st.write(
                reason
            )

            if suggestion:

                st.caption(
                    f"Suggestion: {suggestion}"
                )


    else:

        st.write(
            "No major advisory issue detected."
        )


    # =====================================================
    # Human Review Notes
    # =====================================================

    review_notes = evaluation.get(
        "review_notes",
        [],
    )


    if review_notes:

        st.markdown(
            "### Human Review Notes"
        )

        for note in review_notes:

            st.write(
                f"• {note}"
            )


# =========================================================
# Header
# =========================================================

st.title(
    "🚀 GrowthPilot"
)

st.caption(
    "Policy-Grounded AI Marketing "
    "Content Governance"
)

st.markdown(
    """
GrowthPilot separates **blocking compliance issues**
from **non-blocking quality suggestions**.

Rules determine what must be fixed.
AI suggests what could be improved.
Humans retain the final publishing decision.
"""
)


# =========================================================
# Model Selection
# =========================================================

st.header(
    "1. Model Configuration"
)


selected_model_name = st.selectbox(
    "Candidate Model",
    options=list(
        MODEL_OPTIONS.keys()
    ),
)


selected_model_key = (
    MODEL_OPTIONS[
        selected_model_name
    ]
)


judge_display_name = (
    MODEL_DISPLAY_NAMES.get(
        JUDGE_MODEL_KEY,
        JUDGE_MODEL_KEY,
    )
)


st.info(
    f"Generator / Reviser: "
    f"{selected_model_name}  |  "
    f"Primary Demo Judge: "
    f"{judge_display_name}"
)


# =========================================================
# Inputs
# =========================================================

st.header(
    "2. Campaign Input"
)


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


policy_context = st.text_area(
    "Additional Policy Context (Optional)",
    value="",
    height=180,
    placeholder=(
        "Paste applicable advertising rules, "
        "platform policies, or internal brand "
        "policies here. Only supplied policy "
        "can be used as hard external compliance basis."
    ),
)


st.caption(
    "Later this field can be automatically "
    "populated through a RAG policy knowledge base."
)


# =========================================================
# Generate V1
# =========================================================

st.header(
    "3. Generate & Review"
)


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

        st.session_state.saved_policy_context = (
            policy_context
        )

        st.session_state.saved_model_key = (
            selected_model_key
        )

        st.session_state.saved_model_name = (
            selected_model_name
        )


        try:

            with st.spinner(
                f"Generating with "
                f"{selected_model_name}..."
            ):

                start = time.perf_counter()

                content = generate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    model_key=selected_model_key,
                )

                generation_latency = (
                    time.perf_counter()
                    - start
                )


            st.session_state.v1_content = (
                content
            )

            st.session_state.v1_generation_latency = (
                round(
                    generation_latency,
                    2,
                )
            )


            with st.spinner(
                f"Reviewing with "
                f"{judge_display_name}..."
            ):

                start = time.perf_counter()

                evaluation = evaluate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    generated_content=content,
                    policy_context=policy_context,
                )

                evaluation_latency = (
                    time.perf_counter()
                    - start
                )


            st.session_state.v1_evaluation = (
                evaluation
            )

            st.session_state.v1_evaluation_latency = (
                round(
                    evaluation_latency,
                    2,
                )
            )


        except Exception as error:

            st.error(
                f"Generation or review failed: {error}"
            )


# =========================================================
# Display V1
# =========================================================

if st.session_state.v1_content:

    st.subheader(
        "V1 Content"
    )

    st.write(
        st.session_state.v1_content
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(
        "Generation Latency",
        f"{st.session_state.v1_generation_latency:.2f}s",
    )


    col2.metric(
        "Review Latency",
        f"{st.session_state.v1_evaluation_latency:.2f}s",
    )


if st.session_state.v1_evaluation:

    show_evaluation(
        st.session_state.v1_evaluation,
        "V1 Review",
    )


# =========================================================
# Actions
# =========================================================

if st.session_state.v1_evaluation:

    evaluation = (
        st.session_state.v1_evaluation
    )


    compliance_findings = evaluation.get(
        "compliance_findings",
        [],
    )


    advisory_findings = evaluation.get(
        "advisory_findings",
        [],
    )


    st.header(
        "4. Actions"
    )


    # =====================================================
    # Compliance Fix
    # =====================================================

    if compliance_findings:

        st.warning(
            "Blocking compliance findings require "
            "attention before publishing."
        )


        if st.button(
            "Fix Compliance Issues",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Fixing compliance issues..."
                ):

                    start = time.perf_counter()

                    v2_content = (
                        fix_compliance_issues(
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

                            policy_context=(
                                st.session_state
                                .saved_policy_context
                            ),

                            model_key=(
                                st.session_state
                                .saved_model_key
                            ),
                        )
                    )

                    revision_latency = (
                        time.perf_counter()
                        - start
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

                st.session_state.v2_mode = (
                    "Compliance Fix"
                )


                with st.spinner(
                    "Re-checking revised content..."
                ):

                    start = time.perf_counter()

                    v2_evaluation = (
                        evaluate_content(
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

                            policy_context=(
                                st.session_state
                                .saved_policy_context
                            ),
                        )
                    )

                    evaluation_latency = (
                        time.perf_counter()
                        - start
                    )


                st.session_state.v2_evaluation = (
                    v2_evaluation
                )

                st.session_state.v2_evaluation_latency = (
                    round(
                        evaluation_latency,
                        2,
                    )
                )


            except Exception as error:

                st.error(
                    f"Compliance fix failed: {error}"
                )


    # =====================================================
    # Optional Quality Optimization
    # =====================================================

    elif advisory_findings:

        st.info(
            "No blocking compliance issue detected. "
            "Quality optimization is optional."
        )


        if st.button(
            "Optimize Quality (Optional)",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Optimizing quality..."
                ):

                    start = time.perf_counter()

                    v2_content = optimize_quality(
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

                        policy_context=(
                            st.session_state
                            .saved_policy_context
                        ),

                        model_key=(
                            st.session_state
                            .saved_model_key
                        ),
                    )

                    revision_latency = (
                        time.perf_counter()
                        - start
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

                st.session_state.v2_mode = (
                    "Optional Quality Optimization"
                )


                with st.spinner(
                    "Reviewing optimized content..."
                ):

                    start = time.perf_counter()

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

                        policy_context=(
                            st.session_state
                            .saved_policy_context
                        ),
                    )

                    evaluation_latency = (
                        time.perf_counter()
                        - start
                    )


                st.session_state.v2_evaluation = (
                    v2_evaluation
                )

                st.session_state.v2_evaluation_latency = (
                    round(
                        evaluation_latency,
                        2,
                    )
                )


            except Exception as error:

                st.error(
                    f"Quality optimization failed: {error}"
                )


    else:

        st.success(
            "No blocking issue or major quality "
            "suggestion was identified."
        )


# =========================================================
# Display V2
# =========================================================

if st.session_state.v2_content:

    st.header(
        "5. Revised Content"
    )


    st.caption(
        f"Revision Mode: "
        f"{st.session_state.v2_mode}"
    )


    st.write(
        st.session_state.v2_content
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(
        "Revision Latency",
        f"{st.session_state.v2_revision_latency:.2f}s",
    )


    col2.metric(
        "Re-check Latency",
        f"{st.session_state.v2_evaluation_latency:.2f}s",
    )


if st.session_state.v2_evaluation:

    show_evaluation(
        st.session_state.v2_evaluation,
        "V2 Review",
    )


    v1_count = len(
        st.session_state
        .v1_evaluation[
            "compliance_findings"
        ]
    )


    v2_count = len(
        st.session_state
        .v2_evaluation[
            "compliance_findings"
        ]
    )


    st.subheader(
        "Revision Result"
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(
        "Blocking Findings",
        f"{v2_count}",
        delta=(
            v2_count - v1_count
        ),
        delta_color="inverse",
    )


    score_change = round(
        st.session_state
        .v2_evaluation[
            "heuristic_composite_score"
        ]

        - st.session_state
        .v1_evaluation[
            "heuristic_composite_score"
        ],

        1,
    )


    col2.metric(
        "Diagnostic Score Change",
        f"{score_change:+.1f}",
    )


    if (
        v1_count > 0
        and v2_count == 0
    ):

        st.success(
            "The previously detected blocking "
            "compliance findings were removed "
            "in the current re-check."
        )


    elif v2_count < v1_count:

        st.warning(
            "Some blocking findings were removed, "
            "but additional compliance attention "
            "may still be required."
        )


    elif (
        v1_count > 0
        and v2_count >= v1_count
    ):

        st.error(
            "The compliance revision did not "
            "eliminate the blocking findings."
        )


st.divider()

st.caption(
    "GrowthPilot MVP · "
    "Policy-Grounded Compliance Review + "
    "Human-in-the-Loop Quality Advisory"
)