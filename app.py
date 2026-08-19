import pandas as pd
import streamlit as st

from src.generator import generate_content
from src.evaluator import evaluate_content
from src.reviser import revise_content


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="GrowthPilot",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 GrowthPilot")
st.subheader("AI Marketing Content Generation & Evaluation Platform")

st.divider()


# =========================
# Session State
# =========================

if "v1_content" not in st.session_state:
    st.session_state.v1_content = None

if "v1_evaluation" not in st.session_state:
    st.session_state.v1_evaluation = None

if "v2_content" not in st.session_state:
    st.session_state.v2_content = None

if "v2_evaluation" not in st.session_state:
    st.session_state.v2_evaluation = None

if "saved_brand_info" not in st.session_state:
    st.session_state.saved_brand_info = None

if "saved_campaign_brief" not in st.session_state:
    st.session_state.saved_campaign_brief = None


# =========================
# Helper Function
# =========================

def show_evaluation(evaluation: dict, title: str):

    st.subheader(title)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Brand Alignment",
        f"{evaluation['brand_alignment']}/10"
    )

    col2.metric(
        "Tone Match",
        f"{evaluation['tone_match']}/10"
    )

    col3.metric(
        "Selling Points",
        f"{evaluation['selling_point_coverage']}/10"
    )

    col4.metric(
        "Fact Consistency",
        f"{evaluation['factual_consistency']}/10"
    )

    col5.metric(
        "Claim Risk",
        f"{evaluation['unsupported_claim_risk']}/10"
    )

    st.metric(
        "Overall Score",
        f"{evaluation['overall_score']}/10"
    )

    st.markdown("### Issues")

    if evaluation["issues"]:
        for issue in evaluation["issues"]:
            st.write(f"• {issue}")
    else:
        st.write("No major issues detected.")

    st.markdown("### Suggestions")

    if evaluation["suggestions"]:
        for suggestion in evaluation["suggestions"]:
            st.write(f"• {suggestion}")
    else:
        st.write("No major improvements required.")


# =========================
# User Input
# =========================

brand_info = st.text_area(
    "Brand Information",
    placeholder="请输入品牌介绍、产品卖点、品牌风格等...",
    height=180
)

campaign_brief = st.text_area(
    "Campaign Brief",
    placeholder="请输入营销目标、目标用户、平台、内容形式等...",
    height=180
)


# =========================
# Generate V1
# =========================

if st.button("Generate Content", type="primary"):

    if not brand_info or not campaign_brief:

        st.warning(
            "Please provide both Brand Information and Campaign Brief."
        )

    else:

        try:

            # Save the original inputs
            st.session_state.saved_brand_info = brand_info
            st.session_state.saved_campaign_brief = campaign_brief

            # Clear previous V2 result
            st.session_state.v2_content = None
            st.session_state.v2_evaluation = None

            # Generate V1
            with st.spinner("Generating V1 content..."):

                v1_content = generate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief
                )

            # Evaluate V1
            with st.spinner("Evaluating V1 content..."):

                v1_evaluation = evaluate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    generated_content=v1_content
                )

            # Save results
            st.session_state.v1_content = v1_content
            st.session_state.v1_evaluation = v1_evaluation

        except Exception as e:

            st.error(
                f"Generation or evaluation failed: {e}"
            )


# =========================
# Display V1
# =========================

if st.session_state.v1_content is not None:

    st.divider()

    st.header("📝 Version 1")

    st.subheader("Generated Content")

    st.write(
        st.session_state.v1_content
    )

    show_evaluation(
        st.session_state.v1_evaluation,
        "🔍 V1 AI Evaluation"
    )


    # =========================
    # Revise Button
    # =========================

    if st.button("✨ Revise with AI"):

        try:

            # Generate V2
            with st.spinner(
                "Revising content based on evaluation feedback..."
            ):

                v2_content = revise_content(
                    brand_info=st.session_state.saved_brand_info,
                    campaign_brief=st.session_state.saved_campaign_brief,
                    original_content=st.session_state.v1_content,
                    evaluation=st.session_state.v1_evaluation
                )

            # Evaluate V2
            with st.spinner(
                "Evaluating revised content..."
            ):

                v2_evaluation = evaluate_content(
                    brand_info=st.session_state.saved_brand_info,
                    campaign_brief=st.session_state.saved_campaign_brief,
                    generated_content=v2_content
                )

            # Save results
            st.session_state.v2_content = v2_content
            st.session_state.v2_evaluation = v2_evaluation

        except Exception as e:

            st.error(
                f"Revision failed: {e}"
            )


# =========================
# Display V2
# =========================

if st.session_state.v2_content is not None:

    st.divider()

    st.header("✨ Version 2 — AI Revised")

    st.subheader("Revised Content")

    st.write(
        st.session_state.v2_content
    )

    show_evaluation(
        st.session_state.v2_evaluation,
        "🔍 V2 AI Evaluation"
    )


    # =========================
    # V1 vs V2 Comparison
    # =========================

    st.divider()

    st.header("📊 V1 vs V2 Comparison")

    v1 = st.session_state.v1_evaluation
    v2 = st.session_state.v2_evaluation

    comparison_data = [
        {
            "Metric": "Brand Alignment",
            "V1": v1["brand_alignment"],
            "V2": v2["brand_alignment"],
            "Improvement": round(
                v2["brand_alignment"]
                - v1["brand_alignment"],
                1
            )
        },
        {
            "Metric": "Tone Match",
            "V1": v1["tone_match"],
            "V2": v2["tone_match"],
            "Improvement": round(
                v2["tone_match"]
                - v1["tone_match"],
                1
            )
        },
        {
            "Metric": "Selling Point Coverage",
            "V1": v1["selling_point_coverage"],
            "V2": v2["selling_point_coverage"],
            "Improvement": round(
                v2["selling_point_coverage"]
                - v1["selling_point_coverage"],
                1
            )
        },
        {
            "Metric": "Factual Consistency",
            "V1": v1["factual_consistency"],
            "V2": v2["factual_consistency"],
            "Improvement": round(
                v2["factual_consistency"]
                - v1["factual_consistency"],
                1
            )
        },
        {
            "Metric": "Claim Risk",
            "V1": v1["unsupported_claim_risk"],
            "V2": v2["unsupported_claim_risk"],
            "Improvement": round(
                v1["unsupported_claim_risk"]
                - v2["unsupported_claim_risk"],
                1
            )
        },
        {
            "Metric": "Overall Score",
            "V1": v1["overall_score"],
            "V2": v2["overall_score"],
            "Improvement": round(
                v2["overall_score"]
                - v1["overall_score"],
                1
            )
        }
    ]

    comparison_df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )