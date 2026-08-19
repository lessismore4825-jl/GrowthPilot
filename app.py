import streamlit as st

from src.generator import generate_content
from src.evaluator import evaluate_content


st.set_page_config(
    page_title="GrowthPilot",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 GrowthPilot")
st.subheader("AI Marketing Content Generation & Evaluation Platform")

st.divider()

# -------------------------
# User Input
# -------------------------

brand_info = st.text_area(
    "Brand Information",
    placeholder="请输入品牌介绍、产品卖点、品牌风格等..."
)

campaign_brief = st.text_area(
    "Campaign Brief",
    placeholder="请输入营销目标、目标用户、平台、内容形式等..."
)


# -------------------------
# Generate + Evaluate
# -------------------------

if st.button("Generate Content"):

    if brand_info and campaign_brief:

        try:
            # 1. Generate Content
            with st.spinner("Generating content..."):
                result = generate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief
                )

            st.subheader("Generated Content")
            st.write(result)

            st.divider()

            # 2. Evaluate Content
            st.subheader("🔍 AI Evaluation")

            with st.spinner("Evaluating content..."):
                evaluation = evaluate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief,
                    generated_content=result
                )

            # 3. Evaluation Metrics
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

            # 4. Issues
            st.subheader("Issues")

            for issue in evaluation["issues"]:
                st.write(f"• {issue}")

            # 5. Suggestions
            st.subheader("Suggestions")

            for suggestion in evaluation["suggestions"]:
                st.write(f"• {suggestion}")

        except Exception as e:
            st.error(f"Processing failed: {e}")

    else:
        st.warning(
            "Please provide both Brand Information and Campaign Brief."
        )