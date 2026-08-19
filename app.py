from src.generator import generate_content
import streamlit as st

st.set_page_config(
    page_title="GrowthPilot",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 GrowthPilot")
st.subheader("AI Marketing Content Generation & Evaluation Platform")

st.divider()

brand_info = st.text_area(
    "Brand Information",
    placeholder="请输入品牌介绍、产品卖点、品牌风格等..."
)

campaign_brief = st.text_area(
    "Campaign Brief",
    placeholder="请输入营销目标、目标用户、平台、内容形式等..."
)

if st.button("Generate Content"):
    if brand_info and campaign_brief:

        with st.spinner("Generating content..."):
            try:
                result = generate_content(
                    brand_info=brand_info,
                    campaign_brief=campaign_brief
                )

                st.subheader("Generated Content")
                st.write(result)

            except Exception as e:
                st.error(f"Generation failed: {e}")

    else:
        st.warning(
            "Please provide both Brand Information and Campaign Brief."
        )