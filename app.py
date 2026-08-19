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
        st.success("输入成功！下一步我们将在这里接入 LLM。")
    else:
        st.warning("请先填写 Brand Information 和 Campaign Brief。")