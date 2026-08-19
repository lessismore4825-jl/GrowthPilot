import json

from src.evaluator import evaluate_content


brand_info = """
Brand: LumiSkin

Product: Barrier Repair Moisturizer

Key selling points:
- Contains ceramides and hyaluronic acid
- Designed for dry and sensitive skin
- Lightweight and non-greasy texture
- Provides up to 24 hours of hydration

Brand tone:
Professional, calm, trustworthy and modern.

Restrictions:
Do not claim that the product treats or cures skin diseases.
Do not use exaggerated medical claims.
"""


campaign_brief = """
Platform: Xiaohongshu

Target audience:
Women aged 20-30 with dry or sensitive skin.

Campaign objective:
Introduce the Barrier Repair Moisturizer.

Content style:
Natural recommendation rather than aggressive advertising.
"""


bad_content = """
LumiSkin屏障修护面霜采用全球领先医学配方，
可以治疗湿疹和敏感肌问题。

独家临床研究证明，使用一次即可修复受损肌肤，
并提供72小时深层保湿。

现在购买即可彻底告别敏感肌！
"""


print("1. 开始调用 Gemini Evaluator...")

evaluation = evaluate_content(
    brand_info=brand_info,
    campaign_brief=campaign_brief,
    generated_content=bad_content
)

print("2. Gemini 返回成功")

print(
    json.dumps(
        evaluation,
        ensure_ascii=False,
        indent=2
    )
)