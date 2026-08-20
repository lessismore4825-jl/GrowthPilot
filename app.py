from locales import get_text

import time

import pandas as pd
import streamlit as st

from src.generator import generate_content
from src.llm_client import generate_text
from src.reviser import fix_mandatory_issues, optimize_quality
from src.review_orchestrator import (
    FAST_REVIEW,
    CROSS_JUDGE_REVIEW,
    NO_MANDATORY_ACTION,
    COMPLIANCE_ACTION,
    REQUIREMENT_ACTION,
    COMPLIANCE_AND_REQUIREMENT_ACTION,
    HUMAN_REVIEW_REQUIRED,
    REVIEW_ERROR,
    review_content,
)


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="GrowthPilot",
    page_icon="🚀",
    layout="wide",
)


# =========================================================
# Language Configuration
# =========================================================

selected_language = st.sidebar.selectbox(
    "Language / 语言",
    options=[
        "English",
        "中文",
    ],
    index=0,
    key="ui_language_selector",
)

_locale_text = get_text(
    selected_language
)


def tr(
    key: str,
    fallback: str | None = None,
    **kwargs,
) -> str:
    """
    Translate UI text only.

    Submitted content, evidence, policy basis,
    source quotes and model-generated findings
    remain untranslated.
    """

    value = _locale_text(
        key
    )

    if (
        value == key
        and fallback is not None
    ):
        value = fallback

    if kwargs:

        try:

            value = value.format(
                **kwargs
            )

        except (
            KeyError,
            IndexError,
            ValueError,
        ):

            pass

    return value


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

    # Create workflow
    "create_guidance": None,
    "create_reference_draft": None,
    "create_guidance_latency": None,
    "create_generation_latency": None,

    "create_review_result": None,
    "create_evaluation": None,
    "create_evaluation_latency": None,

    "create_v2_content": None,
    "create_v2_review_result": None,
    "create_v2_evaluation": None,
    "create_v2_revision_latency": None,
    "create_v2_evaluation_latency": None,
    "create_v2_mode": None,

    # Review workflow
    "review_original_content": None,

    "review_review_result": None,
    "review_evaluation": None,
    "review_evaluation_latency": None,

    "review_v2_content": None,
    "review_v2_review_result": None,
    "review_v2_evaluation": None,
    "review_v2_revision_latency": None,
    "review_v2_evaluation_latency": None,
    "review_v2_mode": None,

    # Frozen context snapshot
    "saved_brand_context": None,
    "saved_campaign_context": None,
    "saved_policy_context": None,
    "saved_requirements": None,
    "saved_model_key": None,
    "saved_model_name": None,

    # Review-mode snapshots
    "create_saved_review_mode": None,
    "review_saved_review_mode": None,
}

for key, value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# =========================================================
# Input / Context Helpers
# =========================================================

def parse_line_items(
    text: str,
) -> list[str]:

    items = []

    for raw_line in str(
        text or ""
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        line = (
            line
            .lstrip("-•* ")
            .strip()
        )

        if line:

            items.append(
                line
            )

    return items


def parse_structured_requirements(
    text: str,
) -> list[dict]:

    raw_items = parse_line_items(
        text
    )

    requirements = []

    for index, item in enumerate(
        raw_items,
        start=1,
    ):

        match_mode = "SEMANTIC"
        content = item

        if "|" in item:

            left, right = item.split(
                "|",
                1,
            )

            candidate_mode = (
                left
                .strip()
                .upper()
            )

            if candidate_mode in {
                "EXACT",
                "SEMANTIC",
            }:

                match_mode = (
                    candidate_mode
                )

                content = (
                    right.strip()
                )

        elif item.upper().startswith(
            "EXACT:"
        ):

            match_mode = "EXACT"

            content = (
                item.split(
                    ":",
                    1,
                )[1]
                .strip()
            )

        elif item.upper().startswith(
            "SEMANTIC:"
        ):

            match_mode = "SEMANTIC"

            content = (
                item.split(
                    ":",
                    1,
                )[1]
                .strip()
            )

        if not content:
            continue

        requirements.append(
            {
                "requirement_id":
                    f"R{index}",

                "content":
                    content,

                "match_mode":
                    match_mode,
            }
        )

    return requirements


def requirement_contents(
    requirements: list[dict],
) -> list[str]:

    return [
        str(
            item.get(
                "content",
                "",
            )
        ).strip()

        for item in requirements

        if str(
            item.get(
                "content",
                "",
            )
        ).strip()
    ]


def format_list_section(
    title: str,
    values: list[str],
) -> str:

    if not values:

        return (
            f"{title}:\n"
            "- None specified"
        )

    return "\n".join(
        [
            f"{title}:"
        ]
        + [
            f"- {value}"
            for value in values
        ]
    )


def build_brand_context(
    brand_info: str,
    product_info: str,
) -> str:

    return (
        f"{brand_info.strip()}\n\n"
        "VERIFIED PRODUCT INFORMATION:\n"
        f"{product_info.strip()}"
    ).strip()


def build_creator_profile(
    creator_category: str,
    creator_audience: str,
    creator_style: str,
    creator_characteristics: str,
) -> str:

    fields = [
        (
            "Creator Category",
            creator_category,
        ),
        (
            "Creator Audience",
            creator_audience,
        ),
        (
            "Creator Style",
            creator_style,
        ),
        (
            "Content Characteristics",
            creator_characteristics,
        ),
    ]

    lines = [
        "CREATOR PROFILE:"
    ]

    for label, value in fields:

        value = str(
            value or ""
        ).strip()

        if value:

            lines.append(
                f"- {label}: {value}"
            )

    if len(lines) == 1:

        lines.append(
            "- No creator profile supplied"
        )

    return "\n".join(
        lines
    )


def build_campaign_context(
    campaign_brief: str,
    platform: str,
    content_type: str,
    creator_category: str,
    creator_audience: str,
    creator_style: str,
    creator_characteristics: str,
    requirements: list[dict],
    must_avoid: list[str],
) -> str:

    must_mention = requirement_contents(
        requirements
    )

    sections = [
        "ORIGINAL CAMPAIGN BRIEF:",
        campaign_brief.strip(),
        "",
        (
            "PLATFORM:\n"
            f"{platform.strip() or 'Not specified'}"
        ),
        "",
        (
            "CONTENT TYPE:\n"
            f"{content_type.strip() or 'Not specified'}"
        ),
        "",
        build_creator_profile(
            creator_category=
                creator_category,

            creator_audience=
                creator_audience,

            creator_style=
                creator_style,

            creator_characteristics=
                creator_characteristics,
        ),
        "",
        format_list_section(
            "CAMPAIGN MUST MENTION",
            must_mention,
        ),
        "",
        format_list_section(
            "CAMPAIGN MUST AVOID",
            must_avoid,
        ),
    ]

    return "\n".join(
        sections
    ).strip()


def save_context_snapshot(
    brand_context: str,
    campaign_context: str,
    policy_context: str,
    requirements: list[dict],
    model_key: str,
    model_name: str,
):

    st.session_state.saved_brand_context = (
        brand_context
    )

    st.session_state.saved_campaign_context = (
        campaign_context
    )

    st.session_state.saved_policy_context = (
        policy_context
    )

    st.session_state.saved_requirements = (
        requirements
    )

    st.session_state.saved_model_key = (
        model_key
    )

    st.session_state.saved_model_name = (
        model_name
    )


# =========================================================
# Reset Helpers
# =========================================================

def reset_create_results():

    for key in [
        "create_guidance",
        "create_reference_draft",
        "create_guidance_latency",
        "create_generation_latency",

        "create_review_result",
        "create_evaluation",
        "create_evaluation_latency",

        "create_v2_content",
        "create_v2_review_result",
        "create_v2_evaluation",
        "create_v2_revision_latency",
        "create_v2_evaluation_latency",
        "create_v2_mode",
    ]:

        st.session_state[
            key
        ] = None


def reset_review_results():

    for key in [
        "review_original_content",

        "review_review_result",
        "review_evaluation",
        "review_evaluation_latency",

        "review_v2_content",
        "review_v2_review_result",
        "review_v2_evaluation",
        "review_v2_revision_latency",
        "review_v2_evaluation_latency",
        "review_v2_mode",
    ]:

        st.session_state[
            key
        ] = None


# =========================================================
# Creator Guidance Generator
# =========================================================

def generate_creator_guidance(
    brand_context: str,
    campaign_context: str,
    policy_context: str,
    model_key: str,
) -> str:

    policy_display = (
        policy_context.strip()

        if policy_context.strip()

        else
        "No additional policy context supplied."
    )

    prompt = f"""
You are the campaign content planning assistant inside GrowthPilot.

Your task is to produce PRE-CREATION GUIDANCE for a creator or content team.

Use only the supplied Brand, Product, Campaign, Creator and Policy context.

Do not invent:

- product facts
- product claims
- creator experiences
- legal rules
- platform rules
- research
- certifications
- performance numbers

The campaign's structured Must Mention and Must Avoid items are authoritative.

Do not create new mandatory requirements.


============================================================
BRAND + VERIFIED PRODUCT CONTEXT
============================================================

{brand_context}


============================================================
CAMPAIGN + CREATOR CONTEXT
============================================================

{campaign_context}


============================================================
ADDITIONAL POLICY CONTEXT
============================================================

{policy_display}


============================================================
OUTPUT
============================================================

Return concise Markdown with exactly these sections:

## Creative Direction

A short description of the content angle and intended communication approach.

## Key Messages

3-5 message priorities based only on supplied facts and campaign goals.

## Tone Guidance

Concrete tone and style guidance grounded in the supplied brand / creator context.

## Creator Adaptation

How this creator should make the content feel natural for their audience while
preserving brand consistency.

## Platform Execution Notes

Practical format / readability / structure suggestions.

If a suggestion relies on general marketing knowledge rather than supplied
context, clearly label it as "General heuristic".

Do not write the final full campaign copy in this guidance response.
"""

    return generate_text(
        prompt=prompt,
        model_key=model_key,
        temperature=0.2,
    )


# =========================================================
# Cross-Judge Product Display
# =========================================================

def judge_summary(
    evaluation: dict,
) -> tuple[int, int, int]:

    compliance_count = len(
        evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    requirement_count = len(
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    advisory_count = len(
        evaluation.get(
            "advisory_findings",
            [],
        )
        or []
    )

    return (
        compliance_count,
        requirement_count,
        advisory_count,
    )


def show_cross_judge_disagreements(
    review_result: dict,
):

    details = (
        review_result.get(
            "cross_judge"
        )
        or {}
    )

    first_only_compliance = (
        details.get(
            "first_only_compliance",
            [],
        )
        or []
    )

    second_only_compliance = (
        details.get(
            "second_only_compliance",
            [],
        )
        or []
    )

    first_only_requirements = (
        details.get(
            "first_only_requirements",
            [],
        )
        or []
    )

    second_only_requirements = (
        details.get(
            "second_only_requirements",
            [],
        )
        or []
    )

    disagreements = (
        first_only_compliance
        + second_only_compliance
        + first_only_requirements
        + second_only_requirements
    )

    if not disagreements:
        return

    with st.expander(
        tr(
            "judge_disagreement_details",
            "Judge Disagreement Details",
        ),
        expanded=True,
    ):

        if first_only_compliance:

            st.markdown(
                "**Step-only Compliance Findings**"
            )

            for finding in (
                first_only_compliance
            ):

                st.write(
                    "• "
                    + str(
                        finding.get(
                            "evidence",
                            "",
                        )
                    )
                )

        if second_only_compliance:

            st.markdown(
                "**Qwen-only Compliance Findings**"
            )

            for finding in (
                second_only_compliance
            ):

                st.write(
                    "• "
                    + str(
                        finding.get(
                            "evidence",
                            "",
                        )
                    )
                )

        if first_only_requirements:

            st.markdown(
                "**Step-only Requirement Findings**"
            )

            for finding in (
                first_only_requirements
            ):

                st.write(
                    "• "
                    + str(
                        finding.get(
                            "requirement",
                            "",
                        )
                    )
                )

        if second_only_requirements:

            st.markdown(
                "**Qwen-only Requirement Findings**"
            )

            for finding in (
                second_only_requirements
            ):

                st.write(
                    "• "
                    + str(
                        finding.get(
                            "requirement",
                            "",
                        )
                    )
                )


def show_review_routing(
    review_result: dict | None,
):

    if not review_result:
        return

    review_mode = review_result.get(
        "review_mode",
        FAST_REVIEW,
    )

    final_route = review_result.get(
        "final_route",
        NO_MANDATORY_ACTION,
    )

    judge_errors = (
        review_result.get(
            "judge_errors",
            {},
        )
        or {}
    )

    if review_mode == FAST_REVIEW:

        st.markdown(
            f"### {tr('review_mode_result', 'Review Decision')}"
        )

        st.info(
            tr(
                "fast_review_result_message",
                (
                    "Fast Review used one Judge. "
                    "This mode prioritizes responsiveness."
                ),
            )
        )

        return


    # =====================================================
    # Cross-Judge Mode
    # =====================================================

    st.markdown(
        f"### 🛡️ {tr('cross_judge_decision', 'Cross-Judge Decision')}"
    )

    judge_results = (
        review_result.get(
            "judge_results",
            {},
        )
        or {}
    )

    step_result = (
        judge_results.get(
            "step"
        )
    )

    qwen_result = (
        judge_results.get(
            "qwen"
        )
    )

    col1, col2 = (
        st.columns(
            2
        )
    )


    # -----------------------------------------------------
    # Step Judge
    # -----------------------------------------------------

    with col1:

        st.markdown(
            "#### Step-3.5-Flash"
        )

        if step_result:

            (
                compliance_count,
                requirement_count,
                advisory_count,
            ) = judge_summary(
                step_result
            )

            metric1, metric2, metric3 = (
                st.columns(
                    3
                )
            )

            metric1.metric(
                tr(
                    "compliance_short",
                    "Compliance",
                ),
                compliance_count,
            )

            metric2.metric(
                tr(
                    "requirements_short",
                    "Requirements",
                ),
                requirement_count,
            )

            metric3.metric(
                tr(
                    "advisory_short",
                    "Advisory",
                ),
                advisory_count,
            )

        elif "step" in judge_errors:

            st.error(
                judge_errors[
                    "step"
                ]
            )


    # -----------------------------------------------------
    # Qwen Judge
    # -----------------------------------------------------

    with col2:

        st.markdown(
            "#### Qwen3.5-35B-A3B"
        )

        if qwen_result:

            (
                compliance_count,
                requirement_count,
                advisory_count,
            ) = judge_summary(
                qwen_result
            )

            metric1, metric2, metric3 = (
                st.columns(
                    3
                )
            )

            metric1.metric(
                tr(
                    "compliance_short",
                    "Compliance",
                ),
                compliance_count,
            )

            metric2.metric(
                tr(
                    "requirements_short",
                    "Requirements",
                ),
                requirement_count,
            )

            metric3.metric(
                tr(
                    "advisory_short",
                    "Advisory",
                ),
                advisory_count,
            )

        elif "qwen" in judge_errors:

            st.error(
                judge_errors[
                    "qwen"
                ]
            )


    # -----------------------------------------------------
    # Final Routing
    # -----------------------------------------------------

    st.markdown(
        (
            f"#### "
            f"{tr('final_routing', 'Final Routing')}"
        )
    )

    if final_route == HUMAN_REVIEW_REQUIRED:

        st.warning(
            tr(
                "human_review_required_message",
                (
                    "The two Judges disagree on at least one "
                    "mandatory-layer finding. Automatic mandatory "
                    "revision is disabled. Human review is required."
                ),
            )
        )

        show_cross_judge_disagreements(
            review_result
        )

    elif final_route == COMPLIANCE_ACTION:

        st.error(
            tr(
                "consensus_compliance_message",
                (
                    "Both Judges confirmed one or more "
                    "compliance findings. Mandatory compliance "
                    "correction is authorized."
                ),
            )
        )

    elif final_route == REQUIREMENT_ACTION:

        st.warning(
            tr(
                "consensus_requirement_message",
                (
                    "Both Judges confirmed one or more missing "
                    "campaign requirements. Mandatory completion "
                    "is authorized."
                ),
            )
        )

    elif (
        final_route
        == COMPLIANCE_AND_REQUIREMENT_ACTION
    ):

        st.error(
            tr(
                "consensus_both_message",
                (
                    "Both Judges confirmed mandatory compliance "
                    "and campaign-requirement findings."
                ),
            )
        )

    elif final_route == NO_MANDATORY_ACTION:

        st.success(
            tr(
                "consensus_clear_message",
                (
                    "The Cross-Judge review found no "
                    "mandatory action requiring escalation."
                ),
            )
        )

    elif final_route == REVIEW_ERROR:

        st.error(
            tr(
                "review_error_message",
                (
                    "The review could not be completed."
                ),
            )
        )


# =========================================================
# Evaluation Display
# =========================================================

def show_evaluation(
    evaluation: dict,
    title: str,
    review_result: dict | None = None,
):

    st.subheader(
        title
    )

    compliance_findings = (
        evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    requirement_findings = (
        evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    advisory_findings = (
        evaluation.get(
            "advisory_findings",
            [],
        )
        or []
    )

    review_notes = (
        evaluation.get(
            "review_notes",
            [],
        )
        or []
    )

    mandatory_count = (
        len(
            compliance_findings
        )
        + len(
            requirement_findings
        )
    )

    review_result = (
        review_result
        or {}
    )

    requires_human_review = bool(
        review_result.get(
            "requires_human_review",
            False,
        )
    )

    cross_judge_details = (
        review_result.get(
            "cross_judge"
        )
        or {}
    )

    compliance_disagreement = bool(
        cross_judge_details.get(
            "compliance_disagreement",
            False,
        )
    )

    requirement_disagreement = bool(
        cross_judge_details.get(
            "requirement_disagreement",
            False,
        )
    )


    # =====================================================
    # Review Status
    # =====================================================

    st.markdown(
        f"### {tr('review_status', 'Review Status')}"
    )

    if requires_human_review:

        st.warning(
            tr(
                "human_review_status_message",
                (
                    "Mandatory-layer disagreement requires "
                    "Human Review. No automatic clearance "
                    "has been granted."
                ),
            )
        )

        st.caption(
            tr(
                "human_review_status_explanation",
                (
                    "A zero consensus-finding count does not "
                    "mean the content has passed. At least one "
                    "Judge raised a mandatory concern that was "
                    "not independently confirmed by the other Judge."
                ),
            )
        )

    elif mandatory_count > 0:

        st.error(
            tr(
                "mandatory_detected_message",
                (
                    "{total} mandatory action(s) detected: "
                    "{compliance} compliance + "
                    "{requirement} requirement."
                ),
                total=mandatory_count,
                compliance=len(
                    compliance_findings
                ),
                requirement=len(
                    requirement_findings
                ),
            )
        )

    else:

        st.success(
            tr(
                "no_mandatory_correction",
                (
                    "No mandatory correction or completion "
                    "was detected by the current review."
                ),
            )
        )

    st.caption(
        tr(
            "human_decision_disclaimer",
            (
                "AI pre-review is not legal approval "
                "and does not replace final human "
                "publishing judgment."
            ),
        )
    )


    # =====================================================
    # Mandatory Fixes
    # =====================================================

    st.markdown(
        f"### {tr('mandatory_fixes', 'Mandatory Fixes')}"
    )


    # =====================================================
    # Compliance Findings
    # =====================================================

    st.markdown(
        (
            "#### 1. "
            f"{tr('compliance_issues', 'Compliance Issues')}"
        )
    )

    if compliance_findings:

        for index, finding in enumerate(
            compliance_findings,
            start=1,
        ):

            with st.expander(
                (
                    f"{tr('compliance_issue', 'Compliance Issue')} "
                    f"{index}"
                ),
                expanded=True,
            ):

                st.markdown(
                    (
                        f"**{tr('problematic_content', 'Problematic Content')}**"
                    )
                )

                st.write(
                    finding.get(
                        "evidence",
                        "",
                    )
                )

                st.markdown(
                    (
                        f"**{tr('policy_source', 'Policy Source')}**"
                    )
                )

                st.write(
                    finding.get(
                        "policy_source",
                        "",
                    )
                )

                st.markdown(
                    (
                        f"**{tr('policy_basis', 'Policy Basis')}**"
                    )
                )

                st.write(
                    finding.get(
                        "policy_basis",
                        "",
                    )
                )

                st.markdown(
                    (
                        f"**{tr('why_conflicts', 'Why It Conflicts')}**"
                    )
                )

                st.write(
                    finding.get(
                        "reason",
                        "",
                    )
                )

                st.markdown(
                    (
                        f"**{tr('required_action', 'Required Action')}**"
                    )
                )

                st.write(
                    finding.get(
                        "required_action",
                        "",
                    )
                )

    elif (
        requires_human_review
        and compliance_disagreement
    ):

        st.warning(
            tr(
                "no_consensus_compliance",
                (
                    "No compliance finding was confirmed by "
                    "both Judges, but an unresolved compliance "
                    "disagreement remains. Human Review is required."
                ),
            )
        )

    else:

        st.success(
            tr(
                "no_compliance_issue",
                (
                    "No direct, source-grounded "
                    "blocking conflict detected."
                ),
            )
        )


    # =====================================================
    # Requirement Findings
    # =====================================================

    st.markdown(
        (
            "#### 2. "
            f"{tr('requirement_issues', 'Missing Campaign Requirements')}"
        )
    )

    if requirement_findings:

        for index, finding in enumerate(
            requirement_findings,
            start=1,
        ):

            requirement = finding.get(
                "requirement",
                "",
            )

            mode = finding.get(
                "match_mode",
                "SEMANTIC",
            )

            with st.expander(
                (
                    f"{tr('requirement_label', 'Requirement')} "
                    f"{index}: {requirement}"
                ),
                expanded=True,
            ):

                cols = st.columns(
                    2
                )

                cols[0].metric(
                    tr(
                        "requirement_id",
                        "Requirement ID",
                    ),
                    finding.get(
                        "requirement_id",
                        "",
                    ),
                )

                cols[1].metric(
                    tr(
                        "match_mode",
                        "Match Mode",
                    ),
                    mode,
                )

                st.markdown(
                    (
                        f"**{tr('why_missing', 'Why It Is Missing')}**"
                    )
                )

                st.write(
                    finding.get(
                        "reason",
                        "",
                    )
                )

                st.markdown(
                    (
                        f"**{tr('required_action', 'Required Action')}**"
                    )
                )

                st.write(
                    finding.get(
                        "required_action",
                        "",
                    )
                )

                verification = (
                    finding.get(
                        "verification_mode",
                        "",
                    )
                )

                if verification:

                    st.caption(
                        (
                            f"{tr('verification', 'Verification')}: "
                            f"{verification}"
                        )
                    )

    elif (
        requires_human_review
        and requirement_disagreement
    ):

        st.warning(
            tr(
                "no_consensus_requirement",
                (
                    "No missing campaign requirement was "
                    "confirmed by both Judges, but an unresolved "
                    "requirement disagreement remains. "
                    "Human Review is required."
                ),
            )
        )

    else:

        st.success(
            tr(
                "no_requirement_missing",
                (
                    "No structured Must Mention "
                    "requirement is missing."
                ),
            )
        )


    # =====================================================
    # Human Review Gate
    # =====================================================

    if requires_human_review:

        st.warning(
            tr(
                "human_review_gate",
                (
                    "This content has not been automatically "
                    "cleared. Resolve the Judge disagreement "
                    "before any mandatory AI revision or "
                    "publishing decision."
                ),
            )
        )


    # =====================================================
    # Advisory
    # =====================================================

    st.markdown(
        (
            f"### "
            f"{tr('optional_improvements', 'Optional Improvements')}"
        )
    )

    if advisory_findings:

        st.info(
            tr(
                "advisory_count_message",
                (
                    "{count} non-blocking advisory "
                    "finding(s) were generated."
                ),
                count=len(
                    advisory_findings
                ),
            )
        )

        for index, finding in enumerate(
            advisory_findings,
            start=1,
        ):

            area = finding.get(
                "area",
                "General",
            )

            with st.expander(
                (
                    f"{tr('advisory_label', 'Advisory')} "
                    f"{index}: {area}"
                ),
                expanded=False,
            ):

                evidence = finding.get(
                    "evidence",
                    "",
                )

                if evidence:

                    st.markdown(
                        (
                            f"**{tr('relevant_content', 'Relevant Content')}**"
                        )
                    )

                    st.write(
                        evidence
                    )

                st.markdown(
                    (
                        f"**{tr('why_it_matters', 'Why It Matters')}**"
                    )
                )

                st.write(
                    finding.get(
                        "reason",
                        "",
                    )
                )

                suggestion = finding.get(
                    "suggestion",
                    "",
                )

                if suggestion:

                    st.markdown(
                        (
                            f"**{tr('edit_direction', 'Edit Direction')}**"
                        )
                    )

                    st.write(
                        suggestion
                    )

                basis_type = finding.get(
                    "basis_type",
                    "GENERAL_HEURISTIC",
                )

                st.markdown(
                    (
                        f"**{tr('advice_basis', 'Advice Basis')}**"
                    )
                )

                if basis_type == "SUPPLIED_CONTEXT":

                    source = finding.get(
                        "basis_source",
                        "",
                    )

                    quote = finding.get(
                        "basis_quote",
                        "",
                    )

                    st.write(
                        (
                            "SUPPLIED_CONTEXT · "
                            f"{source}"
                        )
                    )

                    if quote:

                        st.caption(
                            (
                                f"{tr('source_quote', 'Source quote')}: "
                                f'"{quote}"'
                            )
                        )

                elif (
                    basis_type
                    == "SYSTEM_GROUNDING_REVIEW"
                ):

                    st.write(
                        (
                            "SYSTEM_GROUNDING_REVIEW · "
                            f"{tr('grounding_review_signal', 'Manual review signal')}"
                        )
                    )

                else:

                    st.write(
                        (
                            "GENERAL_HEURISTIC · "
                            f"{tr('general_heuristic_guidance', 'General AI / marketing guidance')}"
                        )
                    )

    else:

        st.write(
            tr(
                "no_optional_improvement",
                (
                    "No major optional improvement "
                    "was identified."
                ),
            )
        )


    # =====================================================
    # Diagnostic Scores
    # =====================================================

    st.markdown(
        (
            f"### "
            f"{tr('diagnostic_signals', 'Diagnostic Signals')}"
        )
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    col1.metric(
        tr(
            "heuristic_composite",
            "Heuristic Composite",
        ),
        (
            f"{evaluation['heuristic_composite_score']:.1f}"
            "/10"
        ),
        help=tr(
            "diagnostic_score_help",
            (
                "Diagnostic comparison signal only. "
                "It is not a calibrated pass/fail threshold."
            ),
        ),
    )

    col2.metric(
        tr(
            "factual_consistency",
            "Factual Consistency",
        ),
        (
            f"{evaluation['factual_consistency']}"
            "/10"
        ),
    )

    col3.metric(
        tr(
            "unsupported_claim_risk",
            "Unsupported Claim Risk",
        ),
        (
            f"{evaluation['unsupported_claim_risk']}"
            "/10"
        ),
        help=tr(
            "lower_is_better",
            "Lower is better.",
        ),
    )

    score_df = pd.DataFrame(
        {
            tr(
                "dimension",
                "Dimension",
            ): [
                tr(
                    "brand_alignment",
                    "Brand Alignment",
                ),
                tr(
                    "tone_match",
                    "Tone Match",
                ),
                tr(
                    "selling_point_coverage",
                    "Selling Point Coverage",
                ),
                tr(
                    "factual_consistency",
                    "Factual Consistency",
                ),
                tr(
                    "unsupported_claim_risk",
                    "Unsupported Claim Risk",
                ),
            ],

            tr(
                "score",
                "Score",
            ): [
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
    # Human Review Notes
    # =====================================================

    if review_notes:

        st.markdown(
            (
                f"### "
                f"{tr('human_review_notes', 'Human Review Notes')}"
            )
        )

        for note in review_notes:

            st.write(
                f"• {note}"
            )
# =========================================================
# Revision Comparison
# =========================================================

def show_revision_comparison(
    original_evaluation: dict,
    revised_evaluation: dict,
):

    original_compliance = len(
        original_evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    revised_compliance = len(
        revised_evaluation.get(
            "compliance_findings",
            [],
        )
        or []
    )

    original_requirements = len(
        original_evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    revised_requirements = len(
        revised_evaluation.get(
            "requirement_findings",
            [],
        )
        or []
    )

    original_mandatory = (
        original_compliance
        + original_requirements
    )

    revised_mandatory = (
        revised_compliance
        + revised_requirements
    )

    score_change = round(
        revised_evaluation[
            "heuristic_composite_score"
        ]
        - original_evaluation[
            "heuristic_composite_score"
        ],
        1,
    )

    st.subheader(
        tr(
            "revision_result",
            "Revision Result",
        )
    )

    col1, col2, col3 = (
        st.columns(
            3
        )
    )

    col1.metric(
        tr(
            "mandatory_findings",
            "Mandatory Findings",
        ),
        revised_mandatory,
        delta=(
            revised_mandatory
            - original_mandatory
        ),
        delta_color="inverse",
    )

    col2.metric(
        tr(
            "compliance_findings",
            "Compliance Findings",
        ),
        revised_compliance,
        delta=(
            revised_compliance
            - original_compliance
        ),
        delta_color="inverse",
    )

    col3.metric(
        tr(
            "diagnostic_score_change",
            "Diagnostic Score Change",
        ),
        f"{score_change:+.1f}",
    )

    if (
        original_mandatory > 0
        and revised_mandatory == 0
    ):

        st.success(
            tr(
                "all_mandatory_cleared",
                (
                    "All previously detected mandatory "
                    "findings were cleared in the current "
                    "re-check."
                ),
            )
        )

    elif (
        revised_mandatory
        < original_mandatory
    ):

        st.warning(
            tr(
                "some_mandatory_remaining",
                (
                    "Some mandatory findings were removed, "
                    "but additional attention is still required."
                ),
            )
        )

    elif (
        original_mandatory > 0
        and revised_mandatory
        >= original_mandatory
    ):

        st.error(
            tr(
                "mandatory_not_cleared",
                (
                    "The mandatory revision did not clear "
                    "all mandatory findings."
                ),
            )
        )


# =========================================================
# Header
# =========================================================

st.title(
    tr(
        "app_title",
        "🚀 GrowthPilot",
    )
)

st.caption(
    tr(
        "app_caption",
        (
            "AI Campaign Content Copilot "
            "for Brand & Agency Teams"
        ),
    )
)

st.markdown(
    tr(
        "app_description",
        """
**Policy decides what must be corrected.  
Campaign requirements decide what must be completed.  
AI advises what could be improved.  
Humans decide what should ultimately be used.**
""",
    )
)


# =========================================================
# Sidebar — Model + Review Configuration
# =========================================================

with st.sidebar:

    st.caption(
        tr(
            "language_scope_note",
            (
                "UI language only. Submitted content, evidence, "
                "source quotes and model findings are not "
                "automatically translated."
            ),
        )
    )

    st.divider()

    st.header(
        tr(
            "model_configuration",
            "Model Configuration",
        )
    )

    selected_model_name = st.selectbox(
        tr(
            "generator_reviser_model",
            "Generator / Reviser Model",
        ),
        options=list(
            MODEL_OPTIONS.keys()
        ),
    )

    selected_model_key = (
        MODEL_OPTIONS[
            selected_model_name
        ]
    )


    # =====================================================
    # Review Mode
    # =====================================================

    st.markdown(
        (
            f"### "
            f"{tr('review_mode', 'Review Mode')}"
        )
    )

    selected_review_mode = st.radio(
        tr(
            "review_mode_selector",
            "Evaluation Strategy",
        ),
        options=[
            FAST_REVIEW,
            CROSS_JUDGE_REVIEW,
        ],
        format_func=lambda mode: (
            tr(
                "fast_review",
                "⚡ Fast Review",
            )

            if mode == FAST_REVIEW

            else tr(
                "cross_judge_review",
                "🛡️ Cross-Judge Review",
            )
        ),
        key="review_mode_selector",
    )


    # =====================================================
    # Fast Review UI
    # =====================================================

    if selected_review_mode == FAST_REVIEW:

        st.info(
            (
                f"{tr('generator_reviser_model', 'Generator / Reviser Model')}: "
                f"{selected_model_name}\n\n"
                f"{tr('primary_demo_judge', 'Primary Demo Judge')}: "
                f"Step-3.5-Flash"
            )
        )

        st.caption(
            tr(
                "fast_review_description",
                (
                    "Fast Review uses one Judge and prioritizes "
                    "interactive response speed."
                ),
            )
        )


    # =====================================================
    # Cross-Judge UI
    # =====================================================

    else:

        st.info(
            (
                f"{tr('generator_reviser_model', 'Generator / Reviser Model')}: "
                f"{selected_model_name}\n\n"
                f"{tr('cross_judge_models', 'Cross-Judge Models')}:\n\n"
                "Step-3.5-Flash + Qwen3.5-35B-A3B"
            )
        )

        st.caption(
            tr(
                "cross_judge_description",
                (
                    "Two independent Judges evaluate the same content. "
                    "Only cross-judge consensus can authorize automatic "
                    "mandatory correction. Mandatory disagreement is "
                    "escalated to Human Review."
                ),
            )
        )

        st.warning(
            tr(
                "cross_judge_latency_note",
                (
                    "Cross-Judge Review may take significantly longer "
                    "because both models must complete evaluation."
                ),
            )
        )


# =========================================================
# Shared Campaign Context
# =========================================================

st.header(
    tr(
        "campaign_context",
        "Campaign Context",
    )
)


# =========================================================
# Brand / Product
# =========================================================

with st.expander(
    tr(
        "brand_product",
        "Brand & Product",
    ),
    expanded=True,
):

    default_brand_info = """
Brand: LumiSkin

Brand positioning:
Dermatology-inspired daily skincare focused on barrier support and hydration.

Brand tone:
Professional, calm, trustworthy and modern.

Restrictions:
- Do not claim to treat or cure diseases
- Do not invent clinical studies
- Do not guarantee medical results
"""

    default_product_info = """
Product: Barrier Repair Moisturizer

Verified product facts:
- Contains ceramides
- Contains hyaluronic acid
- Designed for dry and sensitive skin
- Lightweight texture
- Non-greasy
- Provides up to 24 hours of hydration
"""

    brand_col, product_col = (
        st.columns(
            2
        )
    )

    with brand_col:

        brand_info = st.text_area(
            tr(
                "brand_information",
                "Brand Information",
            ),
            value=(
                default_brand_info.strip()
            ),
            height=260,
        )

    with product_col:

        product_info = st.text_area(
            tr(
                "verified_product_information",
                "Verified Product Information",
            ),
            value=(
                default_product_info.strip()
            ),
            height=260,
        )


# =========================================================
# Campaign / Requirements
# =========================================================

with st.expander(
    tr(
        "campaign_brief_requirements",
        "Campaign Brief & Requirements",
    ),
    expanded=True,
):

    default_campaign_brief = """
Objective:
Introduce the moisturizer for daily hydration.

Target audience:
Women aged 20-30.

Campaign style:
Natural lifestyle recommendation.

Length:
Approximately 150 Chinese characters.
"""

    campaign_brief = st.text_area(
        tr(
            "campaign_brief",
            "Campaign Brief",
        ),
        value=(
            default_campaign_brief.strip()
        ),
        height=190,
    )

    campaign_col1, campaign_col2 = (
        st.columns(
            2
        )
    )

    with campaign_col1:

        platform = st.text_input(
            tr(
                "platform",
                "Platform",
            ),
            value="Xiaohongshu",
        )

    with campaign_col2:

        content_type = st.text_input(
            tr(
                "content_type",
                "Content Type",
            ),
            value="Creator social post",
        )

    requirement_col, avoid_col = (
        st.columns(
            2
        )
    )

    with requirement_col:

        must_mention_text = st.text_area(
            tr(
                "must_mention",
                "Must Mention",
            ),
            value=(
                "SEMANTIC | Ceramides\n"
                "SEMANTIC | Up to 24 hours "
                "of hydration"
            ),
            height=150,
            help=tr(
                "must_mention_help",
                (
                    "One requirement per line. "
                    "Default is SEMANTIC. "
                    "Use EXACT | #BrandCampaign "
                    "when exact wording is required."
                ),
            ),
        )

    with avoid_col:

        must_avoid_text = st.text_area(
            tr(
                "must_avoid",
                "Must Avoid",
            ),
            value=(
                "Disease treatment or cure claims\n"
                "Invented clinical evidence\n"
                "Guaranteed medical outcomes"
            ),
            height=150,
        )


# =========================================================
# Creator Context
# =========================================================

with st.expander(
    tr(
        "creator_context",
        "Creator Context",
    ),
    expanded=False,
):

    creator_col1, creator_col2 = (
        st.columns(
            2
        )
    )

    with creator_col1:

        creator_category = st.text_input(
            tr(
                "creator_category",
                "Creator Category",
            ),
            value=(
                "Skincare lifestyle creator"
            ),
        )

        creator_audience = st.text_input(
            tr(
                "creator_audience",
                "Creator Audience",
            ),
            value=(
                "Young adults interested in "
                "sensitive-skin routines"
            ),
        )

    with creator_col2:

        creator_style = st.text_input(
            tr(
                "creator_style",
                "Creator Style",
            ),
            value=(
                "Natural, experience-led, "
                "ingredient-aware"
            ),
        )

        creator_characteristics = st.text_input(
            tr(
                "content_characteristics",
                "Content Characteristics",
            ),
            value=(
                "Concise lifestyle sharing with "
                "practical product details"
            ),
        )


# =========================================================
# Policy Context
# =========================================================

with st.expander(
    tr(
        "additional_policy_context",
        "Additional Policy Context",
    ),
    expanded=False,
):

    policy_context = st.text_area(
        tr(
            "policy_optional",
            "Additional Policy Context (Optional)",
        ),
        value="",
        height=160,
        placeholder=tr(
            "policy_placeholder",
            (
                "Paste applicable internal brand rules, "
                "advertising policy, or platform requirements here. "
                "Only supplied rules may be used as hard external "
                "compliance basis."
            ),
        ),
    )

    st.caption(
        tr(
            "rag_note",
            (
                "A future RAG policy layer can populate "
                "this context automatically."
            ),
        )
    )


# =========================================================
# Build Structured Context
# =========================================================

requirements = (
    parse_structured_requirements(
        must_mention_text
    )
)

must_avoid = (
    parse_line_items(
        must_avoid_text
    )
)

brand_context = (
    build_brand_context(
        brand_info=
            brand_info,

        product_info=
            product_info,
    )
)

campaign_context = (
    build_campaign_context(
        campaign_brief=
            campaign_brief,

        platform=
            platform,

        content_type=
            content_type,

        creator_category=
            creator_category,

        creator_audience=
            creator_audience,

        creator_style=
            creator_style,

        creator_characteristics=
            creator_characteristics,

        requirements=
            requirements,

        must_avoid=
            must_avoid,
    )
)


# =========================================================
# Workflow Tabs
# =========================================================

create_tab, review_tab = st.tabs(
    [
        tr(
            "create_tab",
            (
                "✍️ Create Guidance / "
                "Generate Draft"
            ),
        ),

        tr(
            "review_tab",
            "🔎 Review Creator Draft",
        ),
    ]
)


# =========================================================
# Workflow 1 — Create
# =========================================================

with create_tab:

    st.header(
        tr(
            "create_title",
            "Create Guidance / Generate Draft",
        )
    )

    st.write(
        tr(
            "create_description",
            (
                "Generate creator-specific campaign "
                "guidance and a reference draft before "
                "content production."
            ),
        )
    )


    # -----------------------------------------------------
    # Requirements Preview
    # -----------------------------------------------------

    if requirements or must_avoid:

        req_col, avoid_col = (
            st.columns(
                2
            )
        )

        with req_col:

            st.markdown(
                (
                    "#### "
                    f"{tr('must_mention', 'Must Mention')}"
                )
            )

            if requirements:

                for item in requirements:

                    st.write(
                        (
                            "• "
                            f"[{item['match_mode']}] "
                            f"{item['content']}"
                        )
                    )

            else:

                st.write(
                    tr(
                        "none_specified",
                        "None specified.",
                    )
                )

        with avoid_col:

            st.markdown(
                (
                    "#### "
                    f"{tr('must_avoid', 'Must Avoid')}"
                )
            )

            if must_avoid:

                for item in must_avoid:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(
                    tr(
                        "none_specified",
                        "None specified.",
                    )
                )


    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    if st.button(
        tr(
            "generate_guidance_button",
            (
                "Generate Guidance & "
                "Reference Draft"
            ),
        ),
        type="primary",
        use_container_width=True,
        key="create_generate_button",
    ):

        if not brand_info.strip():

            st.error(
                tr(
                    "missing_brand",
                    "Please provide Brand Information.",
                )
            )

        elif not product_info.strip():

            st.error(
                tr(
                    "missing_product",
                    (
                        "Please provide Verified "
                        "Product Information."
                    ),
                )
            )

        elif not campaign_brief.strip():

            st.error(
                tr(
                    "missing_campaign",
                    "Please provide a Campaign Brief.",
                )
            )

        else:

            reset_create_results()

            save_context_snapshot(
                brand_context=
                    brand_context,

                campaign_context=
                    campaign_context,

                policy_context=
                    policy_context,

                requirements=
                    requirements,

                model_key=
                    selected_model_key,

                model_name=
                    selected_model_name,
            )

            try:

                with st.spinner(
                    tr(
                        "creating_guidance",
                        (
                            "Creating guidance with "
                            "{model}..."
                        ),
                        model=(
                            selected_model_name
                        ),
                    )
                ):

                    start = time.perf_counter()

                    guidance = (
                        generate_creator_guidance(
                            brand_context=
                                brand_context,

                            campaign_context=
                                campaign_context,

                            policy_context=
                                policy_context,

                            model_key=
                                selected_model_key,
                        )
                    )

                    guidance_latency = (
                        time.perf_counter()
                        - start
                    )

                st.session_state.create_guidance = (
                    guidance
                )

                st.session_state.create_guidance_latency = round(
                    guidance_latency,
                    2,
                )


                with st.spinner(
                    tr(
                        "generating_reference_draft",
                        (
                            "Generating reference draft "
                            "with {model}..."
                        ),
                        model=(
                            selected_model_name
                        ),
                    )
                ):

                    start = time.perf_counter()

                    reference_draft = (
                        generate_content(
                            brand_info=
                                brand_context,

                            campaign_brief=
                                campaign_context,

                            model_key=
                                selected_model_key,
                        )
                    )

                    generation_latency = (
                        time.perf_counter()
                        - start
                    )

                st.session_state.create_reference_draft = (
                    reference_draft
                )

                st.session_state.create_generation_latency = round(
                    generation_latency,
                    2,
                )

            except Exception as error:

                st.error(
                    tr(
                        "guidance_generation_failed",
                        (
                            "Guidance or draft generation "
                            "failed: {error}"
                        ),
                        error=error,
                    )
                )


    # -----------------------------------------------------
    # Guidance
    # -----------------------------------------------------

    if st.session_state.create_guidance:

        st.subheader(
            tr(
                "creator_guidance",
                "Creator Guidance",
            )
        )

        st.markdown(
            st.session_state.create_guidance
        )

        st.caption(
            tr(
                "guidance_advisory_note",
                (
                    "AI-generated guidance is advisory. "
                    "Structured Must Mention / Must Avoid "
                    "inputs remain authoritative."
                ),
            )
        )


    # -----------------------------------------------------
    # Reference Draft
    # -----------------------------------------------------

    if st.session_state.create_reference_draft:

        st.subheader(
            tr(
                "reference_draft",
                "Reference Draft",
            )
        )

        st.text_area(
            tr(
                "generated_reference_draft",
                "Generated Reference Draft",
            ),
            value=(
                st.session_state
                .create_reference_draft
            ),
            height=240,
            key=(
                "create_reference_"
                "draft_display"
            ),
            disabled=True,
        )

        latency_col1, latency_col2 = (
            st.columns(
                2
            )
        )

        latency_col1.metric(
            tr(
                "guidance_latency",
                "Guidance Latency",
            ),
            (
                f"{st.session_state.create_guidance_latency:.2f}s"
            ),
        )

        latency_col2.metric(
            tr(
                "draft_generation_latency",
                "Draft Generation Latency",
            ),
            (
                f"{st.session_state.create_generation_latency:.2f}s"
            ),
        )


        # -------------------------------------------------
        # Review Reference Draft
        # -------------------------------------------------

        if st.button(
            tr(
                "review_reference_draft",
                "Review Reference Draft",
            ),
            use_container_width=True,
            key=(
                "create_review_"
                "reference_button"
            ),
        ):

            try:

                st.session_state.create_saved_review_mode = (
                    selected_review_mode
                )

                with st.spinner(
                    tr(
                        "running_review",
                        (
                            "Running content review..."
                        ),
                    )
                ):

                    start = time.perf_counter()

                    review_result = review_content(
                        brand_info=(
                            st.session_state
                            .saved_brand_context
                        ),

                        campaign_brief=(
                            st.session_state
                            .saved_campaign_context
                        ),

                        generated_content=(
                            st.session_state
                            .create_reference_draft
                        ),

                        policy_context=(
                            st.session_state
                            .saved_policy_context
                        ),

                        requirements=(
                            st.session_state
                            .saved_requirements
                        ),

                        content_origin=
                            "generated",

                        review_mode=
                            selected_review_mode,
                    )

                    evaluation_latency = (
                        time.perf_counter()
                        - start
                    )

                st.session_state.create_review_result = (
                    review_result
                )

                st.session_state.create_evaluation = (
                    review_result.get(
                        "evaluation"
                    )
                )

                st.session_state.create_evaluation_latency = round(
                    evaluation_latency,
                    2,
                )

            except Exception as error:

                st.error(
                    tr(
                        "reference_review_failed",
                        (
                            "Reference draft review "
                            "failed: {error}"
                        ),
                        error=error,
                    )
                )


    # -----------------------------------------------------
    # Review Result
    # -----------------------------------------------------

    if st.session_state.create_review_result:

        show_review_routing(
            st.session_state
            .create_review_result
        )

    if st.session_state.create_evaluation:

        show_evaluation(
            st.session_state
            .create_evaluation,

            tr(
                "reference_draft_review",
                "Reference Draft Review",
            ),

            review_result=(
                st.session_state
                .create_review_result
            ),
        )

        st.metric(
            tr(
                "review_latency",
                "Review Latency",
            ),
            (
                f"{st.session_state.create_evaluation_latency:.2f}s"
            ),
        )

        create_review_result = (
            st.session_state
            .create_review_result
        )

        create_advisories = (
            st.session_state
            .create_evaluation
            .get(
                "advisory_findings",
                [],
            )
            or []
        )

        st.markdown(
            (
                f"### "
                f"{tr('actions', 'Actions')}"
            )
        )


        # -------------------------------------------------
        # Human Review Required
        # -------------------------------------------------

        if (
            create_review_result
            .get(
                "requires_human_review",
                False,
            )
        ):

            st.warning(
                tr(
                    "automatic_revision_disabled",
                    (
                        "Automatic mandatory revision is disabled "
                        "because the Judges disagree. Resolve the "
                        "disagreement through Human Review first."
                    ),
                )
            )


        # -------------------------------------------------
        # Consensus Mandatory Fix
        # -------------------------------------------------

        elif (
            create_review_result
            .get(
                "can_auto_fix",
                False,
            )
        ):

            st.warning(
                tr(
                    "mandatory_before_optional",
                    (
                        "Cross-Judge consensus authorized "
                        "mandatory correction. Mandatory findings "
                        "should be resolved before optional optimization."
                    ),
                )
            )

            if st.button(
                tr(
                    "mandatory_fix",
                    "Apply Mandatory Fix",
                ),
                type="primary",
                use_container_width=True,
                key=(
                    "create_mandatory_"
                    "fix_button"
                ),
            ):

                try:

                    with st.spinner(
                        tr(
                            "applying_mandatory_revision",
                            (
                                "Applying the minimum "
                                "mandatory revision..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_content = (
                            fix_mandatory_issues(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                original_content=(
                                    st.session_state
                                    .create_reference_draft
                                ),

                                evaluation=(
                                    st.session_state
                                    .create_evaluation
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

                    st.session_state.create_v2_content = (
                        revised_content
                    )

                    st.session_state.create_v2_revision_latency = round(
                        revision_latency,
                        2,
                    )

                    st.session_state.create_v2_mode = (
                        "Mandatory Fix"
                    )


                    with st.spinner(
                        tr(
                            "rechecking_revised_draft",
                            (
                                "Re-checking the revised "
                                "draft using the same review mode..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_review_result = (
                            review_content(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                generated_content=
                                    revised_content,

                                policy_context=(
                                    st.session_state
                                    .saved_policy_context
                                ),

                                requirements=(
                                    st.session_state
                                    .saved_requirements
                                ),

                                content_origin=
                                    "generated",

                                review_mode=(
                                    st.session_state
                                    .create_saved_review_mode
                                ),
                            )
                        )

                        recheck_latency = (
                            time.perf_counter()
                            - start
                        )

                    st.session_state.create_v2_review_result = (
                        revised_review_result
                    )

                    st.session_state.create_v2_evaluation = (
                        revised_review_result.get(
                            "evaluation"
                        )
                    )

                    st.session_state.create_v2_evaluation_latency = round(
                        recheck_latency,
                        2,
                    )

                except Exception as error:

                    st.error(
                        tr(
                            "mandatory_revision_failed",
                            (
                                "Mandatory revision "
                                "failed: {error}"
                            ),
                            error=error,
                        )
                    )


        # -------------------------------------------------
        # Optional Optimization
        # -------------------------------------------------

        elif create_advisories:

            st.info(
                tr(
                    "optional_optimization_available",
                    (
                        "No mandatory action is required. "
                        "Quality optimization remains optional "
                        "and must be triggered by the user."
                    ),
                )
            )

            if st.button(
                tr(
                    "quality_optimization",
                    "Optimize Quality (Optional)",
                ),
                use_container_width=True,
                key="create_optimize_button",
            ):

                try:

                    with st.spinner(
                        tr(
                            "applying_optional_improvements",
                            (
                                "Applying optional "
                                "advisory improvements..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_content = (
                            optimize_quality(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                original_content=(
                                    st.session_state
                                    .create_reference_draft
                                ),

                                evaluation=(
                                    st.session_state
                                    .create_evaluation
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

                    st.session_state.create_v2_content = (
                        revised_content
                    )

                    st.session_state.create_v2_revision_latency = round(
                        revision_latency,
                        2,
                    )

                    st.session_state.create_v2_mode = (
                        "Optional Quality Optimization"
                    )


                    with st.spinner(
                        tr(
                            "reviewing_optimized_content",
                            (
                                "Reviewing optimized content "
                                "using the same review mode..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_review_result = (
                            review_content(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                generated_content=
                                    revised_content,

                                policy_context=(
                                    st.session_state
                                    .saved_policy_context
                                ),

                                requirements=(
                                    st.session_state
                                    .saved_requirements
                                ),

                                content_origin=
                                    "generated",

                                review_mode=(
                                    st.session_state
                                    .create_saved_review_mode
                                ),
                            )
                        )

                        recheck_latency = (
                            time.perf_counter()
                            - start
                        )

                    st.session_state.create_v2_review_result = (
                        revised_review_result
                    )

                    st.session_state.create_v2_evaluation = (
                        revised_review_result.get(
                            "evaluation"
                        )
                    )

                    st.session_state.create_v2_evaluation_latency = round(
                        recheck_latency,
                        2,
                    )

                except Exception as error:

                    st.error(
                        tr(
                            "quality_optimization_failed",
                            (
                                "Quality optimization "
                                "failed: {error}"
                            ),
                            error=error,
                        )
                    )

        else:

            st.success(
                tr(
                    "no_mandatory_issue",
                    (
                        "No mandatory issue or major "
                        "advisory was identified."
                    ),
                )
            )


    # -----------------------------------------------------
    # Revised Reference Draft
    # -----------------------------------------------------

    if st.session_state.create_v2_content:

        st.header(
            tr(
                "revised_reference_draft",
                "Revised Reference Draft",
            )
        )

        st.caption(
            (
                f"{tr('revision_mode', 'Revision Mode')}: "
                f"{st.session_state.create_v2_mode}"
            )
        )

        st.text_area(
            tr(
                "revised_content",
                "Revised Content",
            ),
            value=(
                st.session_state
                .create_v2_content
            ),
            height=240,
            key="create_v2_display",
            disabled=True,
        )

        revision_col1, revision_col2 = (
            st.columns(
                2
            )
        )

        revision_col1.metric(
            tr(
                "revision_latency",
                "Revision Latency",
            ),
            (
                f"{st.session_state.create_v2_revision_latency:.2f}s"
            ),
        )

        revision_col2.metric(
            tr(
                "recheck_latency",
                "Re-check Latency",
            ),
            (
                f"{st.session_state.create_v2_evaluation_latency:.2f}s"
            ),
        )


    if st.session_state.create_v2_review_result:

        show_review_routing(
            st.session_state
            .create_v2_review_result
        )


    if st.session_state.create_v2_evaluation:

        show_revision_comparison(
            original_evaluation=(
                st.session_state
                .create_evaluation
            ),

            revised_evaluation=(
                st.session_state
                .create_v2_evaluation
            ),
        )

        show_evaluation(
            st.session_state
            .create_v2_evaluation,

            tr(
                "revised_draft_review",
                "Revised Draft Review",
            ),

            review_result=(
                st.session_state
                .create_v2_review_result
            ),
        )


# =========================================================
# Workflow 2 — Review Creator Draft
# =========================================================

with review_tab:

    st.header(
        tr(
            "review_creator_title",
            "Review Creator Draft",
        )
    )

    st.write(
        tr(
            "review_creator_description",
            (
                "Paste a creator-returned draft for "
                "pre-publication review. First-person creator "
                "language is treated as creator-authored context, "
                "while factual claims and supplied campaign rules "
                "are reviewed normally."
            ),
        )
    )

    default_creator_draft = """
最近换季我会用 LumiSkin Barrier Repair Moisturizer 做日常保湿。里面有神经酰胺和透明质酸，质地很轻，不会觉得油腻，保湿感可以维持最长 24 小时。作为日常通勤护肤我觉得很方便。
"""

    creator_draft = st.text_area(
        tr(
            "creator_draft",
            "Creator Draft",
        ),
        value=(
            default_creator_draft.strip()
        ),
        height=260,
        key="creator_draft_input",
    )


    # -----------------------------------------------------
    # Review Creator Draft
    # -----------------------------------------------------

    if st.button(
        tr(
            "review_creator_button",
            "Review Creator Draft",
        ),
        type="primary",
        use_container_width=True,
        key="review_creator_button",
    ):

        if not creator_draft.strip():

            st.error(
                tr(
                    "missing_creator_draft",
                    "Please paste a Creator Draft.",
                )
            )

        elif not brand_info.strip():

            st.error(
                tr(
                    "missing_brand",
                    "Please provide Brand Information.",
                )
            )

        elif not product_info.strip():

            st.error(
                tr(
                    "missing_product",
                    (
                        "Please provide Verified "
                        "Product Information."
                    ),
                )
            )

        elif not campaign_brief.strip():

            st.error(
                tr(
                    "missing_campaign",
                    "Please provide a Campaign Brief.",
                )
            )

        else:

            reset_review_results()

            save_context_snapshot(
                brand_context=
                    brand_context,

                campaign_context=
                    campaign_context,

                policy_context=
                    policy_context,

                requirements=
                    requirements,

                model_key=
                    selected_model_key,

                model_name=
                    selected_model_name,
            )

            st.session_state.review_saved_review_mode = (
                selected_review_mode
            )

            st.session_state.review_original_content = (
                creator_draft
            )

            try:

                with st.spinner(
                    tr(
                        "running_review",
                        "Running content review...",
                    )
                ):

                    start = time.perf_counter()

                    review_result = (
                        review_content(
                            brand_info=
                                brand_context,

                            campaign_brief=
                                campaign_context,

                            generated_content=
                                creator_draft,

                            policy_context=
                                policy_context,

                            requirements=
                                requirements,

                            content_origin=
                                "creator_draft",

                            review_mode=
                                selected_review_mode,
                        )
                    )

                    evaluation_latency = (
                        time.perf_counter()
                        - start
                    )

                st.session_state.review_review_result = (
                    review_result
                )

                st.session_state.review_evaluation = (
                    review_result.get(
                        "evaluation"
                    )
                )

                st.session_state.review_evaluation_latency = round(
                    evaluation_latency,
                    2,
                )

            except Exception as error:

                st.error(
                    tr(
                        "creator_review_failed",
                        (
                            "Creator Draft review "
                            "failed: {error}"
                        ),
                        error=error,
                    )
                )


    # -----------------------------------------------------
    # Review Result
    # -----------------------------------------------------

    if st.session_state.review_review_result:

        show_review_routing(
            st.session_state
            .review_review_result
        )


    if st.session_state.review_evaluation:

        show_evaluation(
            st.session_state
            .review_evaluation,

            tr(
                "creator_draft_review",
                "Creator Draft Review",
            ),

            review_result=(
                st.session_state
                .review_review_result
            ),
        )

        st.metric(
            tr(
                "review_latency",
                "Review Latency",
            ),
            (
                f"{st.session_state.review_evaluation_latency:.2f}s"
            ),
        )

        review_result = (
            st.session_state
            .review_review_result
        )

        review_advisories = (
            st.session_state
            .review_evaluation
            .get(
                "advisory_findings",
                [],
            )
            or []
        )

        st.markdown(
            (
                f"### "
                f"{tr('actions', 'Actions')}"
            )
        )


        # -------------------------------------------------
        # Human Review
        # -------------------------------------------------

        if review_result.get(
            "requires_human_review",
            False,
        ):

            st.warning(
                tr(
                    "automatic_revision_disabled",
                    (
                        "Automatic mandatory revision is disabled "
                        "because the Judges disagree. Human Review "
                        "is required before content can proceed."
                    ),
                )
            )


        # -------------------------------------------------
        # Consensus Mandatory Fix
        # -------------------------------------------------

        elif review_result.get(
            "can_auto_fix",
            False,
        ):

            st.warning(
                tr(
                    "creator_mandatory_warning",
                    (
                        "Cross-Judge consensus confirmed mandatory "
                        "findings. The revision below applies only "
                        "the minimum mandatory changes and still "
                        "requires final human approval."
                    ),
                )
            )

            if st.button(
                tr(
                    "mandatory_fix",
                    "Apply Mandatory Fix",
                ),
                type="primary",
                use_container_width=True,
                key=(
                    "review_mandatory_"
                    "fix_button"
                ),
            ):

                try:

                    with st.spinner(
                        tr(
                            "applying_mandatory_revision",
                            (
                                "Applying the minimum "
                                "mandatory revision..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_content = (
                            fix_mandatory_issues(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                original_content=(
                                    st.session_state
                                    .review_original_content
                                ),

                                evaluation=(
                                    st.session_state
                                    .review_evaluation
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

                    st.session_state.review_v2_content = (
                        revised_content
                    )

                    st.session_state.review_v2_revision_latency = round(
                        revision_latency,
                        2,
                    )

                    st.session_state.review_v2_mode = (
                        "Mandatory Fix"
                    )


                    with st.spinner(
                        tr(
                            "rechecking_creator_draft",
                            (
                                "Re-checking revised Creator Draft "
                                "using the same review mode..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_review_result = (
                            review_content(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                generated_content=
                                    revised_content,

                                policy_context=(
                                    st.session_state
                                    .saved_policy_context
                                ),

                                requirements=(
                                    st.session_state
                                    .saved_requirements
                                ),

                                content_origin=
                                    "creator_draft",

                                review_mode=(
                                    st.session_state
                                    .review_saved_review_mode
                                ),
                            )
                        )

                        recheck_latency = (
                            time.perf_counter()
                            - start
                        )

                    st.session_state.review_v2_review_result = (
                        revised_review_result
                    )

                    st.session_state.review_v2_evaluation = (
                        revised_review_result.get(
                            "evaluation"
                        )
                    )

                    st.session_state.review_v2_evaluation_latency = round(
                        recheck_latency,
                        2,
                    )

                except Exception as error:

                    st.error(
                        tr(
                            "mandatory_revision_failed",
                            (
                                "Mandatory revision "
                                "failed: {error}"
                            ),
                            error=error,
                        )
                    )


        # -------------------------------------------------
        # Optional Optimization
        # -------------------------------------------------

        elif review_advisories:

            st.info(
                tr(
                    "creator_optional_optimization",
                    (
                        "No mandatory action is required. "
                        "Advisory optimization remains optional "
                        "and is only applied when triggered by the user."
                    ),
                )
            )

            if st.button(
                tr(
                    "quality_optimization",
                    "Optimize Quality (Optional)",
                ),
                use_container_width=True,
                key="review_optimize_button",
            ):

                try:

                    with st.spinner(
                        tr(
                            "applying_optional_improvements",
                            (
                                "Applying optional "
                                "advisory improvements..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_content = (
                            optimize_quality(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                original_content=(
                                    st.session_state
                                    .review_original_content
                                ),

                                evaluation=(
                                    st.session_state
                                    .review_evaluation
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

                    st.session_state.review_v2_content = (
                        revised_content
                    )

                    st.session_state.review_v2_revision_latency = round(
                        revision_latency,
                        2,
                    )

                    st.session_state.review_v2_mode = (
                        "Optional Quality Optimization"
                    )


                    with st.spinner(
                        tr(
                            "reviewing_optimized_creator_draft",
                            (
                                "Reviewing optimized Creator Draft "
                                "using the same review mode..."
                            ),
                        )
                    ):

                        start = time.perf_counter()

                        revised_review_result = (
                            review_content(
                                brand_info=(
                                    st.session_state
                                    .saved_brand_context
                                ),

                                campaign_brief=(
                                    st.session_state
                                    .saved_campaign_context
                                ),

                                generated_content=
                                    revised_content,

                                policy_context=(
                                    st.session_state
                                    .saved_policy_context
                                ),

                                requirements=(
                                    st.session_state
                                    .saved_requirements
                                ),

                                content_origin=
                                    "creator_draft",

                                review_mode=(
                                    st.session_state
                                    .review_saved_review_mode
                                ),
                            )
                        )

                        recheck_latency = (
                            time.perf_counter()
                            - start
                        )

                    st.session_state.review_v2_review_result = (
                        revised_review_result
                    )

                    st.session_state.review_v2_evaluation = (
                        revised_review_result.get(
                            "evaluation"
                        )
                    )

                    st.session_state.review_v2_evaluation_latency = round(
                        recheck_latency,
                        2,
                    )

                except Exception as error:

                    st.error(
                        tr(
                            "quality_optimization_failed",
                            (
                                "Quality optimization "
                                "failed: {error}"
                            ),
                            error=error,
                        )
                    )

        else:

            st.success(
                tr(
                    "no_mandatory_issue",
                    (
                        "No mandatory issue or major "
                        "advisory was identified."
                    ),
                )
            )


    # -----------------------------------------------------
    # Revised Creator Draft
    # -----------------------------------------------------

    if st.session_state.review_v2_content:

        st.header(
            tr(
                "revised_creator_draft",
                (
                    "AI-Assisted Revised "
                    "Creator Draft"
                ),
            )
        )

        st.caption(
            (
                f"{tr('revision_mode', 'Revision Mode')}: "
                f"{st.session_state.review_v2_mode}"
            )
        )

        st.text_area(
            tr(
                "revised_content",
                "Revised Content",
            ),
            value=(
                st.session_state
                .review_v2_content
            ),
            height=260,
            key="review_v2_display",
            disabled=True,
        )

        revision_col1, revision_col2 = (
            st.columns(
                2
            )
        )

        revision_col1.metric(
            tr(
                "revision_latency",
                "Revision Latency",
            ),
            (
                f"{st.session_state.review_v2_revision_latency:.2f}s"
            ),
        )

        revision_col2.metric(
            tr(
                "recheck_latency",
                "Re-check Latency",
            ),
            (
                f"{st.session_state.review_v2_evaluation_latency:.2f}s"
            ),
        )


    if st.session_state.review_v2_review_result:

        show_review_routing(
            st.session_state
            .review_v2_review_result
        )


    if st.session_state.review_v2_evaluation:

        show_revision_comparison(
            original_evaluation=(
                st.session_state
                .review_evaluation
            ),

            revised_evaluation=(
                st.session_state
                .review_v2_evaluation
            ),
        )

        show_evaluation(
            st.session_state
            .review_v2_evaluation,

            tr(
                "revised_creator_review",
                (
                    "Revised Creator "
                    "Draft Review"
                ),
            ),

            review_result=(
                st.session_state
                .review_v2_review_result
            ),
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    tr(
        "footer",
        (
            "GrowthPilot MVP v2 · "
            "Creator-Specific Guidance + "
            "Cross-Judge Policy Review + "
            "Human-in-the-Loop Approval"
        ),
    )
)