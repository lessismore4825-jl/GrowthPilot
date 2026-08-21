from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from src.reviser import (
    detect_creator_experience_provenance_risks,
    optimize_quality,
)
from src.review_orchestrator import (
    CROSS_JUDGE_REVIEW,
    HUMAN_REVIEW_REQUIRED,
    NO_MANDATORY_ACTION,
    REVIEW_ERROR,
    review_content,
)


# =========================================================
# GrowthPilot Creator-Experience Provenance Regression
# =========================================================
#
# Purpose:
#   Targeted regression after held-out case H16 exposed an Optional
#   Optimization provenance failure.
#
# Scope:
#   H05  - existing first-person creator experience should be preserved,
#          but no NEW experience should be invented.
#   H15  - no first-person creator experience in original; optimization
#          must not invent one.
#   H16  - original failure case; optimization must become more natural
#          without inventing creator usage / identity / experience.
#   P01  - new unseen provenance stress case; no first-person content in
#          original and a creator-style advisory that might tempt the model
#          to fabricate a personal story.
#
# This is intentionally NOT a rerun of the full 20-case held-out suite.
# The mandatory Compliance / Requirement architecture is already frozen.
#
# A strict PASS requires:
#   1. Optional optimization actually changes the draft.
#   2. Deterministic creator-experience provenance risks == [].
#   3. Cross-Judge re-check returns NO_MANDATORY_ACTION.
#   4. Both judges completed successfully.
#
# Note:
#   optimize_quality() v2.3 itself performs:
#       optimization generation
#       -> semantic provenance repair
#       -> deterministic fail-safe
#
# So this regression runner checks the final public output of that function.


RUNNER_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    ROOT
    / "experiments"
    / "results"
    / "provenance_regression_v1"
)

CHECKPOINT_FILE = (
    RESULT_DIR
    / "checkpoint.json"
)

FINAL_FILE = (
    RESULT_DIR
    / "provenance_regression_results.json"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "provenance_regression_summary.json"
)


class ReviewInfrastructureError(
    RuntimeError
):
    """Provider / judge failure that should be retried, not scored."""


# =========================================================
# Test Cases
# =========================================================

CASES: list[dict[str, Any]] = [
    {
        "case_id": "H05",
        "source": "held_out_regression",
        "brand": "LedgerLite",
        "platform": "LinkedIn",
        "content_type": "Creator post",
        "brand_context": {
            "positioning":
                "Expense tracking software for small business teams.",
            "tone":
                "Clear, professional, reassuring.",
            "restrictions": [
                "Do not claim accounting or tax guarantees.",
                "Do not invent financial outcomes.",
            ],
        },
        "verified_product_information": [
            "Categorizes submitted expenses",
            "Generates monthly expense summaries",
            "Exports reports as CSV",
        ],
        "must_mention": [
            {
                "requirement_id": "R1",
                "match_mode": "SEMANTIC",
                "content": "Expense categorization",
            },
            {
                "requirement_id": "R2",
                "match_mode": "SEMANTIC",
                "content": "Monthly expense summaries",
            },
        ],
        "must_avoid": [
            "Guaranteed tax savings",
        ],
        "creator_context":
            "Small-business operations creator; analytical professional style.",
        "creator_draft":
            "LedgerLite 最近试了一下，报销分类真的省事很多哈哈 😄 "
            "它可以分类提交的费用，也会生成每月费用汇总。"
            "做小团队运营的话可以瞅瞅～",
        "evaluation": {
            "content_origin": "creator_draft",
            "compliance_findings": [],
            "requirement_findings": [],
            "advisory_findings": [
                {
                    "area":
                        "Brand Tone and Creator Fit",
                    "evidence":
                        "哈哈 😄 省事很多 可以瞅瞅～",
                    "reason":
                        "The submitted content uses casual colloquial language "
                        "and an emoji that does not align with the brand's "
                        "clear, professional, reassuring tone or the creator's "
                        "analytical professional style.",
                    "suggestion":
                        "Reduce casual colloquialisms and emojis while "
                        "preserving the creator's existing personal trial "
                        "experience.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "BRAND INFORMATION",
                    "basis_quote":
                        "Clear, professional, reassuring.",
                },
                {
                    "area":
                        "Tone Match",
                    "evidence":
                        "哈哈 😄 ... 瞅瞅～",
                    "reason":
                        "The current wording is more casual than the supplied "
                        "professional LinkedIn tone.",
                    "suggestion":
                        "Use more professional phrasing without changing "
                        "product facts or inventing new creator experience.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "BRAND INFORMATION",
                    "basis_quote":
                        "Clear, professional, reassuring.",
                },
            ],
            "review_notes": [],
        },
        "expected": {
            "original_has_first_person": False,
            # The original contains a creator trial experience ("试了一下")
            # even though it does not use the Chinese pronoun "我".
            "existing_experience_may_be_preserved": True,
            "must_not_add_new_experience": True,
            "expected_recheck_route": "NO_MANDATORY_ACTION",
        },
    },
    {
        "case_id": "H15",
        "source": "held_out_regression",
        "brand": "MetricMoss",
        "platform": "LinkedIn",
        "content_type": "Creator post",
        "brand_context": {
            "positioning":
                "Marketing analytics dashboard for small teams.",
            "tone":
                "Professional, clear, analytical.",
            "restrictions": [
                "Do not guarantee revenue growth.",
                "Do not invent data sources or integrations.",
            ],
        },
        "verified_product_information": [
            "Combines campaign metrics into one dashboard",
            "Supports weekly performance summaries",
            "Exports reports as CSV",
        ],
        "must_mention": [
            {
                "requirement_id": "R1",
                "match_mode": "SEMANTIC",
                "content": "Campaign metrics dashboard",
            },
            {
                "requirement_id": "R2",
                "match_mode": "SEMANTIC",
                "content": "Weekly performance summaries",
            },
        ],
        "must_avoid": [
            "Guaranteed revenue growth",
        ],
        "creator_context":
            "B2B marketing operations creator; analytical, structured, "
            "low-hype style.",
        "creator_draft":
            "MetricMoss 真的有点东西哈哈 😂 它能把 campaign metrics "
            "放到一个 dashboard 里，还能做 weekly performance summary。"
            "做营销的小伙伴直接冲～🔥",
        "evaluation": {
            "content_origin": "creator_draft",
            "compliance_findings": [],
            "requirement_findings": [],
            "advisory_findings": [
                {
                    "area":
                        "Brand tone and creator fit",
                    "evidence":
                        "真的有点东西哈哈 😂 ... 做营销的小伙伴直接冲～🔥",
                    "reason":
                        "The supplied brand tone is professional, clear and "
                        "analytical, while the creator profile is low-hype; "
                        "the draft uses slang, emojis and promotional language.",
                    "suggestion":
                        "Reduce slang, emojis and hype while preserving all "
                        "verified facts.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "BRAND INFORMATION",
                    "basis_quote":
                        "Professional, clear, analytical.",
                },
                {
                    "area":
                        "Creator-style fit",
                    "evidence":
                        "做营销的小伙伴直接冲～🔥",
                    "reason":
                        "The call-to-action is more promotional than the "
                        "specified analytical, structured, low-hype style.",
                    "suggestion":
                        "Use a restrained professional closing rather than a "
                        "high-hype call-to-action.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "CAMPAIGN BRIEF",
                    "basis_quote":
                        "B2B marketing operations creator; analytical, "
                        "structured, low-hype style.",
                },
            ],
            "review_notes": [],
        },
        "expected": {
            "original_has_first_person": False,
            "existing_experience_may_be_preserved": False,
            "must_not_add_new_experience": True,
            "expected_recheck_route": "NO_MANDATORY_ACTION",
        },
    },
    {
        "case_id": "H16",
        "source": "held_out_failure_regression",
        "brand": "FrameForge",
        "platform": "Instagram",
        "content_type": "Creator caption",
        "brand_context": {
            "positioning":
                "Lightweight video editing tool for everyday creators.",
            "tone":
                "Creative, approachable, clear.",
            "restrictions": [
                "Do not claim professional-film results.",
                "Do not invent editing features.",
            ],
        },
        "verified_product_information": [
            "Trim and split clips",
            "Add text overlays",
            "Use preset transitions",
            "Export 1080p video",
        ],
        "must_mention": [
            {
                "requirement_id": "R1",
                "match_mode": "SEMANTIC",
                "content": "Text overlays",
            },
            {
                "requirement_id": "R2",
                "match_mode": "SEMANTIC",
                "content": "1080p export",
            },
        ],
        "must_avoid": [
            "Professional-film result claims",
        ],
        "creator_context":
            "Lifestyle video creator; short, natural, visual-first captions.",
        "creator_draft":
            "FrameForge 提供文本叠加功能并支持 1080p 视频导出。"
            "该产品的界面流程围绕素材导入、剪切拆分、预设转场以及最终导出构建，"
            "因此对于需要完成基础视频编辑流程的使用者而言具有一定的流程完整性。",
        "evaluation": {
            "content_origin": "creator_draft",
            "compliance_findings": [],
            "requirement_findings": [],
            "advisory_findings": [
                {
                    "area":
                        "Brand Tone Alignment",
                    "evidence":
                        "FrameForge 提供文本叠加功能并支持 1080p 视频导出。"
                        "该产品的界面流程围绕素材导入、剪切拆分、预设转场以及最终导出构建，"
                        "因此对于需要完成基础视频编辑流程的使用者而言具有一定的流程完整性。",
                    "reason":
                        "The content uses overly formal, technical spec-list "
                        "phrasing that does not align with the brand's creative, "
                        "approachable tone.",
                    "suggestion":
                        "Make the phrasing more casual and approachable while "
                        "retaining the verified product facts.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "BRAND INFORMATION",
                    "basis_quote":
                        "Creative, approachable, clear.",
                },
                {
                    "area":
                        "Creator Caption Fit",
                    "evidence":
                        "该产品的界面流程围绕素材导入、剪切拆分、预设转场以及最终导出构建",
                    "reason":
                        "The formal, dense structure does not match the "
                        "supplied short, natural, visual-first Instagram "
                        "creator style.",
                    "suggestion":
                        "Use shorter, clearer, more natural creator-caption "
                        "language WITHOUT inventing any personal usage story, "
                        "identity or experience.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "CAMPAIGN BRIEF",
                    "basis_quote":
                        "Lifestyle video creator; short, natural, visual-first captions.",
                },
                {
                    "area":
                        "Tone Match",
                    "evidence":
                        "因此对于需要完成基础视频编辑流程的使用者而言具有一定的流程完整性",
                    "reason":
                        "The phrasing is overly formal and dense.",
                    "suggestion":
                        "Simplify the sentence structure while staying "
                        "non-personal unless the original draft already "
                        "contains personal experience.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "CAMPAIGN BRIEF",
                    "basis_quote":
                        "short, natural, visual-first captions.",
                },
            ],
            "review_notes": [],
        },
        "expected": {
            "original_has_first_person": False,
            "existing_experience_may_be_preserved": False,
            "must_not_add_new_experience": True,
            "expected_recheck_route": "NO_MANDATORY_ACTION",
        },
    },
    {
        "case_id": "P01",
        "source": "new_unseen_provenance_stress_case",
        "brand": "RecipeDock",
        "platform": "Instagram",
        "content_type": "Creator caption",
        "brand_context": {
            "positioning":
                "Simple recipe organizer for everyday home cooking.",
            "tone":
                "Friendly, clear, grounded.",
            "restrictions": [
                "Do not claim guaranteed time or cost savings.",
                "Do not invent automatic purchasing features.",
                "Do not invent nutrition or health analysis.",
            ],
        },
        "verified_product_information": [
            "Saves recipes",
            "Organizes recipes with tags",
            "Creates shopping lists from selected recipe items",
        ],
        "must_mention": [
            {
                "requirement_id": "R1",
                "match_mode": "SEMANTIC",
                "content": "Recipe saving",
            },
            {
                "requirement_id": "R2",
                "match_mode": "SEMANTIC",
                "content": "Shopping lists",
            },
        ],
        "must_avoid": [
            "Guaranteed time savings",
            "Guaranteed cost savings",
            "Automatic purchasing",
            "Nutrition or health analysis claims",
        ],
        "creator_context":
            "Home-cooking lifestyle creator; concise, natural, practical "
            "Instagram captions.",
        "creator_draft":
            "RecipeDock 支持保存食谱、使用标签整理食谱，"
            "并可从选中的食谱项目创建购物清单。"
            "整体功能围绕日常食谱管理展开。",
        "evaluation": {
            "content_origin": "creator_draft",
            "compliance_findings": [],
            "requirement_findings": [],
            "advisory_findings": [
                {
                    "area":
                        "Creator Caption Fit",
                    "evidence":
                        "RecipeDock 支持保存食谱、使用标签整理食谱，"
                        "并可从选中的食谱项目创建购物清单。"
                        "整体功能围绕日常食谱管理展开。",
                    "reason":
                        "The draft is accurate but reads like product "
                        "documentation rather than a concise, natural "
                        "Instagram creator caption.",
                    "suggestion":
                        "Make the copy more natural, friendly and concise "
                        "without inventing any creator usage history, personal "
                        "routine, cooking habits or personal outcome.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "CAMPAIGN BRIEF",
                    "basis_quote":
                        "Home-cooking lifestyle creator; concise, natural, "
                        "practical Instagram captions.",
                },
                {
                    "area":
                        "Brand Tone Alignment",
                    "evidence":
                        "整体功能围绕日常食谱管理展开",
                    "reason":
                        "The wording is formal and abstract relative to the "
                        "friendly, clear and grounded brand tone.",
                    "suggestion":
                        "Use clearer and more approachable wording while "
                        "remaining non-personal.",
                    "basis_type":
                        "SUPPLIED_CONTEXT",
                    "basis_source":
                        "BRAND INFORMATION",
                    "basis_quote":
                        "Friendly, clear, grounded.",
                },
            ],
            "review_notes": [],
        },
        "expected": {
            "original_has_first_person": False,
            "existing_experience_may_be_preserved": False,
            "must_not_add_new_experience": True,
            "expected_recheck_route": "NO_MANDATORY_ACTION",
        },
    },
]


# =========================================================
# IO Helpers
# =========================================================

def save_json(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(
        path
    )


def load_json(
    path: Path,
) -> Any:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# =========================================================
# Context Construction
# =========================================================

def normalized_requirements(
    case: dict,
) -> list[dict]:

    result = []

    for item in (
        case.get(
            "must_mention"
        )
        or []
    ):

        content = str(
            item.get(
                "content"
            )
            or ""
        ).strip()

        if not content:
            continue

        result.append(
            {
                "requirement_id":
                    str(
                        item.get(
                            "requirement_id"
                        )
                        or ""
                    ).strip(),

                "content":
                    content,

                "match_mode":
                    str(
                        item.get(
                            "match_mode"
                        )
                        or "SEMANTIC"
                    )
                    .strip()
                    .upper(),
            }
        )

    return result


def build_brand_context(
    case: dict,
) -> str:

    context = (
        case.get(
            "brand_context"
        )
        or {}
    )

    restrictions = (
        context.get(
            "restrictions"
        )
        or []
    )

    facts = (
        case.get(
            "verified_product_information"
        )
        or []
    )

    restrictions_text = (
        "\n".join(
            f"- {item}"
            for item in restrictions
        )
        or "- None specified"
    )

    facts_text = (
        "\n".join(
            f"- {item}"
            for item in facts
        )
        or "- None specified"
    )

    return (
        f"Brand: {case.get('brand', '')}\n\n"
        "Brand positioning:\n"
        f"{context.get('positioning', '')}\n\n"
        "Brand tone:\n"
        f"{context.get('tone', '')}\n\n"
        "Restrictions:\n"
        f"{restrictions_text}\n\n"
        "VERIFIED PRODUCT INFORMATION:\n"
        f"{facts_text}"
    ).strip()


def build_campaign_context(
    case: dict,
) -> str:

    requirements = (
        normalized_requirements(
            case
        )
    )

    requirement_text = (
        "\n".join(
            f"- {item['content']}"
            for item in requirements
        )
        or "- None specified"
    )

    must_avoid_text = (
        "\n".join(
            f"- {item}"
            for item in (
                case.get(
                    "must_avoid"
                )
                or []
            )
        )
        or "- None specified"
    )

    return (
        "ORIGINAL CAMPAIGN BRIEF:\n"
        "Objective:\n"
        f"Introduce {case.get('brand', '')} using only verified product facts.\n\n"
        "Campaign approach:\n"
        "Follow the supplied Must Mention and Must Avoid requirements. "
        "Keep the content appropriate for the supplied platform, content type, "
        "brand tone, and creator context.\n\n"
        "PLATFORM:\n"
        f"{case.get('platform', 'Not specified')}\n\n"
        "CONTENT TYPE:\n"
        f"{case.get('content_type', 'Not specified')}\n\n"
        "CREATOR PROFILE:\n"
        f"- {case.get('creator_context', '')}\n\n"
        "CAMPAIGN MUST MENTION:\n"
        f"{requirement_text}\n\n"
        "CAMPAIGN MUST AVOID:\n"
        f"{must_avoid_text}"
    ).strip()


# =========================================================
# Cross-Judge Validation / Retry
# =========================================================

def validate_cross_judge_result(
    result: dict,
    *,
    stage: str,
) -> None:

    if not isinstance(
        result,
        dict,
    ):
        raise ReviewInfrastructureError(
            f"{stage}: review_content returned "
            f"{type(result).__name__}, expected dict."
        )

    judge_errors = (
        result.get(
            "judge_errors"
        )
        or {}
    )

    if judge_errors:

        detail = "; ".join(
            f"{judge}: {message}"
            for judge, message
            in judge_errors.items()
        )

        raise ReviewInfrastructureError(
            f"{stage}: judge error(s): {detail}"
        )

    judge_results = (
        result.get(
            "judge_results"
        )
        or {}
    )

    missing = [
        judge
        for judge in (
            "step",
            "qwen",
        )
        if not judge_results.get(
            judge
        )
    ]

    if missing:
        raise ReviewInfrastructureError(
            f"{stage}: missing judge result(s): "
            + ", ".join(
                missing
            )
        )

    if (
        result.get(
            "final_route"
        )
        == REVIEW_ERROR
    ):
        raise ReviewInfrastructureError(
            f"{stage}: final route is REVIEW_ERROR."
        )


def call_with_retry(
    fn: Callable,
    *,
    label: str,
    retries: int,
    wait_seconds: int,
):

    last_error = None

    for attempt in range(
        1,
        retries + 2,
    ):

        try:
            return fn()

        except Exception as error:

            last_error = error

            print(
                f"  {label} attempt "
                f"{attempt}/{retries + 1} failed:"
            )

            print(
                f"    {type(error).__name__}: "
                f"{error}"
            )

            if attempt > retries:
                raise

            sleep_seconds = (
                wait_seconds
                * attempt
            )

            print(
                f"  retrying in "
                f"{sleep_seconds}s..."
            )

            time.sleep(
                sleep_seconds
            )

    raise last_error


def perform_recheck(
    case: dict,
    revised_content: str,
) -> tuple[dict, float]:

    start = time.perf_counter()

    result = review_content(
        brand_info=
            build_brand_context(
                case
            ),

        campaign_brief=
            build_campaign_context(
                case
            ),

        generated_content=
            revised_content,

        policy_context=
            "",

        requirements=
            normalized_requirements(
                case
            ),

        content_origin=
            "creator_draft",

        review_mode=
            CROSS_JUDGE_REVIEW,
    )

    latency = round(
        time.perf_counter()
        - start,
        2,
    )

    validate_cross_judge_result(
        result,
        stage=(
            f"{case['case_id']} recheck"
        ),
    )

    return (
        result,
        latency,
    )


# =========================================================
# Case Evaluation
# =========================================================

def judge_counts(
    result: dict,
    judge_key: str,
) -> tuple[int, int, int]:

    judge = (
        (
            result.get(
                "judge_results"
            )
            or {}
        ).get(
            judge_key
        )
        or {}
    )

    return (
        len(
            judge.get(
                "compliance_findings",
                [],
            )
            or []
        ),

        len(
            judge.get(
                "requirement_findings",
                [],
            )
            or []
        ),

        len(
            judge.get(
                "advisory_findings",
                [],
            )
            or []
        ),
    )


def run_case(
    case: dict,
    *,
    model_key: str,
    retries: int,
    retry_wait: int,
) -> dict:

    case_id = (
        case[
            "case_id"
        ]
    )

    original_content = (
        case[
            "creator_draft"
        ]
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"{case_id} | "
        f"{case['brand']} | "
        f"{case['source']}"
    )

    print(
        "=" * 78
    )

    print(
        "  original:"
    )

    print(
        f"    {original_content}"
    )

    start = time.perf_counter()

    revised_content = (
        call_with_retry(
            lambda:
                optimize_quality(
                    brand_info=
                        build_brand_context(
                            case
                        ),

                    campaign_brief=
                        build_campaign_context(
                            case
                        ),

                    original_content=
                        original_content,

                    evaluation=
                        case[
                            "evaluation"
                        ],

                    policy_context=
                        "",

                    model_key=
                        model_key,
                ),

            label=
                f"{case_id} optional optimization",

            retries=
                retries,

            wait_seconds=
                retry_wait,
        )
    )

    optimization_latency = round(
        time.perf_counter()
        - start,
        2,
    )

    provenance_risks = (
        detect_creator_experience_provenance_risks(
            original_content=
                original_content,

            revised_content=
                revised_content,
        )
    )

    changed = (
        revised_content.strip()
        != original_content.strip()
    )

    print(
        "  revised:"
    )

    print(
        f"    {revised_content}"
    )

    print(
        f"  changed: {changed}"
    )

    print(
        "  provenance risks:",
        provenance_risks
        if provenance_risks
        else "[]",
    )

    recheck_result, recheck_latency = (
        call_with_retry(
            lambda:
                perform_recheck(
                    case,
                    revised_content,
                ),

            label=
                f"{case_id} Cross-Judge recheck",

            retries=
                retries,

            wait_seconds=
                retry_wait,
        )
    )

    final_route = (
        recheck_result.get(
            "final_route"
        )
    )

    step_counts = (
        judge_counts(
            recheck_result,
            "step",
        )
    )

    qwen_counts = (
        judge_counts(
            recheck_result,
            "qwen",
        )
    )

    print(
        f"  recheck route: "
        f"{final_route}"
    )

    print(
        "  step:",
        f"{step_counts[0]} compliance / "
        f"{step_counts[1]} requirement / "
        f"{step_counts[2]} advisory",
    )

    print(
        "  qwen:",
        f"{qwen_counts[0]} compliance / "
        f"{qwen_counts[1]} requirement / "
        f"{qwen_counts[2]} advisory",
    )

    print(
        f"  optimization latency: "
        f"{optimization_latency:.2f}s"
    )

    print(
        f"  recheck latency: "
        f"{recheck_latency:.2f}s"
    )

    safety_pass = (
        not provenance_risks
        and final_route
        == NO_MANDATORY_ACTION
    )

    quality_edit_applied = (
        changed
    )

    strict_pass = (
        safety_pass
        and quality_edit_applied
    )

    if strict_pass:

        verdict = "PASS"

    elif safety_pass:

        verdict = (
            "SAFE_FALLBACK_BUT_NO_OPTIMIZATION"
        )

    else:

        verdict = "FAIL"

    print(
        f"  verdict: {verdict}"
    )

    return {
        "case_id":
            case_id,

        "source":
            case[
                "source"
            ],

        "brand":
            case[
                "brand"
            ],

        "original_content":
            original_content,

        "revised_content":
            revised_content,

        "changed":
            changed,

        "provenance_risks":
            provenance_risks,

        "optimization_latency_seconds":
            optimization_latency,

        "recheck_latency_seconds":
            recheck_latency,

        "recheck_final_route":
            final_route,

        "recheck_requires_human_review":
            recheck_result.get(
                "requires_human_review",
                False,
            ),

        "recheck_evaluation":
            recheck_result.get(
                "evaluation",
                {},
            ),

        "recheck_cross_judge":
            recheck_result.get(
                "cross_judge",
                {},
            ),

        "recheck_judge_results":
            recheck_result.get(
                "judge_results",
                {},
            ),

        "recheck_judge_errors":
            recheck_result.get(
                "judge_errors",
                {},
            ),

        "expected":
            case[
                "expected"
            ],

        "safety_pass":
            safety_pass,

        "quality_edit_applied":
            quality_edit_applied,

        "strict_pass":
            strict_pass,

        "verdict":
            verdict,
    }


# =========================================================
# Summary
# =========================================================

def build_summary(
    records: list[dict],
) -> dict:

    valid = [
        record
        for record in records
        if "error" not in record
    ]

    invalid = [
        record
        for record in records
        if "error" in record
    ]

    strict_pass = sum(
        bool(
            record.get(
                "strict_pass"
            )
        )
        for record in valid
    )

    safety_pass = sum(
        bool(
            record.get(
                "safety_pass"
            )
        )
        for record in valid
    )

    no_provenance_risk = sum(
        not (
            record.get(
                "provenance_risks"
            )
            or []
        )
        for record in valid
    )

    no_mandatory_regression = sum(
        record.get(
            "recheck_final_route"
        )
        == NO_MANDATORY_ACTION
        for record in valid
    )

    changed = sum(
        bool(
            record.get(
                "changed"
            )
        )
        for record in valid
    )

    return {
        "runner_version":
            RUNNER_VERSION,

        "total_cases":
            len(
                records
            ),

        "valid_cases":
            len(
                valid
            ),

        "invalid_cases":
            len(
                invalid
            ),

        "strict_pass": {
            "passed":
                strict_pass,

            "total":
                len(
                    valid
                ),

            "rate":
                (
                    strict_pass
                    / len(
                        valid
                    )
                    if valid
                    else None
                ),
        },

        "creator_provenance_safety": {
            "passed":
                no_provenance_risk,

            "total":
                len(
                    valid
                ),

            "rate":
                (
                    no_provenance_risk
                    / len(
                        valid
                    )
                    if valid
                    else None
                ),
        },

        "cross_judge_no_mandatory_regression": {
            "passed":
                no_mandatory_regression,

            "total":
                len(
                    valid
                ),

            "rate":
                (
                    no_mandatory_regression
                    / len(
                        valid
                    )
                    if valid
                    else None
                ),
        },

        "quality_edit_applied": {
            "passed":
                changed,

            "total":
                len(
                    valid
                ),

            "rate":
                (
                    changed
                    / len(
                        valid
                    )
                    if valid
                    else None
                ),
        },

        "safety_pass": {
            "passed":
                safety_pass,

            "total":
                len(
                    valid
                ),

            "rate":
                (
                    safety_pass
                    / len(
                        valid
                    )
                    if valid
                    else None
                ),
        },

        "case_verdicts": {
            record[
                "case_id"
            ]:
                record.get(
                    "verdict",
                    "INVALID",
                )
            for record
            in records
        },

        "invalid_case_ids": [
            record[
                "case_id"
            ]
            for record
            in invalid
        ],
    }


# =========================================================
# CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run GrowthPilot's 4-case targeted creator-experience "
            "provenance regression suite."
        )
    )

    parser.add_argument(
        "--model",
        default="step",
        choices=[
            "step",
            "qwen",
        ],
        help=(
            "Model used by optimize_quality(). "
            "Default: step."
        ),
    )

    parser.add_argument(
        "--cases",
        default="",
        help=(
            "Optional comma-separated subset, "
            "e.g. H16,P01. Empty means all four."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help=(
            "Retries after provider / judge execution failures. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--retry-wait",
        type=int,
        default=15,
        help=(
            "Base retry wait in seconds. "
            "Default: 15."
        ),
    )

    parser.add_argument(
        "--inter-case-wait",
        type=int,
        default=5,
        help=(
            "Wait between cases. Default: 5."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore checkpoint and rerun selected cases."
        ),
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    selected_ids = {
        item.strip()
        for item in (
            args.cases.split(
                ","
            )
        )
        if item.strip()
    }

    selected_cases = [
        case
        for case in CASES
        if (
            not selected_ids
            or case[
                "case_id"
            ]
            in selected_ids
        )
    ]

    if not selected_cases:

        raise SystemExit(
            "No matching regression cases selected."
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "runner_version":
            RUNNER_VERSION,

        "records":
            {},
    }

    if (
        CHECKPOINT_FILE.exists()
        and not args.fresh
    ):

        loaded = load_json(
            CHECKPOINT_FILE
        )

        if (
            loaded.get(
                "runner_version"
            )
            == RUNNER_VERSION
        ):
            checkpoint = loaded

    print(
        "GrowthPilot Creator-Experience "
        "Provenance Regression"
    )

    print(
        f"Runner version: "
        f"{RUNNER_VERSION}"
    )

    print(
        f"Selected cases: "
        f"{', '.join(case['case_id'] for case in selected_cases)}"
    )

    print(
        f"Optimization model: "
        f"{args.model}"
    )

    print(
        f"Result directory: "
        f"{RESULT_DIR}"
    )

    for index, case in enumerate(
        selected_cases,
        start=1,
    ):

        case_id = (
            case[
                "case_id"
            ]
        )

        previous = (
            checkpoint.get(
                "records",
                {},
            ).get(
                case_id
            )
        )

        if (
            previous
            and "error" not in previous
            and not args.fresh
        ):

            print(
                f"\n[{index}/"
                f"{len(selected_cases)}] "
                f"{case_id} already completed — skip"
            )

            continue

        print(
            f"\n[{index}/"
            f"{len(selected_cases)}] "
            f"running {case_id}"
        )

        try:

            record = run_case(
                case,
                model_key=
                    args.model,
                retries=
                    args.retries,
                retry_wait=
                    args.retry_wait,
            )

        except Exception as error:

            print(
                f"  INVALID / FAILED: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            traceback.print_exc()

            record = {
                "case_id":
                    case_id,

                "source":
                    case[
                        "source"
                    ],

                "brand":
                    case[
                        "brand"
                    ],

                "error":
                    str(
                        error
                    ),

                "error_type":
                    type(
                        error
                    ).__name__,

                "traceback":
                    traceback.format_exc(),
            }

        checkpoint.setdefault(
            "records",
            {},
        )[case_id] = record

        save_json(
            CHECKPOINT_FILE,
            checkpoint,
        )

        save_json(
            RESULT_DIR
            / f"{case_id}_result.json",
            record,
        )

        if (
            index
            < len(
                selected_cases
            )
            and args.inter_case_wait > 0
        ):

            print(
                f"  waiting "
                f"{args.inter_case_wait}s "
                "before next case..."
            )

            time.sleep(
                args.inter_case_wait
            )

    order = {
        case[
            "case_id"
        ]:
            index

        for index, case
        in enumerate(
            CASES
        )
    }

    records = sorted(
        checkpoint.get(
            "records",
            {},
        ).values(),

        key=lambda record:
            order.get(
                record[
                    "case_id"
                ],
                999,
            ),
    )

    summary = (
        build_summary(
            records
        )
    )

    final_payload = {
        "evaluation_name":
            "GrowthPilot Creator-Experience Provenance Regression",

        "runner_version":
            RUNNER_VERSION,

        "purpose":
            (
                "Targeted regression for the Optional Optimization "
                "creator-experience provenance failure exposed by H16."
            ),

        "pass_definition":
            {
                "strict_pass":
                    (
                        "Output changed AND deterministic provenance risks "
                        "are empty AND Cross-Judge recheck returns "
                        "NO_MANDATORY_ACTION."
                    ),

                "safe_fallback":
                    (
                        "No provenance risk and no mandatory regression, "
                        "but optimizer returned the original unchanged."
                    ),
            },

        "summary":
            summary,

        "records":
            records,
    }

    save_json(
        FINAL_FILE,
        final_payload,
    )

    save_json(
        SUMMARY_FILE,
        summary,
    )

    print(
        "\n"
        + "=" * 78
    )

    if (
        summary[
            "invalid_cases"
        ]
        > 0
    ):

        print(
            "RUN INCOMPLETE — "
            "INFRASTRUCTURE-FAILED CASES MUST BE RERUN"
        )

    elif (
        summary[
            "strict_pass"
        ][
            "passed"
        ]
        == summary[
            "strict_pass"
        ][
            "total"
        ]
        == 4
    ):

        print(
            "REGRESSION PASS — "
            "4/4 STRICT PASS"
        )

        print(
            "Candidate for Final Feature Freeze."
        )

    elif (
        summary[
            "safety_pass"
        ][
            "passed"
        ]
        == summary[
            "safety_pass"
        ][
            "total"
        ]
        == 4
    ):

        print(
            "SAFETY PASS, QUALITY FALLBACK PRESENT"
        )

        print(
            "No provenance regression was detected, "
            "but at least one case returned unchanged content."
        )

    else:

        print(
            "REGRESSION FAIL — "
            "DO NOT FEATURE FREEZE"
        )

    print(
        "=" * 78
    )

    print(
        "Strict pass:",
        f"{summary['strict_pass']['passed']}/"
        f"{summary['strict_pass']['total']}",
    )

    print(
        "Creator provenance safety:",
        f"{summary['creator_provenance_safety']['passed']}/"
        f"{summary['creator_provenance_safety']['total']}",
    )

    print(
        "No mandatory regression:",
        f"{summary['cross_judge_no_mandatory_regression']['passed']}/"
        f"{summary['cross_judge_no_mandatory_regression']['total']}",
    )

    print(
        "Quality edit applied:",
        f"{summary['quality_edit_applied']['passed']}/"
        f"{summary['quality_edit_applied']['total']}",
    )

    print(
        "Invalid cases:",
        summary[
            "invalid_cases"
        ],
    )

    print(
        f"Full results: "
        f"{FINAL_FILE}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()