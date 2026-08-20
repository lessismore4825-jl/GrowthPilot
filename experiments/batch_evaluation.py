import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.evaluator import (
    compare_contents_pairwise,
    evaluate_content,
)
from src.generator import generate_content
from src.llm_client import MODELS
from src.reviser import fix_compliance_issues


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation_cases.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_RESULTS_FILE = (
    RESULTS_DIR
    / "batch_results.csv"
)

JUDGE_RESULTS_FILE = (
    RESULTS_DIR
    / "judge_results.csv"
)

PAIRWISE_RESULTS_FILE = (
    RESULTS_DIR
    / "pairwise_results.csv"
)


# =========================================================
# Experiment Configuration
# =========================================================

CANDIDATE_MODELS = [
    "step",
    "qwen",
]

JUDGE_MODELS = [
    "step",
    "qwen",
]


# One normal attempt + one retry.
JUDGE_MAX_ATTEMPTS = 2


# Balance A/B positions across pairwise comparisons.
#
# This helps reduce systematic Position Bias
# without doubling the number of Judge calls.
BALANCE_PAIRWISE_ORDER = True


# Evidence similarity threshold used when combining
# Cross-Judge compliance findings.
#
# This is NOT a quality threshold.
#
# It is only used to determine whether two Judges
# are referring to essentially the same problematic
# piece of generated content.
FINDING_EVIDENCE_SIMILARITY_THRESHOLD = 0.85


# =========================================================
# Evaluation Metrics
# =========================================================

SCORE_FIELDS = [
    "brand_alignment",
    "tone_match",
    "selling_point_coverage",
    "factual_consistency",
    "unsupported_claim_risk",
    "heuristic_composite_score",
]


# =========================================================
# Load Evaluation Cases
# =========================================================

def load_cases() -> list:
    """
    Load evaluation cases from:

    data/evaluation_cases.json

    Supports either:

    [
        {...},
        {...}
    ]

    or:

    {
        "cases": [
            {...},
            {...}
        ]
    }
    """

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{DATA_FILE}"
        )


    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )


    if isinstance(
        data,
        list,
    ):

        cases = data


    elif (
        isinstance(
            data,
            dict,
        )
        and "cases" in data
    ):

        cases = data[
            "cases"
        ]


    else:

        raise ValueError(
            "evaluation_cases.json must contain "
            "a list of cases or a "
            "{'cases': [...]} object."
        )


    if not cases:

        raise ValueError(
            "No evaluation cases were found."
        )


    return cases


# =========================================================
# Case Helpers
# =========================================================

def get_case_id(
    case: dict,
    index: int,
) -> str:
    """
    Return case ID.
    """

    return (
        case.get(
            "case_id"
        )
        or case.get(
            "id"
        )
        or f"CASE_{index:03d}"
    )


def get_challenge(
    case: dict,
) -> str:
    """
    Return case challenge / description.
    """

    return (
        case.get(
            "challenge"
        )
        or case.get(
            "description"
        )
        or ""
    )


def get_policy_context(
    case: dict,
) -> str:
    """
    Return optional external policy context.

    Current cases may leave this empty.

    Later this field can contain:

    - advertising regulations
    - platform rules
    - internal brand policies
    - RAG-retrieved policy evidence

    Brand Information and Campaign Brief
    are still separately supplied to the Judge.
    """

    return str(
        case.get(
            "policy_context",
            "",
        )
    ).strip()


def validate_case(
    case: dict,
    case_id: str,
):
    """
    Validate required case fields.
    """

    required_fields = [
        "brand_info",
        "campaign_brief",
    ]


    for field in required_fields:

        if field not in case:

            raise ValueError(
                f"{case_id} is missing "
                f"required field: {field}"
            )


        if not str(
            case[
                field
            ]
        ).strip():

            raise ValueError(
                f"{case_id} has an empty "
                f"required field: {field}"
            )


# =========================================================
# Generic Helpers
# =========================================================

def timed_call(
    function,
    **kwargs,
):
    """
    Execute a function and return:

    result,
    latency_seconds
    """

    start_time = (
        time.perf_counter()
    )


    result = function(
        **kwargs
    )


    latency = (
        time.perf_counter()
        - start_time
    )


    return (
        result,
        round(
            latency,
            2,
        ),
    )


def get_heuristic_score(
    evaluation: dict,
):
    """
    Read the new heuristic diagnostic score.

    overall_score fallback is retained only
    for safer migration from the older schema.

    New evaluator should return:

    heuristic_composite_score
    """

    if (
        "heuristic_composite_score"
        in evaluation
    ):

        return evaluation[
            "heuristic_composite_score"
        ]


    if (
        "overall_score"
        in evaluation
    ):

        return evaluation[
            "overall_score"
        ]


    raise KeyError(
        "Evaluation contains neither "
        "heuristic_composite_score "
        "nor overall_score."
    )


def get_blocking_count(
    evaluation: dict,
) -> int:
    """
    Return number of blocking
    compliance findings.
    """

    if (
        "blocking_compliance_issue_count"
        in evaluation
    ):

        return int(
            evaluation[
                "blocking_compliance_issue_count"
            ]
        )


    return len(
        evaluation.get(
            "compliance_findings",
            [],
        )
    )


def safe_numeric_mean(
    series: pd.Series,
):
    """
    Return numeric mean.

    Invalid / empty values are ignored.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    if numeric.empty:

        return None


    return float(
        numeric.mean()
    )


# =========================================================
# Cross-Judge Evaluation
# =========================================================

def run_all_judges(
    case_id: str,
    challenge: str,
    candidate_model: str,
    version: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
    content: str,
    judge_rows: list,
) -> dict:
    """
    Evaluate one output with:

    Step Judge
        +
    Qwen Judge

    Each Judge returns:

    1. Policy-grounded compliance findings
    2. Non-blocking advisory findings
    3. Diagnostic dimension scores

    Each Judge may retry when API / parsing
    errors occur.
    """

    results = {}


    for judge_model in JUDGE_MODELS:

        print(
            f"\n      Judge: "
            f"{judge_model}"
        )


        evaluation = None

        final_error = None

        successful_attempt = None

        total_judge_latency = 0.0


        # =================================================
        # Retry Loop
        # =================================================

        for attempt in range(
            1,
            JUDGE_MAX_ATTEMPTS + 1,
        ):

            print(
                f"      Attempt "
                f"{attempt}/"
                f"{JUDGE_MAX_ATTEMPTS}"
            )


            attempt_start = (
                time.perf_counter()
            )


            try:

                evaluation = evaluate_content(
                    brand_info=brand_info,

                    campaign_brief=campaign_brief,

                    generated_content=content,

                    policy_context=policy_context,

                    judge_model_key=judge_model,
                )


                attempt_latency = (
                    time.perf_counter()
                    - attempt_start
                )


                total_judge_latency += (
                    attempt_latency
                )


                successful_attempt = (
                    attempt
                )


                final_error = None


                print(
                    f"      ✅ "
                    f"Heuristic="
                    f"{get_heuristic_score(evaluation)}"
                    f" | Risk="
                    f"{evaluation['unsupported_claim_risk']}"
                    f" | Blocking="
                    f"{get_blocking_count(evaluation)}"
                    f" | Attempt latency="
                    f"{attempt_latency:.2f}s"
                )


                break


            except Exception as error:

                attempt_latency = (
                    time.perf_counter()
                    - attempt_start
                )


                total_judge_latency += (
                    attempt_latency
                )


                final_error = str(
                    error
                )


                print(
                    f"      ⚠️ Attempt "
                    f"{attempt} failed after "
                    f"{attempt_latency:.2f}s: "
                    f"{error}"
                )


                if (
                    attempt
                    < JUDGE_MAX_ATTEMPTS
                ):

                    print(
                        "      Retrying Judge..."
                    )


        total_judge_latency = round(
            total_judge_latency,
            2,
        )


        # =================================================
        # Judge SUCCESS
        # =================================================

        if evaluation is not None:

            results[
                judge_model
            ] = {
                "evaluation":
                    evaluation,

                "latency":
                    total_judge_latency,

                "error":
                    None,

                "attempts":
                    successful_attempt,
            }


            judge_rows.append(
                {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "version":
                        version,

                    "judge_model":
                        judge_model,

                    "judge_model_id":
                        MODELS[
                            judge_model
                        ],

                    "status":
                        "SUCCESS",

                    "attempts":
                        successful_attempt,

                    "evaluation_latency":
                        total_judge_latency,

                    "brand_alignment":
                        evaluation[
                            "brand_alignment"
                        ],

                    "tone_match":
                        evaluation[
                            "tone_match"
                        ],

                    "selling_point_coverage":
                        evaluation[
                            "selling_point_coverage"
                        ],

                    "factual_consistency":
                        evaluation[
                            "factual_consistency"
                        ],

                    "unsupported_claim_risk":
                        evaluation[
                            "unsupported_claim_risk"
                        ],

                    "heuristic_composite_score":
                        get_heuristic_score(
                            evaluation
                        ),

                    "blocking_count":
                        get_blocking_count(
                            evaluation
                        ),

                    "blocking_flag":
                        (
                            get_blocking_count(
                                evaluation
                            )
                            > 0
                        ),

                    "compliance_status":
                        evaluation.get(
                            "compliance_status",
                            "",
                        ),

                    "compliance_findings":
                        json.dumps(
                            evaluation.get(
                                "compliance_findings",
                                [],
                            ),
                            ensure_ascii=False,
                        ),

                    "advisory_findings":
                        json.dumps(
                            evaluation.get(
                                "advisory_findings",
                                [],
                            ),
                            ensure_ascii=False,
                        ),

                    "review_notes":
                        json.dumps(
                            evaluation.get(
                                "review_notes",
                                [],
                            ),
                            ensure_ascii=False,
                        ),

                    "error":
                        "",
                }
            )


        # =================================================
        # Judge FAILED
        # =================================================

        else:

            print(
                f"      ❌ Judge failed "
                f"after "
                f"{JUDGE_MAX_ATTEMPTS} "
                f"attempts."
            )


            results[
                judge_model
            ] = {
                "evaluation":
                    None,

                "latency":
                    total_judge_latency,

                "error":
                    final_error,

                "attempts":
                    JUDGE_MAX_ATTEMPTS,
            }


            judge_rows.append(
                {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "version":
                        version,

                    "judge_model":
                        judge_model,

                    "judge_model_id":
                        MODELS[
                            judge_model
                        ],

                    "status":
                        "FAILED",

                    "attempts":
                        JUDGE_MAX_ATTEMPTS,

                    "evaluation_latency":
                        total_judge_latency,

                    "brand_alignment":
                        None,

                    "tone_match":
                        None,

                    "selling_point_coverage":
                        None,

                    "factual_consistency":
                        None,

                    "unsupported_claim_risk":
                        None,

                    "heuristic_composite_score":
                        None,

                    "blocking_count":
                        None,

                    "blocking_flag":
                        None,

                    "compliance_status":
                        "",

                    "compliance_findings":
                        "",

                    "advisory_findings":
                        "",

                    "review_notes":
                        "",

                    "error":
                        final_error or "",
                }
            )


    return results


# =========================================================
# Cross-Judge Aggregation
# =========================================================

def aggregate_judges(
    judge_results: dict,
) -> dict:
    """
    Aggregate Judge results without using
    an arbitrary numerical pass/fail threshold.

    Compliance states:

    CONSENSUS_BLOCKING
        Both Judges detect one or more
        blocking compliance findings.

    CONSENSUS_NO_BLOCKING
        Both Judges detect no
        blocking compliance findings.

    JUDGE_DISAGREEMENT
        One Judge detects blocking findings
        while the other does not.

    Judge disagreement is routed to
    Human Review rather than automatic revision.
    """

    valid_evaluations = [
        result[
            "evaluation"
        ]

        for result
        in judge_results.values()

        if result.get(
            "evaluation"
        ) is not None
    ]


    valid_judge_count = len(
        valid_evaluations
    )


    expected_judge_count = len(
        JUDGE_MODELS
    )


    panel_complete = (
        valid_judge_count
        == expected_judge_count
    )


    # =====================================================
    # Incomplete Judge Panel
    # =====================================================

    if not panel_complete:

        return {
            "valid_judge_count":
                valid_judge_count,

            "expected_judge_count":
                expected_judge_count,

            "panel_complete":
                False,

            "avg_heuristic_score":
                None,

            "avg_claim_risk":
                None,

            "avg_blocking_count":
                None,

            "judge_score_gap":
                None,

            "any_blocking":
                None,

            "all_blocking":
                None,

            "blocking_agreement":
                None,

            "compliance_decision":
                "INCOMPLETE_PANEL",
        }


    # =====================================================
    # Complete Judge Panel
    # =====================================================

    heuristic_scores = [
        get_heuristic_score(
            evaluation
        )

        for evaluation
        in valid_evaluations
    ]


    claim_risks = [
        evaluation[
            "unsupported_claim_risk"
        ]

        for evaluation
        in valid_evaluations
    ]


    blocking_counts = [
        get_blocking_count(
            evaluation
        )

        for evaluation
        in valid_evaluations
    ]


    blocking_flags = [
        count > 0

        for count
        in blocking_counts
    ]


    avg_heuristic_score = (
        sum(
            heuristic_scores
        )
        / valid_judge_count
    )


    avg_claim_risk = (
        sum(
            claim_risks
        )
        / valid_judge_count
    )


    avg_blocking_count = (
        sum(
            blocking_counts
        )
        / valid_judge_count
    )


    # -----------------------------------------------------
    # Diagnostic Score Gap
    # -----------------------------------------------------

    step_score = (
        get_heuristic_score(
            judge_results[
                "step"
            ][
                "evaluation"
            ]
        )
    )


    qwen_score = (
        get_heuristic_score(
            judge_results[
                "qwen"
            ][
                "evaluation"
            ]
        )
    )


    judge_score_gap = abs(
        step_score
        - qwen_score
    )


    # -----------------------------------------------------
    # Compliance Agreement
    # -----------------------------------------------------

    any_blocking = any(
        blocking_flags
    )


    all_blocking = all(
        blocking_flags
    )


    blocking_agreement = (
        len(
            set(
                blocking_flags
            )
        )
        == 1
    )


    if all_blocking:

        compliance_decision = (
            "CONSENSUS_BLOCKING"
        )


    elif not any_blocking:

        compliance_decision = (
            "CONSENSUS_NO_BLOCKING"
        )


    else:

        compliance_decision = (
            "JUDGE_DISAGREEMENT"
        )


    return {
        "valid_judge_count":
            valid_judge_count,

        "expected_judge_count":
            expected_judge_count,

        "panel_complete":
            True,

        "avg_heuristic_score":
            round(
                avg_heuristic_score,
                2,
            ),

        "avg_claim_risk":
            round(
                avg_claim_risk,
                2,
            ),

        "avg_blocking_count":
            round(
                avg_blocking_count,
                2,
            ),

        "judge_score_gap":
            round(
                judge_score_gap,
                2,
            ),

        "any_blocking":
            any_blocking,

        "all_blocking":
            all_blocking,

        "blocking_agreement":
            blocking_agreement,

        "compliance_decision":
            compliance_decision,
    }


# =========================================================
# Add Judge Results to Main Row
# =========================================================

def add_judge_scores_to_row(
    row: dict,
    version: str,
    judge_results: dict,
):
    """
    Add each Judge's individual scores
    and findings to the wide-format
    batch_results.csv row.
    """

    for judge_model in JUDGE_MODELS:

        result = judge_results.get(
            judge_model,
            {},
        )


        evaluation = result.get(
            "evaluation"
        )


        prefix = (
            f"{version}_"
            f"{judge_model}_judge"
        )


        row[
            f"{prefix}_latency"
        ] = result.get(
            "latency"
        )


        row[
            f"{prefix}_attempts"
        ] = result.get(
            "attempts"
        )


        row[
            f"{prefix}_error"
        ] = (
            result.get(
                "error"
            )
            or ""
        )


        # =================================================
        # Judge succeeded
        # =================================================

        if evaluation:

            for field in SCORE_FIELDS:

                if (
                    field
                    == "heuristic_composite_score"
                ):

                    value = (
                        get_heuristic_score(
                            evaluation
                        )
                    )


                else:

                    value = evaluation[
                        field
                    ]


                row[
                    f"{prefix}_{field}"
                ] = value


            row[
                f"{prefix}_blocking_count"
            ] = get_blocking_count(
                evaluation
            )


            row[
                f"{prefix}_blocking_flag"
            ] = (
                get_blocking_count(
                    evaluation
                )
                > 0
            )


            row[
                f"{prefix}_compliance_findings"
            ] = json.dumps(
                evaluation.get(
                    "compliance_findings",
                    [],
                ),
                ensure_ascii=False,
            )


            row[
                f"{prefix}_advisory_findings"
            ] = json.dumps(
                evaluation.get(
                    "advisory_findings",
                    [],
                ),
                ensure_ascii=False,
            )


            row[
                f"{prefix}_review_notes"
            ] = json.dumps(
                evaluation.get(
                    "review_notes",
                    [],
                ),
                ensure_ascii=False,
            )


        # =================================================
        # Judge failed
        # =================================================

        else:

            for field in SCORE_FIELDS:

                row[
                    f"{prefix}_{field}"
                ] = None


            row[
                f"{prefix}_blocking_count"
            ] = None


            row[
                f"{prefix}_blocking_flag"
            ] = None


            row[
                f"{prefix}_compliance_findings"
            ] = ""


            row[
                f"{prefix}_advisory_findings"
            ] = ""


            row[
                f"{prefix}_review_notes"
            ] = ""


# =========================================================
# Compliance Finding Deduplication Helpers
# =========================================================

def normalize_finding_text(
    text: str,
) -> str:
    """
    Normalize text for Cross-Judge finding matching.

    Removes differences caused only by:

    - uppercase / lowercase
    - whitespace
    - punctuation
    - Chinese punctuation
    - quotation marks

    This does NOT alter the original finding
    that is eventually saved to CSV.
    """

    text = str(
        text
        or ""
    ).strip().casefold()


    # Remove whitespace
    text = re.sub(
        r"\s+",
        "",
        text,
    )


    # Keep letters, numbers and Chinese characters.
    # Remove punctuation / quotation marks.
    text = re.sub(
        r"[^\w\u4e00-\u9fff]",
        "",
        text,
    )


    return text


def evidence_similarity(
    evidence_a: str,
    evidence_b: str,
) -> float:
    """
    Measure whether two Judges are referring
    to essentially the same problematic text.

    Uses:

    1. Exact normalized match
    2. Substring containment
    3. SequenceMatcher similarity

    This is used only for deduplication,
    NOT for content quality evaluation.
    """

    normalized_a = normalize_finding_text(
        evidence_a
    )

    normalized_b = normalize_finding_text(
        evidence_b
    )


    if not normalized_a or not normalized_b:

        return 0.0


    if normalized_a == normalized_b:

        return 1.0


    # If one Judge quotes a slightly longer
    # version of the same problematic phrase.
    if (
        normalized_a in normalized_b
        or normalized_b in normalized_a
    ):

        shorter_length = min(
            len(
                normalized_a
            ),
            len(
                normalized_b
            ),
        )


        longer_length = max(
            len(
                normalized_a
            ),
            len(
                normalized_b
            ),
        )


        if longer_length == 0:

            return 0.0


        containment_ratio = (
            shorter_length
            / longer_length
        )


        # Exact phrase contained inside a moderately
        # longer quotation should still count
        # as the same evidence.
        if containment_ratio >= 0.55:

            return max(
                0.90,
                containment_ratio,
            )


    return SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()


def findings_refer_to_same_issue(
    finding_a: dict,
    finding_b: dict,
) -> bool:
    """
    Determine whether two Judge findings
    are essentially about the same issue.

    Primary matching signal:

    same / highly similar evidence.

    Policy source is used as an additional
    safeguard when available.

    policy_basis is intentionally NOT required
    to be identical because Step and Qwen may
    phrase the same governing rule differently.
    """

    evidence_a = finding_a.get(
        "evidence",
        "",
    )

    evidence_b = finding_b.get(
        "evidence",
        "",
    )


    similarity = evidence_similarity(
        evidence_a,
        evidence_b,
    )


    if (
        similarity
        < FINDING_EVIDENCE_SIMILARITY_THRESHOLD
    ):

        return False


    source_a = normalize_finding_text(
        finding_a.get(
            "policy_source",
            "",
        )
    )


    source_b = normalize_finding_text(
        finding_b.get(
            "policy_source",
            "",
        )
    )


    # If both have explicit sources and the sources
    # are completely different, do not merge them.
    #
    # Example:
    #
    # one finding grounded in Brand Information
    # another grounded in Additional Policy Context
    #
    # They may deserve separate provenance.
    if (
        source_a
        and source_b
        and source_a != source_b
    ):

        return False


    return True


def split_merged_text(
    text: str,
) -> list:
    """
    Convert a previously merged text value into
    reusable components.

    Values are joined with:

    " | "

    so repeated merging does not create duplicates.
    """

    text = str(
        text
        or ""
    ).strip()


    if not text:

        return []


    return [
        item.strip()

        for item
        in text.split(
            " | "
        )

        if item.strip()
    ]


def merge_unique_text(
    existing_text: str,
    new_text: str,
) -> str:
    """
    Merge two textual explanations while
    preserving both Judges' useful information
    and avoiding duplicate wording.
    """

    items = []


    for value in [
        existing_text,
        new_text,
    ]:

        for item in split_merged_text(
            value
        ):

            normalized_item = (
                normalize_finding_text(
                    item
                )
            )


            if not normalized_item:

                continue


            duplicate = False


            for existing_item in items:

                existing_normalized = (
                    normalize_finding_text(
                        existing_item
                    )
                )


                if (
                    normalized_item
                    == existing_normalized
                ):

                    duplicate = True

                    break


                similarity = SequenceMatcher(
                    None,
                    normalized_item,
                    existing_normalized,
                ).ratio()


                if similarity >= 0.93:

                    duplicate = True

                    break


            if not duplicate:

                items.append(
                    item
                )


    return " | ".join(
        items
    )


def merge_reported_by(
    existing,
    new_judge: str,
) -> list:
    """
    Merge Judge provenance.

    Example:

    ["step"]
        +
    "qwen"

    →

    ["step", "qwen"]
    """

    if isinstance(
        existing,
        list,
    ):

        reporters = [
            str(
                item
            ).strip()

            for item
            in existing

            if str(
                item
            ).strip()
        ]


    elif existing:

        reporters = [
            str(
                existing
            ).strip()
        ]


    else:

        reporters = []


    if (
        new_judge
        and new_judge
        not in reporters
    ):

        reporters.append(
            new_judge
        )


    return reporters


def merge_compliance_finding(
    existing: dict,
    incoming: dict,
    judge_model: str,
) -> dict:
    """
    Merge two findings that refer to
    the same underlying content issue.

    Important:

    We do NOT discard the second Judge's:

    - policy basis
    - reason
    - required action
    - provenance

    Instead, useful non-duplicate information
    is retained.
    """

    merged = dict(
        existing
    )


    # =====================================================
    # Evidence
    # =====================================================

    existing_evidence = str(
        merged.get(
            "evidence",
            "",
        )
    ).strip()


    incoming_evidence = str(
        incoming.get(
            "evidence",
            "",
        )
    ).strip()


    # Prefer the more complete quotation when
    # one is simply a longer version of the other.
    if (
        len(
            normalize_finding_text(
                incoming_evidence
            )
        )
        > len(
            normalize_finding_text(
                existing_evidence
            )
        )
    ):

        merged[
            "evidence"
        ] = incoming_evidence


    # =====================================================
    # Policy Source
    # =====================================================

    if not merged.get(
        "policy_source"
    ):

        merged[
            "policy_source"
        ] = incoming.get(
            "policy_source",
            "",
        )


    # =====================================================
    # Policy Basis
    # =====================================================

    merged[
        "policy_basis"
    ] = merge_unique_text(
        merged.get(
            "policy_basis",
            "",
        ),

        incoming.get(
            "policy_basis",
            "",
        ),
    )


    # =====================================================
    # Reason
    # =====================================================

    merged[
        "reason"
    ] = merge_unique_text(
        merged.get(
            "reason",
            "",
        ),

        incoming.get(
            "reason",
            "",
        ),
    )


    # =====================================================
    # Required Action
    # =====================================================

    merged[
        "required_action"
    ] = merge_unique_text(
        merged.get(
            "required_action",
            "",
        ),

        incoming.get(
            "required_action",
            "",
        ),
    )


    # =====================================================
    # Judge Provenance
    # =====================================================

    merged[
        "reported_by"
    ] = merge_reported_by(
        merged.get(
            "reported_by"
        ),

        judge_model,
    )


    return merged


# =========================================================
# Combine Compliance Findings
# =========================================================

def combine_compliance_findings(
    judge_results: dict,
) -> list:
    """
    Combine policy-grounded compliance findings
    from all successful Judges.

    This function is called only when
    both Judges agree that blocking
    compliance issues exist.

    IMPORTANT:

    Step and Qwen often identify the same
    problematic wording but describe:

    - policy basis
    - reason
    - required action

    slightly differently.

    The old implementation used:

    evidence + policy_source + policy_basis

    as an exact deduplication key.

    That caused duplicate findings when
    policy_basis wording differed.

    New behavior:

    1. Compare normalized evidence.
    2. Allow small quotation differences.
    3. Require compatible policy provenance.
    4. Merge the two Judges' explanations.
    5. Preserve which Judges reported the issue.
    """

    combined = []


    for judge_model in JUDGE_MODELS:

        evaluation = (
            judge_results
            .get(
                judge_model,
                {},
            )
            .get(
                "evaluation"
            )
        )


        if not evaluation:

            continue


        findings = evaluation.get(
            "compliance_findings",
            [],
        )


        for finding in findings:

            incoming = dict(
                finding
            )


            incoming[
                "reported_by"
            ] = [
                judge_model
            ]


            matched_index = None


            # =============================================
            # Search existing combined findings
            # =============================================

            for index, existing in enumerate(
                combined
            ):

                if findings_refer_to_same_issue(
                    existing,
                    incoming,
                ):

                    matched_index = (
                        index
                    )

                    break


            # =============================================
            # New unique finding
            # =============================================

            if matched_index is None:

                combined.append(
                    incoming
                )


            # =============================================
            # Same underlying issue:
            # merge Judge evidence / rationale
            # =============================================

            else:

                combined[
                    matched_index
                ] = (
                    merge_compliance_finding(
                        existing=combined[
                            matched_index
                        ],

                        incoming=incoming,

                        judge_model=judge_model,
                    )
                )


    return combined


# =========================================================
# Pairwise Order Helper
# =========================================================

def should_swap_pairwise_order(
    case_id: str,
    candidate_model: str,
    judge_model: str,
) -> bool:
    """
    Deterministically vary A/B order.

    This reduces systematic Position Bias
    across the benchmark without doubling
    the number of API calls.

    The same Case × Candidate × Judge
    always receives the same order.
    """

    if not BALANCE_PAIRWISE_ORDER:

        return False


    key = (
        f"{case_id}|"
        f"{candidate_model}|"
        f"{judge_model}"
    )


    digest = hashlib.sha256(
        key.encode(
            "utf-8"
        )
    ).hexdigest()


    return (
        int(
            digest[
                -1
            ],
            16,
        )
        % 2
        == 1
    )


def normalize_pairwise_preference(
    raw_preference: str,
    a_version: str,
    b_version: str,
) -> str:
    """
    Convert:

    A / B / tie

    into:

    v1 / v2 / tie

    regardless of presentation order.
    """

    preference = str(
        raw_preference
    ).strip().lower()


    if preference == "a":

        return a_version


    if preference == "b":

        return b_version


    if preference == "tie":

        return "tie"


    raise ValueError(
        f"Invalid pairwise preference: "
        f"{raw_preference}"
    )


# =========================================================
# Pairwise Cross-Judge Evaluation
# =========================================================

def run_pairwise_judges(
    case_id: str,
    challenge: str,
    candidate_model: str,
    brand_info: str,
    campaign_brief: str,
    policy_context: str,
    v1_content: str,
    v2_content: str,
    pairwise_rows: list,
) -> dict:
    """
    Compare V1 vs V2 with:

    Step Judge
    +
    Qwen Judge

    Numerical scores are NOT used
    to decide the pairwise winner.

    A/B order is balanced across
    comparisons and normalized back
    to:

    v1
    v2
    tie
    """

    results = {}


    for judge_model in JUDGE_MODELS:

        print(
            f"\n      Pairwise Judge: "
            f"{judge_model}"
        )


        swap_order = (
            should_swap_pairwise_order(
                case_id=case_id,

                candidate_model=(
                    candidate_model
                ),

                judge_model=(
                    judge_model
                ),
            )
        )


        if swap_order:

            content_a = (
                v2_content
            )

            content_b = (
                v1_content
            )

            a_version = "v2"

            b_version = "v1"


        else:

            content_a = (
                v1_content
            )

            content_b = (
                v2_content
            )

            a_version = "v1"

            b_version = "v2"


        result = None

        final_error = None

        successful_attempt = None

        total_latency = 0.0


        # =================================================
        # Retry Loop
        # =================================================

        for attempt in range(
            1,
            JUDGE_MAX_ATTEMPTS + 1,
        ):

            print(
                f"      Attempt "
                f"{attempt}/"
                f"{JUDGE_MAX_ATTEMPTS} "
                f"(A={a_version}, "
                f"B={b_version})"
            )


            attempt_start = (
                time.perf_counter()
            )


            try:

                result = (
                    compare_contents_pairwise(
                        brand_info=brand_info,

                        campaign_brief=(
                            campaign_brief
                        ),

                        content_a=content_a,

                        content_b=content_b,

                        policy_context=(
                            policy_context
                        ),

                        judge_model_key=(
                            judge_model
                        ),
                    )
                )


                attempt_latency = (
                    time.perf_counter()
                    - attempt_start
                )


                total_latency += (
                    attempt_latency
                )


                successful_attempt = (
                    attempt
                )


                final_error = None


                normalized_preference = (
                    normalize_pairwise_preference(
                        raw_preference=(
                            result[
                                "preference"
                            ]
                        ),

                        a_version=a_version,

                        b_version=b_version,
                    )
                )


                print(
                    f"      ✅ "
                    f"Raw="
                    f"{result['preference']}"
                    f" | Normalized="
                    f"{normalized_preference}"
                    f" | Attempt latency="
                    f"{attempt_latency:.2f}s"
                )


                break


            except Exception as error:

                attempt_latency = (
                    time.perf_counter()
                    - attempt_start
                )


                total_latency += (
                    attempt_latency
                )


                final_error = str(
                    error
                )


                print(
                    f"      ⚠️ Attempt "
                    f"{attempt} failed after "
                    f"{attempt_latency:.2f}s: "
                    f"{error}"
                )


                if (
                    attempt
                    < JUDGE_MAX_ATTEMPTS
                ):

                    print(
                        "      Retrying "
                        "Pairwise Judge..."
                    )


        total_latency = round(
            total_latency,
            2,
        )


        # =================================================
        # Pairwise SUCCESS
        # =================================================

        if result is not None:

            normalized_preference = (
                normalize_pairwise_preference(
                    raw_preference=(
                        result[
                            "preference"
                        ]
                    ),

                    a_version=a_version,

                    b_version=b_version,
                )
            )


            results[
                judge_model
            ] = {
                "preference":
                    normalized_preference,

                "raw_preference":
                    result[
                        "preference"
                    ],

                "a_version":
                    a_version,

                "b_version":
                    b_version,

                "reason":
                    result.get(
                        "reason",
                        "",
                    ),

                "key_difference":
                    result.get(
                        "key_difference",
                        "",
                    ),

                "latency":
                    total_latency,

                "attempts":
                    successful_attempt,

                "error":
                    None,
            }


            pairwise_rows.append(
                {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "judge_model":
                        judge_model,

                    "judge_model_id":
                        MODELS[
                            judge_model
                        ],

                    "status":
                        "SUCCESS",

                    "attempts":
                        successful_attempt,

                    "evaluation_latency":
                        total_latency,

                    "a_version":
                        a_version,

                    "b_version":
                        b_version,

                    "raw_preference":
                        result[
                            "preference"
                        ],

                    "normalized_preference":
                        normalized_preference,

                    "reason":
                        result.get(
                            "reason",
                            "",
                        ),

                    "key_difference":
                        result.get(
                            "key_difference",
                            "",
                        ),

                    "error":
                        "",
                }
            )


        # =================================================
        # Pairwise FAILED
        # =================================================

        else:

            print(
                f"      ❌ Pairwise Judge "
                f"failed after "
                f"{JUDGE_MAX_ATTEMPTS} "
                f"attempts."
            )


            results[
                judge_model
            ] = {
                "preference":
                    None,

                "raw_preference":
                    None,

                "a_version":
                    a_version,

                "b_version":
                    b_version,

                "reason":
                    "",

                "key_difference":
                    "",

                "latency":
                    total_latency,

                "attempts":
                    JUDGE_MAX_ATTEMPTS,

                "error":
                    final_error,
            }


            pairwise_rows.append(
                {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "judge_model":
                        judge_model,

                    "judge_model_id":
                        MODELS[
                            judge_model
                        ],

                    "status":
                        "FAILED",

                    "attempts":
                        JUDGE_MAX_ATTEMPTS,

                    "evaluation_latency":
                        total_latency,

                    "a_version":
                        a_version,

                    "b_version":
                        b_version,

                    "raw_preference":
                        None,

                    "normalized_preference":
                        None,

                    "reason":
                        "",

                    "key_difference":
                        "",

                    "error":
                        final_error or "",
                }
            )


    # =====================================================
    # Pairwise Aggregation
    # =====================================================

    valid_preferences = [
        result[
            "preference"
        ]

        for result
        in results.values()

        if result.get(
            "preference"
        ) is not None
    ]


    panel_complete = (
        len(
            valid_preferences
        )
        == len(
            JUDGE_MODELS
        )
    )


    agreement = (
        panel_complete

        and len(
            set(
                valid_preferences
            )
        )
        == 1
    )


    if agreement:

        consensus_preference = (
            valid_preferences[
                0
            ]
        )


    else:

        consensus_preference = (
            None
        )


    return {
        "results":
            results,

        "panel_complete":
            panel_complete,

        "agreement":
            agreement,

        "consensus_preference":
            consensus_preference,
    }


# =========================================================
# Save Results
# =========================================================

def save_results(
    batch_rows: list,
    judge_rows: list,
    pairwise_rows: list,
):
    """
    Save intermediate results after
    every Candidate × Case experiment.

    This prevents losing finished work
    if later API requests fail.
    """

    if batch_rows:

        pd.DataFrame(
            batch_rows
        ).to_csv(
            BATCH_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    if judge_rows:

        pd.DataFrame(
            judge_rows
        ).to_csv(
            JUDGE_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    if pairwise_rows:

        pd.DataFrame(
            pairwise_rows
        ).to_csv(
            PAIRWISE_RESULTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )


# =========================================================
# Candidate × Case Experiment
# =========================================================

def run_candidate_case(
    case: dict,
    case_index: int,
    candidate_model: str,
    judge_rows: list,
    pairwise_rows: list,
) -> dict:
    """
    Run one complete policy-grounded experiment.

    V1 Generation
        ↓
    Step + Qwen Review
        ↓

    CASE 1
    Both Judges detect no blocking issue
        ↓
    No mandatory revision

    CASE 2
    Judges disagree
        ↓
    Human Review recommended
        ↓
    No automatic rewrite

    CASE 3
    Both Judges detect blocking issue(s)
        ↓
    Targeted Minimal Compliance Fix
        ↓
    V2 Step + Qwen Re-check
        ↓
    V1 vs V2 Pairwise Evaluation

    Numerical scores remain descriptive only.
    They never determine pass/fail.
    """

    case_id = get_case_id(
        case,
        case_index,
    )


    challenge = get_challenge(
        case
    )


    validate_case(
        case,
        case_id,
    )


    brand_info = case[
        "brand_info"
    ]


    campaign_brief = case[
        "campaign_brief"
    ]


    policy_context = (
        get_policy_context(
            case
        )
    )


    row = {
        "case_id":
            case_id,

        "challenge":
            challenge,

        "candidate_model":
            candidate_model,

        "candidate_model_id":
            MODELS[
                candidate_model
            ],

        "policy_context_present":
            bool(
                policy_context
            ),

        "status":
            "RUNNING",

        "error":
            "",
    }


    # =====================================================
    # Generate V1
    # =====================================================

    print(
        "\n"
        + "-" * 70
    )


    print(
        f"Generating V1 with "
        f"{candidate_model}..."
    )


    try:

        (
            v1_content,
            generation_latency,
        ) = timed_call(
            generate_content,

            brand_info=brand_info,

            campaign_brief=campaign_brief,

            model_key=candidate_model,
        )


        row[
            "v1_content"
        ] = v1_content


        row[
            "v1_generation_latency"
        ] = generation_latency


        print(
            f"✅ V1 generated "
            f"in "
            f"{generation_latency:.2f}s"
        )


    except Exception as error:

        print(
            f"❌ Generation failed: "
            f"{error}"
        )


        row[
            "status"
        ] = "FAILED_GENERATION"


        row[
            "error"
        ] = str(
            error
        )


        return row


    # =====================================================
    # Evaluate V1
    # =====================================================

    print(
        "\nEvaluating V1 "
        "with all Judges..."
    )


    v1_judges = run_all_judges(
        case_id=case_id,

        challenge=challenge,

        candidate_model=candidate_model,

        version="v1",

        brand_info=brand_info,

        campaign_brief=campaign_brief,

        policy_context=policy_context,

        content=v1_content,

        judge_rows=judge_rows,
    )


    add_judge_scores_to_row(
        row=row,

        version="v1",

        judge_results=v1_judges,
    )


    v1_aggregate = aggregate_judges(
        v1_judges
    )


    row.update(
        {
            "v1_valid_judge_count":
                v1_aggregate[
                    "valid_judge_count"
                ],

            "v1_expected_judge_count":
                v1_aggregate[
                    "expected_judge_count"
                ],

            "v1_judge_panel_complete":
                v1_aggregate[
                    "panel_complete"
                ],

            "v1_cross_judge_heuristic":
                v1_aggregate[
                    "avg_heuristic_score"
                ],

            "v1_cross_judge_claim_risk":
                v1_aggregate[
                    "avg_claim_risk"
                ],

            "v1_cross_judge_blocking_count":
                v1_aggregate[
                    "avg_blocking_count"
                ],

            "v1_judge_score_gap":
                v1_aggregate[
                    "judge_score_gap"
                ],

            "v1_any_blocking":
                v1_aggregate[
                    "any_blocking"
                ],

            "v1_all_blocking":
                v1_aggregate[
                    "all_blocking"
                ],

            "v1_blocking_agreement":
                v1_aggregate[
                    "blocking_agreement"
                ],

            "v1_compliance_decision":
                v1_aggregate[
                    "compliance_decision"
                ],
        }
    )


    # =====================================================
    # Stop if V1 Judge Panel is incomplete
    # =====================================================

    if not v1_aggregate[
        "panel_complete"
    ]:

        print(
            "\n❌ V1 Cross-Judge Panel "
            "is incomplete."
        )


        print(
            "Experiment cannot continue "
            "with a valid Cross-Judge "
            "compliance decision."
        )


        row[
            "compliance_fix_triggered"
        ] = None


        row[
            "human_review_recommended"
        ] = True


        row[
            "status"
        ] = "FAILED_V1_JUDGES"


        row[
            "error"
        ] = (
            "One or more V1 Judges "
            "failed after all "
            "retry attempts."
        )


        return row


    # =====================================================
    # Decision 1:
    # Consensus No Blocking
    # =====================================================

    if (
        v1_aggregate[
            "compliance_decision"
        ]
        == "CONSENSUS_NO_BLOCKING"
    ):

        print(
            "\n✅ Both Judges detected "
            "no blocking compliance issue."
        )


        print(
            "Quality advisory findings, "
            "if any, remain optional "
            "for human review."
        )


        row.update(
            {
                "compliance_fix_triggered":
                    False,

                "human_review_recommended":
                    False,

                "v2_content":
                    "",

                "v2_revision_latency":
                    None,

                "v2_cross_judge_heuristic":
                    None,

                "v2_cross_judge_claim_risk":
                    None,

                "v2_cross_judge_blocking_count":
                    None,

                "heuristic_score_change":
                    None,

                "claim_risk_reduction":
                    None,

                "blocking_removed":
                    None,

                "pairwise_panel_complete":
                    None,

                "pairwise_agreement":
                    None,

                "pairwise_consensus_preference":
                    None,

                "status":
                    "SUCCESS_NO_BLOCKING",
            }
        )


        return row


    # =====================================================
    # Decision 2:
    # Judge Disagreement
    # =====================================================

    if (
        v1_aggregate[
            "compliance_decision"
        ]
        == "JUDGE_DISAGREEMENT"
    ):

        print(
            "\n⚠️ Judges disagree on "
            "blocking compliance status."
        )


        print(
            "No automatic rewrite is run. "
            "Human review is recommended."
        )


        row.update(
            {
                "compliance_fix_triggered":
                    False,

                "human_review_recommended":
                    True,

                "v2_content":
                    "",

                "v2_revision_latency":
                    None,

                "v2_cross_judge_heuristic":
                    None,

                "v2_cross_judge_claim_risk":
                    None,

                "v2_cross_judge_blocking_count":
                    None,

                "heuristic_score_change":
                    None,

                "claim_risk_reduction":
                    None,

                "blocking_removed":
                    None,

                "pairwise_panel_complete":
                    None,

                "pairwise_agreement":
                    None,

                "pairwise_consensus_preference":
                    None,

                "status":
                    "SUCCESS_REVIEW_REQUIRED",
            }
        )


        return row


    # =====================================================
    # Decision 3:
    # Consensus Blocking
    # =====================================================

    print(
        "\n⚠️ Both Judges detected "
        "blocking compliance issue(s)."
    )


    print(
        f"Running targeted Minimal "
        f"Compliance Fix with "
        f"{candidate_model}..."
    )


    row[
        "compliance_fix_triggered"
    ] = True


    row[
        "human_review_recommended"
    ] = False


    # =====================================================
    # Combine + Deduplicate Policy-Grounded Findings
    # =====================================================

    combined_findings = (
        combine_compliance_findings(
            v1_judges
        )
    )


    row[
        "combined_compliance_finding_count"
    ] = len(
        combined_findings
    )


    row[
        "combined_compliance_findings"
    ] = json.dumps(
        combined_findings,
        ensure_ascii=False,
    )


    print(
        f"\nCombined unique compliance "
        f"findings: "
        f"{len(combined_findings)}"
    )


    for index, finding in enumerate(
        combined_findings,
        start=1,
    ):

        reporters = finding.get(
            "reported_by",
            [],
        )


        print(
            f"  {index}. "
            f"{finding.get('evidence', '')} "
            f"| reported_by="
            f"{reporters}"
        )


    combined_evaluation = {
        "compliance_findings":
            combined_findings,
    }


    # =====================================================
    # Generate V2 Minimal Compliance Fix
    # =====================================================

    try:

        (
            v2_content,
            revision_latency,
        ) = timed_call(
            fix_compliance_issues,

            brand_info=brand_info,

            campaign_brief=campaign_brief,

            original_content=v1_content,

            evaluation=combined_evaluation,

            policy_context=policy_context,

            model_key=candidate_model,
        )


        row[
            "v2_content"
        ] = v2_content


        row[
            "v2_revision_latency"
        ] = revision_latency


        print(
            f"✅ V2 compliance fix "
            f"generated in "
            f"{revision_latency:.2f}s"
        )


    except Exception as error:

        print(
            f"❌ Compliance fix failed: "
            f"{error}"
        )


        row[
            "status"
        ] = (
            "FAILED_COMPLIANCE_FIX"
        )


        row[
            "error"
        ] = str(
            error
        )


        return row


    # =====================================================
    # Evaluate V2
    # =====================================================

    print(
        "\nEvaluating V2 "
        "with all Judges..."
    )


    v2_judges = run_all_judges(
        case_id=case_id,

        challenge=challenge,

        candidate_model=candidate_model,

        version="v2",

        brand_info=brand_info,

        campaign_brief=campaign_brief,

        policy_context=policy_context,

        content=v2_content,

        judge_rows=judge_rows,
    )


    add_judge_scores_to_row(
        row=row,

        version="v2",

        judge_results=v2_judges,
    )


    v2_aggregate = aggregate_judges(
        v2_judges
    )


    row.update(
        {
            "v2_valid_judge_count":
                v2_aggregate[
                    "valid_judge_count"
                ],

            "v2_expected_judge_count":
                v2_aggregate[
                    "expected_judge_count"
                ],

            "v2_judge_panel_complete":
                v2_aggregate[
                    "panel_complete"
                ],

            "v2_cross_judge_heuristic":
                v2_aggregate[
                    "avg_heuristic_score"
                ],

            "v2_cross_judge_claim_risk":
                v2_aggregate[
                    "avg_claim_risk"
                ],

            "v2_cross_judge_blocking_count":
                v2_aggregate[
                    "avg_blocking_count"
                ],

            "v2_judge_score_gap":
                v2_aggregate[
                    "judge_score_gap"
                ],

            "v2_any_blocking":
                v2_aggregate[
                    "any_blocking"
                ],

            "v2_all_blocking":
                v2_aggregate[
                    "all_blocking"
                ],

            "v2_blocking_agreement":
                v2_aggregate[
                    "blocking_agreement"
                ],

            "v2_compliance_decision":
                v2_aggregate[
                    "compliance_decision"
                ],
        }
    )


    # =====================================================
    # Stop if V2 Judge Panel is incomplete
    # =====================================================

    if not v2_aggregate[
        "panel_complete"
    ]:

        print(
            "\n❌ V2 Cross-Judge Panel "
            "is incomplete."
        )


        row[
            "heuristic_score_change"
        ] = None


        row[
            "claim_risk_reduction"
        ] = None


        row[
            "blocking_removed"
        ] = None


        row[
            "pairwise_panel_complete"
        ] = None


        row[
            "pairwise_agreement"
        ] = None


        row[
            "pairwise_consensus_preference"
        ] = None


        row[
            "human_review_recommended"
        ] = True


        row[
            "status"
        ] = "FAILED_V2_JUDGES"


        row[
            "error"
        ] = (
            "One or more V2 Judges "
            "failed after all "
            "retry attempts."
        )


        return row


    # =====================================================
    # Descriptive V1 → V2 Changes
    # =====================================================

    row[
        "heuristic_score_change"
    ] = round(
        v2_aggregate[
            "avg_heuristic_score"
        ]
        - v1_aggregate[
            "avg_heuristic_score"
        ],
        2,
    )


    row[
        "claim_risk_reduction"
    ] = round(
        v1_aggregate[
            "avg_claim_risk"
        ]
        - v2_aggregate[
            "avg_claim_risk"
        ],
        2,
    )


    row[
        "blocking_removed"
    ] = (
        v1_aggregate[
            "all_blocking"
        ]

        and not v2_aggregate[
            "any_blocking"
        ]
    )


    # =====================================================
    # Pairwise V1 vs V2
    # =====================================================

    print(
        "\nRunning V1 vs V2 "
        "pairwise comparison..."
    )


    pairwise = run_pairwise_judges(
        case_id=case_id,

        challenge=challenge,

        candidate_model=candidate_model,

        brand_info=brand_info,

        campaign_brief=campaign_brief,

        policy_context=policy_context,

        v1_content=v1_content,

        v2_content=v2_content,

        pairwise_rows=pairwise_rows,
    )


    row[
        "pairwise_panel_complete"
    ] = pairwise[
        "panel_complete"
    ]


    row[
        "pairwise_agreement"
    ] = pairwise[
        "agreement"
    ]


    row[
        "pairwise_consensus_preference"
    ] = pairwise[
        "consensus_preference"
    ]


    for judge_model in JUDGE_MODELS:

        judge_pairwise = (
            pairwise[
                "results"
            ].get(
                judge_model,
                {},
            )
        )


        prefix = (
            f"pairwise_"
            f"{judge_model}_judge"
        )


        row[
            f"{prefix}_preference"
        ] = judge_pairwise.get(
            "preference"
        )


        row[
            f"{prefix}_latency"
        ] = judge_pairwise.get(
            "latency"
        )


        row[
            f"{prefix}_attempts"
        ] = judge_pairwise.get(
            "attempts"
        )


        row[
            f"{prefix}_error"
        ] = (
            judge_pairwise.get(
                "error"
            )
            or ""
        )


    # =====================================================
    # Final Routing
    # =====================================================

    if (
        v2_aggregate[
            "compliance_decision"
        ]
        == "CONSENSUS_NO_BLOCKING"
    ):

        row[
            "human_review_recommended"
        ] = False


        row[
            "status"
        ] = (
            "SUCCESS_COMPLIANCE_FIX_CLEARED"
        )


    else:

        # If blocking findings remain
        # OR Judges disagree after repair,
        # route to Human Review.

        row[
            "human_review_recommended"
        ] = True


        row[
            "status"
        ] = (
            "SUCCESS_COMPLIANCE_FIX_"
            "REVIEW_REQUIRED"
        )


    return row


# =========================================================
# Final Benchmark Summary
# =========================================================

def print_summary(
    batch_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
):
    """
    Print descriptive benchmark results.

    IMPORTANT:

    This small test suite is designed for:

    - failure-mode coverage
    - candidate comparison
    - compliance behavior analysis
    - Judge consistency analysis

    It is NOT used to calibrate an absolute
    acceptance threshold.
    """

    print(
        "\n\n"
        + "=" * 80
    )


    print(
        "GROWTHPILOT POLICY-GROUNDED "
        "BENCHMARK SUMMARY"
    )


    print(
        "=" * 80
    )


    print(
        "\nInterpretation note:"
    )


    print(
        "- Heuristic scores are "
        "diagnostic signals only."
    )


    print(
        "- No numerical pass/fail "
        "threshold is used."
    )


    print(
        "- Blocking findings must be "
        "grounded in supplied "
        "policy / facts."
    )


    print(
        "- Judge disagreement routes "
        "to human review, "
        "not auto-rewrite."
    )


    successful = batch_df[
        batch_df[
            "status"
        ].astype(
            str
        ).str.startswith(
            "SUCCESS",
            na=False,
        )
    ]


    if successful.empty:

        print(
            "\nNo successful experiments."
        )

        return


    # =====================================================
    # Candidate Summary
    # =====================================================

    print(
        "\n\nCANDIDATE MODEL SUMMARY"
    )


    print(
        "-" * 80
    )


    for candidate_model in CANDIDATE_MODELS:

        candidate_rows = successful[
            successful[
                "candidate_model"
            ]
            == candidate_model
        ]


        if candidate_rows.empty:

            continue


        print(
            f"\nCandidate: "
            f"{candidate_model}"
        )


        print(
            f"Completed cases: "
            f"{len(candidate_rows)}"
        )


        avg_v1_score = safe_numeric_mean(
            candidate_rows[
                "v1_cross_judge_heuristic"
            ]
        )


        avg_v1_risk = safe_numeric_mean(
            candidate_rows[
                "v1_cross_judge_claim_risk"
            ]
        )


        avg_generation_latency = (
            safe_numeric_mean(
                candidate_rows[
                    "v1_generation_latency"
                ]
            )
        )


        avg_judge_gap = (
            safe_numeric_mean(
                candidate_rows[
                    "v1_judge_score_gap"
                ]
            )
        )


        if avg_v1_score is not None:

            print(
                "Average V1 "
                "Heuristic Score: "
                f"{avg_v1_score:.2f}"
            )


        if avg_v1_risk is not None:

            print(
                "Average V1 "
                "Claim Risk: "
                f"{avg_v1_risk:.2f}"
            )


        if (
            avg_generation_latency
            is not None
        ):

            print(
                "Average Generation "
                "Latency: "
                f"{avg_generation_latency:.2f}s"
            )


        if avg_judge_gap is not None:

            print(
                "Average Judge "
                "Score Gap: "
                f"{avg_judge_gap:.2f}"
            )


        total = len(
            candidate_rows
        )


        consensus_blocking = (
            candidate_rows[
                candidate_rows[
                    "v1_compliance_decision"
                ]
                == "CONSENSUS_BLOCKING"
            ]
        )


        disagreement_rows = (
            candidate_rows[
                candidate_rows[
                    "v1_compliance_decision"
                ]
                == "JUDGE_DISAGREEMENT"
            ]
        )


        no_blocking_rows = (
            candidate_rows[
                candidate_rows[
                    "v1_compliance_decision"
                ]
                == "CONSENSUS_NO_BLOCKING"
            ]
        )


        print(
            "Consensus Blocking Rate: "
            f"{len(consensus_blocking) / total:.1%}"
        )


        print(
            "Judge Disagreement Rate: "
            f"{len(disagreement_rows) / total:.1%}"
        )


        print(
            "Consensus No-Blocking Rate: "
            f"{len(no_blocking_rows) / total:.1%}"
        )


        # =================================================
        # Compliance Fix Analysis
        # =================================================

        fixed_rows = candidate_rows[
            candidate_rows[
                "status"
            ].astype(
                str
            ).str.startswith(
                "SUCCESS_COMPLIANCE_FIX",
                na=False,
            )
        ]


        if not fixed_rows.empty:

            print(
                f"Compliance Fix Runs: "
                f"{len(fixed_rows)}"
            )


            avg_score_change = (
                safe_numeric_mean(
                    fixed_rows[
                        "heuristic_score_change"
                    ]
                )
            )


            avg_risk_reduction = (
                safe_numeric_mean(
                    fixed_rows[
                        "claim_risk_reduction"
                    ]
                )
            )


            blocking_removed_rate = (
                fixed_rows[
                    "blocking_removed"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .mean()
            )


            if (
                avg_score_change
                is not None
            ):

                print(
                    "Average Heuristic "
                    "Score Change: "
                    f"{avg_score_change:+.2f}"
                )


            if (
                avg_risk_reduction
                is not None
            ):

                print(
                    "Average Claim "
                    "Risk Reduction: "
                    f"{avg_risk_reduction:+.2f}"
                )


            print(
                "Blocking Findings "
                "Fully Removed Rate: "
                f"{blocking_removed_rate:.1%}"
            )


            # =============================================
            # Pairwise Agreement
            # =============================================

            valid_pairwise = fixed_rows[
                fixed_rows[
                    "pairwise_panel_complete"
                ]
                == True
            ]


            if not valid_pairwise.empty:

                pairwise_agreement_rate = (
                    valid_pairwise[
                        "pairwise_agreement"
                    ]
                    .fillna(
                        False
                    )
                    .astype(
                        bool
                    )
                    .mean()
                )


                print(
                    "Pairwise Judge "
                    "Agreement Rate: "
                    f"{pairwise_agreement_rate:.1%}"
                )


                for preference in [
                    "v2",
                    "v1",
                    "tie",
                ]:

                    count = len(
                        valid_pairwise[
                            valid_pairwise[
                                "pairwise_consensus_preference"
                            ]
                            == preference
                        ]
                    )


                    print(
                        f"Pairwise Consensus "
                        f"{preference.upper()}: "
                        f"{count}"
                    )


    # =====================================================
    # Cross-Judge Compliance Analysis
    # =====================================================

    print(
        "\n\nCROSS-JUDGE "
        "COMPLIANCE ANALYSIS — V1"
    )


    print(
        "-" * 80
    )


    complete_v1 = successful[
        successful[
            "v1_judge_panel_complete"
        ]
        == True
    ]


    if not complete_v1.empty:

        agreement_rate = (
            complete_v1[
                "v1_blocking_agreement"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
            .mean()
        )


        disagreement_count = len(
            complete_v1[
                complete_v1[
                    "v1_compliance_decision"
                ]
                == "JUDGE_DISAGREEMENT"
            ]
        )


        print(
            "Blocking / No-Blocking "
            "Agreement Rate: "
            f"{agreement_rate:.1%}"
        )


        print(
            f"Judge Disagreement Cases: "
            f"{disagreement_count}/"
            f"{len(complete_v1)}"
        )


    # =====================================================
    # Cross-Judge Diagnostic Analysis
    # =====================================================

    print(
        "\n\nCROSS-JUDGE "
        "DIAGNOSTIC ANALYSIS — V1"
    )


    print(
        "-" * 80
    )


    v1_judges = judge_df[
        (
            judge_df[
                "version"
            ]
            == "v1"
        )

        & (
            judge_df[
                "status"
            ]
            == "SUCCESS"
        )
    ]


    for judge_model in JUDGE_MODELS:

        judge_data = v1_judges[
            v1_judges[
                "judge_model"
            ]
            == judge_model
        ]


        if judge_data.empty:

            continue


        print(
            f"\nJudge: "
            f"{judge_model}"
        )


        for candidate_model in CANDIDATE_MODELS:

            candidate_data = judge_data[
                judge_data[
                    "candidate_model"
                ]
                == candidate_model
            ]


            if candidate_data.empty:

                continue


            score_mean = safe_numeric_mean(
                candidate_data[
                    "heuristic_composite_score"
                ]
            )


            blocking_rate = (
                candidate_data[
                    "blocking_flag"
                ]
                .fillna(
                    False
                )
                .astype(
                    bool
                )
                .mean()
            )


            if score_mean is not None:

                print(
                    f"  Candidate "
                    f"{candidate_model}: "
                    f"heuristic="
                    f"{score_mean:.2f}, "
                    f"blocking_rate="
                    f"{blocking_rate:.1%}"
                )


    # =====================================================
    # Pairwise Analysis
    # =====================================================

    print(
        "\n\nPAIRWISE "
        "V1 VS V2 ANALYSIS"
    )


    print(
        "-" * 80
    )


    if pairwise_df.empty:

        print(
            "\nNo pairwise comparisons "
            "were run."
        )


    else:

        successful_pairwise = pairwise_df[
            pairwise_df[
                "status"
            ]
            == "SUCCESS"
        ]


        for judge_model in JUDGE_MODELS:

            judge_pairwise = (
                successful_pairwise[
                    successful_pairwise[
                        "judge_model"
                    ]
                    == judge_model
                ]
            )


            if judge_pairwise.empty:

                continue


            print(
                f"\nJudge: "
                f"{judge_model}"
            )


            total = len(
                judge_pairwise
            )


            for preference in [
                "v2",
                "v1",
                "tie",
            ]:

                count = len(
                    judge_pairwise[
                        judge_pairwise[
                            "normalized_preference"
                        ]
                        == preference
                    ]
                )


                print(
                    f"  {preference}: "
                    f"{count}/"
                    f"{total} "
                    f"({count / total:.1%})"
                )


    # =====================================================
    # Own-model Score Advantage
    # =====================================================

    print(
        "\n\nOBSERVED "
        "OWN-MODEL SCORE ADVANTAGE"
    )


    print(
        "-" * 80
    )


    print(
        "Exploratory descriptive "
        "signal only; this does NOT "
        "prove self-preference bias."
    )


    for judge_model in JUDGE_MODELS:

        judge_data = v1_judges[
            v1_judges[
                "judge_model"
            ]
            == judge_model
        ]


        own_scores = pd.to_numeric(
            judge_data[
                judge_data[
                    "candidate_model"
                ]
                == judge_model
            ][
                "heuristic_composite_score"
            ],
            errors="coerce",
        ).dropna()


        other_candidates = [
            model

            for model
            in CANDIDATE_MODELS

            if model != judge_model
        ]


        other_scores = pd.to_numeric(
            judge_data[
                judge_data[
                    "candidate_model"
                ].isin(
                    other_candidates
                )
            ][
                "heuristic_composite_score"
            ],
            errors="coerce",
        ).dropna()


        if (
            not own_scores.empty
            and not other_scores.empty
        ):

            advantage = (
                own_scores.mean()
                - other_scores.mean()
            )


            print(
                f"\n{judge_model} Judge:"
            )


            print(
                f"Own candidate avg: "
                f"{own_scores.mean():.2f}"
            )


            print(
                f"Other candidate avg: "
                f"{other_scores.mean():.2f}"
            )


            print(
                f"Observed advantage: "
                f"{advantage:+.2f}"
            )


# =========================================================
# Main Experiment
# =========================================================

if __name__ == "__main__":

    cases = load_cases()


    print(
        "\n"
        + "=" * 80
    )


    print(
        "GrowthPilot Policy-Grounded "
        "Cross-Judge Benchmark"
    )


    print(
        "=" * 80
    )


    print(
        f"\nEvaluation Cases: "
        f"{len(cases)}"
    )


    print(
        f"Candidate Models: "
        f"{CANDIDATE_MODELS}"
    )


    print(
        f"Judge Models: "
        f"{JUDGE_MODELS}"
    )


    print(
        f"Judge Max Attempts: "
        f"{JUDGE_MAX_ATTEMPTS}"
    )


    print(
        "Numerical Quality Threshold: "
        "None"
    )


    print(
        "Compliance Auto-Fix Rule: "
        "both Judges must detect "
        "blocking issues"
    )


    print(
        "Judge Disagreement Rule: "
        "route to human review; "
        "no auto-fix"
    )


    print(
        "Pairwise A/B Order Balancing: "
        f"{BALANCE_PAIRWISE_ORDER}"
    )


    print(
        "Cross-Judge Finding Deduplication: "
        "evidence similarity + merged rationale"
    )


    total_candidate_runs = (
        len(
            cases
        )
        * len(
            CANDIDATE_MODELS
        )
    )


    print(
        f"Candidate-Case Runs: "
        f"{total_candidate_runs}"
    )


    batch_rows = []

    judge_rows = []

    pairwise_rows = []


    run_number = 0


    # =====================================================
    # Run All Cases
    # =====================================================

    for case_index, case in enumerate(
        cases,
        start=1,
    ):

        case_id = get_case_id(
            case,
            case_index,
        )


        challenge = get_challenge(
            case
        )


        for candidate_model in CANDIDATE_MODELS:

            run_number += 1


            print(
                "\n\n"
                + "=" * 80
            )


            print(
                f"[{run_number}/"
                f"{total_candidate_runs}] "
                f"{case_id}"
            )


            print(
                f"Candidate: "
                f"{candidate_model}"
            )


            if challenge:

                print(
                    f"Challenge: "
                    f"{challenge}"
                )


            print(
                "=" * 80
            )


            try:

                row = run_candidate_case(
                    case=case,

                    case_index=case_index,

                    candidate_model=(
                        candidate_model
                    ),

                    judge_rows=(
                        judge_rows
                    ),

                    pairwise_rows=(
                        pairwise_rows
                    ),
                )


            except Exception as error:

                print(
                    f"\n❌ Unexpected failure: "
                    f"{error}"
                )


                row = {
                    "case_id":
                        case_id,

                    "challenge":
                        challenge,

                    "candidate_model":
                        candidate_model,

                    "candidate_model_id":
                        MODELS[
                            candidate_model
                        ],

                    "status":
                        "FAILED_UNEXPECTED",

                    "error":
                        str(
                            error
                        ),
                }


            batch_rows.append(
                row
            )


            # =================================================
            # Save Immediately
            # =================================================

            save_results(
                batch_rows,
                judge_rows,
                pairwise_rows,
            )


            print(
                "\n💾 Intermediate "
                "results saved."
            )


    # =====================================================
    # Final Save
    # =====================================================

    save_results(
        batch_rows,
        judge_rows,
        pairwise_rows,
    )


    batch_df = pd.DataFrame(
        batch_rows
    )


    judge_df = pd.DataFrame(
        judge_rows
    )


    pairwise_df = pd.DataFrame(
        pairwise_rows
    )


    # =====================================================
    # Summary
    # =====================================================

    print_summary(
        batch_df,
        judge_df,
        pairwise_df,
    )


    print(
        "\n\n"
        + "=" * 80
    )


    print(
        "Benchmark finished."
    )


    print(
        f"\nMain results:\n"
        f"{BATCH_RESULTS_FILE}"
    )


    print(
        f"\nJudge-level results:\n"
        f"{JUDGE_RESULTS_FILE}"
    )


    print(
        f"\nPairwise results:\n"
        f"{PAIRWISE_RESULTS_FILE}"
    )


    print(
        "=" * 80
    )