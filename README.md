# GrowthPilot

> **AI Campaign Content Governance Copilot for Brand & Agency Teams**  
> Creator-specific guidance, pre-publication review, evidence-grounded routing, and provenance-safe revision.

GrowthPilot is a portfolio AI product that explores a practical problem in creator marketing workflows:

**How can brands scale creator content review without turning every subjective quality issue into a hard compliance block, while still preventing unsupported claims and campaign requirement misses?**

The product focuses on the stage **after a campaign and creator have already been selected** and **before content is published**.

It separates three fundamentally different review decisions:

1. **Compliance / Hard Constraints** — what must be corrected.
2. **Campaign Requirements** — what must be completed.
3. **Quality Advisories** — what could be improved.

The system then uses a **Cross-Judge permission layer** to decide whether AI is allowed to auto-fix content or whether the case should be escalated to a human reviewer.

---

## Table of Contents

- [1. Product Overview](#1-product-overview)
- [2. Problem Definition](#2-problem-definition)
- [3. Target Users and Jobs To Be Done](#3-target-users-and-jobs-to-be-done)
- [4. Product Principles](#4-product-principles)
- [5. End-to-End Workflow](#5-end-to-end-workflow)
- [6. Evaluation Architecture](#6-evaluation-architecture)
- [7. Review Modes](#7-review-modes)
- [8. Routing Logic](#8-routing-logic)
- [9. Creator Experience Provenance Guardrail](#9-creator-experience-provenance-guardrail)
- [10. Product UI](#10-product-ui)
- [11. Evaluation Methodology](#11-evaluation-methodology)
- [12. Evaluation Results](#12-evaluation-results)
- [13. Key Failure Cases and Iteration](#13-key-failure-cases-and-iteration)
- [14. Tech Stack](#14-tech-stack)
- [15. Project Structure](#15-project-structure)
- [16. Quick Start](#16-quick-start)
- [17. Model Configuration](#17-model-configuration)
- [18. Data, Privacy, and Evaluation Scope](#18-data-privacy-and-evaluation-scope)
- [19. Limitations](#19-limitations)
- [20. Future Work](#20-future-work)
- [21. Project Status](#21-project-status)

---

# 1. Product Overview

GrowthPilot is designed as a **Campaign Content Governance Copilot** rather than a generic AI copy generator.

The core workflow is:

```text
Campaign Brief
    ↓
Creator-specific Guidance / Reference Draft
    ↓
Creator Draft
    ↓
AI Pre-review
    ↓
Permission Decision
    ↓
Mandatory Fix / Requirement Completion / Human Review / Optional Advisory
    ↓
Re-check
    ↓
Human Approval
    ↓
Launch
```

The product wedge is intentionally narrow:

> **Brand-side × Pre-approval × Creator-specific × Evidence-grounded**

GrowthPilot does **not** attempt to replace creator selection, media buying, campaign strategy, or final human approval.

Its purpose is to improve the quality and consistency of the **pre-publication creator review process**.

---

# 2. Problem Definition

Creator marketing teams face a recurring operational tension:

```text
Brand Consistency
        ×
Creator Differentiation
        ×
Campaign Control
```

A brand may need hundreds of creator assets to stay aligned with product facts and campaign requirements, while each creator still needs enough freedom to sound authentic.

A single generic LLM review prompt often collapses very different issues into one bucket:

- a false product claim,
- a missing hashtag,
- a tone mismatch,
- a platform-style preference,
- a subjective wording suggestion.

That creates two product risks:

### Risk A — Over-blocking

Subjective style suggestions may be incorrectly presented as compliance violations.

### Risk B — Under-control

Real factual conflicts or campaign requirements may be treated as optional recommendations.

GrowthPilot addresses this by separating the review problem into independent decision layers before deciding what AI is permitted to change.

---

# 3. Target Users and Jobs To Be Done

## Primary Users

### Brand Marketing / Campaign Teams

Need to review large volumes of creator content while preserving brand facts, campaign requirements, and approval control.

### Agency / Creator Operations Teams

Need a structured review workflow that reduces repetitive manual checking and makes feedback easier to explain to creators.

### Content Reviewers

Need to distinguish:

- what is objectively wrong,
- what is missing,
- what is merely a recommendation.

---

## Jobs To Be Done

When reviewing creator content, the user needs to:

- identify unsupported or prohibited claims;
- identify missing campaign requirements;
- separate mandatory changes from optional quality suggestions;
- understand why a finding was raised;
- decide whether AI can safely fix the issue;
- preserve creator voice during revision;
- avoid AI inventing new creator experiences;
- re-check revised content before human approval.

---

# 4. Product Principles

GrowthPilot is built around one central policy:

> **Policy decides what must be corrected. Campaign requirements decide what must be completed. AI advises what could be improved. Humans decide what should ultimately be used.**

中文：

> **规则决定什么必须纠正；营销活动要求决定什么必须补齐；AI 建议什么可以优化；人最终决定采用什么。**

This principle leads to several product rules.

---

## 4.1 Must Avoid ≠ Must Mention

The two are intentionally asymmetric.

```text
Must Avoid violation
→ Blocking Compliance Finding
→ Mandatory correction

Must Mention omission
→ Requirement Finding
→ Mandatory completion
→ NOT labeled as compliance

Tone / Creator / Platform mismatch
→ Advisory Finding
→ Non-blocking
```

This prevents the system from presenting normal campaign execution requirements as legal or compliance violations.

---

## 4.2 No Numeric Pass/Fail Threshold

GrowthPilot exposes diagnostic scores such as:

- Brand Alignment
- Tone Match
- Selling Point Coverage
- Factual Consistency
- Unsupported Claim Risk

However, these scores are **diagnostic only**.

They do not determine whether content passes or fails.

A content item can have imperfect tone and still require no mandatory action.

---

## 4.3 Cross-Judge Controls Permission

The Cross-Judge system is not positioned as:

> “Two models are automatically smarter than one.”

Its purpose is:

> **to control whether AI has permission to perform an automatic mandatory edit.**

When two judges disagree on a mandatory finding, the product can escalate the case instead of allowing one model to make an irreversible decision.

---

# 5. End-to-End Workflow

## Create

The user enters:

- Brand Information
- Campaign Brief
- Product Facts
- Must Mention requirements
- Must Avoid restrictions
- Platform
- Content Type
- Creator Context

GrowthPilot can generate creator-specific reference content or guidance.

---

## Creator Review

A creator draft is submitted for review.

The system evaluates it across three layers:

```text
Creator Draft
    ↓
Compliance Findings
Requirement Findings
Advisory Findings
    ↓
Review Orchestrator
    ↓
Final Route
```

---

## Action

Depending on the route, the product may allow:

- Minimal Compliance Fix
- Minimal Requirement Completion
- Combined Mandatory Fix
- Human Review
- Optional Quality Optimization

---

## Re-check

Any revised content can be reviewed again using the same review mode.

The product keeps **review status** and **content clearance status** separate so a Cross-Judge disagreement cannot accidentally be displayed as “clear”.

---

# 6. Evaluation Architecture

GrowthPilot uses a layered evaluator.

## Layer 1 — Hard Constraint / Compliance

Examples:

- contradiction with supplied product facts;
- violation of explicit Must Avoid rules;
- unsupported guarantee;
- prohibited health or performance claim.

These findings can authorize mandatory correction.

---

## Layer 2 — Campaign Requirement Coverage

Examples:

- missing required hashtag;
- missing mandatory product attribute;
- missing campaign message;
- missing disclosure.

These findings authorize mandatory completion but are not labeled as compliance.

---

## Layer 3 — Quality Advisory

Examples:

- tone mismatch;
- creator-style mismatch;
- platform-fit issue;
- readability problem;
- excessive hype;
- overly formal wording.

These findings are non-blocking.

Each Advisory can also carry provenance:

```text
SUPPLIED_CONTEXT
```

or

```text
GENERAL_HEURISTIC
```

This helps distinguish source-grounded feedback from model-generated marketing judgment.

---

# 7. Review Modes

GrowthPilot supports two product review modes.

## Fast Review

```text
Step-3.5-Flash
    ↓
Structured Evaluation
    ↓
Route
```

Designed for lower-latency product review.

---

## Cross-Judge Review

```text
Creator Draft
      ↓
┌───────────────┐
│               │
Step          Qwen
│               │
└──────┬────────┘
       ↓
Finding Comparison
       ↓
Deterministic Consensus
       ↓
Permission Gate
       ↓
Final Route
```

Current judges:

- `stepfun-ai/Step-3.5-Flash`
- `Qwen/Qwen3.5-35B-A3B`

Cross-Judge execution is currently sequential.

---

# 8. Routing Logic

The review orchestrator can return:

```text
NO_MANDATORY_ACTION
COMPLIANCE_ACTION
REQUIREMENT_ACTION
COMPLIANCE_AND_REQUIREMENT_ACTION
HUMAN_REVIEW_REQUIRED
REVIEW_ERROR
```

---

## NO_MANDATORY_ACTION

No confirmed mandatory issue exists.

The user may still choose to act on Advisory Findings.

---

## COMPLIANCE_ACTION

A confirmed blocking issue exists.

A minimal compliance edit may be authorized.

---

## REQUIREMENT_ACTION

One or more required campaign elements are missing.

A minimal requirement completion may be authorized.

---

## COMPLIANCE_AND_REQUIREMENT_ACTION

Both mandatory issue types exist.

The system performs one conservative combined revision rather than two unrelated rewrites.

---

## HUMAN_REVIEW_REQUIRED

Used when mandatory findings cannot be deterministically resolved across judges.

Automatic mandatory editing is disabled.

---

## REVIEW_ERROR

Infrastructure/provider failures are not treated as product judgment.

They are surfaced separately.

This distinction became important during evaluation, where provider failures were explicitly invalidated and rerun rather than silently counted as Human Review outcomes.

---

# 9. Creator Experience Provenance Guardrail

A held-out evaluation exposed an important failure mode in Optional Quality Optimization.

The original creator draft contained no personal usage experience.

The AI rewrite introduced statements equivalent to:

```text
"I have been using this recently..."
"As a creator..."
"This is perfect for creators like me..."
```

These were stylistically plausible but not grounded in the creator-submitted draft.

This revealed a **semantic provenance gap**:

```text
Creator Profile
≠
Creator Personal Fact
```

A creator profile can guide:

- tone;
- complexity;
- pacing;
- platform style.

It must not be used as evidence that the creator personally:

- used the product;
- used it for a certain duration;
- has a particular identity;
- follows a particular routine;
- experienced a particular outcome.

---

## Final Provenance Flow

```text
Original Creator Draft
        ↓
Optional Quality Optimization
        ↓
AI Candidate
        ↓
Creator Experience Provenance Check
        ↓
Provenance-safe Rewrite
        ↓
Deterministic Risk Smoke Check
        ↓
Cross-Judge Re-check
```

The final guardrail uses both:

1. prompt-level prevention;
2. post-rewrite provenance validation.

It also includes semantic-category checks for newly introduced creator experience types such as:

```text
TRIAL_OR_PRIOR_USAGE
ONGOING_USAGE
DAILY_OR_REPEATED_ROUTINE
CREATOR_IDENTITY_OR_ROLE
PERSONAL_ATTRIBUTE
PERSONAL_RESULT_OR_OUTCOME
PERSONAL_ENDORSEMENT
```

This avoids a simple exact-string problem where:

```text
Original:
最近试了一下

Rewritten:
最近试用了
```

would otherwise be falsely classified as a new creator experience.

---

# 10. Product UI

GrowthPilot currently provides a bilingual Streamlit interface with:

- Create workflow;
- Creator Review workflow;
- Fast Review / Cross-Judge selector;
- structured Compliance findings;
- structured Requirement findings;
- non-blocking Advisory findings;
- judge disagreement visibility;
- mandatory fix controls;
- Human Review lockout;
- optional quality optimization;
- re-check;
- model selector.

---

## Screenshots

> Screenshots will be added after final visual polishing.

Planned screenshots:

```text
docs/images/
├── 01_create_workflow.png
├── 02_clean_review.png
├── 03_compliance_action.png
├── 04_requirement_action.png
├── 05_human_review_disagreement.png
└── 06_optional_optimization.png
```

---

# 11. Evaluation Methodology

GrowthPilot was evaluated in multiple stages rather than relying on a single demo case.

---

## Stage A — Natural Synthetic Benchmark

Purpose:

- compare candidate models;
- observe generation quality;
- compare latency;
- understand model-specific behavior.

Main finding:

- Step was generally faster and more proactive;
- Qwen was more conservative and lower-risk but slower.

This informed the final product configuration:

```text
Fast Review
→ Step

Cross-Judge Review
→ Step + Qwen
```

---

## Stage B — Controlled Creator-Draft Stress Benchmark

A 10-case controlled development/regression suite tested:

- blocking compliance positives;
- requirement omissions;
- advisory-only cases;
- clean controls.

Evaluation Architecture v2.2 achieved:

```text
10/10 case-level compliance routing agreement
100% blocking finding recall
100% requirement detection on designed cases
0 false mandatory actions
```

This was a controlled development suite and is **not presented as held-out production accuracy**.

---

## Stage C — 20-Case Held-out Synthetic Evaluation

A separate 20-case suite was created after the main architecture had been frozen.

Composition:

```text
4 Compliance
4 Requirement Missing
4 Advisory-only
4 Clean
4 Boundary / Human Review
```

The suite was held out from:

- earlier benchmark cases;
- front-end integration cases.

Boundary cases were intentionally excluded from exact-route accuracy because they were designed to test ambiguous policy edges rather than one uniquely correct answer.

---

## Stage D — Label-Masked AI-Assisted Rubric Review

A structured blind review was conducted using a label-masked packet.

This was **not a human-subject user study**.

The methodology is described as:

> **Label-masked AI-assisted Rubric Review**

Scored dimensions:

- Finding Correctness
- Finding Usefulness
- Revision Correctness
- Voice Preservation
- Advisory Usefulness
- Mandatory Action Appropriateness
- Over-editing
- Output Acceptance

The hidden labels were only used after the ratings had been locked.

---

## Stage E — Targeted Provenance Regression

After the held-out evaluation exposed the H16 Optional Optimization provenance failure, the system was patched and tested on:

```text
H05
H15
H16
P01
```

P01 was a new unseen provenance stress case.

This targeted suite validated the fix without rewriting the frozen mandatory-routing architecture.

---

# 12. Evaluation Results

## 12.1 Held-out Routing Results

On the 20-case held-out synthetic suite:

| Metric | Result |
|---|---:|
| Determinate cases | 16 |
| Exact route agreement | **15/16 = 93.75%** |
| Compliance exact route | **3/4 = 75%** |
| Compliance safety intervention | **4/4 = 100%** |
| Requirement routing | **4/4 = 100%** |
| Advisory-only routing | **4/4 = 100%** |
| Clean routing | **4/4 = 100%** |
| False mandatory actions on Advisory + Clean | **0/8 = 0%** |

The difference between **Compliance exact route** and **Compliance safety intervention** matters.

One compliance-positive case was conservatively escalated to Human Review because the two judges disagreed on part of the mandatory evidence.

The system did not incorrectly clear the content.

---

## 12.2 Revision Results

| Metric | Result |
|---|---:|
| Mandatory revisions | 9 |
| Mandatory revision clearance on re-check | **9/9 = 100%** |
| Optional revisions | 4 |
| Optional revisions with no new mandatory issue | **4/4 = 100%** |

Automated route clearance does not automatically mean the rewrite is qualitatively good.

That distinction was important in H16.

---

## 12.3 Label-Masked AI-Assisted Rubric Review

| Dimension | Result |
|---|---:|
| Finding Correctness | **4.60 / 5** |
| Finding Usefulness | **4.65 / 5** |
| Revision Correctness | **4.54 / 5** |
| Voice Preservation | **4.46 / 5** |
| Advisory Usefulness | **3.75 / 5** |
| Mandatory-action appropriateness | **95%** |
| Output acceptance | **95%** |
| Over-editing among revised cases | **7.7%** |

The weakest area was Advisory quality.

Common issues included:

- generic platform heuristics;
- duplicated suggestions;
- overly prescriptive style recommendations;
- occasional creator-persona drift.

This is why Advisories remain non-blocking and human-triggered.

---

## 12.4 Targeted Provenance Regression

After the provenance fix:

| Metric | Result |
|---|---:|
| Strict pass | **4/4** |
| Creator provenance safety | **4/4** |
| No mandatory regression | **4/4** |
| Quality edit applied | **4/4** |
| Invalid runs | **0** |

The final regression included:

- one case with an existing creator trial experience;
- one low-hype B2B tone case;
- the original H16 failure case;
- one new unseen creator-provenance stress case.

---

# 13. Key Failure Cases and Iteration

GrowthPilot was intentionally evaluated for failures rather than only for positive demo outputs.

Three examples were especially useful.

---

## H12 — Conservative Human Review Escalation

The content contained a confirmed unsupported 4K claim.

Both judges agreed on that violation.

They disagreed on whether a second “optical stabilization” statement had sufficiently grounded blocking provenance.

Instead of automatically modifying the disputed issue, the Cross-Judge system returned:

```text
HUMAN_REVIEW_REQUIRED
```

This was an exact-route mismatch against the designed label, but a safe product behavior.

---

## H16 — Optional Rewrite Provenance Failure

The original issue was only stylistic:

- too formal;
- too technical;
- insufficiently creator-like.

The first version of Optional Optimization produced a more natural caption but invented:

- first-person usage history;
- creator identity;
- personal experience.

The automated re-check still considered the rewrite clear.

This demonstrated that:

> **a downstream evaluator cannot detect provenance it was never explicitly given.**

The fix introduced an original-vs-revised creator experience provenance comparison.

---

## H05 — Regression Detector False Positive

The first provenance guard correctly fixed H16, but the first regression run achieved only:

```text
3/4 strict pass
```

H05 originally contained:

```text
最近试了一下
```

The rewrite contained:

```text
最近试用了
```

A phrase-based detector incorrectly classified the paraphrase as a newly invented experience.

The detector was changed from exact phrase matching to **semantic experience-category matching**.

The next regression run achieved:

```text
4/4 strict pass
```

This iteration reinforced an important evaluation lesson:

> Regression tests should be designed to discover new failure modes, not merely confirm a patch.

---

# 14. Tech Stack

## Application

- Python 3.13
- Streamlit
- pandas
- python-dotenv

## LLM Integration

- OpenAI-compatible Python SDK
- ModelScope inference endpoint

## Models

- Step-3.5-Flash
- Qwen3.5-35B-A3B

## Evaluation

- structured JSON outputs;
- deterministic routing;
- controlled benchmark suites;
- held-out synthetic evaluation;
- Cross-Judge comparison;
- label-masked AI-assisted rubric review;
- targeted regression testing.

---

# 15. Project Structure

```text
GrowthPilot/
├── data/
│   ├── evaluation_cases.json
│   ├── creator_draft_stress_cases.json
│   └── human_eval_cases.json
│
├── experiments/
│   ├── model_screening.py
│   ├── batch_evaluation.py
│   ├── creator_draft_stress_baseline.py
│   ├── creator_draft_stress_v2.py
│   ├── human_evaluation.py
│   ├── provenance_regression.py
│   └── results/
│
├── locales/
│   ├── __init__.py
│   ├── en.py
│   └── zh.py
│
├── src/
│   ├── llm_client.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── reviser.py
│   └── review_orchestrator.py
│
├── app.py
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```

> `.env` is excluded from version control.

---

# 16. Quick Start

## 16.1 Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd GrowthPilot
```

---

## 16.2 Create an environment

Example with Conda:

```bash
conda create -n growthpilot python=3.13
conda activate growthpilot
```

---

## 16.3 Install dependencies

```bash
pip install -r requirements.txt
```

Current core dependencies:

```text
streamlit
pandas
python-dotenv
openai
```

---

## 16.4 Configure environment variables

Create a local `.env` file.

Example:

```env
MODELSCOPE_API_KEY=your_api_key_here
```

Do not commit credentials to Git.

---

## 16.5 Run the application

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

# 17. Model Configuration

The current ModelScope OpenAI-compatible endpoint is configured inside the LLM client.

Current product configuration:

```text
Fast Review
→ Step-3.5-Flash

Cross-Judge Review
→ Step-3.5-Flash + Qwen3.5-35B-A3B

Revision / Optional Optimization
→ selectable model
```

Model names and provider availability may change over time.

---

# 18. Data, Privacy, and Evaluation Scope

All public portfolio evaluation examples are synthetic.

The project does not publish:

- client campaign data;
- confidential agency SOPs;
- private creator content;
- proprietary brand documents.

The held-out suite is a **synthetic evaluation suite**, not production traffic.

The AI-assisted rubric review is not presented as a human user study.

No claim is made that the reported metrics represent real-world production accuracy.

---

# 19. Limitations

GrowthPilot is a portfolio prototype, not a production compliance system.

Important limitations include:

---

## 19.1 No Legal Guarantee

The Compliance layer only evaluates supplied policy/product context.

It is not a substitute for legal, regulatory, medical, or professional compliance review.

---

## 19.2 Synthetic Evaluation

The current benchmark and held-out suites are synthetic.

Real production performance would require:

- real reviewer disagreement data;
- domain-specific policy sets;
- real creator drafts;
- multilingual error analysis;
- reviewer calibration;
- production monitoring.

---

## 19.3 Provider Reliability

Cross-Judge review can experience high tail latency due to model/provider runtime.

The current implementation uses sequential judge calls.

---

## 19.4 Advisory Quality

Advisory suggestions remain the weakest component.

They can be:

- repetitive;
- generic;
- platform-stereotyped;
- overly prescriptive.

For this reason, they are intentionally non-blocking.

---

## 19.5 Provenance Guard Coverage

The final provenance guard reduces a known creator-experience failure mode but does not prove complete semantic provenance safety.

More diverse multilingual regression cases would be needed for production use.

---

## 19.6 No Smart Auto Mode

The product currently avoids dynamically switching review modes based on a heuristic risk score.

This is intentional.

A future Smart Auto mode would need separate evaluation before being trusted with mandatory-action routing.

---

# 20. Future Work

The current version is feature-frozen.

Potential future directions are documented rather than immediately added.

---

## Evaluation

- larger fresh held-out suite;
- real reviewer calibration;
- inter-rater agreement analysis;
- multilingual stress testing;
- adversarial policy cases.

---

## Performance

- parallel Cross-Judge calls;
- provider fallback;
- structured latency monitoring;
- cost/latency routing.

---

## Product

- review history;
- team approval workflow;
- revision diff view;
- reviewer comments;
- campaign-level dashboard;
- policy versioning;
- audit trail.

---

## Provenance

- more robust semantic provenance comparison;
- claim-source linking;
- sentence-level source traceability;
- creator testimony provenance labels.

---

# 21. Project Status

```text
Status: FINAL FEATURE FREEZE
```

Current frozen milestone includes:

```text
✓ Creator content workflow
✓ Bilingual UI
✓ Layered Compliance / Requirement / Advisory review
✓ Fast Review
✓ Cross-Judge Review
✓ Deterministic permission routing
✓ Human Review escalation
✓ Minimal mandatory revision
✓ Optional quality optimization
✓ Creator Experience Provenance Guardrail
✓ Re-check workflow
✓ Controlled benchmark
✓ 20-case held-out synthetic evaluation
✓ Label-masked AI-assisted rubric review
✓ 4-case targeted provenance regression
```

---

## Current Portfolio Summary

GrowthPilot demonstrates an AI product workflow that moves beyond simple prompt generation.

The core product contribution is not the number of LLM calls.

It is the decision architecture around them:

```text
Evidence
    ↓
Structured Findings
    ↓
Layered Product Semantics
    ↓
Cross-Judge Permission Control
    ↓
Conservative Revision
    ↓
Provenance Validation
    ↓
Human Approval
```

The project is designed to show how an AI PM can think about:

- product scope;
- policy semantics;
- human-AI decision boundaries;
- evaluation methodology;
- failure analysis;
- guardrail design;
- model trade-offs;
- regression testing;
- production-oriented product behavior.

---

## Visual Assets To Be Added

The following assets will be added after final visual polishing:

```text
[TODO] Product Workflow Diagram
[TODO] Cross-Judge Architecture Diagram
[TODO] Creator Experience Provenance Diagram
[TODO] UI Screenshots
[TODO] Evaluation Charts
```

These assets are intentionally left out of the current version so the written product logic can be finalized first.
