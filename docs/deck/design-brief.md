# Progress Report — Presentation Design Brief (3 min)

> This is a **design specification**, not slide content. It defines what the
> presentation must communicate and how it should look, so that Claude Code +
> a presentation skill can reliably generate the slides in a later step.

> **Status:** original brief, kept as design rationale. The delivered deck
> (`progress-report-v1.pptx`) has since evolved — Phase 3 is reframed as a
> **Control Perspective** (adaptive-control questions used as a *lens*, not an
> "interpretation"), a **title slide** was added, and wording was tightened.

---

## Project

**Title:** Bio-inspired Adaptive Locomotion via Torque-based Learning: A Case Study of SATA

**Subtitle (optional):** A simulation-based study using adaptive and robust control questions as a lens

---

## Overall Goal

Generate a concise **academic progress presentation** for an **Adaptive Control
Systems** course. In 3 minutes it must convince the instructor that:

1. The problem is clearly defined.
2. The topic fits the course.
3. There is an executable plan.
4. This is **not** a pure AI paper review.

The presentation should communicate:

1. A meaningful **control problem**.
2. **Why** SATA is chosen.
3. The planned **simulation methodology**.
4. How **adaptive control questions** will be used as an analytical *lens* — not
   as a replacement for — the learning framework.

**Tone guardrails**

- Avoid overselling novelty.
- Do **not** claim SATA is classical adaptive control.
- Focus on **understanding adaptive behavior**.

---

## Additional Generation Constraints

This presentation is for a **3-minute academic progress report**, not a final
defense. The audience already understands basic control concepts.

Slides should optimize for:

- fast understanding
- clear research positioning
- feasibility

The presentation should leave the audience with:

> The project is technically meaningful and realistically executable.

### Slide Design Rules

Each slide must satisfy **ONE main message only**, with no more than:

- 25–35 words per content block
- 3 content blocks per slide
- 1 main visual

**Avoid:** paragraphs · excessive equations · excessive references · screenshots of papers.

**Prefer:** diagrams · comparison layouts · process flows · simple annotations.

### Speaker Support

For each slide, generate **visible slide content** *and* **hidden speaker notes**.

Speaker notes should contain:

- intended talking points
- a transition sentence
- estimated speaking time (target **35–60 sec per slide**)

### Academic Positioning Rules

**Do NOT claim:**

- SATA is adaptive control
- RL replaces control theory
- a novel algorithm contribution

**Preferred wording:** *inspired by* · *analyzed from* · *questions used as a lens* · *preliminary exploration*.

### Visual Communication Rules

- Every **figure** must answer: *Why is this figure shown?*
- Every **diagram** must answer: *What should the audience learn?*

### Technical Depth Rules

Prioritize: **Problem → Mechanism → Comparison → Implementation**

Not: Implementation → Framework → Code.

### Deliverables

Generate:

1. Slide outline
2. Slide content
3. Speaker notes
4. Figure suggestions
5. Citation placement

Do **NOT** generate a final script.

---

## Slide Count & Pacing

Target: **4 slides (~3 minutes)**

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Motivation & Problem Definition | 40 sec |
| 2 | SATA Overview | 60 sec |
| 3 | Project Plan | 50 sec |
| 4 | Expected Contribution | 30 sec |

---

## Slide 1 — Motivation & Problem Definition

**Goal:** Convince the audience this is fundamentally a **control problem**, not
an RL presentation.

**Key message:** Position-based locomotion performs well under known conditions
but struggles with **compliance, disturbance handling, unknown terrain, and
sim-to-real robustness**. This motivates **torque-based locomotion**.

**Layout**

- **LEFT:** Problem statement.
- **RIGHT:** Simple conceptual comparison.

  ```
  Position Control                 Torque Control
  command position                 command force
        ↓                                ↓
   rigid behavior                 compliant interaction
        ↓                                ↓
   poor adaptation                 adaptive response
  ```

- **BOTTOM — Research Question:**
  > How can robots achieve safer and more adaptive locomotion in unknown
  > environments?

**Background knowledge to convey briefly**

- *Position control:* policy outputs target joint angle; low-level PD converts to torque.
- *Torque control:* policy outputs torque directly.
- *Compliance:* robot yields instead of resisting.
- *Generalization:* works outside the training distribution.

**References (this slide):** SATA (main); Chen et al. (Learning Torque Control);
Lee et al. (Challenging Terrain); Miki et al. (Perceptive Locomotion in the Wild).

---

## Slide 2 — SATA Overview

**Goal:** Explain *what SATA contributes*. **Do not** explain equations.

**Key message:** SATA introduces **bio-inspired adaptation** into torque-based
locomotion.

**Layout — CENTER flow:**

```
Observation
    ↓
Torque Policy (RL)
    ↓
Biomechanical Layer
    ↓
Torque Output
    ↓
  Robot
    ↑
Fatigue Feedback  (loops back)
```

**Highlight two blocks**

- **Biomechanical Model:** activation · muscle · fatigue.
- **Growth Mechanism:** torque limit · training curriculum · control frequency.

**Background knowledge to convey**

- *Activation:* smooth actuator command.
- *Muscle:* prevent abrupt torque.
- *Fatigue:* avoid overusing joints.
- *Growth:* progressively unlock capability.
- *Zero-shot sim-to-real:* deploy without finetuning.

**References (this slide):** SATA; Hill (muscle dynamics); Liu et al. (activation,
fatigue, recovery); Bellegarda & Ijspeert (CPG-RL).

---

## Slide 3 — Project Plan

**Goal:** Demonstrate feasibility; show this is not only a literature review.

**Key message:** **Understand → Reproduce → Analyze** — not redesign RL.

**Layout — timeline:**

```
Phase 1            Phase 2            Phase 3                 (Optional)
Reproduce SATA  →  Ablation        →  Control Perspective    → Residual Compensation
```

**Per-phase detail**

- **Phase 1 — Reproduce SATA**
  - Environment: CUDA server · container · venv · Isaac Gym.
  - Output: simulation running.
- **Phase 2 — Ablation**
  - Disable fatigue · change torque limit · modify growth schedule · terrain variation.
- **Phase 3 — Control Perspective** *(questions from adaptive control, as a lens)*
  - Growth ↔ gain scheduling.
  - Fatigue ↔ feedback.
  - Torque modulation ↔ robustness.
- **Optional — Residual Compensation**
  - `τ_total = τ_SATA + τ_comp`
  - Only if feasible.

**References (this slide):** SATA; RL2AC; DecAP.

---

## Slide 4 — Expected Contribution

**Goal:** Close the story; avoid exaggerated claims.

**Key message:** This work **studies adaptive behavior** rather than proposing a
new RL algorithm.

**Layout**

- **LEFT — Expected Outputs:**
  - Simulation reproduction.
  - Behavior analysis.
  - Control-perspective comparison.
- **RIGHT — Research Questions:**
  - RQ1: Why torque control?
  - RQ2: What creates adaptation?
  - RQ3: How does SATA address problems traditionally studied in adaptive control?
  - RQ4: Can lightweight compensation help?
- **BOTTOM — one-line conclusion.**

**Suggested closing sentence:**
> This project studies how bio-inspired torque control produces adaptive
> locomotion, and asks how SATA answers questions traditionally posed by
> adaptive and robust control.

---

## Reference Policy

- **Slides:** 5–6 references max.
- **Final report:** 10 references.
- Cite **only where a concept is introduced**.
- Do **not** fill slides with a bibliography.

---

## Visual Principles

**Prefer:** diagrams · arrows · conceptual comparison.

**Avoid:** equations · architecture screenshots · large tables · RL training curves.

**If a figure is used:** annotate the takeaway. Never show figures without
explanation.
