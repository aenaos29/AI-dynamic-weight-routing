# AI Dynamic Weight Routing

**AENAOS research proposal · v0.2 · November 2026**  
**Status:** Conceptual design for public technical review; **not an implemented or experimentally validated system**.

> **Only the best is good enough.** The objective is not to make one checkpoint perform every task equally well, but to select and efficiently activate the best *validated* parameter configuration for the current requirement.

## Abstract

This proposal explores a **single shared base language model** whose active parameter configuration can change during inference. Candidate **Weight Maps** originate from checkpoints at different stages of sequential training **or** independent training branches derived from the same base model. After training, candidates are evaluated by domain or capability; their measured performance determines which tags they receive and which configurations are preferred.

A **Tagger** identifies user intent and relevant conversation history, retrieves context by semantic tags, and routes work to an appropriate Weight Map. A runtime may preload a small working set from a larger shared library and choose between replacing a complete active parameter state and updating only parameters that differ. A single conversation history is preserved even when a user invokes different named configurations (for example, Luna, Sol, or Aurora).

This is a **research proposal**, not a claim that arbitrary checkpoint deltas can already be swapped at token granularity without overhead or regression.

## 1. Motivation and design goals

**Optimization before scaling:** The research objective is to test whether selecting and activating already-trained parameter configurations can improve quality per unit of available compute and memory, rather than treating larger models or stronger hardware as the default response to every performance limitation. This is a hypothesis, not an established resource saving.


Training may improve one capability while reducing performance in another. The most recent checkpoint is therefore not necessarily the best checkpoint for every task. Maintaining a collection of measured training states may allow the system to recover capabilities that are less effective in the final checkpoint.

The architecture aims to:

- Prioritize **fulfillment of the user's requirement**, including correctness, relevance, reliability, and appropriate uncertainty.
- Preserve useful parameter configurations from training history rather than treating only the final checkpoint as valuable.
- Improve **selection and routing** before assuming more training is required.
- Retrieve relevant conversation history without repeatedly processing unrelated material.
- Avoid loading every available specialization into every worker's accelerator memory.
- Permit one base model and one logical conversation to support different user-selectable configurations.
- Measure the actual trade-offs among answer quality, latency, memory use, bandwidth, and operational complexity.

These goals do **not** imply that a single configuration can seamlessly combine every independently learned skill.

## 2. Terminology

| Term | Meaning in this proposal |
| --- | --- |
| **Base model** | A shared model architecture and reference parameter state, `W0`. |
| **Training state / checkpoint** | A parameter state `Wi` obtained at a stage of sequential training or on a training branch. |
| **Weight Map** | A representation of parameter differences associated with a checkpoint and a declared reference state. It includes sufficient metadata to reconstruct or apply the intended parameter configuration. |
| **Capability tag** | A searchable label associated with a domain or skill, assigned using evaluation evidence rather than training intent alone. |
| **Tagger** | The component that indexes/retrieves context and selects or recommends suitable parameter configurations. |
| **Weight Map library** | The collection of available candidate maps and their metadata, not necessarily all resident in GPU memory. |
| **Resident map** | A map preloaded into memory and available for activation. |
| **Active map** | The parameter configuration currently used by a particular inference worker. |
| **Worker** | An inference execution instance; multiple workers may share a map library. |
| **Conversation context** | The persistent logical history, instructions, decisions, constraints, and relevant retrieved passages. |
| **KV cache** | Intermediate attention state computed under a particular model configuration; not interchangeable with persistent conversation history. |

## 3. Sources of Weight Maps

Two complementary training paths are proposed:

### 3.1 Sequential checkpoints

```text
W0 → W1 → W2 → W3 → …
```

A capability may perform best at `W1` even if later checkpoints improve other capabilities.

### 3.2 Independent branches from a shared base

```text
          ┌→ W_language
W0 ───────┼→ W_math
          └→ W_logic
```

Branches share the same initial architecture and base parameters but follow different training paths. A map is not automatically compatible with another map merely because both originate from `W0`; interactions and runtime assumptions must be tested.

### 3.3 Mathematical representation

For a checkpoint `Wi`, one possible representation is:

```text
ΔWi = Wi − W0
Wi  = W0 + ΔWi
```

The map must record its **reference checkpoint**, tensor identities/shapes, numerical format, and reconstruction requirements. Differences may also be stored relative to another explicitly identified checkpoint, such as the first selected tag map.

**Important:** A small numerical delta is not necessarily cheap to store. Dense deltas can require almost as much space as complete checkpoints. Sparsity, compression, quantization, low-rank approximation, and their effect on output quality must be measured rather than assumed.

## 4. Post-training evaluation and tag assignment

**Tags represent measured capability, not merely the stated purpose of training.**

```text
Training / checkpoints
        ↓
Evaluation by capability and domain
        ↓
Comparison against other candidate maps
        ↓
Validated best performer(s) and alternatives
        ↓
Capability tags + performance metadata
        ↓
Weight Map library
```

Each candidate should be evaluated using appropriate held-out tasks, quality criteria, regression tests, and uncertainty estimates. Selection should consider multiple dimensions where relevant: correctness, robustness, calibration, latency, memory, and energy. There may be no single best map across all dimensions; the selection policy must define the trade-off.

Illustrative, **hypothetical** results:

| Map | Mathematics | Greek | Logic |
| --- | ---: | ---: | ---: |
| W1 | 82% | **91%** | 78% |
| W2 | **94%** | 85% | 83% |
| W3 | 88% | 89% | **96%** |

Illustrative assignments: `MATH → W2`, `GREEK → W1`, `LOGIC → W3`. These figures are not experimental findings.

Retain rankings and confidence intervals where possible, not just one winner: a second-best map may be preferable under tighter latency or memory constraints.

## 5. Tagger: context index and parameter router

The Tagger has related but distinct responsibilities:

1. **Interpret the request:** derive semantic tags, constraints, and task boundaries.
2. **Retrieve context:** use tags to locate relevant messages, prior decisions, source passages, and dependencies. The full conversation remains the authoritative record.
3. **Select capabilities:** match task requirements and context to validated map performance.
4. **Plan residency:** predict which maps are likely to be needed and request selective preloading.
5. **Route execution:** choose an active map for a worker, or switch at an appropriate processing boundary.
6. **Evaluate conversation progress:** observe repeated requests, corrections, refinement, and topic shifts as *imperfect* signals of whether the requirement was satisfied.

A repeated or neighboring tag may indicate an unresolved answer; more specific tags may indicate successful progression; a topic change may indicate completion. **None of these signals alone proves user satisfaction.** The system should avoid optimizing for silence or discouraging follow-up questions.

### 5.1 Context is not the same as weights

The persistent history can be represented as messages, tokens, metadata, and semantic indexes. It is independent of the selected Weight Map. However, the transformer's internal **KV cache is computed using particular weights**. Changing weights may require cache invalidation or recomputation. Targeted weight replacement does not automatically make prior KV values valid.

```text
User request + shared conversation history
                  ↓
          Semantic tags
                  ↓
      Relevant context retrieval
                  ↓
     Capability / Weight Map selection
                  ↓
          Inference worker
                  ↓
             Response
                  ↓
     Feedback and routing evaluation
```

Retrieved context must preserve relevant dependencies and constraints; tag matches without supporting source material must not be treated as verified memory.

## 6. Relationship to Mixture of Experts (MoE)

The closest established conceptual analogue is **Mixture of Experts (MoE)** and its **expert router**.

| Aspect | Conventional MoE | Proposed Tagger / Weight Maps |
| --- | --- | --- |
| Selection target | Expert modules within a model | Parameter configurations derived from training checkpoints/branches |
| Selection mechanism | Learned routing, often per token/layer | Tag/context/performance-driven routing; exact granularity to be tested |
| Parameters | Experts are typically designed and trained as part of the architecture | Candidate states arise from sequential or branched training of a shared base |
| Context retrieval | Not inherently a function of MoE routing | Explicit responsibility of the Tagger |
| Switching | Selectively executes resident experts | May require changing active parameters and recomputing internal state |

**This proposal is not equivalent to MoE.** Existing MoE implementations demonstrate the value of conditional computation, but do not establish that arbitrary full-model checkpoint deltas can be switched or combined with MoE-like efficiency. Adapter routing, LoRA, model merging, and sparse parameter updates are additional relevant baselines for comparison.

## 7. Runtime memory and switching

A library might expose **100 maps to 1,000 workers** while each worker keeps only a small, task-specific set resident. These counts are an **illustrative scenario**, not a hardware benchmark or requirement.

```text
Shared library: 100 available Weight Maps
                    ↓
           Tagger / scheduler
                    ↓
       Worker-specific map selection
                    ↓
        Resident map cache (RAM/VRAM)
                    ↓
         Active parameter configuration
```

### 7.1 Two switching strategies

- **Full replacement:** restore or load the target parameter state.
- **Targeted replacement:** update only parameter positions whose values differ between the **currently active** and **target** states.

A targeted transition from `WA` to `WB` must account for **all** positions differing between those states, including positions changed in `WA` that return to their base values in `WB`. Otherwise, stale values can remain.

### 7.2 Two reference choices

- **Main-map reference:** store or reconstruct differences relative to `W0`.
- **First-tag-map reference:** where useful, represent differences relative to the first selected map for the task/session.

**Storage reference and transition source are different concepts.** A map stored relative to `W0` can still be activated by calculating a transition from whichever state is currently resident.

The scheduler should compare estimated **full replacement cost** with **targeted update cost**, including transfer bandwidth, scattered memory writes, synchronization, cache invalidation, and inference restart overhead. Targeted updates may be slower if differences are dense or irregular.

### 7.3 Concurrency and consistency

Workers must not mutate shared live parameters in ways that corrupt concurrent inference requests. Possible implementations require isolated active states, synchronization, immutable base weights with controlled overlays, or another demonstrated safe mechanism. This is an engineering requirement, **not yet a solved component of this proposal**.

## 8. UNION Chat: one model, one conversation, selectable configurations

UNION Chat is an illustrative application concept:

- One underlying model architecture and shared logical conversation history.
- A user may preselect permitted named configurations (metaphorically, different *aprons* for the same worker).
- Selecting a name such as `Luna`, `Sol`, or `Aurora` maps to a particular validated parameter configuration, **not a separate model identity or a new conversation**.
- The Tagger can choose among user-authorized maps and preload likely needs.
- The same design can be deployed as **one worker switching configurations sequentially** or **several workers running concurrently**, subject to hardware and isolation constraints.

```text
Same conversation + same base model
               ↓
        User selects “Sol”
               ↓
        Name → Weight Map
               ↓
   Activate target parameters safely
               ↓
    Process relevant shared context
               ↓
             Answer
```

The logical conversation remains continuous, but any KV cache associated with an earlier configuration may need to be rebuilt. Named configurations should not be mistaken for separate independently trained base models.

## 9. Success criteria and calibration

The **primary objective** is satisfying the user's actual requirement, not maximizing a proxy such as routing confidence or minimizing latency in isolation.

Potential evaluation signals:

- Correctness and task completion against explicit requirements.
- Need for user corrections or repeated clarification of the **same unresolved request**.
- Progression from a general question to a more specific one.
- Confidence calibration and appropriate acknowledgment of uncertainty.
- End-to-end latency, time to first token, VRAM use, and transfer volume.
- Regression rates when switching maps and reusing/rebuilding context.

These are hypotheses for evaluation. User silence, a topic change, or a repeated tag is **not** a definitive success/failure label. The system must distinguish genuine topic deepening from unresolved repetition and must not learn to suppress necessary questions.

## 10. Open technical questions

1. **Map sparsity:** How many parameters differ meaningfully between checkpoints, and at what numerical precision?
2. **Reconstruction fidelity:** Can compressed deltas preserve the measured capabilities?
3. **Switching granularity:** Is routing practical per request, paragraph, sentence, or token? At what cost?
4. **KV cache:** When is invalidation mandatory, and can any safe reuse be established?
5. **Concurrent workers:** How can parameter updates remain isolated and race-free?
6. **Combined skills:** What happens when a task needs capabilities best represented by multiple incompatible states?
7. **Routing quality:** Can a Tagger select better maps than a static baseline, accounting for overhead?
8. **Context retrieval:** Can tags preserve the relevant evidence, decisions, and constraints without hallucinating missing history?
9. **Evaluation reliability:** How do we prevent overfitting to domain benchmarks or misleading conversational feedback?
10. **Prior art:** Which elements are already addressed by MoE, adapters/LoRA, checkpoint merging, dynamic inference, or sparse updates?

## 11. Proposed Kaggle notebook: first experimental evidence

**Status: planned; no proof-of-concept results are claimed here.** Prior training experiments with small language models may provide candidates, but usable weight checkpoints and their exact compatibility must first be confirmed. Training logs or loss values alone are not sufficient to reconstruct Weight Maps.

### Phase A — Two-map switching correctness

1. Choose one base model `W0` and two compatible checkpoints from its training lineage: a lower-performing candidate `W_low` and a higher-performing candidate `W_high` on a clearly specified capability.
2. Evaluate both on the same held-out task set, keeping prompts, decoding settings, scoring and model architecture consistent.
3. Calculate `ΔW_low = W_low − W0` and `ΔW_high = W_high − W0` (or use an explicitly declared reference and exact reconstruction procedure).
4. Run `W0`, `W0 + ΔW_low`, and `W0 + ΔW_high` as separate baseline states.
5. Switch repeatedly on **one inference worker**: `low → high → low → high`, restoring all changed positions and rebuilding KV state as necessary.
6. Verify that each activated state reproduces its corresponding checkpoint's outputs or scores within documented numerical tolerances. Record load/switch times and peak RAM/VRAM.

A successful Phase A demonstrates **correct state restoration and switching**, not necessarily a speed or memory advantage.

### Phase B — Expand to three and four maps

Add compatible checkpoints `W3` and `W4`. Score every candidate **per language** using a selection/validation set. A map is tagged for a language based on its measured performance, **not** merely the language or objective used during training.

Illustrative only — these numbers are **not Kaggle results**:

| Language | W1 | W2 | W3 | W4 | Selected map |
| --- | ---: | ---: | ---: | ---: | --- |
| Greek | 85% | **92%** | 88% | 86% | W2 |
| English | **94%** | 89% | 91% | 90% | W1 |
| German | 81% | 84% | **93%** | 87% | W3 |
| French | 83% | 86% | 88% | **95%** | W4 |

### Phase C — On-the-fly language routing

Implement a minimal Tagger that detects the target language and activates its validated top-performing map at **request boundaries**. Keep a single base model architecture and worker; measure switching and cache-rebuild costs. Compare against each fixed checkpoint, the base model, and a simple non-switching baseline.

Use **separate data for map selection and final evaluation** to avoid reporting selection-set gains as generalization. Report per-language scores and an explicitly defined aggregate (e.g. equal-weight macro-average), alongside latency, memory, switching overhead, and failure cases. A better aggregate score does **not** imply improvement in every language.

Only after request-level routing works should finer-grained routing, predictive preloading, or context-based routing be investigated.

### Reproducibility and reporting

The notebook should publish, where licensing and privacy permit: model and checkpoint identifiers, training lineage, environment and dependency versions, exact evaluation prompts or datasets, deterministic settings/seeds, scoring scripts, raw per-language results, switching traces, hardware specifications, and observed failures. Do not publish restricted model weights or private training data without permission.

**Falsification matters:** If dynamic routing does not improve held-out quality, or the switching overhead outweighs its benefit, document that outcome rather than presenting the architecture as validated.

## 12. What this proposal does *not* claim

- It does not claim a working prototype or verified performance gains.
- It does not claim that every checkpoint has a uniquely isolated skill.
- It does not claim that independent weight deltas can be added together without interference.
- It does not claim that small weight changes imply small KV-cache changes.
- It does not claim that storing differences always saves memory.
- It does not claim that conversational tag transitions alone prove satisfaction.
- It does not claim novelty over prior research without a dedicated literature review.

## 13. Request for community review

Feedback from researchers and engineers is welcome, especially on:

- Closest existing research and implementations.
- Fundamental incompatibilities or missing assumptions.
- Efficient representations and switching of checkpoint-derived parameter states.
- Safe inference-time state management and KV-cache behavior.
- Rigorous comparisons with MoE routing, LoRA/adapters, and model merging.
- Minimal experiments that could falsify the central hypotheses.

Please distinguish **theoretical impossibility**, **engineering difficulty**, **poor cost-benefit**, and **questions that require measurement**. Well-supported disagreement is as valuable as confirmation.

---

### Core principle

> **Train to create capabilities. Evaluate to identify where each capability performs best. Tag to retrieve and route. Activate only what is needed. Judge success by whether the user's requirement is actually met.**

*This README is a proposal for technical discussion, not a statement of established results.*
