#  Agent Workflow Guide

This document explains how to use Athena, Gaia, and Nemesis to efficiently generate the remaining workshop content.

---

##  The Three-Agent Workflow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   ATHENA    │ --> │    GAIA     │ --> │   NEMESIS    │
│  (Planner)  │     │  (Executor) │     │ (Validator)  │
│   Opus      │     │   Haiku     │     │   Sonnet     │
└─────────────┘     └─────────────┘     └──────────────┘
  Day Planning       Content Writing      Quality Checks
     ^                                          |
     |__________________________________________|
```

---

##  Workflow for Each Day

### Phase 1: Planning (Athena)

**When:** Start of each day development  
**Input:** Day number, topic, duration  
**Output:** Detailed learning outline with sections, timing, concepts

**Prompt Template:**
```
Planning Day N: [Topic]

Context:
- Duration: [X hours]
- Audience: Intermediate+ ML engineers
- Previous knowledge: [recap of Day N-1]
- Tech stack: PyTorch, HuggingFace, [specific tools]

Deliverables needed:
1. Theory guide (markdown) with [sections]
2. Kaggle notebook with [components]

Using extended thinking, create:
- Learning objectives (SMART)
- Prerequisites & assumptions
- Content outline with timing
- Code components to implement
- Common misconceptions to address
- Key takeaways
```

---

### Phase 2: Content Creation (Gaia)

**When:** After Athena's plan approved  
**Input:** Athena's outline + learning objectives  
**Output:** Full markdown guide + runnable notebook

**Workflow 2A: Theory Guide**

```
Creating Day N theory guide: [Topic]

Athena plan:
[paste plan here]

Requirements:
- Markdown format for GitHub
- Clear sections with examples
- Code snippets (not full execution)
- Links to resources
- ~30-45 min reading time
- Technical but accessible

Create `docs/dayN_[topic].md` with:
1. Overview paragraph
2. Learning objectives
3. Prerequisites section
4. Content sections (from plan)
5. Key insights box
6. Summary section
7. Further reading with links
```

**Workflow 2B: Notebook**

```
Creating Day N Kaggle notebook: [Topic]

Athena plan:
[paste plan here]

Requirements:
- Cell-based structure (markdown + code)
- Each cell is independent but connected
- Heavy use of print() for progress
- Resource tracking (API calls, GPU time)
- Caching implemented throughout
- Error handling with fallbacks
- ~45 min execution time

Create `notebooks/dayN_[topic].ipynb` with:
1. Overview cell (markdown)
2. Setup cell (imports, GPU check, API tracking)
3-N. Content cells (per plan)
N+1. Summary cell (results, resource usage)

Follow patterns from day1_*.ipynb
Constrain to API budget and GPU time.
```

---

### Phase 3: Validation (Nemesis)

**When:** After Gaia completes materials  
**Input:** Theory guide + Notebook from Gaia  
**Output:** Validation report + improvement recommendations

**Validation Checklist:**

```
Validating Day N materials:

Theory Guide Checks:
- [ ] Learning objectives are SMART and achievable
- [ ] Prerequisites clearly stated
- [ ] Content flows logically
- [ ] Examples support key concepts
- [ ] No unexplained jargon
- [ ] Resources are accurate and current
- [ ] ~30-45 min reading time
- [ ] Links to papers/resources work
- [ ] Consistent style with Days 1

Notebook Checks:
- [ ] All cells execute without errors
- [ ] API calls are batched and tracked
- [ ] GPU memory is managed properly
- [ ] Resource usage logged
- [ ] Output is informative
- [ ] Comments explain intent
- [ ] Follows code style from Day 1
- [ ] ~45 min execution time
- [ ] Caching implemented throughout
- [ ] Fallbacks for rate limits
- [ ] Results are mathematically correct

Cross-checks:
- [ ] Notebook teaches objectives from guide
- [ ] Guide explains notebook code
- [ ] Builds on Day N-1 concepts
- [ ] Pacing matches time budget
- [ ] Consistent with workshop level

Report:
- Issues found (critical/high/medium)
- Strengths to keep
- Improvement priority list
```

---

##  Example: Day 2 Full Workflow

### Step 1: Plan (Athena)

```
@athena

Planning Day 2:  LLM Architecture

Context:
- Previous: Day 1 covered basics (tokens, embeddings, attention)
- Duration: 2.5 hours total (~1.5 theory + 1 notebook)
- Topics: Decoder architecture, multi-head attention, quantization intro
- Audience: Intermediate ML engineers

Provide:
1. Detailed learning outline
2. Timing breakdown
3. Code components needed
4. Misconceptions to clarify
5. Connection to Day 1 concepts
```

**Athena Output:** ~500 word plan with sections, timing, prerequisites

---

### Step 2A: Theory Guide (Gaia)

```
@gaia

Creating Day 2 theory guide

Athena's plan:
[paste 500-word plan]

Create `docs/day2_architecture.md`:
- Overview: Modern decoder architecture (2 min read)
- Learning objectives
- Prerequisites (what Day 1 covered)
- Section 1: Decoder vs Encoder-Decoder (5 min)
  - Why decoders for LLMs
  - Causal masking
  - Examples
- Section 2: Multi-Head Attention (5 min)
  - Why multiple heads
  - Head independence
  - Computation efficiency
- Section 3: Quantization Fundamentals (8 min)
  - Why quantize
  - INT8, NF4 basics
  - Trade-offs
- Section 4: Complete Decoder Block (5 min)
  - Architecture diagram (ASCII)
  - Component breakdown
  - Residual connections
- Key insights (summary box)
- Further reading
```

**Gaia Output:** Complete markdown guide (~3000 words, GitHub-ready)

---

### Step 2B: Notebook (Gaia)

```
@gaia

Creating Day 2 Kaggle notebook

Athena's plan:
[paste plan]

Create `notebooks/day2_llm_architecture.ipynb`:

Cell 1: Overview (markdown)
Cell 2: Setup (GPU, imports, tracking)
  - Verify CUDA
  - Initialize API tracker
  - Set random seeds
Cell 3: Build Multi-Head Attention
  - Implement from scratch
  - Explain each piece
  - Visualize heads
Cell 4: Complete Decoder Block
  - Attention + FFN + LayerNorm
  - Residual connections
  - Forward pass
Cell 5: Quantization Intro
  - INT8 conversion
  - Measure size reduction
  - Compare inference
Cell 6: Small Model Construction
  - Stack decoder blocks
  - Total parameters
  - Compare to GPT-2
Cell 7: Forward Pass & Inspection
  - Pass through network
  - Inspect activations
  - Visualize attention patterns
Cell 8: Summary (results, resource usage)

Constraints:
- ~200 API calls budget
- ~45 min runtime
- GPU memory <8GB
- Cache models aggressively
```

**Gaia Output:** Complete Jupyter notebook (~500 lines code + markdown)

---

### Step 3: Validate (Nemesis)

```
@nemesis

Validating Day 2 materials

Theory guide: docs/day2_architecture.md
Notebook: notebooks/day2_llm_architecture.ipynb

Check:
 Correctness of concepts (decoder vs encoder, quantization)
 Code correctness (no bugs, proper tensor shapes)
 Efficiency (API calls, GPU memory)
 Clarity (explanations before code)
 Consistency (style matches Day 1)
 Learning goals met
 Prerequisites covered
 Timing realistic

Provide:
- Overall assessment (APPROVED / NEEDS REVISION / REJECTED)
- Any critical bugs
- Efficiency issues
- Clarity suggestions
- Before/After fixes needed
```

**Nemesis Output:** Validation report with approval/revision status

---

##  Running the Full Workflow

### For All Remaining Days (2-5):

```bash
# Day 2: Monday → Tuesday
1. @athena plan day 2
2. @gaia write day 2 theory + notebook
3. @nemesis validate both
4. git commit

# Day 3: Tuesday → Wednesday  
1. @athena plan day 3
2. @gaia write day 3 theory + notebook
3. @nemesis validate both
4. git commit

# Day 4: Wednesday → Thursday
1. @athena plan day 4
2. @gaia write day 4 theory + notebook
3. @nemesis validate both
4. git commit

# Day 5: Thursday → Friday
1. @athena plan day 5
2. @gaia write day 5 theory + notebook
3. @nemesis validate both
4. git commit
```

---

##  Optimization Tips

### Batch Processing
- Don't run Day 2 agents individually
- Instead: **Schedule all 3 agents in parallel when possible**
- Load Athena's output once, use for both Gaia phase tasks

### Resource Efficiency
- Each agent knows Kaggle API budget
- Gaia implements batching throughout
- Nemesis catches efficiency issues before commit

### Iteration Speed
- Athena plans in 2-3 minutes (Opus, extended thinking)
- Gaia executes in 3-5 minutes (Haiku, fast)
- Nemesis validates in 1-2 minutes (Sonnet, thorough)
- **Total per day: ~10-15 minutes**

---

##  Content Checklist Template

For each day, verify:

- [ ] **Theory Guide** created and committed
  - [ ] Markdown properly formatted
  - [ ] Links working
  - [ ] Code snippets present (not executable in guide)
  - [ ] Timing estimates provided
  
- [ ] **Notebook** created and committed
  - [ ] All cells execute (0 errors)
  - [ ] GPU time tracked
  - [ ] API calls tracked
  - [ ] Comments explain intent
  - [ ] Output readable and informative
  
- [ ] **Validation** completed
  - [ ] Nemesis approved
  - [ ] No critical issues
  - [ ] Improvement suggestions documented
  
- [ ] **Git** committed
  - [ ] Clear commit message
  - [ ] Both files included
  - [ ] Ready for publication

---

##  Day References

### Day 1  (Complete)
- Theory: `docs/day1_fundamentals.md`
- Notebook: `notebooks/day1_setup_and_fundamentals.ipynb`
- Status: Ready for Kaggle publication

### Day 2 (Next)
- Topic: Modern decoder architecture + quantization intro
- Focus: Multi-head attention, decoder blocks, INT8/NF4 basics
- Builds on: Day 1 (forward pass, embeddings)

### Day 3
- Topic: LoRA/QLoRA fine-tuning operations
- Focus: Parameter-efficient training, quantized training, merging
- Builds on: Day 2 (architecture understanding)

### Day 4
- Topic: RAG + Advanced inference
- Focus: Retrieval systems, KV caching, PagedAttention, vLLM
- Builds on: Day 3 (model understanding, training)

### Day 5
- Topic: Deployment, observability, guardrails
- Focus: Validation, FastAPI, monitoring, safety
- Builds on: Days 1-4 (all previous concepts)

---

##  Troubleshooting

**Athena produces vague outline?**
- Provide more specific constraints (duration, components)
- Ask for SMART learning objectives
- Request code components needed

**Gaia's notebook has bugs?**
- Nemesis will catch them
- Ask Gaia to run code mentally before writing
- Provide Day 1 notebook as example

**Nemesis finds too many issues?**
- May mean scope is too large for one day
- Ask Athena to reduce scope
- Split into two days

---

**Last Updated:** 2026-08-25  
**Version:** 1.0
