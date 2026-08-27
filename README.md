#  LLMOps Workshop: 5-Day Intensive

**A hands-on, code-driven exploration of Large Language Model operations** — from fundamentals through production deployment.

---

##  Quick Start

### What You'll Learn

| Day | Topic | Focus |
|-----|-------|-------|
| **Monday**  | Setup + Peek Inside | GPU setup, tokenization, embeddings, forward pass |
| **Tuesday**  | LLM Architecture | Decoder blocks, attention, quantization fundamentals |
| **Wednesday**  | Fine-Tuning Ops | LoRA, QLoRA, training, merging |
| **Thursday**  | Retrieval + Inference | RAG, KV caching, vLLM, continuous batching |
| **Friday**  | Deploy & Observability | Validation, deployment, metrics, guardrails |

### Getting Started

1. **Prerequisites:**
   - Python 3.10+, PyTorch, HuggingFace Transformers
   - Kaggle account with GPU quota
   - Basic ML/transformer knowledge

2. **For Each Day:**
   - Read the theory guide: `docs/dayN_*.md`
   - Run the notebook: `notebooks/dayN_*.ipynb`
   - Expected time: ~2-2.5 hours per day

3. **Resource Limits:**
   - 1000 Kaggle API requests/day
   - 30 hours GPU usage total (~6 hrs/day)

---

##  Repository Structure

```
LLMOps/
├── README.md                          # This file
├── SYLLABUS.md                        # Complete course overview
├── KAGGLE_GUIDELINES.md               # Best practices for Kaggle
│
├── docs/                              # Theory guides (markdown)
│   ├── day1_fundamentals.md           #  Ready
│   ├── day2_architecture.md           # → Forthcoming
│   ├── day3_finetuning.md             # → Forthcoming
│   ├── day4_rag_inference.md          # → Forthcoming
│   └── day5_deploy.md                 # → Forthcoming
│
├── notebooks/                         # Kaggle notebooks
│   ├── day1_setup_and_fundamentals.ipynb  #  Ready
│   ├── day2_llm_architecture.ipynb        # → Forthcoming
│   ├── day3_finetuning_ops.ipynb          # → Forthcoming
│   ├── day4_rag_and_inference.ipynb       # → Forthcoming
│   └── day5_deploy_observability.ipynb    # → Forthcoming
│
├── .claude/                           # Claude agent configurations
│   ├── agents/
│   │   ├── athena.md                  # Curriculum planner (Opus)
│   │   ├── gaia.md                    # Notebook executor (Haiku)
│   │   └── nemesis.md                 # Quality validator (Sonnet)
│   └── settings.json                  # Claude Code configuration
│
├── assets/                            # Images, diagrams, supplementary
└── .env                               # API keys (gitignored)
```

---

##  Day 1:  Setup + Peek Inside  READY

### What's Included

**Theory:** [`docs/day1_fundamentals.md`](docs/day1_fundamentals.md)
- Kaggle GPU setup & verification
- Tokenization fundamentals (BPE, subword splitting)
- Embedding spaces and semantic geometry
- Positional encoding
- Self-attention mechanism
- Forward pass breakdown

**Notebook:** [`notebooks/day1_setup_and_fundamentals.ipynb`](notebooks/day1_setup_and_fundamentals.ipynb)
-  Environment setup & GPU verification
-  Tokenization exploration
-  Embedding space analysis
-  Positional encoding implementation
-  Self-attention mechanism
-  Complete end-to-end forward pass
-  Resource usage tracking

**Estimated GPU Time:** ~30 min  
**Estimated API Calls:** ~50 (out of 1000/day)

---

##  Resources

### Must-Read Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Day 3
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Day 4

### Recommended References
- [HuggingFace Transformers Course](https://huggingface.co/course/en/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [FastAPI for ML Deployment](https://fastapi.tiangolo.com/)

### Tools We Use
- **PyTorch** — Neural networks
- **HuggingFace Transformers** — Models & tokenizers
- **vLLM** — Efficient inference
- **QLoRA** — Parameter-efficient fine-tuning
- **FAISS/ChromaDB** — Vector databases (Day 4)

---

##  Roadmap

###  Complete
- [x] Day 1 theory guide & notebook
- [x] Course syllabus
- [x] Kaggle guidelines
- [x] Agent definitions

###  In Progress
- [ ] Days 2-5 theory guides
- [ ] Days 2-5 notebooks
- [ ] GitHub markdown theory

###  Planned
- [ ] Kaggle notebook publication
- [ ] Community forum setup
- [ ] Example outputs/benchmarks
- [ ] Troubleshooting guide

---

##  Tips for Success

### Running Notebooks on Kaggle

1. **Activate GPU:**
   - Notebook Settings → Accelerator → GPU
   - Recommended: P100 or higher

2. **Enable Internet:**
   - Settings → Internet → On (needed for model downloads)

3. **Save Checkpoints:**
   - Use `/kaggle/working/` for outputs
   - Use `/kaggle/input/cache/` for persistent storage

### Managing Resources

- **API Calls:** Monitor via `KAGGLE_GUIDELINES.md`
- **GPU Memory:** Use `torch.cuda.memory_summary()` to track
- **Batch Size:** Start small, scale up as comfortable
- **Caching:** Download once, use many times

### When You Get Stuck

1. Check KAGGLE_GUIDELINES.md for rate limit solutions
2. Reduce batch size if OOM
3. Use CPU fallback for non-critical tasks
4. Check notebook runtime limits (max 9 hours on Kaggle)

---

##  Contributing

Found an issue or have suggestions?

- **Bug reports:** Open an issue with reproducible steps
- **Improvements:** PRs welcome — follow the agent definitions
- **Questions:** Check existing issues first

---

##  Support

- **Kaggle Issues:** See KAGGLE_GUIDELINES.md troubleshooting
- **General Questions:** Check day-specific theory guides
- **Technical:** Review notebook comments and error messages

---

##  License

MIT License — Free to use, modify, and share.

---

##  Next Steps

**Ready to begin?**

1. Start with [**Day 1 Theory**](docs/day1_fundamentals.md) (read: 30 min)
2. Run [**Day 1 Notebook**](notebooks/day1_setup_and_fundamentals.ipynb) (execute: 45 min)
3. Move to Day 2 when ready!

---

**Last Updated:** 2026-08-25  
**Status:** Day 1 Complete   
**Next:** Day 2 (Architecture)

Enjoy the workshop! 
