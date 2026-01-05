# FairHire Auditor

> The open source bias audit platform for recruitment AI

Companies use AI to screen candidates but can't explain rejections. Existing bias detection tools (IBM AIF360, Microsoft Fairlearn) are libraries for data scientists, not platforms for HR teams.

**FairHire Auditor** fills this gap: a multi-agent bias detection platform that wraps proven tools (AIF360, LIME, SHAP) in an intelligent system. Upload hiring data, get candidate explanations, track bias over time, prove EU AI Act compliance all self-hosted.

## Status

**Current Phase:** Architecture complete, starting MVP  


## Features to be added

- **Multi-agent system**: Specialized agents for data bias, model bias, explanations, audits
- **Candidate facing explanations**: LIME/SHAP translated to plain English ("You were rejected because...")
- **Bias trend tracking**: See if fairness improves over time (Q1: 15% disparity → Q3: 3%)
- **advanced RAG**: Hybrid search (BM25 + vector) finds relevant fairness regulations
- **agentic memory**: Tracks bias patterns across audits, learns from past corrections



## Architecture

![System Design](docs/system-Design.png)

**Core Design ideas:**
- **Multi-agent orchestrator** (LangGraph) routes tasks to specialized agents
- **data bias agent** runs statistical tests (Chi-square, KS, Z-test via AIF360)
- **model bias agent** computes fairness metrics (demographic parity, equal opportunity via Fairlearn)
- **explainability agent** generates LIME/SHAP explanations in plain English
- **audit trail agent** logs every decision for compliance (EU AI Act 5-year requirement)
- **report generator** synthesizes findings into technical/executive reports

**Advanced RAG:**
- Hybrid search (BM25 + vector) retrieves fairness regulations
- Cross-encoder reranker scores relevance
- Knowledge base: EU AI Act, EEOC guidelines, case studies

**Agentic Memory (A-MEM):**
- Tier 1 (RAM): Current audit context
- Tier 2 (Redis): Recent audits (hot recall)
- Tier 3 (PostgreSQL + ChromaDB): Historical archive (semantic search)
- Context compilation: Summarizes 12 audits into 1 paragraph when needed



See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete C4 Level 3 diagram.

## Roadmap

### Phase 0: Design ✓
- [x] C4 Level 3 architecture (multi-agent + RAG + memory)


### Phase 1: MVP (4 weeks)
**Goal:** Prove multi-agent bias detection works

- [ ] Multi-agent orchestrator (LangGraph)
- [ ] Data Bias Agent (Chi-square, Z-test via AIF360)
- [ ] Model Bias Agent (demographic parity only via Fairlearn)
- [ ] Basic memory (Redis only, no PostgreSQL archival yet)
- [ ] Explainability Agent (LIME only, skip SHAP)
- [ ] Report Generator (technical report only)
- [ ] REST API (FastAPI)
- [ ] Simple dashboard (Streamlit)

**Not in MVP:** Audit Trail Agent, counterfactuals, trend analysis, WebSocket updates

### Phase 2: Production 
- [ ] PostgreSQL archival (Tier 3 memory)
- [ ] ChromaDB vector search (semantic audit retrieval)
- [ ] SHAP explanations (in addition to LIME)
- [ ] Audit Trail Agent (compliance logs)
- [ ] Trend analysis (track bias over time)
- [ ] React dashboard (replace Streamlit)
- [ ] WebSocket (real-time progress updates)
- [ ] Celery workers (async processing)

### Phase 3: Enterprise 
- [ ] Counterfactual explanations ("If you had X, you'd be accepted")
- [ ] Multi-tenant (isolate company data)
- [ ] Horizontal scaling (PostgreSQL read replicas, Redis cluster)
- [ ] Monitoring (Prometheus + Grafana)
- [ ] Advanced RAG (cross-encoder reranker, query augmentation)
- [ ] PDF reports with charts

## Why Multi-Agent Architecture?

Bias detection isn't one task it's multiple specialized tasks that need coordination.

**Single-agent approach (fails):**
```
Upload data → One LLM tries to do everything → Often hallucinates metrics → Not auditable
```


Each agent has ONE job, uses proven tools (AIF360, LIME), logs decisions. The orchestrator ensures they work together without hallucinating or skipping steps.


MIT - see [LICENSE](LICENSE)

Free to use commercially, modify, and distribute. Just keep the license notice.

---

**"Bias detection shouldn't require a PhD in fairness metrics."**


