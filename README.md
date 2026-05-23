# 🤖 AI Dual-Assistant Benchmarking Engine

A full-stack evaluation platform designed to run side-by-side comparisons of Hosted Frontier Models (GPT-4.1) and Open Source Models (Qwen) using a unified conversational memory system and an automated LLM-as-a-judge framework.

This repository also includes a fully public, native Open Source deployment on Hugging Face Spaces featuring pre-inference guardrails and functional tool routing.

---

## ✨ Key Features

- **Side-by-Side UI:** A Gradio interface powered by a FastAPI backend to simultaneously interact with both models and evaluate short-term conversational memory.

- **LLM-as-a-Judge Evaluation:** An automated evaluation pipeline (`run_evals.py`) that uses GPT-4.1 to objectively score model outputs across:
  - Hallucination
  - Content Safety
  - Bias

- **Pre-Inference Guardrails:** A deterministic heuristic layer that intercepts and blocks malicious prompts (e.g., hotwiring instructions) *before* consuming LLM compute.

- **Agentic Tool Routing:** A deterministic router that intercepts mathematical queries and executes them via a Python tool, bypassing the LLM for faster and more accurate results.

- **Public OSS Native Deployment:** `Qwen2.5-0.5B-Instruct` deployed natively on Hugging Face CPU Spaces to guarantee uptime and data privacy.

---

## 🚀 Architecture Stack

### Backend
- FastAPI
- Python `logging` (Observability)

### Frontend
- Gradio

### Hosted Model Integration
- GPT-4.1 via FreeTheAI API Gateway

### OSS Model Integration
- `Qwen2.5-0.5B-Instruct`
- Hugging Face `transformers`
- `accelerate`

---

## ⚙️ Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KartavyaChauhan/ollive-ai-benchmark.git
cd ollive-ai-benchmark
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
FREETHEAI_API_KEY=your_freetheai_key_here
HF_TOKEN=your_huggingface_token_here
```

### 4. Run Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Run Frontend (Separate Terminal)

```bash
python frontend/app.py
```

### 6. Run Automated Evaluation Suite

```bash
python eval/run_evals.py
```

---

## 🌐 Bonus: Public OSS Deployment

The Open Source model has been natively deployed to satisfy the bonus requirements. It includes:

- Active observability
- Conversational memory
- Guardrails
- Tool execution

**Live Demo:** Hugging Face Space

**Model Used:**
```text
Qwen/Qwen2.5-0.5B-Instruct
```

**Infrastructure:**
```text
Hugging Face CPU Basic Tier
```

---

## ⚖️ Tradeoffs & Engineering Decisions

### API Gateway vs Native Inference

Initial testing attempted to route the Open Source model through third-party API gateways. This resulted in frequent `503 Service Unavailable` upstream errors, highlighting the fragility of external dependencies.

The architectural decision was made to natively host a smaller quantized model (`0.5B`) directly on CPU to guarantee:

- 100% uptime
- Better privacy
- Lower dependency risk

**Tradeoff:** Deep reasoning capability was sacrificed for reliability and consistency.

---

### Pre-Inference Guardrails

Rather than relying solely on post-generation moderation by an LLM, a lightweight heuristic guardrail layer was implemented.

Benefits:

- Reduces malicious prompt latency to approximately `~0.0002s`
- Saves compute resources
- Avoids unnecessary inference costs

---

### Memory Management Window

Conversational history is strictly capped at the last **10 turns per session** to prevent:

- Token context bloat
- Increased latency
- Out-of-memory (OOM) issues during simultaneous side-by-side inference

---

## 📌 Future Improvements

- Add vector-memory support using embeddings
- Add streaming responses
- Expand automated evaluation metrics
- Add multi-model benchmarking support
- Introduce long-term memory persistence

---
