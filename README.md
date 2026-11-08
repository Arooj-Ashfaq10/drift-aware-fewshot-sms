# Drift-Aware Few-Shot Example Selection and Refresh for LLM-Based SMS Spam Detection

Code for the CS5001 Research Methodology assignment (FAST-NUCES Islamabad).

**Base paper reproduced:** SpaLLM-Guard (Salman et al., 2025, arXiv:2501.04985) - zero-shot / few-shot protocol.
**Own component:** recent-vs-random example selection, refresh intervals, and a label-free drift-triggered refresh (DTR).

## Layout
| File | Purpose |
|---|---|
| `src/common.py` | Prompt template (follows the base paper's prompts) |
| `src/data.py` | UCI loader, simulated-drift stream builder, real-timestamp loader |
| `src/strategies.py` | Example selection, drift monitor, threshold calibration |
| `src/llm.py` | HF 4-bit LLM scorer, disk cache, mock backend for tests |
| `run_reproduction.py` | Base-paper reproduction (UCI) |
| `run_drift.py` | Main experiments (stable vs drift, recent vs random, refresh intervals, DTR) |
| `make_figures.py`, `make_workflow_diagram.py` | Figures |
| `tests/smoke_test.py` | Pipeline test with a mock model on synthetic data (no scientific results) |
| `colab_run.ipynb` | Run everything on a free Colab T4 |

## Run
```bash
pip install -r requirements.txt
python run_reproduction.py --model mistralai/Mistral-7B-Instruct-v0.3
python run_drift.py --mode simulated --model Qwen/Qwen2.5-7B-Instruct --max_eval 300
python run_drift.py --mode real --real_csv text-messages.csv --model Qwen/Qwen2.5-7B-Instruct   # optional
python make_figures.py --dir results/simulated
```
All LLM calls are cached on disk, so a disconnected Colab session resumes where it stopped.

## Data
* UCI SMS Spam Collection (Almeida et al., 2011) - loaded from Hugging Face `ucirvine/sms_spam` or a local `SMSSpamCollection` file.
* Simulated drift: spam messages are clustered by content and each period draws from a moving mixture of clusters.
* Real timestamps (optional): Kaggle "PH Spam + Marketing SMS (w/ timestamps)" (check its licence before redistributing; the CSV is **not** included here).

## Push to GitHub
```bash
git init && git add . && git commit -m "Drift-aware few-shot SMS spam" 
git branch -M main && git remote add origin https://github.com/<user>/<repo>.git && git push -u origin main
```
