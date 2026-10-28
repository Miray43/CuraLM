# CuraLM: AI-Driven Personalized Medicine

**CuraLM** is an AI clinical assistant that helps doctors with **initial diagnosis** and **treatment plans**, and explains each recommendation. It fine-tunes the biomedical LLM **BioMistral-7B** with **QLoRA** and grounds its answers using a **hybrid Retrieval-Augmented Generation (RAG)** pipeline.

> 🎓 Graduation project, Computer Science Department, Jamoum University College, **Umm Al-Qura University**. Team of 4.

---

## ✨ Highlights

- **Fine-tuned BioMistral-7B-DARE** with 4-bit quantization (BitsAndBytes) and **QLoRA / PEFT** (LoRA r = 32, α = 64), so it trains within Google Colab GPU limits.
- **Hybrid RAG**: medical records are embedded with `all-MiniLM-L6-v2` and indexed in **FAISS**. Retrieval is applied only to the task it helps (drug details) and skipped where it hurt quality (top-drug recommendation).
- **Two learning tasks** built from a Kaggle drugs and medical-conditions dataset:
  - `drug_details`: predict a drug's generic name and side effects.
  - `condition_to_top_drugs`: recommend top medications by effectiveness, rating and reviews.
- **Doctor-facing web app**: Flask backend and an HTML / Tailwind CSS / JavaScript UI (dashboard, chat-based new diagnosis, treatment-plan generator, diagnosis history).
- **Explainability**: every diagnosis or treatment suggestion comes with a structured explanation.

## 📊 Results (final hybrid strategy, from the project report)

| Metric | Score |
|---|---|
| Overall five-metric accuracy | **92.65%** |
| Generic-name exact match | **95.68%** |
| Drug-hit accuracy | **100%** |
| Condition-answer F1 | **87.98%** |
| BERT answer similarity | **91.37%** |

Side-effect prediction was the hardest task, because side effects are described with inconsistent wording and overlapping terms.

### From the notebook in this repo

The notebook is one of the experiment runs (run 17), so its numbers are slightly different from the final report.

| Training progress | Model report card |
|---|---|
| ![Training curves](docs/figures/fig04.png) | ![Report card](docs/figures/fig09.png) |

More plots (EDA, token lengths, per-task scores, error analysis) are in [`docs/figures`](docs/figures).

## 🏗️ Architecture

```
Doctor (Web UI) ──► Flask API ──► NLP preprocessing
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
          FAISS retrieval (RAG)              BioMistral-7B + QLoRA adapter
          (drug_details task)                (generation)
                    └─────────────────┬──────────────────┘
                                      ▼
                     Diagnosis / treatment plan + explanation
```

## 📁 Repository Structure

```
├── notebook/
│   └── CuraLM_BioMistral_QLoRA_RAG.ipynb   # data prep → QLoRA training → RAG → evaluation
├── ui/
│   ├── dashboard.html                      # doctor dashboard & history
│   └── new-diagnosis.html                  # chat-based diagnosis page
└── docs/figures/                           # plots exported from the notebook
```

## 🚀 Running the Notebook

1. Open `notebook/CuraLM_BioMistral_QLoRA_RAG.ipynb` in **Google Colab** and use a GPU runtime (T4 or better).
2. Run the cells in order. The notebook installs its own dependencies (`transformers`, `peft`, `bitsandbytes`, `faiss-cpu`, `sentence-transformers`, `bert-score`, …), downloads the dataset with `kagglehub`, then trains and evaluates the model.

To view the UI, open the files in `ui/` in a browser.

## 🛠️ Tech Stack

Python · PyTorch · Hugging Face Transformers · PEFT / QLoRA · BitsAndBytes · FAISS · Sentence-Transformers · BERTScore · Pandas · Matplotlib / Seaborn · Flask · HTML · Tailwind CSS · JavaScript

## ⚠️ Disclaimer

CuraLM is an academic prototype meant to **support** clinicians. It is not a medical device and must not replace professional medical judgment.

## 👩‍💻 Author

**Mira Alharbi**, developed with the CuraLM graduation team at Umm Al-Qura University.
