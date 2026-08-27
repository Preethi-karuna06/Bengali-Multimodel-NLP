# 🇧🇩 Bengali Multimodal NLP System

**A unified summarization, visual question answering, extractive QA, and speech recognition pipeline for the Bengali language.**

## Overview

Bengali is spoken by **230M+ people** but remains a low-resource language for NLP. This project builds a **single, unified pipeline** that accepts Bengali input in four different modalities — plain text, scanned documents/images, images with questions, and speech — and converts each into a common Bengali text representation for downstream summarization and question answering.

The system is organized as four independent-but-connected modules, all exposed through one [Gradio](https://gradio.app/) web interface:

| Module | Task | Best Model | Headline Result |
|---|---|---|---|
| **A. OCR + Summarization** | Extract text from scans/PDFs/DOCX and summarize | CNN Sentence Scorer | BLEU 1.86 · chrF++ 30.61 |
| **B. Visual QA (VQA)** | Answer natural-language questions about an image | MCRAN | 63.75% overall test accuracy |
| **C. Extractive QA** | Extract an answer span from a Bengali passage | BanglaBERT (fine-tuned) | 54.0% EM · 67.9% F1 |
| **D. Speech Recognition (ASR)** | Transcribe spoken Bengali to text | Shrutimala Bangla (Wav2Vec-BERT + CTC) | Competitive WER |

code: [Open in Google Colab](https://colab.research.google.com/drive/1z9cXqot8m6TCcJoEc1_VWrwV-e5WgvsG?usp=sharing)

---

## Table of Contents

- [Architecture](#architecture)
- [Modules](#modules)
  - [A. Text Summarization & OCR](#a-text-summarization--ocr)
  - [B. Visual Question Answering](#b-visual-question-answering-vqa)
  - [C. Extractive Question Answering](#c-extractive-question-answering)
  - [D. Automatic Speech Recognition](#d-automatic-speech-recognition-asr)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Limitations & Future Work](#limitations--future-work)
- [Team](#team)
- [License](#license)

---

## Code

All training and the unified Gradio app live in one Colab notebook (no local setup required):

**👉 [Open the notebook in Google Colab](https://colab.research.google.com/drive/1z9cXqot8m6TCcJoEc1_VWrwV-e5WgvsG?usp=sharing)**

Run it top to bottom on a T4 GPU runtime (`Runtime → Change runtime type → T4 GPU`) to reproduce training, or use the "load from Drive" cells to skip straight to launching the app with saved checkpoints.

---

## Architecture

```
                     ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
                     │    Text    │  │  Document  │  │  Image + Q │  │   Speech   │
                     │Plain Bengali│  │Scan/PDF/DOCX│ │ Visual query│ │Bengali audio│
                     └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
                            └───────────────┬┴────────────────┬─────────────┘
                                             ▼
                                   ┌───────────────────┐
                                   │   Modality Router  │
                                   │  Detects input type │
                                   └──────────┬──────────┘
              ┌──────────────────┬────────────┴────────────┬──────────────────┐
              ▼                  ▼                          ▼                  ▼
     ┌─────────────────┐ ┌──────────────────┐     ┌──────────────────┐ ┌──────────────────┐
     │  Module A: OCR + │ │  Module B: Visual │     │ Module C:        │ │ Module D: Speech │
     │  Summarization   │ │  QA (VQA)         │     │ Extractive QA     │ │ Recognition (ASR)│
     ├─────────────────┤ ├──────────────────┤     ├──────────────────┤ ├──────────────────┤
     │ Tesseract/EasyOCR│ │ ResNet-50 image   │     │ NFC normalize +   │ │ Wav2Vec-BERT      │
     │ text extraction  │ │ encoder (768-d)   │     │ tokenize           │ │ self-supervised   │
     │ ↓                │ │ ↓                 │     │ ↓                  │ │ encoder            │
     │ TF-IDF/TextRank/ │ │ MCRAN fusion:     │     │ Span prediction:   │ │ ↓                  │
     │ BiLSTM/CNN/      │ │ ICAR + TCAR +     │     │ BiLSTM / BanglaBERT│ │ CTC decoder →      │
     │ BanglaBERT/mT5   │ │ MMAR attention    │     │                    │ │ Bengali text        │
     └────────┬─────────┘ └─────────┬────────┘     └─────────┬─────────┘ └─────────┬─────────┘
              └──────────────────┬──┴─────────────────────────┴──────────────────┘
                                  ▼
                        ┌───────────────────┐
                        │  Bengali Text Layer │
                        │(unified representation)│
                        └──────────┬──────────┘
                                   ▼
                        ┌───────────────────┐
                        │   NLP Processing    │
                        │Summarize·Answer·Transcribe│
                        └──────────┬──────────┘
                                   ▼
                        ┌───────────────────┐
                        │  Gradio Interface   │
                        │Summary·Answer·Transcript│
                        └───────────────────┘
```

Each module operates independently but shares a common Bengali text preprocessing pipeline (NFC normalization, Bengali Unicode block filtering `U+0980–U+09FF`, whitespace/danda handling), so outputs from OCR, VQA, and ASR all funnel into the same summarization/QA logic.

---

## Modules

### A. Text Summarization & OCR

- **Dataset:** [XL-Sum Bengali](https://huggingface.co/datasets/csebuetnlp/xlsum) — 8,102 BBC Bengali article-summary pairs (7,200 train / 450 val / 452 test), ~8.8:1 compression ratio.
- **OCR pipeline:** Tesseract (`ben.traineddata`, LSTM engine) for scans, EasyOCR for degraded images, PyMuPDF for native PDFs, python-docx for Word files.
- **Models compared:**
  - Extractive: TF-IDF, TextRank, BiLSTM+Attention, CNN Sentence Scorer, BanglaBERT
  - Abstractive: mT5-small (fine-tuned)

| Model | BLEU | chrF++ | BERTScore-F1 | Type |
|---|---|---|---|---|
| TF-IDF | 0.82 | 27.41 | 0.836 | Extractive |
| TextRank | 1.38 | 30.39 | 0.846 | Extractive |
| BiLSTM+Attention | 1.74 | 30.53 | 0.845 | Extractive |
| **CNN Sentence Scorer** | **1.86** | **30.61** | 0.845 | Extractive |
| BanglaBERT | 1.46 | 30.01 | 0.844 | Extractive |
| mT5-small | 1.46 | 30.30 | 0.725 | Abstractive |

### B. Visual Question Answering (VQA)

- **Dataset:** BVQA — ~17,800 Bengali QA pairs generated via an LLM-based pipeline over ~3,500 everyday-scene images (9,557 train / 720 val / 774 test), 140-class answer vocabulary.
- **Model — MCRAN** (Multimodal Cross-modal Representation Attention Network):
  - Image encoder: ResNet-50 → 2048-d → projected to 768-d
  - Text encoder: multilingual BERT (mBERT)
  - Three cross-modal attention branches: **ICAR** (image-weighted), **TCAR** (text-weighted), **MMAR** (token-level), combined via a learned gate → softmax classifier
  - Trained 15 epochs, Adam, lr=1e-4, batch size 32

| Question Category | Test Accuracy | Test Samples |
|---|---|---|
| Yes/No | 80.6% | 77 |
| Counting | 57.0% | 313 |
| Open-Ended | 66.0% | 384 |
| **Overall** | **63.75%** | 774 |

### C. Extractive Question Answering

- **Dataset:** [TyDi QA](https://ai.google.com/research/tydiqa) — Bengali partition (2,390 train / 113 val), Wikipedia-sourced.
- **Preprocessing:** NFC normalization, Bengali Unicode filtering, danda (`।`) retained as sentence boundary, custom 300-d Word2Vec embeddings (35,572-word vocab).
- **Models compared:** BiLSTM, BiLSTM+Attention, CNN+BiLSTM, XLM-RoBERTa (fine-tuned), BanglaBERT (fine-tuned).

| Model | Exact Match | F1 | Type |
|---|---|---|---|
| BiLSTM (baseline) | ~2.5% | ~10% | Custom |
| BiLSTM+Attention | 5.31% | 14.79% | Custom |
| CNN+BiLSTM | 6.2% | 16.8% | Custom |
| XLM-RoBERTa (FT) | 49.6% | 61.0% | Pre-trained |
| **BanglaBERT (FT)** | **54.0%** | **67.9%** | Pre-trained |

### D. Automatic Speech Recognition (ASR)

- **Model:** Shrutimala Bangla ASR — Wav2Vec-BERT 2.0 encoder + CTC decoding.
- **Pipeline:** 16kHz mono waveform → self-supervised speech features → transformer encoder → CTC output over Bengali character vocabulary → beam-search decoding.
- **Training:** two-stage — self-supervised pre-training on unlabeled Bengali audio, then supervised fine-tuning on transcribed pairs.
- Feeds directly into the VQA and summarization modules for speech-driven queries.

---

## Results Summary

| Module | Best Model | Key Metric | Result |
|---|---|---|---|
| Summarization | CNN Sentence Scorer | BLEU / chrF++ | 1.86 / 30.61 |
| VQA | MCRAN | Overall Accuracy | 63.75% |
| Extractive QA | BanglaBERT (FT) | Exact Match / F1 | 54.0% / 67.9% |
| ASR | Shrutimala Bangla | WER (relative) | Competitive |

---

## Repository Structure


```

> **Note:** Notebooks, trained model checkpoints, and datasets are not hosted in this repository. All code runs directly from the [Colab notebook](https://colab.research.google.com/drive/1z9cXqot8m6TCcJoEc1_VWrwV-e5WgvsG?usp=sharing) linked above.

---

## Getting Started

### Prerequisites

- A Google account (to run the Colab notebook)
- A CUDA-capable GPU runtime — the notebook was developed/run on Google Colab with a T4 GPU
- `tesseract-ocr` with the Bengali language pack (`ben.traineddata`) for the OCR module — installed automatically in the notebook's setup cell

### Running the project

1. Open the **[Colab notebook](https://colab.research.google.com/drive/1z9cXqot8m6TCcJoEc1_VWrwV-e5WgvsG?usp=sharing)**.
2. Set the runtime to GPU: `Runtime → Change runtime type → T4 GPU`.
3. Run the install/setup cells at the top to pull in dependencies.
4. Follow the in-notebook instructions to either:
   - train each module from scratch on its dataset (XL-Sum Bengali, BVQA, TyDi QA Bengali), or
   - mount Google Drive and load the saved checkpoints to skip straight to the app.
5. Run the final cells to launch the Gradio interface — a public link is generated for the session.

For local/offline development against the same code, `requirements.txt` in this repo lists all the dependencies used.

---

## Usage

Once models are trained (or loaded from Drive), the **Gradio interface** exposes four tabs:

- **Summarization** — paste text, or upload a text file / PDF / scanned image (OCR runs automatically); pick a model and get a Bengali summary.
- **VQA** — upload an image and type a Bengali question; get the top-5 predicted answers with confidence scores.
- **QA** — paste a Bengali passage and a question; get the extracted answer span highlighted in context.
- **ASR** — upload audio or record from the microphone; get the Bengali transcript, which can then be summarized or queried directly.

Launching the interface (from notebook 04) generates a public Gradio link for the session, so the whole system is reachable from a browser with no local setup.

---

## Key Findings

- **Language-specific pre-training wins:** BanglaBERT consistently outperforms multilingual models (mBERT, XLM-RoBERTa) across summarization and QA.
- **Custom architectures are competitive in extreme low-resource regimes:** with only 2,390 QA training examples, BiLSTM+Attention rivals large transformers that don't get enough fine-tuning signal.
- **Cross-modal attention beats naive fusion:** MCRAN's three parallel attention branches (ICAR/TCAR/MMAR) outperform simple feature concatenation for VQA.
- **Low absolute summarization scores reflect task difficulty, not failure:** matching professional journalist summaries is inherently hard at this data scale (max BLEU 1.86 across all models).

---

## Limitations & Future Work

- All four tasks are constrained by dataset scarcity — TyDi QA Bengali (2,390 examples) and BVQA (17,800 pairs) are small relative to English-language benchmarks.
- Abstractive summarization (mT5) achieves near-zero lexical scores; larger Bengali summarization corpora are needed for stable encoder-decoder fine-tuning.
- OCR degrades on handwritten Bengali and historical/non-standard typefaces.
- ASR has not been systematically evaluated on dialectal variation (e.g., Sylheti, Chittagonian Bengali).

Planned improvements: larger/more diverse Bengali VQA data to reduce the MCRAN train/val accuracy gap (~85% vs ~63%), dialect-robust ASR fine-tuning, and exploring larger mT5 checkpoints for abstractive summarization.

---



## License

This project is licensed under the [MIT License](LICENSE).
