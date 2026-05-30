# Formal Language Bias in Large Language Models (LLMs)

An empirical investigation into register bias within modern instruction-tuned language models. This project establishes a dual-measurement evaluation framework (lexical and learned neural layers) to quantify how severely assistant models over-index on elevated, formal prose relative to human baselines, followed by a low-rank adaptation (LoRA) mitigation pipeline.

Developed by **Oliver Krisetya** & **Warren Nguyen** for *CSCI 6515 — Natural Language Understanding* (April 2026) at The George Washington University.

## Executive Overview

Modern Large Language Models often manifest a systematic stylistic preference toward elevated, formal registers rather than matching ordinary conversational language—a phenomenon known as **The "Delve" Problem**. When instruction-tuned assistants consistently default to complex, document-style prose during interactive tasks, they create a distinct **Conversational Register Mismatch**. This structural padding alters text naturalness and can induce negative downstream impacts in human-centric subdomains:

```
              ┌──────────────────────────────────────────┐
              │        Instruction-Tuned LLM Output      │
              │ "A rigorous critique reveals paradigm..."│
              └────────────────────┬─────────────────────┘
                                   │
                     [ Register Mismatch Friction ]
                                   │
                                   ▼
    ┌──────────────────────────────┼──────────────────────────────┐
    │                              │                              │
    ▼                              ▼                              ▼
[ Education ]                [ Mental Health ]            [ Accessibility ]
Overly formal tutoring       Clinical, sterile tone       High literacy barriers
alienates core learners.     reduces patient alignment.   stifle comprehension.
```

### Core Research Questions

1. **Do LLM outputs exhibit a higher formal register** than human conversational corpora across standard benchmarks?
2. **Can lexical and learned neural metrics isolate** which specific model architectures show the strongest formality tendencies?
3. **Does register bias originate** within raw pretraining corpora or from subsequent RLHF/DPO instruction fine-tuning?
4. **Can parameter-efficient fine-tuning (PEFT)** on conversational data reduce this bias without destroying base fluency and general performance?

---

## Methodology & Measurement Framework

To analyze orthogonal dimensions of register shift, this project utilizes a **Dual-Measurement Framework** that decouples localized surface choice from deep contextual syntax.

```
                  ┌──────────────────────────────┐
                  │    Input Response Corpus     │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│     Lexical Pipeline (FR)     │         │    Neural Classifier Layer    │
├───────────────────────────────┤         ├───────────────────────────────┤
│ • Strict String Tokenization  │         │ • RoBERTa-base-formality      │
│ • Multi-word Phrase Tracking  │         │ • Contextual Layer Embeddings │
│ • Explicit Tier Hit Ratios    │         │ • Probability Softmax Score   │
└──────────────┬────────────────┘         └──────────────┬────────────────┘
               ▼                                         ▼
      [ Surface-Level Count ]                 [ Broad Syntactic Vibe ]
```

### 1. Formality Rate (FR) Lexicon Baseline

The `Formality Rate` computes explicit surface-level choices via a non-ML, deterministic frequency metric normalized per 1,000 tokens:

$$FR = \frac{\text{Formal Lexical Hits}}{\text{Total Tokens}} \times 1000$$

The engine tracks formal tokens across a prioritized multi-word structure followed by a tiered dictionary:

* **Tier 1 — Classic LLM-isms (`TIER1_WORDS`):** High-frequency buzzwords favored by modern models (*delve*, *utilize*, *leverage*, *robust*, *comprehensive*, *intricate*, *elucidate*).
* **Tier 2 — Broader Formal Vocabulary (`TIER2_WORDS`):** Traditional academic, bureaucratic, and legal markers (*henceforth*, *aforementioned*, *stipulate*, *ascertain*, *corroborate*, *paradigm*, *methodology*).

### 2. Learned Sentence-Level Classifier

To track structural patterns beyond explicit dictionary hits, we utilize a neural **RoBERTa Register Classifier** (`s-nlp/roberta-base-formality-ranker`). Pretrained on the **GYAFC (Grammarly Yahoo Answers Formality Corpus)** benchmark and fine-tuned using continuous **Pavlick Formality Scores**, this model extracts dense sentence representations to output a true probability distribution ($P(\text{Formal})$) across conversational windows.

## Evaluation Data Architecture

The evaluation pipeline unifies cross-family model distributions alongside multi-genre human-generated baseline corpora:

### Human Conversational Baselines
* **Reddit Corpus:** Unstructured, casual web-text providing a true low-formality lower boundary.
* **BlendedSkillTalk:** Scripted human-to-human dialogue frames evaluating conversational flow.
* **ELI5 (Explain Like I'm Five):** Informative Q&A text representing explanatory prose, serving as a robust contextual comparator.

### LLM Responses
* **LMSYS-Chat-1M Dataset:** Over 1,000,000 evaluations split across model lineages, filtered to isolate dialogic responses and strip system markup/assistant artifacts.

### Baseline Data Distribution Summary

| Corpus / Group | Mean Formality Rate ($FR$) | Mean RoBERTa $P(\text{Formal})$ | Structural Characteristics |
| :--- | :---: | :---: | :--- |
| **Human: BlendedSkillTalk** | 0.24 $\pm$ 3.60 | 0.271 $\pm$ 0.337 | Highly conversational, minor structural variance. |
| **Human: Reddit** | 0.52 $\pm$ 4.07 | 0.245 $\pm$ 0.325 | Idiomatic, slang-heavy, low lexical density. |
| **Human: ELI5** | 2.33 $\pm$ 7.03 | 0.801 $\pm$ 0.271 | High explanatory focus; elevated structural context. |
| **LLM: LMSYS-Chat** | **7.62 $\pm$ 12.02** | **0.859 $\pm$ 0.271** | **Systematic inflation across all register axes.** |

---

## Research Insights & Findings

### 1. Cross-Family Formality Universality
Register bias is **systemic and universal** across modern foundation models. Evaluations spanning the GPT, PaLM, LLaMA, Mistral, Claude, and Vicuna families demonstrate that nearly all model outputs significantly exceed human conversational register boundaries.

* **Highest Formality Signatures:** PaLM and GPT variants demonstrate the highest over-indexing on elevated academic phrasing.
* **Lowest Formality Signatures:** Claude family models exhibit the closest alignment with human baseline conversational syntax, though they remain significantly more formal than raw casual text.
* **Causal Drivers:** Statistical clustering suggests bias stems from a combination of academic pretraining composition and a reinforcement of sterile text styles during RLHF preference tuning.

---

```
Mean Formality Rate by Corpus Layer
Human (BlendedSkillTalk) █ 0.24
Human (Reddit)           ██ 0.52
Human (ELI5)             ███████████ 2.33
LLM (LMSYS-Chat)         █████████████████████████████████ 7.62
```

### 2. Contextual Topic Drift

Splitting the LMSYS corpus by prompt intent reveals that while explicit user requests for task execution or general summaries shift models into peak formality layers ($P(\text{Formal}) \approx 0.89$), even explicitly **Casual Conversational Prompts** result in inflated registers compared to matching human dialogue:

* **Casual Prompts:** $FR$: 2.447 | RoBERTa Score: 0.614
* **General Prompts:** $FR$: 5.253 | RoBERTa Score: 0.763
* **Task-Driven Prompts:** $FR$: 4.681 | RoBERTa Score: 0.880

---

## Register Control via PEFT Mitigation

To test if register bias can be decoupled from core intelligence, we built a Parameter-Efficient Fine-Tuning (PEFT) pipeline using **LoRA (Low-Rank Adaptation)** applied to **Qwen2.5-3B-Instruct**.

```
[ Freezed Base Weights ]  ──┐
  (Qwen2.5-3B-Instruct)     │
                            ├─► [ Muffled Register Stream ]
[ Trained Adapter ΔW ]    ──┘       Reduced Formal Overhead
  (Low-Formality SFT Data)
```

LoRA freezes the foundational language knowledge vectors while inserting trainable rank-decomposition matrices into the attention layers, mitigating the expensive training footprint and preventing catastrophic forgetting.

### Mitigation Performance Results

| Model Iteration | Mean Formality Rate ($FR$) | Mean RoBERTa $P(\text{Formal})$ | General Similarity (ROUGE-L) |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-3B-Instruct (Base)** | 6.20 | 0.868 | Baseline |
| **Qwen2.5-3B-Instruct (LoRA Fine-Tuned)** | **5.46 (-12.0%)** | **0.794 (-8.5%)** | **0.0766** |

* **Register Steering:** Low-formality supervised fine-tuning (SFT) successfully dampens explicit lexical markers ($12\%$ decrease in surface $FR$) and softens contextual tone ($8.5\%$ drop in RoBERTa probability).
* **Fluency Preservation:** Structural text metrics show stable ROUGE-L alignment ($0.0727 \rightarrow 0.0766$), proving basic syntactic flow and model coherence remain intact under lightweight register steering.
* **Statistical Context:** These preliminary results are evaluated on a restricted evaluation set ($N=50$ held-out prompt pairs); a paired t-test indicates directional register malleability rather than absolute, definitive debiasing.

---

## Limitations & Future Work

* **Measurement Boundaries:** Both $FR$ and RoBERTa metrics function as proxy models; they quantify linguistic markers but do not completely isolate a human's perception of "conversational warmth."
* Fine-Tuning Scope: The current LoRA configuration serves as an early proof-of-concept. Future architectures will scale the mitigation pipeline to larger models (e.g., Mistral-7B, LLaMA-3-8B) using Direct Preference Optimization (DPO).
* Human-Centered Evaluation: We plan to establish extensive human-in-the-loop tests to measure trust, readability, and user preference across pre- and post-mitigated outputs.Multilingual Drift: Tracking how formal register bias shifts across translation layers in non-English assistant prompts remains an open research direction.

## Core Academic References

* **Pavlick, E. & Tetreault, J. (2016):** _An Empirical Analysis of Formality in Online Text._ Transactions of the Association for Computational Linguistics (TACL). Established our lexical verification framework.
* **Rao, S. & Tetreault, J. (2018):** _Dear Sir or Madam, May I Introduce the GYAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer._ NAACL. Foundation for our neural classification layer.
* **Khojah et al. (2024):** _Register Variation in Human-Bot Interaction._ Alware. Documented system-level human vs. conversational agent stylistic gaps.
* **Zhao et al. (2021):** _Ethical and Linguistic Biases in Large Pre-Trained Language Models._ ICML. Motivated our training composition hypothesis.
