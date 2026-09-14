<div align="center">

# LOVER: Long-to-Short Video Evidence Reasoning for Grounded Question Answering

<p>
  <b>EMNLP 2026</b>
</p>

<p>
  <a href="PAPER_LINK">📄 Paper</a> &nbsp;&nbsp; | &nbsp;&nbsp;
  <a href="CODE_LINK">💻 Code</a> &nbsp;&nbsp; | &nbsp;&nbsp;
  <a href="MODEL_LINK">🤗 Model</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Conference-EMNLP%202026-blue" />
  <img src="https://img.shields.io/badge/Task-Video%20Grounded%20QA-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

</div>

---

## 📢 News

* **[2026.XX]** 🎉 LOVER is accepted by **EMNLP 2026**.
* **[2026.XX]** 🚀 Code and model checkpoints are released.

---

## 🔥 Overview

Large Vision-Language Models (LVLMs) have demonstrated strong capabilities in video understanding and question answering. However, accurately identifying the **temporal evidence** supporting an answer remains challenging, particularly for long videos.

We introduce **LOVER**, a **Long-to-Short Video Evidence Reinforced** framework for grounded question answering.

LOVER progressively improves temporal evidence reasoning through:

* 🧠 **Long-to-Short Video Evidence Curriculum**
* 🎯 **Intersection-over-Prediction (IoP) Reward**
* ⏱️ **Adaptive Timestamp Rendering**

Given a long video and a question, LOVER jointly reasons about **what happens** and **when the supporting evidence occurs**, producing both an answer and its corresponding temporal evidence.

---

## ✨ Key Contributions

### 1. Long-to-Short Video Evidence Curriculum

We propose a **long-to-short curriculum learning strategy** that progressively introduces temporal grounding examples with different evidence durations.

The curriculum encourages the model to first learn coarse temporal reasoning and gradually improve its ability to identify short and fine-grained evidence intervals.

### 2. Intersection-over-Prediction Reward

We introduce an **IoP-based reward** to provide more effective supervision for temporal evidence localization.

Compared with conventional IoU-based rewards, IoP is particularly suitable for cases where the ground-truth evidence interval is short and the predicted interval partially overlaps with the target evidence.

### 3. Adaptive Timestamp Rendering

We propose **Adaptive Timestamp Rendering**, which dynamically represents temporal information during video reasoning and improves the model's ability to distinguish fine-grained temporal boundaries.

---

## 🖼️ Method

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

**Overview of LOVER.**
LOVER combines long-to-short curriculum learning, IoP-based reinforcement learning, and adaptive timestamp rendering for grounded video question answering.

---

## 📊 Results

### ReXTime

| Method    | Acc@GQA ↑ |
| :-------- | --------: |
| Baseline  |     38.04 |
| Time-R1   |     42.94 |
| **LOVER** | **43.06** |

### NExT-GQA

| Method    |   mIoP ↑ |   mIoU ↑ | IoP@0.5 ↑ | IoU@0.5 ↑ |    GQA ↑ |
| :-------- | -------: | -------: | --------: | --------: | -------: |
| Time-R1   |    26.06 |        – |         – |         – |        – |
| **LOVER** | **42.7** | **33.5** |  **41.1** |  **29.7** | **32.7** |

> Please refer to the paper for the complete experimental results and detailed comparisons.

---

## 🎥 Qualitative Results

<p align="center">
  <img src="assets/qualitative.png" width="95%">
</p>

LOVER is able to identify the temporal evidence supporting the answer while performing video question answering.

---

# 🚀 Getting Started

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/LOVER.git
cd LOVER

conda create -n lover python=3.10 -y
conda activate lover

pip install -r requirements.txt
```

---

# 💻 Code

The implementation of LOVER is organized as follows:

```text
LOVER/
├── configs/
├── scripts/
├── lover/
│   ├── models/
│   ├── datasets/
│   ├── rewards/
│   ├── trainers/
│   └── utils/
├── tools/
├── examples/
├── requirements.txt
└── README.md
```

### Inference

```bash
bash scripts/inference.sh
```

### Evaluation

```bash
bash scripts/evaluate.sh
```

### Training

```bash
bash scripts/train.sh
```

More detailed instructions will be provided in the corresponding directories.

---

# 🤗 Model

We provide pretrained LOVER checkpoints for research and reproducibility.

| Model     | Base Model | Download                      |
| :-------- | :--------- | :---------------------------- |
| **LOVER** | Qwen-VL    | 🤗 [Hugging Face](MODEL_LINK) |

### Download

```bash
huggingface-cli download YOUR_USERNAME/LOVER
```

Please refer to the model repository for detailed inference instructions and model-specific requirements.

---

# 📚 Datasets

LOVER is evaluated on multiple video grounded question answering benchmarks, including:

* **ReXTime**
* **NExT-GQA**
* **CG-Bench**

Please download the datasets from their official sources and follow their respective licenses and usage policies.

---

# 📝 Citation

If you find LOVER useful in your research, please consider citing:

```bibtex
@inproceedings{chen2026lover,
  title     = {Long to Short Video Evidence Reasoning for Grounded Question Answering},
  author    = {Chen, Kaiyan and Others},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

---

# 📄 License

This project is released under the MIT License.

Please also check the licenses of the third-party models, datasets, and dependencies used in this project.
