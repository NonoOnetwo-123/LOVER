<div align="center">

# Long-to-Short Video Evidence Reasoning for Grounded Question Answering

<p>
  <b>EMNLP 2026</b>
</p>

<p>
  <a href="https://arxiv.org/abs/2609.15224">📄 Paper</a> &nbsp;&nbsp; | &nbsp;&nbsp;
  <a href="https://github.com/NonoOnetwo-123/LOVER">💻 Code</a> &nbsp;&nbsp; | &nbsp;&nbsp;
  <a href="MODEL_LINK">🤗 LOVER-7B</a> &nbsp;&nbsp; | &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset">🤗 Training Data</a>
</p>


<p>
  <img src="https://img.shields.io/badge/Conference-EMNLP%202026-blue" />
  <img src="https://img.shields.io/badge/Task-Video%20Grounded%20QA-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

</div>

---

## 📢 News

* **[2026.8.26]** 🎉 LOVER is accepted by **EMNLP 2026**.
* **[2026.9.19]** 🚀 Code and model checkpoints are released.

---

## 🔥 Overview

Large Vision-Language Models (LVLMs) have demonstrated strong capabilities in video understanding and question answering. However, accurately identifying the **temporal evidence** supporting an answer remains challenging, particularly for long videos.

We introduce **LOVER**, a **Long-to-Short Video Evidence Reinforced** framework for grounded question answering.

LOVER progressively improves temporal evidence reasoning through:

* 🧠 **Long-to-short Video Evidence Curriculum Learning**
* 🎯 **GQA Rewards**
* ⏱️ **Adaptive Timestamp Rendering**

Given a long video and a question, LOVER jointly reasons about **what happens** and **when the supporting evidence occurs**, producing both an answer and its corresponding temporal evidence.

---


## 🖼️ Method

<p align="center">
  <img src="assets/overreview.png" width="95%">
</p>

**Overview of LOVER.**
LOVER combines long-to-short curriculum learning, IoP-based reinforcement learning, and adaptive timestamp rendering for grounded video question answering.

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


## 🙏 Acknowledgements

LOVER is built upon and inspired by the open-source **[Time-R1](https://github.com/xiaomi-research/time-r1)** framework. We sincerely appreciate the authors for making their code and research publicly available, which provided a valuable foundation for our work.

> **Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding**
> *NeurIPS 2025*

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
