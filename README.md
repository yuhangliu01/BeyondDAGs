# BeyondDAGs: A Latent Partial Causal Model for Multimodal Learning

This repository contains code for the ICLR 2026 paper on a latent partial causal model for multimodal learning. The implementation shows how using ICA-processed features from CLIP-style models can help downstream tasks that need disentangled representations.


## Insights
This project implements a latent partial causal model for multimodal learning.
A key takeaway is that **pre‑trained multimodal models (CLIP‑like) with ICA-processed features yield disentangled representations**, particularly in scenarios where disentanglement is critical. Several robust variants showcase how ICA improves feature extraction and linear probing across different data distributions.

## CLIP-like Few-Shot Experiments
For experiments related to CLIP-based models for few‑shot learning (see Fig. 4 in the paper), please refer to our complementary work:

**[Causal CLIP Adapter](https://github.com/tianjiao-j/CCA)** – ICCV 2025

---

## Citation
If you find this work helpful, please cite the following papers:

```bibtex
@inproceedings{liu2026beyond,
  title={Beyond {DAG}s: A Latent Partial Causal Model for Multimodal Learning},
  author={Yuhang Liu and Zhen Zhang and Dong Gong and Erdun Gao and Biwei Huang and Mingming Gong and Anton van den Hengel and Kun Zhang and Javen Qinfeng Shi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=bZqCBgm2N0}
}
```

For complementary few-shot learning work:

```bibtex
@inproceedings{jiang2025causal,
  title={Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning},
  author={Jiang, Tianjiao and Zhang, Zhen and Liu, Yuhang and Shi, Javen Qinfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={890--900},
  year={2025}
}
```
