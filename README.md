# BeyondDAGs: A Latent Partial Causal Model for Multimodal Learning

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-contact%20authors-lightgrey.svg)](LICENSE)

> **BeyondDAGs** implements the latent partial causal model described in our ICLR 2026 paper. The code demonstrates how ICA‑processed features from CLIP‑like multimodal models can boost downstream tasks that demand disentangled representations.

---

## 🔍 Table of Contents
1. [Insights](#insights)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [CLIP‑like Few‑Shot Experiments](#clip-like-few-shot-experiments)
6. [Citation](#citation)
7. [Requirements](#requirements)
8. [Contributing](#contributing)
9. [License](#license)

---

## Insights
This project implements a latent partial causal model for multimodal learning.
A key takeaway is that **pre‑trained multimodal models (CLIP‑like) with ICA-processed features yield more robust representations**, particularly in scenarios where disentanglement is critical. Several robust variants showcase how ICA improves feature extraction and linear probing across different data distributions.

## Repository Structure
- `feat_extractor.py` – Feature extraction module for multimodal data
- `linear_probe_robust.py` – Robust linear probing with advanced techniques
- `linear_probe_robust_PCA.py` – Linear probing with PCA-based robustness
- `linear_probe_robust_ica.py` – Linear probing with ICA-based robustness

## Installation

```bash
# clone the repo
git clone https://github.com/yuhangliu01/BeyondDAGs.git
cd BeyondDAGs

# install dependencies
pip install -r requirements.txt  # (not provided yet; install numpy torch scikit-learn manually)
```

> ⚠️ Requirements file coming soon – in the meantime, ensure Python 3.7+ and the libraries listed below are installed.

## Usage
Detailed usage instructions will be added soon. For now, inspect the docstrings in each module, e.g.:

```python
from feat_extractor import extract_features
# ...
```

## CLIP-like Few-Shot Experiments
For experiments related to CLIP-like models and few‑shot learning (see Fig. 4 in the paper), please refer to our complementary work:

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

## Requirements
- Python 3.7+
- NumPy
- PyTorch
- scikit-learn

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
Please contact the authors for licensing information.
