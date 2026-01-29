# BeyondDAGs: A Latent Partial Causal Model for Multimodal Learning

This repository contains the implementation of **"Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning"** paper.

## Insights

This project implements a latent partial causal model for multimodal learning. A key insight from our work is that **pre-trained multimodal models (CLIP-like models) with ICA-processed features can significantly improve downstream tasks that rely on disentangled representations**. The implementation includes several robust variants demonstrating how ICA-based feature disentanglement enhances feature extraction and linear probing across various data distributions and scenarios.

## Files

- **feat_extractor.py**: Feature extraction module for multimodal data
- **linear_probe_robust.py**: Robust linear probing with advanced techniques
- **linear_probe_robust_PCA.py**: Linear probing with PCA-based robustness
- **linear_probe_robust_ica.py**: Linear probing with ICA-based robustness

## CLIP-like Models for Few-Shot Learning

For experiments related to CLIP-like models and few-shot learning (e.g., Fig. 4 in the paper), please refer to our complementary work on contrastive learning:

**[CCA: Complementary Contrastive Analysis](https://github.com/tianjiao-j/CCA)** - ICCV 2025

This repository provides additional implementations for few-shot learning scenarios using contrastive approaches.

## Citation

If you find this work helpful in your research, please cite the following papers:

```bibtex
@article{liu2024beyonddags,
  title={Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning},
  author={Liu, Yuhang and others},
  year={2024}
}
```

For complementary few-shot learning work, please also cite:

```bibtex
@inproceedings{jiangcca2025,
  title={CCA: Complementary Contrastive Analysis},
  author={Jiang, Tianjiao and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- scikit-learn

## Usage

Detailed usage instructions coming soon. For now, refer to the individual module docstrings for function-level documentation.

## License

Please contact the authors for licensing information.