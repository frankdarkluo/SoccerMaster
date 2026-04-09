# SoccerMaster: A Vision Foundation Model for Soccer Understanding
Official implementation of **SoccerMaster: A Vision Foundation Model for Soccer Understanding** (CVPR 2026 Oral)

[*Haolin Yang*](https://haolinyang-hlyang.github.io/),
[*Jiayuan Rao*](https://jyrao.github.io/),
[*Haoning Wu*](https://haoningwu3639.github.io/),
[*Weidi Xie*](https://weidixie.github.io/)


<div style="line-height: 1;">
  <a href="https://haolinyang-hlyang.github.io/SoccerMaster/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website🌐-SoccerMaster-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/pdf/2512.11016" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv📄-SoccerMaster-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center">
   <img src="./images/teaser.jpg">
</div>
<p><em><strong>SoccerMaster</strong> is a unified soccer-specific vision foundation model that leverages diverse soccer content, including images and videos, to support a wide range of soccer understanding tasks, such as commentary generation, detection, tracking, classification, <em>etc</em>.</em></p>
<div align="center">
   <img src="./images/arch.jpg">
</div>
<p><em><strong>SoccerMaster Architecture.</strong> (a) The architecture of SoccerMaster, which encodes both soccer videos and images through spatial and temporal attention modules to generate semantically rich representations. (b) The pretraining tasks and downstream adaptations of SoccerMaster across both spatial perception and semantic understanding tasks.</em></p>

## Quick Start

### 1. Download Pretrained Models

First, clone the SigLIP2 backbone model to the pretrained models directory:

```bash
cd SoccerMaster/codes/SoccerMaster/pretrained_models/google
git lfs install
git clone https://huggingface.co/google/siglip2-large-patch16-512
```

### 2. Download SoccerMaster Checkpoints

Clone the SoccerMaster model checkpoints from Hugging Face:

```bash
cd SoccerMaster/codes/SoccerMaster/pretrained_models
git lfs install
git clone https://huggingface.co/xleprime/SoccerMaster
```

After completing these steps, you should have the following directory structure:
```
root/codes/SoccerMaster/pretrained_models/
├── google/
│   └── siglip2-large-patch16-512/
└── SoccerMaster/
```

## TODO
- [x] Add pretraining code.
- [] Refine pretraining code.
- [x] Release SoccerMaster checkpoints.
- [ ] Add instructions for quick start.
- [ ] Release datasets.
- [ ] Release data pepeline.

## Citations
If you find our work useful, please cite:
```bibtex
@article{yang2025soccermaster,
  title={SoccerMaster: A Vision Foundation Model for Soccer Understanding},
  author={Yang, Haolin and Rao, Jiayuan and Wu, Haoning and Xie, Weidi},
  journal={arXiv preprint arXiv:2512.11016},
  year={2025}
}
```
