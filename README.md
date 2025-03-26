```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

# Clone the repository (using HTTPS instead of SSH)
git clone https://github.com/IamCreateAI/LayerAnimate.git
cd LayerAnimate

# Create and activate conda environment
conda create -n layeranimate python=3.10 -y
conda activate layeranimate
# Install ipykernel and register the environment with a display name
pip install ipykernel
python -m ipykernel install --user --name layeranimate --display-name "LayerAnimate"

# Install requirements
pip install -r requirements.txt
```

# LayerAnimate: Layer-level Control for Animation

[Yuxue Yang](https://yuxueyang1204.github.io/)<sup>1,2</sup>, [Lue Fan](https://lue.fan/)<sup>2</sup>, [Zuzeng Lin](https://www.researchgate.net/scientific-contributions/Zuzeng-Lin-2192777418)<sup>3</sup>, [Feng Wang](https://happynear.wang/)<sup>4</sup>, [Zhaoxiang Zhang](https://zhaoxiangzhang.net)<sup>1,2†</sup>

<sup>1</sup>UCAS&emsp; <sup>2</sup>CASIA&emsp; <sup>3</sup>TJU&emsp; <sup>4</sup>CreateAI&emsp; <sup>†</sup>Corresponding author

<a href='https://arxiv.org/abs/2501.08295'><img src='https://img.shields.io/badge/arXiv-2501.08295-b31b1b.svg'></a> &nbsp;
<a href='https://layeranimate.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://www.bilibili.com/video/BV1EycqeaEqF/'><img src='https://img.shields.io/badge/BiliBili-Video-479fd1.svg'></a> &nbsp;
<a href='https://youtu.be/b_bvVKigky4'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a> &nbsp;


<div align="center"> <img src='__assets__/figs/demos.gif'></img></div>

**Videos on the [project website](https://layeranimate.github.io) vividly introduces our work and presents qualitative results for an enhanced view experience.**

## Updates

- [25-03-22] Release the checkpoint and the inference script. **We update layer curation pipeline and support trajectory control for a flexible composition of various layer-level controls.**
- [25-01-15] Release the project page and the arXiv preprint.

## Inference

### Installation

```bash
git clone git@github.com:IamCreateAI/LayerAnimate.git
conda create -n layeranimate python=3.10 -y
conda activate layeranimate
pip install -r requirements.txt
```

### Checkpoint preparation

Download the pretrained [layeranimate](https://huggingface.co/Yuppie1204/LayerAnimate-Mix) weights and put them in `checkpoints/` directory as follows:

```bash
checkpoints/
└─ LayerAnimate-Mix
```

### Inference script

Run the following command to generate a video from input images:

```bash
python scripts/animate_Layer.py --config scripts/demo1.yaml --savedir outputs/sample1

python scripts/animate_Layer.py --config scripts/demo2.yaml --savedir outputs/sample2

python scripts/animate_Layer.py --config scripts/demo3.yaml --savedir outputs/sample3

python scripts/animate_Layer.py --config scripts/demo4.yaml --savedir outputs/sample4

python scripts/animate_Layer.py --config scripts/demo5.yaml --savedir outputs/sample5
```

Note that the layer-level controls are prepared in `__assets__/demos`. A more user-friendly interface with gradio will be uploaded in huggingface spaces soon.

## Todo

- [x] Release the code and checkpoint of LayerAnimate.
- [ ] Release checkpoints trained under single control modality with better performance.
- [ ] Upload a gradio script and UI in huggingface spaces.
- [ ] Release layer curation pipeline.
- [ ] Training script for LayerAnimate.
- [ ] DiT-based model LayerAnimate.

## Acknowledgements

We sincerely thank the great work [ToonCrafter](https://doubiiu.github.io/projects/ToonCrafter/), [LVCD](https://luckyhzt.github.io/lvcd), and [AniDoc](https://yihao-meng.github.io/AniDoc_demo/) for their inspiring work and contributions to the animation community.

## Citation

Please consider citing our work as follows if it is helpful.
```bib
@article{yang2025layeranimate,
  author    = {Yang, Yuxue and Fan, Lue and Lin, Zuzeng and Wang, Feng and Zhang, Zhaoxiang},
  title     = {LayerAnimate: Layer-level Control for Animation},
  journal   = {arXiv preprint arXiv:2501.08295},
  year      = {2025},
}
```
