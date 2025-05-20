# A-DMA: **A**ccelerating Diffusion-based Text-to-Speech Model Training with **D**ual **M**odality **A**lignment

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://opendemopages.github.io/ademo/)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)

## 📜 News
🚀 [2025.5] We release all the code to promote the research of accelerating diffusion-based TTS models.

🚀 [2025.5.19] Our paper is accepted to [Interspeech 2025](https://www.interspeech2025.org/home), hope to see you in the conference!

## 💡 Highlights
1. **Dual Modality Alignment**: A novel training paradigm that aligns the text and audio modalities in a dual manner, enhancing the model's ability to generate fluent and faithful speech.
2. **Plug and Play for diffusion-based TTS**: A-DMA can be easily integrated into existing diffusion-based TTS models, providing a simple yet effective way to improve their performance.
3. **Accelerated Training**: A-DMA significantly reduces the training time of diffusion-based TTS models, making it feasible to train large-scale models on limited resources.
4. **Open-Source**: A-DMA is open-sourced to promote research in the field of TTS and to provide a baseline for future work.
## 🛠️ Usage
### 1. Install environment and dependencies
```bash
# We recommend using conda to create a new environment.
conda create -n adma python=3.10
conda activate adma

git clone https://github.com/ZhikangNiu/A-DMA.git
cd A-DMA

# Install PyTorch >= 2.2.0, e.g.,
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Install editable version of A-DMA
pip install -e .
```

### 2. Docker usage also available
We provide a docker image for easy usage. 
```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Run from GitHub Container Registry
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main
```



### 3.Training
> Our training process is based on the [F5-TTS](https://github.com/SWivid/F5-TTS), if you have any questions, please check its issues and README first.
#### Prepare datasets
You can follow the [instructions](src/f5_tts/train) to prepare datasets for training, and our experiments are based on the LibriTTS dataset.
```bash
python src/f5_tts/train/datasets/prepare_libritts.py
```
#### Train the model
Once your datasets are prepared, you can start the training process.

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config
# if you want to save the config to a specific path, you can use:
# accelerate config --config_file /path/to/config.yaml

accelerate launch src/f5_tts/train/train_adma.py --config-name F5TTS_v1_Small.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
```


Read [training & finetuning guidance](src/f5_tts/train) for more instructions.


## [Evaluation](src/f5_tts/eval)


## Development

Use pre-commit to ensure code quality (will run linters and formatters automatically):

```bash
pip install pre-commit
pre-commit install
```

When making a pull request, before each commit, run: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation.


## ❤️ Acknowledgments
Our work is built upon the following open-source project [F5-TTS](https://github.com/SWivid/F5-TTS). Thanks to the authors for their great work, and if you have any questions, you can first check them on F5-TTS issues.

## ✒️ Citation and License
Our code is released under MIT License. If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
```

```



