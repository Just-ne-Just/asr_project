# ASR project barebones

## Installation guide

```shell
pip install -r ./requirements.txt
```
## LM model weights
```
https://www.kaggle.com/datasets/leonbebra/lm-model
```
## Описание проекта
ASR английской речи

## Структура репозитория
```
train.py - скрипт, с помощью которого запускается обучение модели
```
```
test.py - скрипт, с помощью которого запускается инференс модели на тестовом датасете
```
```
asr_project/config.json - основной конфиг, который используется для теста и обучения 
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
