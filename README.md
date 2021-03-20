# inference
Quantifiers and monotonicity in reasoning tasks

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
source .venv/bin/activate
# Finetuning
python -m sarn.train --output-dir models/bart-mq --log-dir logs/bart-mq facebook/bart-large-mnli data/training.csv
# Inference of two sequences (forwards)
python -m sarn.classify --model models/bart-mq "All dogs jumped over the fence." "Some small dogs jumped over the fence."
```

For `--model`, any valid [Huggingface model](https://huggingface.co/transformers/pretrained_models.html) (local or remote) can be specified that has been [finetuned for sequence classification](https://huggingface.co/models?pipeline_tag=text-classification), e.g., [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli), [`microsoft/deberta-large-mnli`](https://huggingface.co/microsoft/deberta-large-mnli) or a local path like `models/bart-mq`. 
