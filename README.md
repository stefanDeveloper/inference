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
python -m sarn.train --model facebook/bart-large-mnli --output-dir models/mybart --log-dir logs/mybart
# Inference of two sequences
python -m sarn.classify --model models/mybart "All dogs jumped over the fence." "Some small dogs jumped over the fence."
```

For `--model`, any valid Huggingface model (local or remote) can be specified that has been finetuned for sequence classification, e.g., `facebook/bart-large-mnli`, `microsoft/deberta-large-mnli` or a local path like `models/mybart`. 
