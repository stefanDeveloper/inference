import argparse
from sarn.data import load_training_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

# Base models:
# facebook/bart-large-mnli
# microsoft/deberta-large-mnli

parser = argparse.ArgumentParser(
    description="Finetuning of a specified model on the specified dataset"
)
parser.add_argument("model")
parser.add_argument("dataset")
parser.add_argument("--output-dir")
parser.add_argument("--log-dir")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--log-frequency", type=int, default=10)

args = parser.parse_args()

print("Loading model", args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

print("Loading dataset", args.dataset)
train_dataset, test_dataset = load_training_dataset(args.dataset, tokenizer)

print("Creating training instance")
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=args.log_dir,
    logging_steps=args.log_frequency,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Starting training")
trainer.train()
print(trainer.evaluate())
trainer.save_model()
