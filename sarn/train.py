import argparse
from sarn.data import load_training_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

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
# More than 1 epochs lead to overfitting in our experiments
parser.add_argument("--epochs", type=float, default=1)
# Google Colab seems to be fine up until a batch size of 4 for the
# Kepler GPUs (11gb VRAM) and until 10 for the Tesla GPUs (16gb VRAM).
# If you're training on CPU and have _lots_ of RAM, use a higher value (e.g., 64).
parser.add_argument("--train-batch-size", type=int, default=4)
parser.add_argument("--test-batch-size", type=int, default=16)
parser.add_argument("--log-frequency", type=int, default=10)
parser.add_argument("--max-checkpoints", type=int, default=2)
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    "--cpu", action="store_true"
)  # Whether to only train on CPU, not GPU

if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading model", args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading dataset", args.dataset)
    train_dataset, test_dataset = load_training_dataset(args.dataset, tokenizer)

    if args.resume:
        checkpoint = get_last_checkpoint(args.output_dir)
    else:
        checkpoint = None

    print("Creating training instance")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,
        logging_dir=args.log_dir,
        logging_steps=args.log_frequency,
        save_total_limit=args.max_checkpoints,
        no_cuda=args.cpu,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Starting training")
    trainer.train(checkpoint)
    print(trainer.evaluate())
    trainer.save_model()
