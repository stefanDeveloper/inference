#!/usr/bin/env bash
#SBATCH --job-name=train_deberta
#SBATCH -p students
#SBATCH -o train_deberta.txt

source .venv/bin/activate
python -m sarn.train --output-dir "models/deberta" --log-dir "logs/deberta" "microsoft/deberta-large-mnli" "data/training.csv"
deactivate
