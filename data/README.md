# Data

Datasets:

- `training.csv`: based on the [MED dataset](https://github.com/verypluming/MED/blob/master/MED.tsv) and [HELP dataset](https://github.com/verypluming/HELP/blob/master/output_en/pmb_train_v1.0.tsv), converted to fit our schema (see `sarn/convert/med.py` and `sarn/convert/help.py`) and then concatenated
- `evaluation.csv`: based on the [SuperGlue diagnostics dataset](https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1) filtered by the Logic categories "Quantification" and "Monotonicity" and the [FraCaS problem set](https://nlp.stanford.edu/~wcmac/downloads/fracas.xml) filtered by the category "1 GENERALIZED QUANTIFIERS", converted to fit our schema (see `sarn/convert/superglue.py` and `sarn/convert/fracas.py`) and then concatenated
