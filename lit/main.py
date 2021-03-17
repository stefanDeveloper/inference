import argparse

import lit_nlp

from sarn.data import load_training_dataset


def main(_):
    parser = argparse.ArgumentParser(
        description="Finetuning of a specified model on the specified dataset"
    )

    parser.add_argument("model")
    parser.add_argument("dataset")

    args = parser.parse_args()

    model_name = args.model
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print("Loading dataset", args.dataset)
    #train_dataset, test_dataset = load_training_dataset(args.dataset, tokenizer)

    #datasets = {
    #    'mnli_matched': MultiNLIData('/path/to/dev_matched.tsv'),
    #    'mnli_mismatched': MultiNLIData('/path/to/dev_mismatched.tsv'),
    #}

    #models = {
    #    'model_bert': NLIModel('../path/to/model/foo/files'),
    #    'model_deberta': NLIModel('/path/to/model/bar/files'),
    #}

    # TODO Load models by parameters
    lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
    return lit_demo.serve()


if __name__ == '__main__':
    main()
