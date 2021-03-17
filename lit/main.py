import argparse
from lit_nlp import dev_server


def main():
    parser = argparse.ArgumentParser(
        description="Finetuning of a specified model on the specified dataset"
    )

    parser.add_argument("models", nargs="+", help='<Required> Path of models')
    parser.add_argument("dataset", nargs="+", help='<Required> Path of datasets')

    args = parser.parse_args()

    print("Loading datasets", args.datasets)
    datasets = {}
    for i, dataset in enumerate(args.datasets):
        setattr(datasets, f'dataset_{i}', dataset)

    print("Loading models", args.datasets)
    models = {}
    for i, model in enumerate(args.models):
        setattr(models, f'model_{i}', model)

    lit_demo = dev_server.Server(models, datasets, port=4321)
    return lit_demo.serve()


if __name__ == '__main__':
    main()
