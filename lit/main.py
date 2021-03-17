import argparse
from lit_nlp import dev_server


def main():
    parser = argparse.ArgumentParser(
        description="Finetuning of a specified model on the specified dataset"
    )

    parser.add_argument("-m", "--models", nargs="+", help='<Required> Path of models', required=True)
    parser.add_argument("-d", "--datasets", nargs="+", help='<Required> Path of datasets', required=True)

    args = parser.parse_args()

    print("Loading datasets", args.datasets)
    datasets = {}
    for i, dataset in enumerate(args.datasets):
        datasets[f'dataset_{i}'] = dataset

    print("Loading models", args.models)
    models = {}
    for i, model in enumerate(args.models):
        models[f'model_{i}'] = model

    lit_demo = dev_server.Server(models, datasets, port=4321)
    return lit_demo.serve()


if __name__ == '__main__':
    main()
