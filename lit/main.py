import argparse

import lit_nlp


def main(_):
    parser = argparse.ArgumentParser(
        description="Finetuning of a specified model on the specified dataset"
    )

    parser.add_argument("-m", "--models", nargs="+", help='<Required> Path of models', required=True)
    parser.add_argument("-d", "--dataset", nargs="+", help='<Required> Path of datasets', required=True)

    args = parser.parse_args()

    print("Loading datasets", args.datasets)
    datasets = {}
    for i, dataset in enumerate(args.datasets):
        setattr(datasets, f'dataset_{i}', dataset)

    print("Loading models", args.datasets)
    models = {}
    for i, model in enumerate(args.models):
        setattr(models, f'model_{i}', model)

    # TODO Load models by parameters
    lit_demo = lit_nlp.dev_server.Server(models, datasets, port=4321)
    return lit_demo.serve()


if __name__ == '__main__':
    main()
