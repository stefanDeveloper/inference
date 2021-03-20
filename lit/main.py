"""
Example Usage:

  python3 -m lit.main

"""

import argparse
from lit_nlp import dev_server
from lit_nlp import server_flags
from absl import logging

from .data import Data
from .model import Model


def clean_name(path):
    path = path.replace("data", "")
    path = path.replace("models", "")
    path = path.replace(".csv", "")
    path = path.replace("/", "")
    path = path.replace(".", "")
    path = path.replace("-mq", "")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Finetuning of a specified model on the specified dataset"
    )

    parser.add_argument("-m", "--models", nargs="+", help='<Required> Path of models', required=True)
    parser.add_argument("-d", "--datasets", nargs="+", help='<Required> Path of datasets', required=True)
    parser.add_argument("-c", "--cache_dir", help='<Required> Cache directory', required=True)
    parser.add_argument("-v", "--verbose", help="Also print debug statements", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

    logging.info("Loading datasets: %s", args.datasets)
    datasets = {}
    for i, dataset in enumerate(args.datasets):
        datasets[f'dataset_{clean_name(dataset)}'] = Data(dataset)

    logging.info("Loading models: %s", args.models)
    models = {}
    for i, model_name in enumerate(args.models):
        models[f'model_{clean_name(model_name)}'] = Model(model_name)

    server_kw = server_flags.get_flags()
    server_kw["data_dir"] = args.cache_dir
    lit_demo = dev_server.Server(models, datasets, **server_kw)
    return lit_demo.serve()


if __name__ == '__main__':
    main()
