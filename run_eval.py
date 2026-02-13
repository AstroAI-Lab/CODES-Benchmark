from argparse import ArgumentParser

from codes.benchmark import (
    check_benchmark,
    check_surrogate,
    compare_models,
    get_surrogate,
    run_benchmark,
)
from codes.utils import download_data, nice_print, read_yaml_config


def main(args):
    """
    Main function to run the benchmark. It reads the config file, checks the benchmark
    configuration, runs the benchmark for each surrogate model, and compares the models
    if specified in the config file.

    Args:
        args (Namespace): The command line arguments.
    """

    config = read_yaml_config(args.config)
    check_benchmark(config)
    download_data(config["dataset"]["name"], verbose=config.get("verbose", False))
    surrogates = config["surrogates"]
    # Create dictionary to store metrics for all surrogate models
    all_metrics = {surrogate: {} for surrogate in surrogates}

    # Run benchmark for each surrogate model
    for surrogate_name in surrogates:
        surrogate_class = get_surrogate(surrogate_name)
        if surrogate_class is not None:
            nice_print(f"Running benchmark for {surrogate_name}")
            check_surrogate(surrogate_name, config)
            metrics = run_benchmark(surrogate_name, surrogate_class, config)
            all_metrics[surrogate_name] = metrics
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")

    # Compare models
    if config.get("compare", False):
        if len(surrogates) < 2:
            nice_print("At least two surrogate models are required to compare.")
        else:
            nice_print("Comparing models")
            compare_models(all_metrics, config)

    nice_print("Evaluation completed.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    parser.add_argument(
        "--device",
        default="None",
        type=str,
        help="Device to run the benchmark on. Can be 'cpu', 'cuda', or 'None'.",
    )
    args = parser.parse_args()
    main(args)
