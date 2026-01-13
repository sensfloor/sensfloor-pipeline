import argparse

from configs import get_hyper_param_configs, run_config


def main(do_train: bool, do_test: bool) -> None:
    all_configs = get_hyper_param_configs()
    for hyper_params in all_configs:
        run_config(do_train, do_test, hyper_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="Include flag if model should train")
    parser.add_argument("--test", action="store_true", help="Include flag if model should create test predictions")

    args = parser.parse_args()

    main(args.train, args.test)
