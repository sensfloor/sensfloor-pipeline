from training.configs import get_hyper_param_configs
from training.training_pipeline import train


def main() -> None:
    all_configs = get_hyper_param_configs()
    for config in all_configs:
        train(config)


if __name__ == "__main__":
    main()
