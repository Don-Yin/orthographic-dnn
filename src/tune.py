import torch
import torch.nn.functional as functional

from config import DIR_PARAMS, load_model, MODEL_REGISTRY
from modules.train import Train


def get_hyperparameters(model_name: str):
    """builds the hyperparameter dict for a given model name."""
    model = load_model(model_name)
    params_path = DIR_PARAMS / f"{model_name}.pth"
    if params_path.exists():
        model.load_state_dict(torch.load(params_path, weights_only=True))
        print(f"params loaded: {model_name}")

    return {
        "model": model,
        "model_tag": model_name,
        "function_optimizer": torch.optim.Adam,
        "function_loss": functional.cross_entropy,
        "rate_learning": 1e-5,
        "size_batch": 32,
        "num_epochs": 3,
        "weight_decay": 0.0001,
        "min_moving_average_threshold": 0.2,
        "batch_report_every": 8,
        "batch_save_every": 128,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    print("cuda available: ", torch.cuda.is_available())
    for model_name in MODEL_REGISTRY:
        Train(get_hyperparameters(model_name)).fit()
