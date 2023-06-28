import logging
from pathlib import Path

import numpy as np
import torch


def inference(model, dataset, load_path=Path.cwd() / "models",
              save_path=Path.cwd() / "models", valid_prop=0.1, test_prop=0.1,
              num_samples=1, device=torch.device("cpu")):
    """
    Perform inference on a dataset using a pretrained model and save the results.

    Args:
        model: A pretrained PyTorch model.
        dataset: The dataset to perform inference on.
        load_path: The path to load the model from.
        save_path: The path to save the results to.
        valid_prop: The proportion of the dataset used for validation.
        test_prop: The proportion of the dataset used for testing.
        num_samples: The number of samples for computing performance metrics.
        device: The device to use for model execution.
    """
    model_path = load_path / "checkpoint.pt"

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint[0])
    except FileNotFoundError:
        print(f'No model found at path: {model_path}')

    model.to(device)
    model.eval()

    embeddings = model.predict_embeddings(dataset, valid_prop=valid_prop, test_prop=test_prop)

    nll, aucroc, ap, mse_to, mse_td = model.predict_auc_roc_precision(
        dataset,
        valid_prop=valid_prop,
        test_prop=test_prop,
        num_samples=num_samples
    )

    report = (f"train nll {nll['train']} aucroc {aucroc['train']} ap {ap['train']}| "
              f"valid nll {nll['valid']} aucroc {aucroc['valid']} ap {ap['valid']} | "
              f"test nll {nll['test']} aucroc {aucroc['test']} ap {ap['test']} | "
              f"mse topological overlap {mse_to} mse temporal degree {mse_td}")

    print(report)
    print("Saving embeddings.")

    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Saving subject embeddings to {save_path}."
              f"Existing saved results will be overridden.")

    model_save_path = Path(save_path) / "results_inference.npy"

    np.save(model_save_path,
            {
                'nll': nll,
                'aucroc': aucroc,
                'ap': ap,
                'mse_to': mse_to,
                'mse_td': mse_td,
                'embeddings': embeddings
            })

    print("Performance metrics saved.")
