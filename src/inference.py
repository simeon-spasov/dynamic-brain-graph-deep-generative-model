from pathlib import Path

import numpy as np
import torch


def inference(model,
              dataset,
              load_path=Path.cwd() / "models",
              save_path=Path.cwd() / "models",
              valid_prop=0.1, test_prop=0.1,
              device=torch.device("cpu")
              ):
    """
    Performs inference on a dataset using a pre-trained model and saves the results.

    Args:
        model (nn.Module): The pre-trained PyTorch model for inference.
        dataset (list): The dataset on which to perform inference.
        load_path (Path, optional): The directory from where the pre-trained model will be loaded. Default is the current working directory under the 'models' subdirectory.
        save_path (Path, optional): The directory where the results of the inference will be saved. Default is the current working directory under the 'models' subdirectory.
        valid_prop (float, optional): The proportion of the dataset to be used for validation during inference. Default is 0.1.
        test_prop (float, optional): The proportion of the dataset to be used for testing during inference. Default is 0.1.
        device (torch.device, optional): The device to use for model execution (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns: None. The function saves the results of the inference to the specified save_path. The embeddings, negative log-likelihoods, AUC-ROC, AP are saved. The results are saved as 'results_inference.npy'.
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

    nll, aucroc, ap = model.predict_auc_roc_precision(
        dataset,
        valid_prop=valid_prop,
        test_prop=test_prop
    )

    report = (f"train nll {nll['train']} aucroc {aucroc['train']} ap {ap['train']}| "
              f"valid nll {nll['valid']} aucroc {aucroc['valid']} ap {ap['valid']} | "
              f"test nll {nll['test']} aucroc {aucroc['test']} ap {ap['test']}")

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
                'embeddings': embeddings
            })

    print("Performance metrics saved.")
