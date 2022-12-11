import logging
from pathlib import Path

import numpy as np
import torch


def inference(model, dataset,
              load_path=Path.cwd() / "models",
              save_path=Path.cwd() / "results",
              model_name="fmri_model.pt",
              valid_prop=0.1,
              test_prop=0.1,
              device=torch.device("cpu")
              ):
    model_path = load_path / model_name

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        logging.error(f'No model found at path: {model_path}')

    model.to(device)

    model.eval()

    subjects_data = model.predict_embeddings(dataset)

    nll, aucroc, ap = model.predict_auc_roc_precision(
        dataset,
        valid_prop=valid_prop,
        test_prop=test_prop)

    logging.info(
        f"train nll {nll['train']} aucroc {aucroc['train']} ap {ap['train']}| "
        f"valid nll {nll['valid']} aucroc {aucroc['valid']} ap {ap['valid']} | "
        f"test nll {nll['test']} aucroc {aucroc['test']} ap {ap['test']}")

    logging.info("Saving results.")

    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logging.warning(f"Saving subject embeddings to {save_path}."
                        f"Existing saved results will be overridden.")

    model_save_path = save_path / "subject_embeddings.npy"

    with open(model_save_path, 'wb') as f:
        np.save(f, subjects_data)

    logging.info("Results saved.")
