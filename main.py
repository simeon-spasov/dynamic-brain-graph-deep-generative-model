import hydra
import wandb
import pathlib
import uuid

from omegaconf import OmegaConf

from src.datasets import get_dataloaders
from src.models import get_model
from src.loss import get_loss
from src.train import train

from src.utils import set_random_seed, get_device, create_dirs, load_config, save_config


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg):
    
    # set random seed for reproducibility
    set_random_seed(seed=cfg.experiment.seed, 
                    is_gpu=cfg.experiment.is_gpu)

    # get training backend
    device = get_device(is_gpu=cfg.experiment.is_gpu, 
                        gpu_number=cfg.experiment.gpu_number)

    # initialise logging
    if cfg.experiment.wb_logging:
        wandb.init(project=cfg.experiment.wb_project)
        run_id = wandb.run.id
    else:
        run_id = uuid.uuid4().hex[:8]

    # create experiment directories
    models_dir, results_dir = create_dirs(run_id=run_id,
                                          prefix=cfg.experiment.prefix)

    # initalise datasets and dataloaders
    dataloaders = get_dataloaders(**cfg.data, 
                                  device=device,
                                  seed=cfg.experiment.seed + cfg.experiment.run) 
    
    # initialise model
    model = get_model(**cfg.model,
                      num_samples=dataloaders["train"].dataset.num_samples,
                      num_nodes=dataloaders["train"].dataset.num_nodes,
                      num_timesteps=dataloaders["train"].dataset.num_timesteps,
                      criterion=get_loss(**cfg.loss),
                      device=device)
 
    trained_model = train(**cfg.train,
                          model=model, 
                          train_loader=dataloaders["train"],
                          valid_loader=dataloaders["valid"], 
                          save_dir=models_dir,
                          wb_logging=cfg.experiment.wb_logging,
                          device=device)
    save_config(cfg, models_dir)


if __name__ == "__main__":
    main()