# DBGDGM Model for fMRI Data

This repository contains the code for the DBGDGM model, a machine learning model for analyzing functional magnetic resonance imaging (fMRI) data. 

## Link to paper:
The paper related to this project is available at https://openreview.net/forum?id=WHS3Zv9pxz.


## Setup repo
1. Make sure you have pipenv installed.
2. Clone this repo:
```shell
git clone https://github.com/simeon-spasov/dynamic-brain-graph-deep-generative-model.git
```
4. Install dependencies from Pipfile:
```shell
cd dynamic-brain-graph-deep-generative-model
pipenv install
```

## Datasets

The current version of the model supports two datasets: 'ukb' and 'hcp'. You can specify the dataset you want to use via the `--dataset` argument when running the model.

## Model Training

To train the model, run the following command:

```shell
python main.py --dataset <dataset_name> --categorical-dim <dim> --valid-prop <valid_prop> --test-prop <test_prop> --trial <trial_num> --gpu <gpu_id>
```

Where:

- `<dataset_name>` is the name of the dataset you want to use. Possible values are 'ukb' and 'hcp'.
- `<dim>` is the categorical dimension.
- `<valid_prop>` is the validation set proportion (default value is 0.1).
- `<test_prop>` is the test set proportion (default value is 0.1).
- `<trial_num>` is the trial number.
- `<gpu_id>` is the ID of the GPU you want to use for training (optional). If not specified or GPU is not available, the model will run on CPU.

For example:

```shell
python main.py --dataset ukb --categorical-dim 8 --valid-prop 0.1 --test-prop 0.1 --trial 1 --gpu 0
```

This command will start training the model on the 'ukb' dataset, with a categorical dimension of 8, validation and test set proportions of 0.1, using GPU 0. 

## Model Inference

To perform inference with a trained model, use the `inference` command:

```shell
python main.py --dataset <dataset_name> --categorical-dim <dim> --valid-prop <valid_prop> --test-prop <test_prop> --trial <trial_num> --gpu <gpu_id> inference
```

All the parameters for the inference command are the same as for the model training command. 

For example:

```shell
python main.py --dataset ukb --categorical-dim 8 --valid-prop 0.1 --test-prop 0.1 --trial 1 --gpu 0 inference
```

This command will start the inference process using the model trained on the 'ukb' dataset with the specified parameters, using GPU 0. Naturally, you need to have trained the model with these specified parameters first.

## Gender classification
See Jupyter notebook in notebooks directory.


## Logging

The script logs various useful information during the training and inference processes. The log file is saved as `fMRI_<dataset>_<trial>.log` in the root directory.

For any issues or queries, please open a Github issue in the repository.

## Model Training and Outputs

Our model training process saves multiple artifacts during and after the training process. These artifacts are saved under a directory specified by `save_path` (default is the current working directory under the 'models' subdirectory).

Here is what you can expect:

- The training process automatically creates the directory defined in `save_path` if it doesn't already exist.
- The model parameters are saved every time the validation negative log-likelihood improves. These are saved as a tuple of the model and the optimizer state dictionaries in a file named 'checkpoint.pt'.
- For the best model according to validation negative log-likelihood, the negative log-likelihoods, AUC-ROC, AP, and embeddings are saved in a file named 'results.npy'. This is a NumPy structured array.
- Additionally, the model parameters and results are saved for the best model according to training negative log-likelihood as 'checkpoint_best_train.pt' and 'results_best_train.npy' respectively.
- Please note, if there are existing model parameters with the same name in the directory, they will be overwritten.

The contents of 'results.npy' and 'results_best_train.npy' are dictionaries with the following keys:

- 'nll': Negative log-likelihood for train, validation, and test data.
- 'aucroc': Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for train, validation, and test data.
- 'ap': Average precision score for train, validation, and test data.
- 'embeddings': Predicted node and community embeddings for all snapshots (train, val and test) for all the subject graphs.

These results files are in NumPy's `.npy` format and can be loaded using `numpy.load()` function with the parameter `allow_pickle=True`.


