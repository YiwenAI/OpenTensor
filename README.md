# OpenTensor

This is the code implement of our paper "OpenTensor: Reproducing Faster Matrix Multiplication Discovering Algorithms" in the 37th Conference on Neural Information Processing Systems Workshop (NeurIPS 2023).

We provide the codes to generate synthetic tensors, train OpenTensor and perform tensor decomposition.

## Config

All configs should be contained in a yaml file. We provide some config templates in the `./config` folder. For example, `./config/S_4.yaml` is the config file for decomposing $4 \times 4 \times 4$ matrix multiplication tensor, which is equivalent to discovering the $2 \times 2$ matrix multiplication algorithm.

## Generating synthetic data

```
mkdir data
python main.py --config ./config/S_4.yaml --mode generate_data
```

This command generates 100000 synthetic tensors and saves it to the `./data` folder.

## Training OpenTensor

```
mkdir exp
python main.py --config ./config/S_4.yaml --mode train
```

The model parameters and the tensorboard log files are all saved in the subfolders of `./exp`.

## Testing OpenTensor

```
python main.py --config ./config/S_4.yaml --mode infer --run_dir $run_dir
```

where `$run_dir` is the subfolders of `./exp`, which contains the model parameters of OpenTensor. This command discovers descomposition of the matrix multiplication tensor with the OpenTensor model.
