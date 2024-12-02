# Computer vision project

## Environment Setup

### Installation of Miniconda
For the initial setup, please follow the instructions for downloading and installing Miniconda available at the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Environment Configuration
1. **Creating the Environment**: Navigate to the code directory in your terminal and create the environment using the provided `.yml` file by executing:

        conda env create -f deepsatmodels_env.yml

2. **Activating the Environment**: Activate the newly created environment with:

        source activate deepsatmodels

3. **PyTorch Installation**: Install the required version of PyTorch along with torchvision and torchaudio by running:

        conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly

## Configs

### Data > model > data model configuration

```
MODEL:
  architecture: "ConvBiRNN"
  backbone: "ConvGRU"
  shape_pattern: 'NTHWC'
  img_res: 48
  max_seq_len: 16
  num_channels: 14
  num_classes: 20
  dropout: 0.

DATASETS:
  train:
    dataset: 'T31TFM_1618'
    label_map:  'labels_20k2k'
    max_seq_len: 16
    batch_size: 32
    extra_data:
    num_workers: 4  # 10
  eval:
    dataset: 'T31TFM_1618'
    label_map:  'labels_20k2k'
    max_seq_len: 16
    batch_size: 32
    extra_data:
    num_workers: 4  # 10

SOLVER:
  num_epochs: 50
  loss_function: masked_cross_entropy
  class_weights:
  lr_scheduler:
  lr_base: 0.0001
  lr_decay: 0.975
  reset_lr: True   # resets lr to base value when loading pretrained model

CHECKPOINT:
  load_from_checkpoint:
  partial_restore: False
  save_path: "./models/saved_models/France/MTLCC_BiconvGRU"
  train_metrics_steps: 2000
  eval_steps: 4000
  save_steps: 8000

```

## Data

### Data loader | Load data from location

torch.utils.data.DataLoader is a PyTorch class used to load datasets in a more manageable way, especially when working with large datasets that can't fit into memory all at once. It provides features like batching, shuffling, and parallel data loading using multiple workers.

Key Features:
1. Batching: It can divide the dataset into batches, which is useful when training models, as it processes data in chunks instead of loading everything at once.
2. Shuffling: It can shuffle the data each time it loads a batch, which helps to avoid overfitting by ensuring that the model doesn’t learn any particular order of data.
3. Parallel Loading: It can use multiple workers (threads or processes) to load data in parallel, which speeds up the data loading process.
4. Custom Sampling: You can create custom data samplers to control the way data is fetched (e.g., weighted sampling, random sampling).


```python

from torch.utils.data import DataLoader, TensorDataset


dataset = TensorDataset(data, labels)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

return dataloader

```


### data_transformer | Transform data





### Matrices

Model evaluation matrices




### Models

Models

### Train and evaluation

#### Semantic Segmentation

To train for semantic segmentation, execute the following command, replacing `**` with the appropriate directory names:

Train ** data to TSViT model.
```
python train_and_eval/segmentation_training_transf.py --config configs/France/TSViT.yaml --device 0
```

#### Object Classification

For object classification tasks, use the command below:

```

python train_and_eval/classification_train_transf.py --config configs/France/TSViT_cls.yaml --device 0

```
