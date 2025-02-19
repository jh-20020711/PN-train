# PN-Train

*This implementation of PN-Train uses Python 3.11.5 and PyTorch 2.1.1, tested on Ubuntu 18.04 with an NVIDIA A100 80GB GPU.*

## Requirements

PN-Train requires the following dependencies:

* Python 3.11.5 and its dependencies
* CUDA 11.4 or a later version (**cuDNN** is highly recommended).
* The environment.yaml is available.

## Dataset & Checkpoints
The datasets and checkpoints for Metro-Traffic, Pedestrian are available at [Google Drive](https://drive.google.com/drive/folders/1wZgMhWcLIFn17iKw0SIMr-zKK04zYtI9?usp=drive_link).

Please place the datasets in the `./datasets` folder and checkpoints in the `./checkpoints` folder.


## Model Training
Main input arguments:
- data: specifies which dataset to use
- mode: options are detect, verify, finetune, train or test
- finetune_sample_num: sample size for pattern neuron optimization
- detect_sample_num: sample size for pattern neuron detection
- select_ratio: selection ratio for neurons with high attribution scores
- deactivate_type: the way to deactivate neurons in the network
- learning_rate: learning rate
- finetune_learning_rate: pattern neuron optimizer learning rate
- finetune_epochs: the maximum number of finetune epochs
- seq_len: the length of the observed segment
- pred_len: the length of the prediction segment
- root_path: the root path for the data file
- save_path: the path where output is saved
- checkpoint_path: path to the pretrained model


## Run Pattern Neurons Detector (PND)

Examples for pattern neuron detection:

* Example 1 (PN-Train with default settings on Metro-Traffic dataset):

```
python main.py --method pn --mode detect --data metro-traffic --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

* Example 2 (PN-Train with default settings on Pedestrian dataset):

```
python main.py --method pn --mode detect --data pedestrian --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/pedestrian/checkpoint.pth
```

* Example 3 (PN-Train with arbitrary settings on Pedestrian dataset):
```
python main.py --method pn --mode detect --data pedestrian --detect_sample_num 40 --select_ratio 0.6 --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/pedestrian/checkpoint.pth
```


## Run Pattern Neurons Verifier (PNV)

Examples for pattern neuron verifier:

* Example 1 (PN-Train without deactivating any neurons on Metro-Traffic dataset):

```
python main.py --method pn --mode verify --deactivate_type none --data metro-traffic --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

* Example 2 (PN-Train with pattern neuron deactivation on Metro-Traffic dataset):

```
python main.py --method pn --mode verify --deactivate_type pn_train --data metro-traffic --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

* Example 3 (PN-Train with random neuron deactivation on Metro-Traffic dataset):

```
python main.py --method pn --mode verify --deactivate_type random --data metro-traffic --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

* Example 4 (PN-Train with pattern neuron deactivation under arbitrary settings on Metro-Traffic dataset):

```
python main.py --method pn --mode verify --deactivate_type pn_train --select_ratio 0.6 --data metro-traffic --root_path ./datasets --test_bsz 1 --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

## Pattern Neuron Optimizer (PNO)

Examples for pattern neuron optimizer:

* Example 1 (PN-Train with default settings on Metro-Traffic dataset):

```
python main.py --method pn --mode finetune --data metro-traffic --root_path ./datasets --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

* Example 2 (PN-Train with arbitrary settings on Metro-Traffic dataset):

```
python main.py --method pn --mode finetune --finetune_sample_num 20 --data metro-traffic --root_path ./datasets --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

## Model Training

Examples for training the network:

* Example 1 (PN-Train with default settings on Metro-Traffic dataset):

```
python main.py --method pn --mode train --data metro-traffic --root_path ./datasets --save_path results/ --checkpoint_path ./checkpoints/traffic/checkpoint.pth
```

* Example 2 (PN-Train with default settings on Pedestrian dataset):

```
python main.py --method pn --mode train --data pedestrian --root_path ./datasets --save_path results/ --checkpoint_path ./checkpoints/pedestrian/checkpoint.pth
```

* Example 3 (PN-Train with arbitrary settings on Pedestrian dataset):
```
python main.py --method pn --mode train --data pedestrian --finetune_sample_num 40 --root_path ./datasets --save_path results/ --checkpoint_path ./checkpoints/pedestrian/checkpoint.pth