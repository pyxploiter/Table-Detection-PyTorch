# Table-Detection-PyTorch

## Getting Started

### Setup
```
$ git clone https://github.com/xploiter-projects/Table-Detection-PyTorch.git
$ cd Table-Detection-PyTorch
$ pip install -r requirements.txt
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ python setup.py build_ext install
```

### Training
```
usage: train.py [-h] [-p TRAIN_PATH] [-l TRAIN_LABEL] [--hf] [-e NUM_EPOCHS]
                [--cf CHECK_FREQ] [-o OUTPUT_WEIGHT_PATH]
                [-i INPUT_WEIGHT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -p TRAIN_PATH, --path TRAIN_PATH
                        Path to training data images.
  -l TRAIN_LABEL, --label TRAIN_LABEL
                        Path to training data labels.
  --hf                  Augment with horizontal flips in training.
                        (Default=false).
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs.
  --cf CHECK_FREQ, --check_freq CHECK_FREQ
                        Checkpoint frequency.
  -o OUTPUT_WEIGHT_PATH, --output_weight_path OUTPUT_WEIGHT_PATH
                        Output path for weights.
  -i INPUT_WEIGHT_PATH, --input_weight_path INPUT_WEIGHT_PATH
                        Input path for weights.
```
### Testing
```
usage: infer.py [-h] [-p TEST_PATH] -c INPUT_CHECKPOINT

optional arguments:
  -h, --help           show this help message and exit
  -p TEST_PATH         Path to test data images.
  -c INPUT_CHECKPOINT  Input checkpoint file path.
```

