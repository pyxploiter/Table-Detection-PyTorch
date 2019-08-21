# Table-Detection-PyTorch

## Getting Started

### Setup
```
$ git clone https://github.com/xploiter-projects/Table-Detection-PyTorch.git
$ cd Table-Detection-PyTorch
$ pip install -r requirements.txt
```

### Training
```
usage: train.py [-h] [-p TRAIN_IMAGES_PATH] [-l TRAIN_LABELS_PATH] [--hf] [-e NUM_EPOCHS]
                [--cf CHECK_FREQ] [-o OUTPUT_WEIGHT_PATH]
                [-i INPUT_WEIGHT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -p TRAIN_IMAGES_PATH, --path TRAIN_IMAGES_PATH
                        Path to training data images.
  -l TRAIN_LABELS_PATH, --label TRAIN_LABELS_PATH
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

```example: python train.py -p ./data/train -l ./data/train.csv -e 70 --cf 10```

### Testing
```
usage: infer.py [-h] [-p TEST_IMAGES_PATH] [-c CHECKPOINT_PATH]

optional arguments:
  -h, --help           show this help message and exit
  -p TEST_IMAGES_PATH         Path to test data images.
  -c CHECKPOINT_PATH  Input checkpoint file path.
```

```example: python infer.py -p ./data/test -c saved_model/model_ep60.pth```

### Evaluating

```example: python evaluate.py ./data/test.csv evaluations/predictions.csv ./data/images/ --ocr ./data/torchocr/```
