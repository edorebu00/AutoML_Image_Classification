# AutoML Image Classification

An end-to-end image classification pipeline developed using TensorFlow and AutoKeras.

## Features

- AutoML image classification
- Batch image prediction
- Automatic preprocessing
- Top-3 predictions
- CSV export
- Error handling
- Prediction statistics
- Image visualization

## Technologies

- Python
- TensorFlow
- AutoKeras
- NumPy
- Matplotlib

## Project Structure

```
AutoML_Image_Classification/
│
├── images/
├── saved_models/
├── predict.py
├── train_autokeras.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_autokeras.py
```

## Prediction

```bash
python predict.py
```

The script automatically:

- scans all images in the `images` folder
- preprocesses them
- predicts the class
- shows the Top-3 predictions
- exports the results to `risultati.csv`

## Dataset

The model is trained on the CIFAR-10 dataset.

## Author

Edoardo Rebughini