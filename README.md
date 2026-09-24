# Oxford-IIIT Pet Breed Classification

## Project Overview

This project is an image classification system that identifies the breed of a pet from an image using the **Oxford-IIIT Pet Dataset**.

The dataset contains 37 different cat and dog breeds. Three approaches were implemented and compared:

* Custom CNN
* ResNet18 using Transfer Learning
* Fine-Tuned ResNet18

A simple GUI is also included, allowing users to upload a pet image and predict its breed.

## Dataset

The project uses the Oxford-IIIT Pet Dataset.

* Total images: 7,393
* Number of breeds: 37
* Cat breeds: 12
* Dog breeds: 25

The dataset was downloaded from the Oxford VGG website.

## Project Workflow

1. Dataset loading
2. Dataset inspection
3. Exploratory Data Analysis
4. Image dimension analysis
5. Data preprocessing
6. Data augmentation
7. Train/validation/test splitting
8. Custom CNN training
9. Transfer learning using ResNet18
10. Fine-tuning ResNet18
11. Model evaluation
12. Error analysis
13. Model saving
14. GUI-based prediction

## Models

### Custom CNN

A CNN was built from scratch using convolutional layers, ReLU activation, max pooling, dropout, and fully connected layers.

Test Accuracy: 16.32%

### ResNet18 Transfer Learning

A pre-trained ResNet18 model was used. The pre-trained layers were frozen and a new classification layer was trained for the 37 pet breeds.

Test Accuracy: 85.66%

### Fine-Tuned ResNet18

The final convolutional block and classification layer were unfrozen and trained with a smaller learning rate to adapt the pre-trained model to the pet-breed dataset.

Test Accuracy: 89.18%

## Model Comparison

| Model                      | Test Accuracy |
| -------------------------- | ------------: |
| Custom CNN                 |        16.32% |
| ResNet18 Transfer Learning |        85.66% |
| Fine-Tuned ResNet18        |        89.18% |

## Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Error analysis was also performed to examine incorrectly classified images and understand difficult breed predictions.

## GUI

A separate Tkinter-based GUI is included in `CNN_GUI.py`.

The GUI allows the user to:

1. Upload a pet image
2. Display the image
3. Predict the pet breed
4. Display the predicted breed and confidence score

## Project Structure

```text
Oxford-Pet-CNN/
├── Oxford_Pet_CNN.ipynb
├── CNN_GUI.py
├── README.md
└── models/
    └── resnet_pet.pth
```

## Requirements

The main libraries used in this project are:

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow
* Tkinter

## How to Run

### Notebook

Open `Oxford_Pet_CNN.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab and run the cells in order.

The notebook downloads the dataset and performs the complete training and evaluation process.

### GUI

Make sure the trained model is present at:

```text
models/resnet_pet.pth
```

Then run:

```bash
python CNN_GUI.py
```

The GUI will open and allow you to upload a pet image for breed prediction.

## Conclusion

This project demonstrates image classification using a Custom CNN, transfer learning, and fine-tuning with ResNet18.

The fine-tuned ResNet18 achieved
