# Avocado_CNN

This project utilizes a Convolutional Neural Network (CNN) to classify images into three different classes. The model is trained on an image dataset where class labels are extracted from file names. Additionally, data augmentation techniques and model visualization are implemented.


## Features

Loading images from subfolders and automatically assigning labels based on file names (.jpg).  
Data preprocessing and augmentation, including random flips, brightness and contrast adjustments, and 90-degree rotations.  
Implementation of a CNN model with multiple convolutional and pooling layers.  
Training the model and evaluating results using accuracy metrics and loss functions.  
Generating and visualizing a confusion matrix to analyze classification errors.  
Visualization of filters in the first convolutional layer and activation maps of the neural network.  

## Technologies Used
- Python  
- TensorFlow/Keras  
- NumPy  
- Matplotlib  
- Sklearn  

## Results Visualization

The project generates several plots to help analyze the model's performance:
   - Confusion matrix, which shows misclassified instances.  
   - Accuracy and loss plots over training epochs.  
   - Visualization of filters in the first convolutional layer.  
   - Activation maps, showing which image features are most important to the model.  

### Possible Improvements
- Implementing transfer learning to improve model performance.  
- Experimenting with different CNN architectures.  
- Optimizing hyperparameters to enhance classification accuracy.  
- Expanding the dataset for better model generalization.

### Model results:

Model accuracy:
![dokladnosc](https://github.com/user-attachments/assets/802542d5-1b9a-471a-b024-d6441c5d48c6)

Model loss:
![strata](https://github.com/user-attachments/assets/b3f7daab-b876-4813-a969-c7f1a43938e2)

Confusion matrix:
![macierz_pomylek](https://github.com/user-attachments/assets/7f6ae718-8686-426e-ab83-c0b9d14d4162)

Map filters:
![filtry](https://github.com/user-attachments/assets/c870f01e-4d68-47b6-9997-f09a73edf541)
