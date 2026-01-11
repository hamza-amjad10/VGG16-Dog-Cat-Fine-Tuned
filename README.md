# VGG16 Dog vs Cat Classifier

This repository contains a TensorFlow/Keras implementation of a **VGG16-based classifier** to distinguish between dog and cat images. The project demonstrates **fine-tuning** a pre-trained model and highlights the effects of **overfitting** on the training dataset.

---

## Features

- Fine-tunes the **VGG16** convolutional base on a custom dataset of dog and cat images.
- Shows overfitting: training accuracy reaches near 100%, while validation accuracy plateaus around 95%.
- Includes strategies for improvement:
  - **Data Augmentation**: Random rotations, flips, zooms to increase dataset variability.
  - **Dropout**: Reduces over-reliance on specific neurons.
  - **Batch Normalization**: Helps in faster and stable convergence.

---

## Dataset

The dataset is downloaded from Kaggle: [Dog and Cat Classification Dataset](https://www.kaggle.com/bhavikjikadara/dog-and-cat-classification-dataset).  
It is split into training and testing sets with an 80:20 ratio.

```
PetImages/
├── train/
│ ├── Cat/
│ └── Dog/
└── test/
├── Cat/
└── Dog/
```

---

## Model Architecture

- **Pre-trained VGG16** (weights from ImageNet)
- **Flatten Layer**
- **Dense Layer**: 256 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation for binary classification

**Fine-tuning:** Only layers from `block5_conv1` onwards are trainable.

---

## Training

- **Optimizer**: RMSprop with learning rate 1e-5
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Image Size**: 150x150
- **Epochs**: 10

**Observations:**

- Training accuracy reaches almost **100%**
- Validation accuracy plateaus around **95%**
- Validation loss increases after some epochs → **sign of overfitting**

---

## How to Improve

1. **Data Augmentation**
```
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   train_datagen = ImageDataGenerator(
       rescale=1./255,
       rotation_range=40,
       width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       fill_mode='nearest'
   )
```
## Conclusion

This project is a demonstration of transfer learning and fine-tuning using VGG16. It clearly shows overfitting on a small dataset and highlights the techniques (augmentation, dropout, batch normalization) that can help improve generalization.


## Author

Hamza Amjad
AI, Machine Learning, NLP & Deep Learning Enthusiast
