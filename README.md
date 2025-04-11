# 🧠 Brain Tumor Detection Using CNN 🔬

## 🌟 Overview

This project presents a **deep learning-based approach** for detecting brain tumors using **Convolutional Neural Networks (CNNs)**. 🏥 The model is trained on MRI images and classifies them into **tumor** (✅ Yes) and **non-tumor** (❌ No) categories. The goal is to provide an **automated and accurate** diagnosis to assist medical professionals in detecting brain tumors at an early stage. ⚕️

## 🚀 Features

- 📸 **Image Preprocessing** using `ImageDataGenerator`
- 🏗 **Deep CNN Architecture** with **Conv2D, MaxPooling, BatchNormalization, Dropout, and Dense layers**
- 🛑 **Early Stopping & Model Checkpointing** to prevent overfitting
- 📊 **Graphical representation** of model performance (accuracy & loss)
- 🎯 **Real-time prediction** of MRI images for tumor detection

---

## 🗂 Dataset

The dataset consists of MRI brain scans categorized into two classes:

- ✅ **Yes**: Images with a brain tumor
- ❌ **No**: Images without a brain tumor

### 📊 Dataset Distribution:

- **Training Set**: 70% 📚
- **Validation Set**: 15% 📖
- **Test Set**: 15% 🎯

### 📁 Folder Structure:

```
/content/train/yes/   # 🧠 MRI images with tumors
/content/train/no/    # 🧠 MRI images without tumors
/content/validation/yes/
/content/validation/no/
/content/test/yes/
/content/test/no/
```

---



## 🔄 CNN Process

The **Convolutional Neural Network (CNN) process** for brain tumor detection consists of the following steps:

1. **🖼 Input Image Processing**:
   - 📏 MRI images are resized to **224x224 pixels**.
   - 🎨 Images are **normalized** by rescaling pixel values between 0 and 1.
   - 🔄 Data augmentation techniques like **rotation, zoom, and flipping** are applied.

2. **🔍 Feature Extraction using Convolutional Layers**:
   - 🏗 The first layer applies **Conv2D filters** to detect edges and features.
   - 📈 **ReLU activation** introduces non-linearity.
   - 🎛 **Batch Normalization** stabilizes training by normalizing inputs.

3. **📉 Dimensionality Reduction using Pooling**:
   - 🏊 **MaxPooling2D** reduces spatial dimensions while retaining important features.
   - ⚡ Makes computations **efficient** and reduces overfitting.

4. **🔗 Fully Connected Layers for Classification**:
   - 🛠 The extracted feature maps are **flattened** into a single vector.
   - 🔢 **Dense layers** process these features and classify MRI images.
   - 🎯 The final layer uses a **Sigmoid activation function** to output probabilities.

5. **📈 Optimization and Training**:
   - 🚀 **Adam optimizer** minimizes binary cross-entropy loss.
   - 🛑 **Early Stopping** prevents overfitting.
   - 💾 **Model Checkpointing** saves the best model.

---

## 🏗 Model Architecture

The CNN model consists of:

1. 🏗 **Conv2D** - Extracts spatial features from MRI images
2. 🔧 **BatchNormalization** - Normalizes inputs to stabilize training
3. 🏊 **MaxPooling2D** - Reduces spatial dimensions
4. 🚫 **Dropout** - Prevents overfitting
5. 🔄 **Flatten** - Converts 2D feature maps into a 1D vector
6. 🔢 **Dense** - Fully connected layers for classification
7. 🎯 **Sigmoid Activation** - Outputs probability of tumor presence

---

## 🏞 Image Preprocessing

MRI images undergo preprocessing to enhance model accuracy:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocessingImage(path):
    image_data = ImageDataGenerator(
        rescale=1/255,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )
    return image_data.flow_from_directory(
        directory=path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
```

---

## 🏋️ Model Training

Model compilation and training process:

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hs = model.fit(
    train_data,
    steps_per_epoch=8,
    epochs=30,
    validation_data=val_data,
    validation_steps=16,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5),
               ModelCheckpoint(filepath="/content/bestmodel.h5", save_best_only=True)]
)
```

---

## 📊 Model Evaluation

Evaluate model performance on test data:

```python
acc = model.evaluate(test_data)
print(f"The accuracy of the model is {acc[1]*100} %")
```

---

## 📈 Performance Visualization

### 📊 Accuracy Graph:

```python
plt.plot(hs.history['accuracy'], label='Train Accuracy')
plt.plot(hs.history['val_accuracy'], label='Validation Accuracy', c='red')
plt.title('Accuracy vs Validation Accuracy')
plt.legend()
plt.show()
```

### 📉 Loss Graph:

```python
plt.plot(hs.history['loss'], label='Train Loss')
plt.plot(hs.history['val_loss'], label='Validation Loss', c='red')
plt.title('Loss vs Validation Loss')
plt.legend()
plt.show()
```

---

## 🔍 Making Predictions on New MRI Images

Predict if an MRI image has a brain tumor:

```python
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

path = "/content/brain-mri-images-for-brain-tumor-detection/no/14 no.jpg"
img = load_img(path, target_size=(224,224))
input_arr = img_to_array(img)/255
input_arr = np.expand_dims(input_arr, axis=0)

pred = model.predict(input_arr)[0][0]
threshold = 0.5
if pred < threshold:
    print("✅ The person does NOT have a brain tumor")
else:
    print("⚠️ The person HAS a brain tumor")
```

---

## 🎯 Results & Conclusion

- ✅ The trained CNN model achieves an accuracy of **80%** on the test dataset.
- 🧠 The model successfully detects brain tumors from MRI images.
- 🚀 Further improvements can be made by **expanding the dataset, tuning hyperparameters, and utilizing Transfer Learning**.

---

## 🔮 Future Enhancements

- 🔄 **Implement Transfer Learning** using pre-trained models (e.g., VGG16, ResNet)
- 🌍 **Develop a web-based interface** for easier accessibility
- 🏆 **Optimize the dataset** to improve generalization

---

## 👥 Contributor

- **Dipendra Raj Bhatt** - Developer & Researcher 👨‍💻

---



