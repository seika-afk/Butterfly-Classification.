
# 🦋 Butterfly Species Image Classification

This project builds a Convolutional Neural Network (CNN) model to classify butterfly species from images using TensorFlow/Keras.

---

## 📂 Dataset

- Dataset source: [Kaggle – Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- Contains ~6000 labeled images
- 75 butterfly species (classes)

---

## 🛠️ Setup

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
kaggle datasets download phucthaiv02/butterfly-image-classification
unzip butterfly-image-classification.zip
```

---

## 🧹 Preprocessing

- Images are loaded using OpenCV
- Converted to RGB format
- Labels are extracted and encoded using `LabelEncoder`
- Optional: Data augmentation using `ImageDataGenerator` (currently commented out)

```python
X = []
for filename in df['filename']:
    img = cv2.imread(os.path.join(image_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(img)
```

---

## ⚙️ Compilation & Training

```python
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)
```

---

## 🧠 Notes

- You can improve performance and robustness with:
  - Transfer Learning (e.g., MobileNet, ResNet)
  - Regularization and Dropout tweaks
  - Data augmentation (rotation, zoom, shift, flip)
---

## 📜 License

This project is for educational and research purposes only. Dataset license is subject to Kaggle’s terms.
