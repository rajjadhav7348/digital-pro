```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```


```python
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```


```python
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)
```


```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

    /usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)



```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```


```python
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_ckpt = ModelCheckpoint('best_model.h5', save_best_only=True)

```


```python
from sklearn.model_selection import train_test_split

# Split training data into train + validation sets
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

# Create augmented data generator for training data only
train_generator = datagen.flow(x_train_split, y_train_split, batch_size=64)

# Train model with validation data explicitly provided
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, model_ckpt]
)

```

    /usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
      self._warn_if_super_not_called()


    Epoch 1/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - accuracy: 0.7073 - loss: 0.8779

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 81ms/step - accuracy: 0.7075 - loss: 0.8775 - val_accuracy: 0.9803 - val_loss: 0.0651
    Epoch 2/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - accuracy: 0.9321 - loss: 0.2287

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m68s[0m 80ms/step - accuracy: 0.9321 - loss: 0.2287 - val_accuracy: 0.9847 - val_loss: 0.0497
    Epoch 3/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - accuracy: 0.9512 - loss: 0.1620

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m68s[0m 81ms/step - accuracy: 0.9512 - loss: 0.1620 - val_accuracy: 0.9863 - val_loss: 0.0418
    Epoch 4/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - accuracy: 0.9594 - loss: 0.1350

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m67s[0m 79ms/step - accuracy: 0.9594 - loss: 0.1350 - val_accuracy: 0.9895 - val_loss: 0.0400
    Epoch 5/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - accuracy: 0.9649 - loss: 0.1119

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m67s[0m 79ms/step - accuracy: 0.9649 - loss: 0.1119 - val_accuracy: 0.9887 - val_loss: 0.0359
    Epoch 6/20
    [1m843/844[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 76ms/step - accuracy: 0.9681 - loss: 0.1058

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m67s[0m 79ms/step - accuracy: 0.9681 - loss: 0.1057 - val_accuracy: 0.9917 - val_loss: 0.0291
    Epoch 7/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m66s[0m 79ms/step - accuracy: 0.9715 - loss: 0.0956 - val_accuracy: 0.9905 - val_loss: 0.0322
    Epoch 8/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m66s[0m 78ms/step - accuracy: 0.9750 - loss: 0.0836 - val_accuracy: 0.9887 - val_loss: 0.0362
    Epoch 9/20
    [1m844/844[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m66s[0m 78ms/step - accuracy: 0.9765 - loss: 0.0756 - val_accuracy: 0.9910 - val_loss: 0.0303



```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

```

    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 12ms/step - accuracy: 0.9900 - loss: 0.0268
    
    ✅ Test Accuracy: 99.21%



```python
pred_probs = model.predict(x_test)
pred_labels = np.argmax(pred_probs, axis=1)
```

    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 9ms/step



```python
print("\nClassification Report:\n")
print(classification_report(y_test, pred_labels))
```

    
    Classification Report:
    
                  precision    recall  f1-score   support
    
               0       0.99      1.00      1.00       980
               1       1.00      0.99      1.00      1135
               2       0.99      0.99      0.99      1032
               3       0.99      1.00      0.99      1010
               4       0.99      0.99      0.99       982
               5       1.00      0.99      0.99       892
               6       0.99      0.99      0.99       958
               7       0.99      0.99      0.99      1028
               8       0.99      0.99      0.99       974
               9       0.99      0.99      0.99      1009
    
        accuracy                           0.99     10000
       macro avg       0.99      0.99      0.99     10000
    weighted avg       0.99      0.99      0.99     10000
    



```python
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```


    
![png](output_11_0.png)
    



```python
def plot_examples(preds, actuals, images, correct=True, count=5):
    idxs = np.where(preds == actuals)[0] if correct else np.where(preds != actuals)[0]
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(idxs[:count]):
        plt.subplot(1, count, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray')
        plt.title(f"P: {preds[idx]}, A: {actuals[idx]}")
        plt.axis('off')
    title = "Correct Predictions" if correct else "Incorrect Predictions"
    plt.suptitle(title)
    plt.show()
```


```python
plot_examples(pred_labels, y_test, x_test, correct=True)
plot_examples(pred_labels, y_test, x_test, correct=False)
```


    
![png](output_13_0.png)
    



    
![png](output_13_1.png)
    



```python

```
