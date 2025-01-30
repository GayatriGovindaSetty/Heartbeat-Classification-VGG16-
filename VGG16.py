import os
import glob
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import pandas as pd

# Audio
import librosa
import librosa.display

# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import class_weight

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.applications import VGG16  # Updated to VGG16
from keras._tf_keras.keras.models import Model, Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras._tf_keras.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Preparing Dataset
dataset = []
for folder in ["/content/drive/MyDrive/Stethoscope project/Deekshitha M/new dataset like set_a and set_b/set_a/**", "/content/drive/MyDrive/Stethoscope project/Deekshitha M/new dataset like set_a and set_b/set_b/**"]:
    for filename in glob.iglob(folder):
        if os.path.exists(filename):
            label = os.path.basename(filename).split("_")[0]
            duration = librosa.get_duration(filename=filename)
            if duration >= 3:  # Skip short audio
                slice_size = 3
                iterations = int((duration - slice_size) / (slice_size - 1)) + 1
                initial_offset = (duration - ((iterations * (slice_size - 1)) + 1)) / 2
                if label not in ["Aunlabelledtest", "Bunlabelledtest", "artifact"]:
                    for i in range(iterations):
                        offset = initial_offset + i * (slice_size - 1)
                        label_category = "normal" if label == "normal" else "abnormal"
                        dataset.append({
                            "filename": filename,
                            "label": label_category,
                            "offset": offset
                        })

dataset = pd.DataFrame(dataset)
dataset = shuffle(dataset, random_state=42)
dataset.info()

plt.figure(figsize=(4, 6))
dataset.label.value_counts().plot(kind='bar', title="Dataset Distribution")
plt.show()

train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Visualization
plt.figure(figsize=(20, 10))
idx = 0
for label in dataset.label.unique():
    y, sr = librosa.load(dataset[dataset.label == label].filename.iloc[0], duration=3)
    print(dataset[dataset.label == label].filename.iloc[0])

    # Wave Plot
    idx += 1
    plt.subplot(2, 3, idx)
    plt.title(f"{label} Waveplot")
    librosa.display.waveshow(y, sr=sr)

    # Mel Spectrogram
    idx += 1
    plt.subplot(2, 3, idx)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.title(f"{label} Mel Spectrogram")

    # MFCC
    idx += 1
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    plt.subplot(2, 3, idx)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title(f"{label} MFCC")
plt.show()

# Feature Extraction
def extract_features(audio_path, offset):
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    return mfccs

x_train, x_test = [], []
for idx in tqdm(range(len(train))):
    x_train.append(extract_features(train.filename.iloc[idx], train.offset.iloc[idx]))
for idx in tqdm(range(len(test))):
    x_test.append(extract_features(test.filename.iloc[idx], test.offset.iloc[idx]))

x_train, x_test = np.asarray(x_train), np.asarray(x_test)
print("X Train:", x_train.shape)
print("X Test:", x_test.shape)

# Encode Labels
encoder = LabelEncoder()
encoder.fit(train.label)

y_train = encoder.transform(train.label)
y_test = encoder.transform(test.label)

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights)) # Convert to dictionary

# Reshape Data and One-Hot Encode Labels
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# VGG16 Model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(x_train.shape[1], x_train.shape[2], 3))  # Updated to VGG16
base_model.trainable = False  # Freeze layers

model = Sequential([
    tf.keras.layers.Conv2D(3, (3, 3), padding='same', input_shape=(x_train.shape[1], x_train.shape[2], 1)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(encoder.classes_), activation='softmax')
])

model.summary()

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=300,
                    validation_data=(x_test, y_test),
                    class_weight=class_weights_dict,
                    shuffle=True)

# Plot Loss and Accuracy
plt.figure(figsize=[14, 10])
plt.subplot(211)
plt.plot(history.history['loss'], '#d62728', linewidth=3.0)
plt.plot(history.history['val_loss'], '#1f77b4', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.subplot(212)
plt.plot(history.history['accuracy'], '#d62728', linewidth=3.0)
plt.plot(history.history['val_accuracy'], '#1f77b4', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

# Evaluate Model
scores = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {scores[0]}")
print(f"Test Accuracy: {scores[1]}")

# Classification Report
predictions = model.predict(x_test, verbose=1)
y_true, y_pred = [], []
classes = encoder.classes_
for idx, prediction in enumerate(predictions):
    y_true.append(classes[np.argmax(y_test[idx])])
    y_pred.append(classes[np.argmax(prediction)])

print(classification_report(y_pred, y_true))

# Save Model
model.save("heartbeat_classifier_vgg16.h5")
