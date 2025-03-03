import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load dataset from CSV
csv_file = "path/to/your.csv"  # Update with your actual CSV file path
df = pd.read_csv(csv_file)

# Assuming the last column is the label
X = df.iloc[:, :-1].values  # Pixel values
y = df.iloc[:, -1].values   # Labels

# Normalize pixel values (0-255 → 0-1)
X = X / 255.0

# Reshape images (assuming 28x28 grayscale, modify for your case)
X = X.reshape(-1, 28, 28, 1)  # Change (28,28) to actual image size

# Convert labels to categorical
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Pre-trained MobileNetV2 Model
base_model = MobileNetV2(input_shape=(28, 28, 3), include_top=False, weights='imagenet')

# Convert grayscale to 3-channel (if using MobileNetV2)
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Freeze base model layers (optional)
base_model.trainable = False

# Build Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjust output layer based on classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save Model
model.save("image_classifier_transfer_learning.h5")

# Plot Training Performance
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
