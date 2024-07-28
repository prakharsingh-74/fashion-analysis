# PREVIOUS MODEL WITH THE CODE -->

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Step 1: Loading the CSV files
train_df = pd.read_csv('fashion_mnist/fashion-mnist_train.csv')
test_df = pd.read_csv('fashion_mnist/fashion-mnist_test.csv')

# Step 2: Exploration of the Dataset
# Display the first few rows of the training and test DataFrames
print("Training DataFrame:")
print(train_df.head())

print("\nTesting DataFrame:")
print(test_df.head())

# summary statistics of the training and testing DataFrames
print("\nTraining DataFrame Info:")
print(train_df.info())

print("\nTesting DataFrame Info:")
print(test_df.info())

# Checking for null values in the training and test DataFrames
print("\nMissing values in Training DataFrame:")
print(train_df.isnull().sum())

print("\nMissing values in Testing DataFrame:")
print(test_df.isnull().sum())

# Display the distribution of labels in the training and test sets
print("\nLabel distribution in Training DataFrame:")
print(train_df['label'].value_counts())

print("\nLabel distribution in Test DataFrame:")
print(test_df['label'].value_counts())

# Step 3: Data Cleaning and Preprocessing
# Extract labels and images from the DataFrames
training_labels = train_df['label'].values
training_images = train_df.drop(columns=['label']).values
testing_labels = test_df['label'].values
testing_images = test_df.drop(columns=['label']).values

# Reshape the data
training_images = train_images.reshape((-1, 28, 28))
testing_images = test_images.reshape((-1, 28, 28))

# Normalize the images to a range of 0 to 1
training_images = train_images / 255.0
testing_images = test_images / 255.0

# Add a single channel dimension
training_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
testing_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Step 4: Building the CNN model
input_shape = (28, 28, 1)
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 5: Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Training the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Step 7: Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Step 8: Visualizing the training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()



## NOW USING THESE METHODS ONLY FOR THE PRACTICING

Data augmentation -> help to immprove the robustness of the model by artificially increasing the size of your training dataset

Model improvement -> using different architectures, adding more layers or trying different types of layers like dropout to prevent overfitting.

hyperparameter tuning -> using libraries like keras to get best hyperparameters


and done many more things for practicing.
