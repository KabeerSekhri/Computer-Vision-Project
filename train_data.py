import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
DATA_DIR = './augmented_data'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Data Generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize images

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training', color_mode='grayscale'
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', color_mode='grayscale'
)

# Model Definition
def cnn_model():
    num_of_classes = train_gen.num_classes
    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Callbacks
filepath = 'sign_model_alpha_comp.keras'
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]

# Model Training
model = cnn_model()
model.summary()

# Train the Model
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks_list)

# Save the final Model
model.save('sign_model.keras')
