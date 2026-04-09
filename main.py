import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator#tensorflow is the liburary that lets us create and train nural networkd
from PIL import Image
import os
import shutil
import random


TRAIN_MODEL = False  # change to True when you want to retrain

# Paths
DATA_DIR = 'data'#images stored here
MODEL_DIR = 'models'#model stored here
test_folder = 'Test'#images you want to see if dog or cat stored here
os.makedirs(MODEL_DIR, exist_ok=True)

# Number of images per class you want to use
IMAGES_PER_CLASS = 500  #number of images I want it to check
# ----------- Test THE MODEL
def predict_image(model, img_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        confidence = pred * 100
        print(f"{img_path} → Dog ({confidence:.1f}%)")
    else:
        confidence = (1 - pred) * 100
        print(f"{img_path} → Cat ({confidence:.1f}%)")

# --------- SCRUB IMAGES ---------
def scrub_images(folder):
    removed = 0
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()  # Verify image is readable
            except Exception:
                print(f"Removing corrupted image: {path}")
                os.remove(path)
                removed += 1
    print(f"Scrubbing complete. {removed} images removed.")


# --------- LIMIT IMAGES PER CLASS ---------
def limit_images(folder, max_per_class):#only use the amount I want it to use for training data
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if len(images) > max_per_class:
                to_remove = random.sample(images, len(images) - max_per_class)#random image so training data isnt only the first or last x images
                for file in to_remove:
                    os.remove(os.path.join(class_path, file))
                print(f"Reduced {class_name} to {max_per_class} images.")


# --------- SIMPLE CNN MODEL ---------
model = tf.keras.models.Sequential([#image pattern model
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


if TRAIN_MODEL:
    scrub_images(DATA_DIR)
    limit_images(DATA_DIR, IMAGES_PER_CLASS)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(150,150),#resizes all images to this size
        batch_size=32,#loads this amount of images at a time
        class_mode='binary',#since 2 options cats vs dogs 
        subset='training' #splits data into 80 for training and 20 for validation
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=25
    )
    model.save(os.path.join(MODEL_DIR, 'cat_dog_classifier.h5'))
    print("Model saved!")
else:
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'cat_dog_classifier.h5'))
    print("Model loaded!")


# --------- TEST THE MODEL ---------

for file in os.listdir(test_folder):
    path = os.path.join(test_folder, file)
    predict_image(model, path)