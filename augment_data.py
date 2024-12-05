import cv2
import numpy as np
import os
import random

DATA_DIR = './data'
AUGMENTED_DIR = './augmented_data'

if not os.path.exists(AUGMENTED_DIR):
    os.makedirs(AUGMENTED_DIR)

# Define augmentation functions
def augment_image(image):
    aug_type = random.choice(["noise", "rotation", "flip"])
    
    if aug_type == "noise":
        # Ensure noise matches the shape of the image
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        image = cv2.add(image, noise)
        return image, "noise"

    elif aug_type == "rotation":
        rows, cols, _ = image.shape
        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
        return image, "rotation"

    elif aug_type == "flip":
        flip_code = random.choice([-1, 0, 1])  # Horizontal, Vertical, or Both
        image = cv2.flip(image, flip_code)
        return image, "flip"

    return image, "none"

# Apply augmentations to each image in the dataset
def augment_dataset(data_dir, augmented_dir):
    for gesture in os.listdir(data_dir):
        gesture_dir = os.path.join(data_dir, gesture)
        if not os.path.isdir(gesture_dir):  # Skip files like `.DS_Store`
            continue

        augmented_gesture_dir = os.path.join(augmented_dir, gesture)
        if not os.path.exists(augmented_gesture_dir):
            os.makedirs(augmented_gesture_dir)
        
        for image_file in os.listdir(gesture_dir):
            img_path = os.path.join(gesture_dir, image_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            for i in range(5):  # Generate 5 augmentations per image
                aug_image, aug_desc = augment_image(image)
                aug_filename = f"{os.path.splitext(image_file)[0]}_aug_{i}_{aug_desc}.jpg"
                cv2.imwrite(os.path.join(augmented_gesture_dir, aug_filename), aug_image)


# Augment the dataset
augment_dataset(DATA_DIR, AUGMENTED_DIR)
print("Data augmentation complete! Augmented images saved to:", AUGMENTED_DIR)
