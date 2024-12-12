import cv2
import numpy as np
import os
import random

DATA_DIR = './data'
AUGMENTED_DIR = './augmented_data'

if not os.path.exists(AUGMENTED_DIR):
    os.makedirs(AUGMENTED_DIR)

augmentations_list = ["noise", "rotation", "flip","brightness","contrast","hue_sat","shearing","scaling",
                      "translation","gaussian", "salt_pep","cutout","perspective",]

# Define augmentation functions
def augment_image(image):
    aug_type = random.choice(augmentations_list)
    
    if aug_type == "noise":
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
        flip_code = random.choice([-1, 0, 1])
        image = cv2.flip(image, flip_code)
        return image, "flip"

    elif aug_type == "brightness":
        factor = random.uniform(0.5, 1.5)
        image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return image, "brightness"

    elif aug_type == "contrast":
        factor = random.uniform(0.5, 1.5)
        image = cv2.addWeighted(image, factor, np.zeros_like(image), 0, 0)
        return image, "contrast"

    elif aug_type == "hue_sat":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-30, 30), 0, 255)
        hsv[..., 0] = np.clip(hsv[..., 0] + random.randint(-15, 15), 0, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image, "hue_sat"

    elif aug_type == "shearing":
        rows, cols, _ = image.shape
        shear_factor = random.uniform(0.2, 0.5)
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M, (cols, rows))
        return image, "shearing"

    elif aug_type == "scaling":
        scale_factor = random.uniform(0.7, 1.3)
        rows, cols, _ = image.shape
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        return image, "scaling"

    elif aug_type == "translation":
        rows, cols, _ = image.shape
        tx, ty = random.randint(-25, 25), random.randint(-25, 25)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (cols, rows))
        return image, "translation"

    elif aug_type == "gaussian":
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image, "gaussian"

    elif aug_type == "salt_pep":
        prob = 0.05
        noisy = np.copy(image)
        num_salt = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        num_pepper = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy, "salt_pep"

    elif aug_type == "cutout":
        rows, cols, _ = image.shape
        size = random.randint(2, 5)
        x, y = random.randint(0, cols - size), random.randint(0, rows - size)
        image[y:y + size, x:x + size] = 0
        return image, "cutout"

    elif aug_type == "perspective":
        rows, cols, _ = image.shape
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        pts2 = np.float32([[random.randint(0, 50), random.randint(0, 50)],
                           [cols - random.randint(0, 50), random.randint(0, 50)],
                           [random.randint(0, 50), rows - random.randint(0, 50)],
                           [cols - random.randint(0, 50), rows - random.randint(0, 50)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (cols, rows))
        return image, "perspective"

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