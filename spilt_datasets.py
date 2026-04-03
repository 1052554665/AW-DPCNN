# This script splits the dataset into training, validation, and test sets, and saves them in separate directories.

import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import random_split

dataset = datasets.ImageFolder(root='./pngData')

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

save_base_dir = './datasets'
os.makedirs(save_base_dir, exist_ok=True)

train_dir = os.path.join(save_base_dir, 'train')
val_dir = os.path.join(save_base_dir, 'val')
test_dir = os.path.join(save_base_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def save_images(indices, target_dir, dataset):
    for idx in indices:
        img_path, label = dataset.imgs[idx]
        class_name = dataset.classes[label]
        label_dir = os.path.join(target_dir, class_name)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(label_dir, os.path.basename(img_path)))

print("Saving training set...")
save_images(train_dataset.indices, train_dir, dataset)
print("Saving validation set...")
save_images(val_dataset.indices, val_dir, dataset)
print("Saving test set...")
save_images(test_dataset.indices, test_dir, dataset)
print("Datasets have been saved to", save_base_dir)
print(f'Training set size: {len(train_dataset)}')
print(f'Validation set size: {len(val_dataset)}')
print(f'Test set size: {len(test_dataset)}')
