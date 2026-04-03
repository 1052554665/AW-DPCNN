# This code generates a JSON file containing the class indices and class names from the dataset.
import json
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='./datasets/train', transform=transform)

class_list = [
    {'index': idx, 'name': name}
    for idx, name in enumerate(train_dataset.classes)
]

with open('class_indices_columns.json', 'w') as f:
    json.dump(class_list, f, indent=4)
