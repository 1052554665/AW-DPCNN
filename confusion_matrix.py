# The confusion matrix is a table that is used to describe the performance of a classification model on a set of test data for which the true values are known.
# The confusion matrix is a 2D array that contains the counts of true positives, true negatives, false positives, and false negatives.
# The confusion matrix can be used to calculate various performance metrics such as accuracy, precision, recall, and F1 score.
# In this code, we will use the confusion matrix to evaluate the performance of our AlexNet model on the test set.
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = datasets.ImageFolder(root='./datasets/val', transform=transform)

model = AlexNet(num_classes=len(val_dataset.classes)).to(device)
model.load_state_dict(torch.load('./alexnet.pth'))
model = model.to(device)

test_dataset = datasets.ImageFolder(root='./datasets/test', transform=transform)
test_loader = DataLoader(test_dataset)

model.eval()

all_preds = []
all_labels = []


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
cm = confusion_matrix(all_labels, all_preds)

print(cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(val_dataset.classes))
plt.xticks(tick_marks, val_dataset.classes, rotation=45)
plt.yticks(tick_marks, val_dataset.classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
