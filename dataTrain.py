import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# Paths
test_dir = "processed_ODIR/test"
resnet_model_path = "resnet_model.pth"
baseline_predictions_path = "baseline_predictions.npy"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test Dataset and DataLoader
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Baseline Predictions and Labels
baseline_predictions_data = np.load(baseline_predictions_path, allow_pickle=True).item()
baseline_preds = baseline_predictions_data['predictions']
baseline_labels = baseline_predictions_data['labels']

# Compute Baseline Metrics Dynamically
baseline_accuracy = accuracy_score(baseline_labels, baseline_preds) * 100
baseline_report = classification_report(baseline_labels, baseline_preds, target_names=test_dataset.classes)
baseline_conf_matrix = confusion_matrix(baseline_labels, baseline_preds)

print(f"Baseline Validation Accuracy: {baseline_accuracy:.2f}%")
print("\nBaseline Classification Report:")
print(baseline_report)

# ====== RESNET18 MODEL ======
# Load the trained ResNet18 model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
resnet_model = torch.load(resnet_model_path)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Evaluate ResNet18 on Test Data
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet_model(images)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

resnet_accuracy = accuracy_score(all_labels, all_preds) * 100
resnet_conf_matrix = confusion_matrix(all_labels, all_preds)
resnet_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

print(f"\nResNet18 Test Accuracy: {resnet_accuracy:.2f}%")
print("\nResNet18 Classification Report:")
print(resnet_report)


# ====== PLOT COMPARISON ======
class ModelComparisonPlotter:
    def __init__(self, resnet_metrics, baseline_metrics):
        self.resnet_metrics = resnet_metrics
        self.baseline_metrics = baseline_metrics

    def plot_accuracy_comparison(self):
        plt.figure(figsize=(8, 6))
        plt.bar(['Baseline (LogReg)', 'ResNet18'],
                [self.baseline_metrics['accuracy'], self.resnet_metrics['accuracy']], color=['blue', 'orange'])
        plt.title('Validation Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.show()

    def plot_classification_comparison(self):
        labels = list(self.resnet_metrics['report'].keys())[:-3]  # Avoid overall metrics like 'accuracy', 'macro avg', etc.
        baseline_f1 = [self.baseline_metrics['report'][label]['f1-score'] for label in labels]
        resnet_f1 = [self.resnet_metrics['report'][label]['f1-score'] for label in labels]

        x = range(len(labels))
        plt.figure(figsize=(10, 6))
        plt.bar(x, baseline_f1, width=0.4, label='Baseline (LogReg)', align='center', color='blue')
        plt.bar([p + 0.4 for p in x], resnet_f1, width=0.4, label='ResNet18', align='center', color='orange')
        plt.xticks([p + 0.2 for p in x], labels, rotation=45)
        plt.title('F1-Score Comparison by Class')
        plt.ylabel('F1-Score')
        plt.legend()
        plt.tight_layout()
        plt.show()


# Prepare Metrics for Plotting
baseline_metrics = {
    'accuracy': baseline_accuracy,
    'conf_matrix': baseline_conf_matrix,
    'report': classification_report(baseline_labels, baseline_preds, target_names=test_dataset.classes, output_dict=True)
}

resnet_metrics = {
    'accuracy': resnet_accuracy,
    'conf_matrix': resnet_conf_matrix,
    'report': classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
}

plotter = ModelComparisonPlotter(resnet_metrics, baseline_metrics)
plotter.plot_accuracy_comparison()
plotter.plot_classification_comparison()