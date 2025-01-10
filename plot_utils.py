import matplotlib.pyplot as plt


class ModelPerformancePlotter:
    def __init__(self, history, baseline_history=None):
        """
        Initialize the plotter with training histories.

        :param history: Dictionary or history object containing training and validation metrics for the trained model.
                        Expected keys: 'train_accuracy', 'val_accuracy', 'train_loss', 'val_loss'
        :param baseline_history: Dictionary or history object containing metrics for the baseline model (optional).
        """
        self.history = history
        self.baseline_history = baseline_history

    def plot_accuracy(self):
        """
        Plot accuracy over epochs for training and validation.
        If baseline history is provided, compare with the baseline model.
        """
        epochs = range(1, len(self.history['train_accuracy']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_accuracy'], label='Train Accuracy')
        plt.plot(epochs, self.history['val_accuracy'], label='Validation Accuracy')

        if self.baseline_history:
            plt.plot(epochs, self.baseline_history['train_accuracy'], linestyle='--', label='Baseline Train Accuracy')
            plt.plot(epochs, self.baseline_history['val_accuracy'], linestyle='--', label='Baseline Val Accuracy')

        plt.title('Model Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_loss(self):
        """
        Plot loss over epochs for training and validation.
        If baseline history is provided, compare with the baseline model.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')

        if self.baseline_history:
            plt.plot(epochs, self.baseline_history['train_loss'], linestyle='--', label='Baseline Train Loss')
            plt.plot(epochs, self.baseline_history['val_loss'], linestyle='--', label='Baseline Val Loss')

        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def compare_models(self):
        """
        Plot side-by-side comparisons of accuracy and loss for the trained and baseline models.
        """
        if not self.baseline_history:
            print("Baseline history not provided. Skipping comparison.")
            return

        epochs = range(1, len(self.history['train_accuracy']) + 1)

        # Accuracy comparison
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['val_accuracy'], label='Trained Model Val Accuracy')
        plt.plot(epochs, self.baseline_history['val_accuracy'], linestyle='--', label='Baseline Val Accuracy')
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        # Loss comparison
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['val_loss'], label='Trained Model Val Loss')
        plt.plot(epochs, self.baseline_history['val_loss'], linestyle='--', label='Baseline Val Loss')
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
