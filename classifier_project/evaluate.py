import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Config
DATA_DIR = "../Images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_val_data():
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True # Must be True to mix classes before splitting, same as train.py
    )
    return val_ds, val_ds.class_names

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("Saved roc_curve.png")

def main():
    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        print("Model not found. Train first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading validation data...")
    val_ds, class_names = load_val_data()
    
    print("Running predictions...")
    y_true = []
    y_pred_probs = []
    
    # Iterate over dataset to get labels and predictions
    # Note: val_ds is not shuffled, so order should match if we process batch by batch
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Save report to text file
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)
    
    print("Generating ROC Curve...")
    plot_roc_curve(y_true, y_pred_probs)

if __name__ == "__main__":
    main()
