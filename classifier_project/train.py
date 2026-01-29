import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Config
DATA_DIR = "../Images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 # Initial epochs
FINE_TUNE_EPOCHS = 10 # Additional epochs
LEARNING_RATE = 1e-3 # Faster initial learning
FINE_TUNE_LR = 1e-5 # Slower fine-tuning

def load_data():
    """
    Load image dataset from directory and split into train/validation sets.
    Returns:
        tuple: (train_ds, val_ds, class_names)
    """
    print("Loading data...")
    # Use validation split for training/validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    
    class_names = train_ds.class_names
    print(f"Class names: {class_names}")

    # Optimize datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

def get_class_weights(data_dir):
    """
    Calculate class weights to handle imbalance in the dataset.
    Args:
        data_dir (str): Path to the dataset directory.
    Returns:
        dict: Class weights map {class_index: weight}
    """
    # Calculate class weights manually since dataset is batched
    # Assumes folders are class names
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_counts = {}
    total_samples = 0
    
    y = []
    
    # We need to map class names to 0 and 1. 
    # image_dataset_from_directory sorts alphabetically by default.
    # AI -> 0, Real -> 1 (or vice versa, check print output)
    
    print("Computing class weights...")
    # Quick count
    counts = []
    for cls in classes:
        p = os.path.join(data_dir, cls)
        count = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
        counts.append(count)
        print(f"Class '{cls}': {count} images")
    
    total = sum(counts)
    # Binary classification weights: total / (2 * n_class)
    weights = {}
    for i, count in enumerate(counts):
        weights[i] = total / (2.0 * count)
    
    print(f"Class Weights: {weights}")
    return weights

def build_model():
    """
    Construct the EfficientNetB0-based model with a custom classification head.
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    print("Building model with EfficientNetB0...")
    
    # Data Augmentation Wrapper
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    
    # Preprocessing for EfficientNet (included in model usually, but good to ensure)
    # EfficientNet expects 0-255 inputs if include_preprocessing is True in modern Keras, 
    # but let's stick to explicit rescaling if needed. 
    # Actually EfficientNetV2/B0 in tf.keras.applications handles this.
    # We will use the built-in preprocess_input if needed, but EfficientNet usually handles raw [0, 255]
    # Check documentation: EfficientNet models expect floats in [0, 255].
    
    base_model = applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="AI_Classifier")
    return model

def compile_model(model):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_history.png')
    print("Saved training_history.png")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    train_ds, val_ds, class_names = load_data()
    class_weights = get_class_weights(DATA_DIR)
    
    model = build_model()
    model = compile_model(model)
    model.summary()
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        'best_model.keras', 
        monitor='val_accuracy', 
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
    
    print("Starting initial training (Frozen Base)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )
    
    # Fine-Tuning Phase
    print("\nUnfreezing base model for Fine-Tuning...")
    base_model = model.layers[1] # Input is 0, Augmentation is maybe wrapped?
    # Let's find the efficientnet layer.
    # In build_model: inputs -> data_augmentation -> base_model -> ...
    # model.layers structure: Input, Sequential(aug), EfficientNet, GlobalAvg, BN, Dropout, Dense
    
    # We can just iterate layers to find the base
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name:
            layer.trainable = True
            print(f"Unfrozen layer: {layer.name}")
    
    # Recompile with low LR
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    
    print("Starting Fine-Tuning...")
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=history.epoch[-1],
        epochs=total_epochs,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )
    
    # Combine histories if needed, or just plot fine-tuning
    # To plot full history, we concat
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    
    # Plot custom
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=EPOCHS, label='Start Fine Tuning', color='green', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=EPOCHS, label='Start Fine Tuning', color='green', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_history.png')
    print("Saved training_history.png")
    
    model.save('final_model.keras')
    print("Model saved.")

if __name__ == "__main__":
    main()
