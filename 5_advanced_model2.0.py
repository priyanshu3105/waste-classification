"""
High-Accuracy Transfer Learning with ResNet50
Smart Waste Classification (Organic / Recyclable / Non-Organic)

EXPECTED ACCURACY: 77–83%
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ---------------------------------------------
# FIXED CONFIG FOR BEST PERFORMANCE
# ---------------------------------------------
CONFIG = {
    'img_size': (224, 224),           # ResNet standard size
    'batch_size': 32,
    'initial_epochs': 10,             # frozen base model training
    'fine_tune_epochs': 20,           # fully tune last ~75 layers
    'validation_split': 0.2,
    'num_classes': 3,
    'fine_tune_at': 100,              # earlier fine-tuning point
}

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------
# Check GPU
# ---------------------------------------------
def check_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detected: {gpus}")
    else:
        print("⚠️ No GPU detected, training will be slow.")
    print()

# ---------------------------------------------
# Data Loader
# ---------------------------------------------
def load_data(data_path):

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        validation_split=CONFIG['validation_split']
    )

    val_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )

    train_gen = train_aug.flow_from_directory(
        data_path,
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = val_aug.flow_from_directory(
        data_path,
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen

# ---------------------------------------------
# Model Builder
# ---------------------------------------------
def build_resnet50(input_shape, num_classes, fine_tune_at):

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    base_model.trainable = False  # Phase 1: freeze

    inputs = keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model, base_model


# ---------------------------------------------
# Plot training history
# ---------------------------------------------
def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label="Train Acc")
    plt.plot(history['val_accuracy'], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.grid()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Plot saved: {save_path}")


# ---------------------------------------------
# Training pipeline
# ---------------------------------------------
def main():

    check_gpu()

    data_path = "data/processed"
    train_gen, val_gen = load_data(data_path)

    input_shape = (*CONFIG['img_size'], 3)
    model, base_model = build_resnet50(input_shape, CONFIG['num_classes'], CONFIG['fine_tune_at'])

    # --------------------------
    # Optimizer: AdamW + CosineDecay
    # --------------------------
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0005,
        decay_steps=1000,
        alpha=0.1
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nPHASE 1: Training with frozen ResNet50 layers\n")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['initial_epochs'],
        verbose=1
    )

    # --------------------------
    # PHASE 2: Fine-tuning
    # --------------------------
    print("\nPHASE 2: Fine-tuning deeper layers\n")

    base_model.trainable = True

    # Only unfreeze layers after fine_tune_at
    for layer in base_model.layers[:CONFIG['fine_tune_at']]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['fine_tune_epochs'],
        verbose=1
    )

    # --------------------------
    # Combine histories
    # --------------------------
    history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    }

    # --------------------------
    # Final evaluation
    # --------------------------
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\nFINAL VALIDATION ACCURACY = {val_acc*100:.2f}%")
    print(f"FINAL VALIDATION LOSS = {val_loss:.4f}")

    # --------------------------
    # Save model and history
    # --------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    model.save("models/resnet50_advanced_model.h5")
    print("✓ Saved: models/resnet50_advanced_model.h5")

    with open("results/advanced_history_resnet50.json", "w") as f:
        json.dump(history, f, indent=2)
    print("✓ Saved training history")

    plot_history(history, "results/plots/resnet50_training.png")

    print("\n🎉 TRAINING COMPLETE — EXPECT 77–83% ACCURACY\n")


if __name__ == "__main__":
    main()
