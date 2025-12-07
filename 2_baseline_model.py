import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


np.random.seed(42)
tf.random.set_seed(42)

CONFIG = {
    'img_size': (64, 64),      
    'batch_size': 64,
    'epochs': 5,                
    'validation_split': 0.2,
    'learning_rate': 0.0005,   
    'num_classes': 3
}

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {gpus}")
    else:
        print("⚠ No GPU detected, using CPU")
    print()

def load_data(data_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,      
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=CONFIG['validation_split']
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )

    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        data_path,
        target_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator

# =========================================================
# BASELINE MODEL 
# =========================================================

def build_baseline_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Ultra-tiny model
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(16, activation='relu'),  
        layers.Dropout(0.4),                   
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy - Baseline CNN')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss - Baseline CNN')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)

    check_gpu()

    data_path = "data/processed"
    if not os.path.exists(data_path):
        print(f"Error: Processed data not found at {data_path}")
        return

    print("Loading data...")
    train_gen, val_gen = load_data(data_path)

    print("\nDataset Info:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Classes: {list(train_gen.class_indices.keys())}\n")

    print("Building baseline model...")
    input_shape = (*CONFIG['img_size'], 3)
    model = build_baseline_model(input_shape, CONFIG['num_classes'])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.5)
    ]

    print("="*60)
    print("TRAINING STARTED")
    print("="*60)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    print("\nTRAINING COMPLETE")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save("models/baseline_fast_model.h5")

    os.makedirs("results/plots", exist_ok=True)
    plot_training_history(history, "results/plots/baseline_fast_training.png")

    print("\n✓ BASELINE TRAINING SUCCESSFUL ✓")

if __name__ == "__main__":
    main()
