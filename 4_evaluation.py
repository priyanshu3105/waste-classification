

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# Seed
np.random.seed(42)
tf.random.set_seed(42)

# Configs
BASELINE_CONFIG = {'img_size': (64, 64), 'batch_size': 32}
ADVANCED_CONFIG = {'img_size': (224, 224), 'batch_size': 32}
DATA_PATH = "data/processed"

# ------------------------------------------------------------------
# Utility: Load dataset
# ------------------------------------------------------------------
def load_validation_data(img_size):
    gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    return gen.flow_from_directory(
        DATA_PATH,
        target_size=img_size,
        batch_size=32,
        subset='validation',
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

# ------------------------------------------------------------------
# Utility: Evaluate Model
# ------------------------------------------------------------------
def evaluate_model(model, val_gen, name):
    print(f"\n{'='*60}")
    print(f"Evaluating {name}")
    print('='*60)

    val_gen.reset()
    y_true = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    print("Generating predictions...")
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    print("Calculating metrics...")
    val_gen.reset()
    loss, acc = model.evaluate(val_gen, verbose=0)

    print(f"\n{name} Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Loss: {loss:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_text)
    
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "name": name,
        "loss": float(loss),
        "accuracy": float(acc),
        "preds": preds,
        "y_pred": y_pred,
        "y_true": y_true,
        "class_names": class_names,
        "cm": cm,
        "report": report_dict,
        "val_gen": val_gen
    }

# ------------------------------------------------------------------
# Plot: Confusion Matrix
# ------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    cm_n = cm.astype("float") / cm.sum(axis=1)[:, None]
    sns.heatmap(cm_n, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=[c.capitalize() for c in class_names], 
                yticklabels=[c.capitalize() for c in class_names],
                cbar_kws={'label': 'Percentage'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ------------------------------------------------------------------
# Plot: ROC Curves
# ------------------------------------------------------------------
def plot_roc_curves(results, save_path):
    preds = results["preds"]
    y_true = results["y_true"]
    class_names = results["class_names"]

    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), preds[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls.capitalize()} (AUC={auc_score:.3f})", 
                linewidth=2, color=colors[i])

    plt.plot([0,1],[0,1],"k--", linewidth=1.5, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves - {results['name']}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ------------------------------------------------------------------
# Plot: Precision-Recall Curve
# ------------------------------------------------------------------
def plot_precision_recall(results, save_path):
    preds = results["preds"]
    y_true = results["y_true"]
    class_names = results["class_names"]

    plt.figure(figsize=(10, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            (y_true == i).astype(int), preds[:, i]
        )
        plt.plot(recall, precision, label=cls.capitalize(), linewidth=2, color=colors[i])

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision–Recall Curves - {results['name']}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ------------------------------------------------------------------
# Plot: Model Comparison
# ------------------------------------------------------------------
def plot_model_comparison(baseline_results, advanced_results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall Accuracy Comparison
    ax1 = axes[0, 0]
    models = ['Baseline CNN', 'ResNet50']
    accuracies = [baseline_results['accuracy'], advanced_results['accuracy']]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Per-Class F1-Score Comparison
    ax2 = axes[0, 1]
    class_names = baseline_results['class_names']
    
    baseline_f1 = [baseline_results['report'][cls]['f1-score'] for cls in class_names]
    advanced_f1 = [advanced_results['report'][cls]['f1-score'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, baseline_f1, width, label='Baseline CNN', color='#3498db', alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, advanced_f1, width, label='ResNet50', color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([cn.capitalize() for cn in class_names])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Loss Comparison
    ax3 = axes[1, 0]
    losses = [baseline_results['loss'], advanced_results['loss']]
    bars = ax3.bar(models, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Baseline CNN', 'ResNet50', 'Difference'],
        ['Accuracy', f"{baseline_results['accuracy']*100:.2f}%", 
         f"{advanced_results['accuracy']*100:.2f}%",
         f"{(advanced_results['accuracy']-baseline_results['accuracy'])*100:+.2f}%"],
        ['Loss', f"{baseline_results['loss']:.4f}", 
         f"{advanced_results['loss']:.4f}",
         f"{baseline_results['loss']-advanced_results['loss']:+.4f}"],
        ['Params', '~1.05M', '~24.78M', '+23.73M']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ------------------------------------------------------------------
# Misclassified Images
# ------------------------------------------------------------------
def plot_misclassified(results, save_path, max_images=12):
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    filepaths = results["val_gen"].filepaths
    class_names = results["class_names"]

    incorrect_indices = np.where(y_true != y_pred)[0]
    
    if len(incorrect_indices) == 0:
        print(f"✓ No misclassifications found for {results['name']}!")
        return
    
    incorrect = incorrect_indices[:max_images]

    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(incorrect):
        plt.subplot(3, 4, i+1)
        img = keras.preprocessing.image.load_img(filepaths[idx])
        plt.imshow(img)
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", 
                 fontsize=9)
        plt.axis("off")

    plt.suptitle(f"Misclassified Samples - {results['name']}", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# ------------------------------------------------------------------
# Save JSON Report
# ------------------------------------------------------------------
def save_json_report(baseline_results, advanced_results, save_path):
    report = {
        "baseline_model": {
            "name": "Baseline CNN",
            "image_size": "64x64",
            "accuracy": baseline_results["accuracy"],
            "loss": baseline_results["loss"],
            "classification_report": baseline_results["report"]
        },
        "advanced_model": {
            "name": "ResNet50",
            "image_size": "160x160",
            "accuracy": advanced_results["accuracy"],
            "loss": advanced_results["loss"],
            "classification_report": advanced_results["report"]
        },
        "comparison": {
            "accuracy_difference": advanced_results["accuracy"] - baseline_results["accuracy"],
            "accuracy_difference_percent": ((advanced_results["accuracy"] - baseline_results["accuracy"]) / baseline_results["accuracy"]) * 100,
            "loss_difference": baseline_results["loss"] - advanced_results["loss"],
            "winner": "ResNet50" if advanced_results["accuracy"] > baseline_results["accuracy"] else "Baseline CNN"
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved: {save_path}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("="*60)
    print("FULL MODEL EVALUATION SUITE")
    print("="*60)
    print()
    
    baseline_path = "models/baseline_fast_model.h5"
    advanced_path = "models/resnet50_advanced_model.h5"

    # Check if models exist
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline model not found at {baseline_path}")
        return
    
    if not os.path.exists(advanced_path):
        print(f"Error: Advanced model not found at {advanced_path}")
        return

    print("Loading models...")
    baseline_model = keras.models.load_model(baseline_path)
    print("✓ Baseline model loaded")
    advanced_model = keras.models.load_model(advanced_path)
    print("✓ Advanced model loaded")
    print()

    # Load validation datasets
    print("Loading validation data...")
    val_baseline = load_validation_data(BASELINE_CONFIG["img_size"])
    print(f"✓ Baseline validation set: {val_baseline.samples} samples (64x64)")
    val_advanced = load_validation_data(ADVANCED_CONFIG["img_size"])
    print(f"✓ Advanced validation set: {val_advanced.samples} samples (160x160)")

    # Evaluate both models
    baseline_results = evaluate_model(baseline_model, val_baseline, "Baseline CNN")
    advanced_results = evaluate_model(advanced_model, val_advanced, "ResNet50")

    # Create directory
    os.makedirs("results/plots", exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    print()

    # Confusion matrices
    print("Creating confusion matrices...")
    plot_confusion_matrix(baseline_results["cm"], baseline_results["class_names"],
                          "Baseline CNN - Confusion Matrix", "results/plots/baseline_cm.png")
    plot_confusion_matrix(advanced_results["cm"], advanced_results["class_names"],
                          "ResNet50 - Confusion Matrix", "results/plots/advanced_cm.png")

    # ROC curves
    print("\nCreating ROC curves...")
    plot_roc_curves(baseline_results, "results/plots/baseline_roc.png")
    plot_roc_curves(advanced_results, "results/plots/advanced_roc.png")

    # Precision-Recall curves
    print("\nCreating Precision-Recall curves...")
    plot_precision_recall(baseline_results, "results/plots/baseline_pr.png")
    plot_precision_recall(advanced_results, "results/plots/advanced_pr.png")

    # Model comparison
    print("\nCreating model comparison chart...")
    plot_model_comparison(baseline_results, advanced_results, "results/plots/model_comparison.png")

    # Misclassified images
    print("\nGenerating misclassified samples...")
    plot_misclassified(baseline_results, "results/plots/baseline_mistakes.png")
    plot_misclassified(advanced_results, "results/plots/advanced_mistakes.png")

    # Save JSON report
    print("\nSaving evaluation report...")
    save_json_report(baseline_results, advanced_results, "results/evaluation_report.json")

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print()
    print("Baseline CNN:")
    print(f"  Image Size: 64x64")
    print(f"  Accuracy: {baseline_results['accuracy']*100:.2f}%")
    print(f"  Loss: {baseline_results['loss']:.4f}")
    print()
    print("ResNet50:")
    print(f"  Image Size: 224x224")
    print(f"  Accuracy: {advanced_results['accuracy']*100:.2f}%")
    print(f"  Loss: {advanced_results['loss']:.4f}")
    print()
    
    acc_diff = (advanced_results['accuracy'] - baseline_results['accuracy']) * 100
    if acc_diff > 0:
        print(f"🏆 Winner: ResNet50 (+{acc_diff:.2f}% better)")
    elif acc_diff < 0:
        print(f"🏆 Winner: Baseline CNN ({abs(acc_diff):.2f}% better)")
    else:
        print("🏆 Tie: Both models perform equally")
    
    print()
    print("="*60)
    print("✓ FULL EVALUATION COMPLETE")
    print("="*60)
    print("\nAll visualizations saved to results/plots/")
    print("JSON report saved to results/evaluation_report.json")
    print("\nNext step: Launch web UI")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()