import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional

import numpy as np
import torch

def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def report_classification_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    *,
    title: Optional[str] = None,
    plot_confusion_matrix: bool = True,
    normalize_confusion_matrix: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single-label multi-class classifier and report common metrics.

    Reports:
      - Accuracy
      - Macro-averaged Precision / Recall / F1-score ("mean" across classes)
      - Confusion matrix

    Args:
        model: PyTorch model returning logits of shape (N, C).
        data_loader: Yields (inputs, targets) with targets as class indices.
        device: torch.device("cuda") or torch.device("cpu").
        class_names: Optional list of class labels (len == num_classes).
        title: Optional label to print/plot.
        plot_confusion_matrix: If True, plots confusion matrix with matplotlib.
        normalize_confusion_matrix:
            None: show counts
            'true': normalize rows (per true class)
            'pred': normalize columns (per predicted class)
            'all': normalize by total

    Returns:
        Dict with accuracy, macro precision/recall/f1, per-class metrics, and confusion matrix.
    """
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        if outputs.ndim != 2:
            raise ValueError(
                f"Expected model outputs with shape (N, C), got shape {tuple(outputs.shape)}"
            )

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    y_pred = torch.cat(all_preds).to(torch.int64).numpy()
    y_true = torch.cat(all_targets).to(torch.int64).numpy()

    if y_true.size == 0:
        raise ValueError("Empty data_loader: no samples to evaluate.")

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    if class_names is not None and len(class_names) != num_classes:
        # still continue, but label mapping might be confusing
        pass

    cm = np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = float(correct / total) if total > 0 else 0.0

    # Per-class metrics from confusion matrix
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    header = f"{title} - " if title else ""
    print(f"{header}Accuracy: {accuracy * 100:.2f}%")
    print(
        f"{header}Macro Precision/Recall/F1: "
        f"{macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}"
    )
    print(f"{header}Confusion Matrix (rows=true, cols=pred):\n{cm}")

    if plot_confusion_matrix:
        cm_to_show = cm.astype(np.float64)
        if normalize_confusion_matrix is not None:
            norm = normalize_confusion_matrix.lower()
            if norm == "true":
                denom = cm_to_show.sum(axis=1, keepdims=True)
                cm_to_show = np.divide(cm_to_show, denom, out=np.zeros_like(cm_to_show), where=denom > 0)
            elif norm == "pred":
                denom = cm_to_show.sum(axis=0, keepdims=True)
                cm_to_show = np.divide(cm_to_show, denom, out=np.zeros_like(cm_to_show), where=denom > 0)
            elif norm == "all":
                denom = cm_to_show.sum()
                cm_to_show = cm_to_show / denom if denom > 0 else cm_to_show
            else:
                raise ValueError("normalize_confusion_matrix must be one of: None, 'true', 'pred', 'all'")

        labels = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_to_show, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix{f' - {title}' if title else ''}")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    per_class = []
    for i in range(num_classes):
        per_class.append(
            {
                "class": class_names[i] if class_names and i < len(class_names) else str(i),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class": per_class,
    }

