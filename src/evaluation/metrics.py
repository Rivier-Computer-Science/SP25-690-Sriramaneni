from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def summarize_predictions(labels, predictions, label_map):
    ordered_labels = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(labels, predictions),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=ordered_labels,
            digits=4,
            zero_division=0,
        ),
    }
