import argparse
import copy
import csv
from pathlib import Path

from main import run_training
from src.utils.config_loader import Config


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MLP, CNN, and multimodal models and print a comparison table."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the base YAML config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for all model runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "cnn", "multimodal"],
        help="Model types to compare.",
    )
    return parser


def format_confusion_matrix(matrix):
    rows = ["[" + ", ".join(str(value) for value in row) + "]" for row in matrix]
    return " ; ".join(rows)


def print_markdown_table(results):
    print("\nComparison Table:")
    print("| Model | Val Accuracy | Macro F1 | Checkpoint |")
    print("|---|---:|---:|---|")
    for result in results:
        metrics = result["best_metrics"]
        print(
            "| "
            f"{result['model_type']} | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | "
            f"{result['checkpoint_path']} |"
        )


def build_markdown_table(results):
    lines = [
        "# Comparison Results",
        "",
        "| Model | Val Accuracy | Macro F1 | Checkpoint |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        metrics = result["best_metrics"]
        lines.append(
            "| "
            f"{result['model_type']} | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | "
            f"{result['checkpoint_path']} |"
        )
    return "\n".join(lines) + "\n"


def print_detailed_summary(results):
    print("\nDetailed Summary:")
    for result in results:
        metrics = result["best_metrics"]
        print(f"\nModel: {result['model_type']}")
        print(f"Feature input: {result['feature_description']}")
        print(f"Best validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Best validation macro F1: {metrics['macro_f1']:.4f}")
        print(f"Confusion matrix: {format_confusion_matrix(metrics['confusion_matrix'])}")


def write_results(results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / "comparison_results.md"
    csv_path = output_path / "comparison_results.csv"

    markdown_content = build_markdown_table(results)
    markdown_path.write_text(markdown_content, encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "val_accuracy",
                "macro_f1",
                "checkpoint",
                "feature_input",
                "confusion_matrix",
            ]
        )
        for result in results:
            metrics = result["best_metrics"]
            writer.writerow(
                [
                    result["model_type"],
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    result["checkpoint_path"],
                    result["feature_description"],
                    format_confusion_matrix(metrics["confusion_matrix"]),
                ]
            )

    print(f"\nSaved markdown results to: {markdown_path}")
    print(f"Saved CSV results to: {csv_path}")


def main():
    args = build_parser().parse_args()
    base_config = Config(args.config).config
    results = []

    for model_type in args.models:
        config = copy.deepcopy(base_config)
        config["model"]["type"] = model_type
        if args.device is not None:
            config["training"]["device"] = args.device
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs

        print(f"\n=== Running {model_type.upper()} ===")
        result = run_training(config)
        results.append(result)

    print_markdown_table(results)
    print_detailed_summary(results)
    write_results(results, output_dir="artifacts")


if __name__ == "__main__":
    main()
