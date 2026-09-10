"""
Run this to get an accuracy report for the trained intent classifier —
useful evidence for the "Model Training & Fine-Tuning" judging criterion.

Usage:  python scripts/evaluate_classifier.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ml_classifier import evaluate

if __name__ == "__main__":
    report = evaluate()
    print(f"\n=== Rasai AI — ML Classifier Evaluation ===")
    print(f"Held-out test set size: {report['n']}")
    print(f"Overall accuracy: {report['accuracy']*100:.1f}%\n")

    print("Per-intent breakdown:")
    for intent, stats in report["per_class"].items():
        if stats["total"] == 0:
            continue
        pct = stats["correct"] / stats["total"] * 100
        print(f"  {intent:12s}: {stats['correct']}/{stats['total']} correct ({pct:.0f}%)")

    print("\nMisclassified examples:")
    misses = [r for r in report["rows"] if not r["correct"]]
    if not misses:
        print("  (none — all test examples classified correctly)")
    for r in misses:
        print(f"  \"{r['text']}\" -> predicted {r['predicted']} (true: {r['true']}, confidence {r['confidence']})")
    print()
