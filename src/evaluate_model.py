"""Print stored evaluation metrics; run train_model.py first."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "outputs" / "evaluation_results" / "metrics.json"
if not path.exists():
    raise SystemExit("Metrics not found. Run: python src/train_model.py")
data = json.loads(path.read_text(encoding="utf-8"))
print("Selected model:", data["selected_model"])
for name, scores in data["models"].items():
    print(f"\n{name}")
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"  {metric}: {scores[metric]:.4f}")
