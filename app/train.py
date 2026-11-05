"""Train a handwritten-digit classifier (8x8 images, digits 0-9) and save it.

Runs during `docker build` (model baked into the image) or in the `trainer`
compose service (model written to the shared `models` volume).
"""
import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    # small neural network: 64 pixels -> 64 hidden units -> 10 digits
    "mlp": lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=42)
    ),
    "rf": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=os.getenv("MODEL_TYPE", "mlp"))
    parser.add_argument("--out", default=os.getenv("MODEL_DIR", "/models"))
    args = parser.parse_args()

    digits = load_digits()  # bundled with scikit-learn, no download
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )

    model = MODELS[args.model]()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    # keep the held-out images so the API can show real examples
    joblib.dump({"X": X_test, "y": y_test}, out / "test_set.joblib")

    metadata = {
        "model_type": args.model,
        "accuracy": round(float(accuracy), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
