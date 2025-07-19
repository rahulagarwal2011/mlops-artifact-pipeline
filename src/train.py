import json
import pickle, os
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    config = load_config("config/config.json")

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"]
        )
    )
    model.fit(X_train, y_train)

    # Save trained model
    os.makedirs("model", exist_ok=True)
    with open("model/train.pkl", "wb") as f:
        pickle.dump(model, f)

    # Evaluate metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    loss = log_loss(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Print Loss: {loss:.4f}")

    # Save metrics
    os.makedirs("result", exist_ok=True)
    result_df = pd.DataFrame([{
        "accuracy": accuracy,
        "f1_score": f1,
        "print_loss": loss
    }])
    result_df.to_csv("result/metrics.csv", index=False)
    print("Metrics saved to result/metrics.csv")    


if __name__ == "__main__":
    main()
