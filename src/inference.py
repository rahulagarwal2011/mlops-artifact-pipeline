import pickle
from sklearn.datasets import load_digits
import os

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_digits(model):
    digits = load_digits()
    predictions = model.predict(digits.data)
    return predictions

def main():
    model_path = "model/train.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)
    predictions = predict_digits(model)

    print(f"predictions: {predictions[:10]}")

if __name__ == "__main__":
    main()
