import json
import os
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.train import load_config
import warnings

@pytest.fixture(scope="module")
def config():
    return load_config("config/config.json")

@pytest.fixture(scope="module")
def digits_data():
    digits = load_digits()
    return train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

def build_model(config):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"]
        )
    )

def test_config_loads(config):
    assert isinstance(config, dict)

def test_config_keys_and_types(config):
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)


def test_model_creation_and_fit(config):
    digits = load_digits()
    model = build_model(config)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(digits.data, digits.target)    
    
    lr_model = model.named_steps['logisticregression']
    assert isinstance(lr_model, LogisticRegression)
    assert hasattr(lr_model, "coef_")
    assert hasattr(lr_model, "classes_")

def test_model_accuracy_threshold(config, digits_data):
    X_train, X_test, y_train, y_test = digits_data
    model = build_model(config)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train, y_train)    
        acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.90
