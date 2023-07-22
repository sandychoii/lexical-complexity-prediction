import os
import pickle

import pandas as pd
import wandb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from autosklearn.regression import AutoSklearnRegressor


FEATURE_DIR = '../data/train'
FEATURE_FILE = os.path.join(FEATURE_DIR, 'original_features.csv')
MODEL_NAME = 'AutoMLRegressor'
WANDB_PROJECT = 'lcp_feature_autoML'
FEATURES_PLOT_FILE = 'regressor_features.png'


def init_wandb(project, model):
    wandb.init(project=project)
    wandb.config.update({"model": model})


def load_data(file):
    df = pd.read_csv(file)
    df = df.fillna(0)
    features = df.drop(["word", "score"], axis=1).to_numpy()
    scores = df["score"].to_numpy()

    return features, scores, df


def train_model(features, scores):
    x_train, x_test, y_train, y_test = train_test_split(features, scores, test_size=0.1, random_state=777)

    regressor = AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30)
    regressor.fit(x_train, y_train)

    return regressor, x_train, x_test, y_train, y_test


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def plot_features(regressor, df):
    if hasattr(regressor, 'feature_importances_'):
        feature_names = list(df.drop(["word", "score"], axis=1))
        plt.barh(feature_names, regressor.feature_importances_)
        plt.savefig(FEATURES_PLOT_FILE, bbox_inches='tight', pad_inches=0.3)


def log_wandb(regressor, x_train, x_test, y_train, y_test):
    wandb.sklearn.plot_regressor(regressor, x_train, x_test, y_train, y_test, model_name=MODEL_NAME)

    models = regressor.show_models()
    wandb.log({"ensemble_models": models})

    stats = regressor.sprint_statistics()
    wandb.log({"statistics": stats})

    num_models = len(regressor.get_models_with_weights())
    wandb.log({"number_of_models": num_models})

    model_info = ""
    for info in regressor.get_models_with_weights():
        model_detail = "model, weight : {}\n{}".format(info[0], "-" * 10 + "\n" + str(info[1]))
        model_info += model_detail + "\n"

    wandb.log({"model_details": model_info})


def main():
    init_wandb(WANDB_PROJECT, MODEL_NAME)

    features, scores, df = load_data(FEATURE_FILE)

    regressor, x_train, x_test, y_train, y_test = train_model(features, scores)

    save_model(regressor, f"{MODEL_NAME}.pkl")

    plot_features(regressor, df)

    log_wandb(regressor, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
