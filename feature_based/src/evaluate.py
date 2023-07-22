'''
Usage:
python evaluate.py

Input: feature list used in training, Trained model
Output: spearman's r, pearson r, MAE, MSE, RMSE score 
'''

import pickle
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_FILE = '../data/train/original_features.csv'
MODEL_NAME = 'AutoMLRegressor.pkl'

def load_data(file):
    df = pd.read_csv(file)
    df = df.fillna(0)
    features = df.drop(["word", "score"], axis=1).to_numpy()
    scores = df["score"].to_numpy()

    return features, scores

def load_model(model_name):
    with open(model_name, 'rb') as f:
        return pickle.load(f)

def evaluate(predictions, y_test):
    # spearman's r
    print('Spearman:', stats.spearmanr(predictions, y_test))

    # pearson r
    print('Pearson:', stats.pearsonr(predictions, y_test))

    # MAE
    print('MAE:', mean_absolute_error(predictions, y_test))

    # MSE
    print('MSE:', mean_squared_error(predictions, y_test))

    # RMSE
    print('RMSE:', np.sqrt(mean_squared_error(predictions, y_test)))

def main():
    features, scores = load_data(FEATURE_FILE)
    x_train, x_test, y_train, y_test = train_test_split(features, scores, test_size=0.1, random_state=777)
    regressor = load_model(MODEL_NAME)
    predictions = regressor.predict(x_test)
    evaluate(predictions, y_test)

if __name__ == "__main__":
    main()
