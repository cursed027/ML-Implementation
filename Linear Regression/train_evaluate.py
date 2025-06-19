import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy import expm1

from src.preprocess import feature_engineering, get_preprocessor
from src.model import CustomLinearRegression
from src.evaluate import evaluate_model, plot_predictions, plot_loss_curve 

def train_and_evaluate():
    # --- Load and preprocess ---
    df = pd.read_csv(r'dataset\Fish.csv')
    df = feature_engineering(df)

    X = df.drop(columns=['Weight'])
    y = df['Weight']
    # ----- log transform to fix right skew in target -----
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --- Train ---
    model = CustomLinearRegression()
    model.fit(X_train_transformed, y_train.values, lr=0.09, epochs=100000)

    # Save loss curve
    pd.DataFrame({'epoch': range(1, len(model.losses) + 1), 'loss': model.losses}).to_csv('loss_curve.csv', index=False)

    # --- Predict ---
    y_pred_log = model.predict(X_test_transformed)
    y_pred = expm1(y_pred_log)
    y_true = expm1(y_test)

    # --- Evaluate ---
    evaluate_model(model, X_test_transformed, y_test)
    plot_predictions(y_true, y_pred)
    plot_loss_curve('loss_curve.csv')

if __name__ == '__main__':
    train_and_evaluate()



