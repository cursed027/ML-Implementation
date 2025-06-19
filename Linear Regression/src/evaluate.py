import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))

    return y_true, y_pred

def plot_predictions(y_true, y_pred, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='royalblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    if save_path:
        plt.savefig(save_path)
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curve(csv_path='loss_curve.csv', save_path=None):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Error] CSV file not found at: {csv_path}")
        return

    if 'epoch' not in df.columns or 'loss' not in df.columns:
        print(f"[Error] CSV must contain 'epoch' and 'loss' columns.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['loss'], color='blue', marker='', linewidth=1.5)
    plt.title('Loss Curve (MSE)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Info] Plot saved to: {save_path}")
    else:
        plt.show()

