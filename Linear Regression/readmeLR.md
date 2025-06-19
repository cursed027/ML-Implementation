---

## 🐟 Linear Regression from Scratch – Fish Weight Prediction

This project implements **Linear Regression from scratch** in Python using NumPy, without relying on Scikit-learn’s built-in estimators. It serves as a part of a broader repository to showcase machine learning algorithms implemented manually — with proper EDA, preprocessing, model training, and evaluation.

---
### 📌 Dataset
The dataset used is the **Fish Market dataset**, commonly used for regression tasks. It contains features like species, lengths, height, and width of fishes, and the target is their weight.
> 📄 Source: [Fish Market Dataset on Kaggle](https://www.kaggle.com/datasets/aungpyaeap/fish-market)
---

### 📁 Project Structure

```
Linear Regression/
├── EDA/
│   ├── EDA Notebook.ipynb  # performed EDA on dataset
│   ├── EDA summary.txt     # Summary of EDA
├── dataset/                # Dataset file (Fish.csv)
├── plots/                  # result plots
├── src/                    # Modular ML pipeline
│   ├── model.py            # Custom LinearRegression class
│   ├── preprocess.py       # EDA-based feature engineering + ColumnTransformer
│   ├── evaluate.py         # Evaluation metrics and plotting
│   └── config.py           # Feature configs
├── train_evaluate.py       # Runs the full pipeline with train and then evaluating
├── req.txt/                # requirement file
├── loss_curve.csv/         # loss curve logs
└── README.md               # Project documentation
```

---

### ⚙️ Features Implemented

* ✅ **Feature Engineering**: Merges multiple length features into a single `AvgLength`
* ✅ **Skew Fix**: `log1p` applied to weight (target)
* ✅ **Preprocessing**:

  * Scaling numerical features using `StandardScaler`
  * One-hot encoding of `Species`
* ✅ **Custom Linear Regression**:

  * Gradient descent optimization
  * Bias term via manual augmentation
* ✅ **Evaluation**:

  * MAE, RMSE, R² score
  * Prediction vs. actual plot
  * Loss Curve

---

### 🚀 How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/cursed027/ML-Implementation.git
   cd ML-Implementation/Linear\ Regression/
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**:

   ```bash
   python main.py
   ```

---

### 📊 Output

* **R² Score**: 84.98183219172603
* **MAE**: 84.98183219172603g
* **RMSE**: 165.7567027959001g
* The model accurately captures fish weight based on physical features using linear regression logic from scratch.


### 🤝 Contributions

This is a learning-focused repository by [Milan Kumar Singh](https://github.com/cursed027), currently an undergraduate passionate about ML, DL, and research implementation.

---
