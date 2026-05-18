import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

filepath = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(filepath)
# print(df.head())

X_num = df[['age', 'avg_glucose_level', 'bmi']].values
col_mean_bmi = np.nanmean(X_num[:, 2])
X_num[:, 2] = np.where(np.isnan(X_num[:, 2]), col_mean_bmi, X_num[:, 2])

features = ['Age', 'Avg_Glucose_Level', 'BMI']
for i, name in enumerate(features):
    print(f"{name} -> Mean: {np.mean(X_num[:, i]):.2f} | Median: {np.median(X_num[:, i]):.2f} | Std: {np.std(X_num[:, i]):.2f}")

X_std = (X_num - np.mean(X_num, axis=0)) / np.std(X_num, axis=0)
cov_matrix = np.cov(X_std, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigenvalues)

y = df['stroke'].values
X_b = np.c_[np.ones((X_std.shape[0], 1)), X_std[:, :2]]
n = len(y)
lr = 0.05
epochs = 150
w = np.zeros(X_b.shape[1])
losses = []

for _ in range(epochs):
    y_pred = X_b.dot(w)
    loss = (1/n) * np.sum((y - y_pred)**2)
    losses.append(loss)
    grad = -(2/n) * X_b.T.dot(y - y_pred)
    w -= lr * grad

plt.figure(figsize=(6, 4))
plt.plot(range(epochs), losses, color='red', lw=2)
plt.title('Loss Convergence via Gradient Descent')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()

df_cleaned = df[df['gender'] != 'Other'].copy()
df_cleaned['bmi'] = df_cleaned['bmi'].fillna(df_cleaned['bmi'].mean())

print("Stroke Rate grouped by Smoking Status:")
print(df_cleaned.groupby('smoking_status')['stroke'].mean().round(4))

df_cleaned['glucose_to_bmi_ratio'] = df_cleaned['avg_glucose_level'] / df_cleaned['bmi']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

num_cols = df_cleaned.select_dtypes(include=[np.number])
sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
axes[0].set_title('Feature Correlation Matrix')

sns.boxplot(x='stroke', y='avg_glucose_level', data=df_cleaned, palette='Set2', ax=axes[1])
axes[1].set_title('Avg Glucose Level Distribution by Stroke Status')

plt.tight_layout()
plt.show()

cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df_cleaned, columns=cat_cols, drop_first=True)
print(f"\nFinal Preprocessed Data Shape: {df_encoded.shape}")