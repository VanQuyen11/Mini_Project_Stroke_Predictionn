
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filepath):

    df = pd.read_csv(filepath)

    return df



def q1_matrix_shape_rank(df):

    matrix = df[['age', 'avg_glucose_level', 'bmi']].fillna(
        df['bmi'].mean()
    ).to_numpy()

    print("Shape:", matrix.shape)
    print("Rank:", np.linalg.matrix_rank(matrix))

    return matrix



def q2_linear_regression(df):

    data = df[['age', 'avg_glucose_level', 'stroke']].dropna()

    X = data[['age', 'avg_glucose_level']].to_numpy()
    y = data['stroke'].to_numpy()

    # Add bias term
    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack((ones, X))

    # Normal Equation
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    print("Model Parameters:")
    print(theta)

    return theta




def q3_conditional_probability(df):

    total = len(df)

    stroke_yes = len(df[df['stroke'] == 1])

    hypertension_yes = len(df[df['hypertension'] == 1])

    both = len(df[
        (df['stroke'] == 1) &
        (df['hypertension'] == 1)
    ])

    p_stroke = stroke_yes / total
    p_hypertension = hypertension_yes / total
    p_stroke_given_hyper = both / hypertension_yes

    print("P(Stroke):", p_stroke)
    print("P(Hypertension):", p_hypertension)
    print("P(Stroke | Hypertension):", p_stroke_given_hyper)




def q4_mse_gradient(df):

    data = df[['age', 'avg_glucose_level', 'stroke']].dropna()

    X = data[['age', 'avg_glucose_level']].to_numpy()
    y = data['stroke'].to_numpy().reshape(-1, 1)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    w = np.zeros((X.shape[1], 1))

    predictions = X @ w

    gradient = (-2 / len(X)) * X.T @ (y - predictions)

    print("Gradient:")
    print(gradient)

    return gradient




def q5_variance_covariance(df):

    data = df[['avg_glucose_level', 'age']].dropna().to_numpy()

    variance = np.var(data[:, 0])

    covariance = np.cov(data[:, 0], data[:, 1])[0, 1]

    print("Variance of avg_glucose_level:", variance)
    print("Covariance:", covariance)




def q6_covariance_eigen(df):

    data = df[['age', 'avg_glucose_level', 'bmi']].fillna(
        df['bmi'].mean()
    ).to_numpy()

    cov_matrix = np.cov(data, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    print("Covariance Matrix:")
    print(cov_matrix)

    print("Eigenvalues:")
    print(eigenvalues)

    print("Eigenvectors:")
    print(eigenvectors)

    return cov_matrix



def q7_svd(df):

    data = df[['age', 'avg_glucose_level', 'bmi']].fillna(
        df['bmi'].mean()
    ).to_numpy()

    U, S, VT = np.linalg.svd(data, full_matrices=False)

    print("U Matrix:")
    print(U[:5])

    print("Singular Values:")
    print(S)

    print("VT Matrix:")
    print(VT)

    return U, S, VT




def q8_gradient_descent(df, lr=0.000001, epochs=100):

    data = df[['age', 'avg_glucose_level', 'stroke']].dropna()

    X = data[['age', 'avg_glucose_level']].to_numpy()
    y = data['stroke'].to_numpy().reshape(-1, 1)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    w = np.zeros((X.shape[1], 1))

    losses = []

    for _ in range(epochs):

        predictions = X @ w

        error = predictions - y

        loss = np.mean(error ** 2)
        losses.append(loss)

        gradient = (2 / len(X)) * X.T @ error

        w = w - lr * gradient

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Loss")
    plt.show()

    print("Optimized Weights:")
    print(w)

    return w




def q9_bayes_theorem(df):

    glucose = df['avg_glucose_level']

    diabetic = glucose > 140

    stroke = df['stroke'] == 1

    p_diabetic = np.mean(diabetic)

    p_stroke = np.mean(stroke)

    p_diabetic_given_stroke = np.mean(diabetic & stroke) / p_stroke

    # Bayes Theorem
    p_stroke_given_diabetic = (p_diabetic_given_stroke * p_stroke) / p_diabetic

    print("P(Stroke | Diabetic):", p_stroke_given_diabetic)




def q10_regularization(df, lambda_value=0.1):

    data = df[['age', 'avg_glucose_level', 'stroke']].dropna()

    X = data[['age', 'avg_glucose_level']].to_numpy()
    y = data['stroke'].to_numpy()

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    # Ridge Regression (L2)
    ridge_weights = np.linalg.inv(
        X.T @ X + lambda_value * np.eye(X.shape[1])
    ) @ X.T @ y

    # Lasso Approximation (L1)
    lasso_weights = ridge_weights - lambda_value * np.sign(ridge_weights)

    print("Ridge Weights:")
    print(ridge_weights)

    print("Lasso Weights:")
    print(lasso_weights)


filepath = 'healthcare-dataset-stroke-data.csv'
if __name__ == '__main__':

   df = load_dataset(filepath)
   q1_matrix_shape_rank(df)
   q2_linear_regression(df)
   q3_conditional_probability(df)
   q4_mse_gradient(df)
   q5_variance_covariance(df)
   q6_covariance_eigen(df)
   q7_svd(df)
   q8_gradient_descent(df)
   q9_bayes_theorem(df)
   q10_regularization(df)
