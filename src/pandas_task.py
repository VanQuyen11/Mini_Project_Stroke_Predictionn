import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs/figures', exist_ok=True)


def q1_load_and_inspect(filepath):
    print("--- Q1: Load & Inspect Data ---")
    df = pd.read_csv(filepath)
    print("Head (first 2 rows):\n", df.head(2))
    print("\nInfo:")
    df.info()
    print("\nDescribe:\n", df.describe().round(2))
    return df


def q2_clean_dataset(df):
    print("\n--- Q2: Clean Dataset ---")
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    initial_shape = df.shape
    df = df[df['gender'] != 'Other']
    print(f"Shape changed from {initial_shape} to {df.shape} (Removed 'Other' gender)")
    return df


def q3_group_smoking_status(df):
    print("\n--- Q3: Group by Smoking Status ---")
    grouped = df.groupby('smoking_status').agg(
        Total_Patients=('id', 'count'),
        Stroke_Rate=('stroke', 'mean')
    )
    print(grouped)


def q4_glucose_bmi_ratio(df):
    print("\n--- Q4: Feature Engineering (Glucose to BMI Ratio) ---")
    df['glucose_to_bmi_ratio'] = df['avg_glucose_level'] / df['bmi']
    corr_with_stroke = df['glucose_to_bmi_ratio'].corr(df['stroke'])
    print(f"Correlation between glucose_to_bmi_ratio and stroke: {corr_with_stroke:.4f}")
    return df


def q5_basic_visualizations(df):
    print("\n--- Q5: Basic Visualizations (Saved to outputs/figures) ---")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Age')

    plt.subplot(1, 2, 2)
    sns.countplot(x='stroke', data=df, palette='Set2')
    plt.title('Stroke Counts')

    plt.tight_layout()
    plt.savefig('outputs/figures/q5_age_stroke_dist.png')
    plt.close()
    print("-> Saved: q5_age_stroke_dist.png")


def q6_pivot_table(df):
    print("\n--- Q6: Pivot Table (Work Type vs Stroke) ---")
    pivot = pd.pivot_table(df, values='id', index='work_type', columns='stroke', aggfunc='count', fill_value=0)
    print(pivot)


def q7_age_groups(df):
    print("\n--- Q7: Age Groups Analysis ---")
    bins = [0, 18, 60, 150]
    labels = ['Child', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    age_analysis = df.groupby('age_group', observed=False)['stroke'].mean().reset_index()
    age_analysis.rename(columns={'stroke': 'stroke_rate'}, inplace=True)
    print(age_analysis)
    return df


def q8_bmi_categories(df):
    print("\n--- Q8: BMI Categories Analysis ---")
    bins = [0, 18.5, 25, 30, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bins, labels=labels, right=False)

    bmi_analysis = df.groupby('bmi_category', observed=False)['stroke'].mean().reset_index()
    bmi_analysis.rename(columns={'stroke': 'stroke_rate'}, inplace=True)
    print(bmi_analysis)
    return df


def q9_advanced_visualizations(df):
    print("\n--- Q9: Advanced Visualizations (Saved to outputs/figures) ---")

    plt.figure(figsize=(8, 6))
    num_cols = df.select_dtypes(include=[np.number])
    sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/figures/q9_correlation_heatmap.png')
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='stroke', y='avg_glucose_level', data=df, palette='Set1')
    plt.title('Avg Glucose Level grouped by Stroke')
    plt.tight_layout()
    plt.savefig('outputs/figures/q9_glucose_boxplot.png')
    plt.close()
    print("-> Saved: q9_correlation_heatmap.png and q9_glucose_boxplot.png")


def q10_feature_encoding(df):
    print("\n--- Q10: Feature Encoding (One-Hot Encoding) ---")
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"Original columns count: {len(df.columns)}")
    print(f"Encoded columns count: {len(df_encoded.columns)}")
    print("Encoded features snippet:\n", df_encoded.columns.tolist()[-5:])

    os.makedirs('data/processed', exist_ok=True)
    df_encoded.to_csv('data/processed/stroke_data_cleaned_encoded.csv', index=False)
    print("-> Saved cleaned and encoded dataset to 'data/processed/stroke_data_cleaned_encoded.csv'")


if __name__ == "__main__":
    filepath = 'healthcare-dataset-stroke-data.csv'

    df = q1_load_and_inspect(filepath)
    df = q2_clean_dataset(df)
    q3_group_smoking_status(df)
    df = q4_glucose_bmi_ratio(df)
    q5_basic_visualizations(df)
    q6_pivot_table(df)
    df = q7_age_groups(df)
    df = q8_bmi_categories(df)
    q9_advanced_visualizations(df)
    q10_feature_encoding(df)