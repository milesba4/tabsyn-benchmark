#Importing adult dataset
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from prdc import compute_prdc
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# Combine features and target into a single DataFrame
adult_df = pd.concat([X, y], axis=1)


#importing synthetic data generated from tabsyn
synthetic_adult_df = pd.read_csv("/content/adult_synthetic.csv")

# Select only numerical columns from the original dataset
adult_df_numerical = adult_df.select_dtypes(include=[np.number])

# Select only numerical columns from the synthetic dataset
synthetic_adult_df_numerical = synthetic_adult_df.select_dtypes(include=[np.number])

# Select only categorical columns from the original dataset
adult_df_categorical = adult_df.select_dtypes(exclude=[np.number])

# Select only categorical columns from the synthetic dataset
synthetic_adult_df_categorical = synthetic_adult_df.select_dtypes(exclude=[np.number])

def compute_metrics(df1, df2, feature_dim, k):
    df2.columns = [col.replace('.', '-') for col in df2.columns]
    # Combine both dataframes to ensure consistent encoding
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_combined = encoder.fit_transform(combined_df.select_dtypes(include=['object', 'category']))

    # Extract numerical columns and concatenate with encoded categorical columns
    numerical_combined = combined_df.select_dtypes(include=[np.number]).to_numpy()
    combined_features = np.concatenate([numerical_combined, encoded_combined], axis=1)

    # Split the combined features back into original and synthetic
    num_original_samples = df1.shape[0]
    original_features = combined_features[:num_original_samples, :]
    synthetic_features = combined_features[num_original_samples:, :]

    # Reduce dimensionality to the specified feature_dim using PCA
    pca = PCA(n_components=feature_dim)
    original_features_reduced = pca.fit_transform(original_features)
    synthetic_features_reduced = pca.transform(synthetic_features)

    # Compute precision, recall, density, and coverage
    metrics = compute_prdc(real_features=original_features_reduced,
                           fake_features=synthetic_features_reduced,
                           nearest_k=k)

    return metrics

# Example usage
metrics = compute_metrics(adult_df, synthetic_adult_df, feature_dim=100, k=5)
print(metrics)
