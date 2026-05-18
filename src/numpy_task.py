import numpy as np


##  Q1
def q1_load_and_process_data(file_path):
    # age, avg_glucose_level, bmi

    data = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=1,
        usecols=(2, 8, 9),
        missing_values='N/A',
        filling_values=np.nan
    )

    # Replace NaN with column mean
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])

    return data

def q2_statistics(data):
  #  compute mean, median, and std

    mean_values = np.mean(data, axis=0)
    median_values = np.median(data, axis=0)
    std_values = np.std(data, axis=0)

    print("Mean:", mean_values)
    print("Median:", median_values)
    print("Standard Deviation:", std_values)

def q3_minmax_normalization(data):
    # Apply Min-Max normalization

    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data

def q4_filter_patients(data):
    # Use boolean indexing to filter patients with age > 50 and avg_glucose_level greater than
    # the dataset average.

    avg_glucose_mean = np.mean(data[:, 1])

    filtered = data[
        (data[:, 0] > 50) &
        (data[:, 1] > avg_glucose_mean)
    ]

    return filtered

def q5_correlation_matrix(data):
    #Compute the correlation matrix

    correlation_matrix = np.corrcoef(data, rowvar=False)

    return correlation_matrix

def q6_euclidean_distance(data):

    sum_square = np.sum(data**2, axis=1)

    distance_squared = (
        sum_square[:, np.newaxis] +
        sum_square[np.newaxis, :] -
        2 * np.dot(data, data.T)
    )

    # Avoid negative values caused by floating point errors
    distance_squared = np.maximum(distance_squared, 0)

    distance_matrix = np.sqrt(distance_squared)

    return distance_matrix

def q7_scaling_methods(data):
    #Implement both Min-Max scaling and Z-score normalization

    # Min-Max Scaling
    minmax = (data - np.min(data, axis=0)) / (
        np.max(data, axis=0) - np.min(data, axis=0)
    )

    # Z-score Normalization
    zscore = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    return minmax, zscore

def q8_cosine_similarity(data, index1=0, index2=1):
    #Compute cosine similarity

    patient1 = data[index1]
    patient2 = data[index2]

    cosine_similarity = np.dot(patient1, patient2) / (
        np.linalg.norm(patient1) * np.linalg.norm(patient2)
    )

    return cosine_similarity


def q9_pca(data):
    # Perform PCA manualy using NumPy

    # Standardize data
    standardized_data = (
        data - np.mean(data, axis=0)
    ) / np.std(data, axis=0)

    # Covariance matrix
    cov_matrix = np.cov(standardized_data, rowvar=False)

    # Eigenvalues & Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort descending
    sorted_index = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[sorted_index]
    eigenvectors = eigenvectors[:, sorted_index]

    # Top 2 principal components
    top2_vectors = eigenvectors[:, :2]

    # Projection
    projected_data = np.dot(standardized_data, top2_vectors)

    return eigenvalues, eigenvectors, projected_data

def q10_batch_processing(data, batch_size=500):
    #Implement batch processing by splitting the NumPy array
    batches = np.array_split(data, len(data) // batch_size)

    batch_means = [np.mean(batch, axis=0) for batch in batches]

    return batch_means


filepath = 'healthcare-dataset-stroke-data.csv'
if __name__ == '__main__':
    data = q1_load_and_process_data(filepath)
    dataset = q1_load_and_process_data(filepath)
    q2_statistics(dataset)
    q3_minmax_normalization(dataset)
    q4_filter_patients(dataset)
    q5_correlation_matrix(dataset)
    q6_euclidean_distance(dataset)
    q7_scaling_methods(dataset)
    q8_cosine_similarity(dataset)
    q9_pca(dataset)
    q10_batch_processing(dataset)