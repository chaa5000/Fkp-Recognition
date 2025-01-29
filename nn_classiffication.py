import numpy as np
from scipy.spatial.distance import cosine, euclidean


def nn_classification(train, train_ids, test, test_ids, n=None, dist='cos', match_kind='all'):
    """
    Performs matching score calculation based on the nearest neighbor classifier.

    Args:
        train (np.ndarray): Training feature-vector matrix of size (B, A),
                            where each column is a feature vector.
        train_ids (np.ndarray): Vector of size (A,) or (1, A), representing class membership of each sample in train.
        test (np.ndarray): Test feature-vector matrix of size (C, D),
                           where each column is a feature vector.
        test_ids (np.ndarray): Vector of size (D,) or (1, D), representing class membership of each sample in test.
        n (int): Number of features to use in matching. Defaults to the number of features in train.
        dist (str): Distance metric ('cos', 'euc'). Defaults to 'cos'.
        match_kind (str): Matching logic ('all', 'sep'). Defaults to 'all'.

    Returns:
        dict: Results containing similarity matrix and other fields.
    """
    if n is None:
        n = train.shape[0]

    # Validate distance metric
    if dist not in ['cos', 'euc']:
        raise ValueError("Distance metric must be 'cos' or 'euc'.")

    # Prepare results dictionary
    results = {
        'mode': match_kind,
        'dist': dist,
        'dim': n
    }

    # Compute similarity matrix
    train = train[:n, :]
    test = test[:n, :]
    num_train_samples = train.shape[1]
    num_test_samples = test.shape[1]

    similarity_matrix = np.zeros((num_test_samples, num_train_samples))

    for i in range(num_test_samples):
        for j in range(num_train_samples):
            if dist == 'cos':
                similarity_matrix[i, j] = -cosine(train[:, j], test[:, i])  # Cosine similarity
            elif dist == 'euc':
                similarity_matrix[i, j] = euclidean(train[:, j], test[:, i])  # Euclidean distance

    results['match_dist'] = similarity_matrix
    results['same_cli_id'] = (train_ids == test_ids[:, None]).astype(int)
    results['horizontal_ids'] = train_ids
    results['vertical_ids'] = test_ids

    return results
