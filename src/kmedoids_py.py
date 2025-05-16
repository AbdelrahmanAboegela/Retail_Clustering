import numpy as np
import random

def kmedoids(X, k=4, seed=42, max_iter=300):
    """
    K-Medoids clustering algorithm implementation using Manhattan distance.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data of shape (n_samples, n_features)
    k : int, default=4
        Number of clusters
    seed : int, default=42
        Random seed for reproducibility
    max_iter : int, default=300
        Maximum number of iterations
    
    Returns:
    --------
    numpy.ndarray
        Cluster labels for each data point
    """
    # Initialize random number generator for reproducibility
    rng = random.Random(seed)
    
    # Get number of samples and initialize indices
    n_samples = len(X)
    all_indices = list(range(n_samples))
    
    # Randomly select initial medoids
    medoid_indices = rng.sample(all_indices, k)
    
    # Compute distance matrix (Manhattan distance)
    distance_matrix = np.abs(X[:, None] - X).sum(axis=2)

    # Iterate until convergence or max_iter
    for _ in range(max_iter):
        # Assign each point to the nearest medoid
        labels = [min(range(k), key=lambda m: distance_matrix[i, medoid_indices[m]]) for i in all_indices]
        
        # Check if medoids need to be updated
        medoids_changed = False
        
        # Update medoids for each cluster
        for cluster_idx in range(k):
            # Get indices of points in current cluster
            cluster_points = [i for i, label in enumerate(labels) if label == cluster_idx]
            
            # Skip empty clusters
            if not cluster_points:
                continue
            
            # Find the point that minimizes the sum of distances to all other points in the cluster
            new_medoid = min(cluster_points, 
                            key=lambda point: sum(distance_matrix[point, other] for other in cluster_points))
            
            # Update medoid if it changed
            if new_medoid != medoid_indices[cluster_idx]:
                medoid_indices[cluster_idx] = new_medoid
                medoids_changed = True
        
        # Stop if no medoids changed
        if not medoids_changed:
            break

    # Final assignment of points to clusters
    labels = [min(range(k), key=lambda m: distance_matrix[i, medoid_indices[m]]) for i in all_indices]
    
    return np.array(labels)
