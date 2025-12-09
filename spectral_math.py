import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csgraph
from scipy.linalg import eigh

class SpectralExtractor:
    def __init__(self, radius=0.70, n_eigenvalues=10):
        self.radius = radius 
        self.k = n_eigenvalues

    def compute_features(self, points, return_math=False):
        """
        Computes the Laplacian Spectral features using the Normalized Laplacian.
        
        Returns:
            eigenvalues: The first k non-zero eigenvalues (1D array)
            eigenvectors: The corresponding eigenvectors (Matrix of shape [N_points, k])
        """
        # We need at least K+1 points to find K eigenvalues (skipping the 0-th)
        if len(points) < self.k + 1:
            # Handle sparse clouds gracefully with zero-padding
            empty_vals = np.zeros(self.k)
            # Default eigenvector shape (N, k) - approximated as (1, k) for safety
            empty_vecs = np.zeros((len(points) if len(points) > 0 else 1, self.k))
            
            if return_math:
                return empty_vals, empty_vecs, np.zeros((1,1)), np.zeros((1,1))
            return empty_vals, empty_vecs

        # 1. Build Adjacency Matrix (The Epsilon-Graph)
        # Calculate Euclidean distances between all pairs of points
        dists = distance_matrix(points, points)
        
        # Create unweighted edges where distance < radius
        adj_matrix = (dists < self.radius).astype(float)
        np.fill_diagonal(adj_matrix, 0) # Remove self-loops

        # 2. Compute Laplacian (Normalized)
        # MATH FIX: Use Normalized Laplacian (L_sym = I - D^-0.5 * A * D^-0.5)
        # This makes the spectrum robust to sampling density variations.
        laplacian = csgraph.laplacian(adj_matrix, normed=True)

        # 3. Eigen Decomposition
        try:
            # Solve generalized eigenvalue problem for Hermitian matrices
            # subset_by_index=[1, k] skips index 0 (constant component, val ~ 0)
            eigvals, eigvecs = eigh(laplacian, eigvals_only=False, subset_by_index=[1, self.k])
            
            # 4. Zero-Padding (Safety Mechanism)
            # If the graph has disconnected components, we might get fewer valid eigenpairs
            if len(eigvals) < self.k:
                pad_width = self.k - len(eigvals)
                eigvals = np.pad(eigvals, (0, pad_width), 'constant')
                # Pad eigenvectors (columns)
                eigvecs = np.pad(eigvecs, ((0,0), (0, pad_width)), 'constant')
                
        except ValueError:
            # Fallback for numerical instability
            eigvals = np.zeros(self.k)
            eigvecs = np.zeros((len(points), self.k))

        if return_math:
            return eigvals, eigvecs, adj_matrix, laplacian
            
        return eigvals, eigvecs