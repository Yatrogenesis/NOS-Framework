"""
Dynamic Representation Space Implementation

This module implements the dynamic representation space concept from the
Neuroplastic Operating Systems framework. It provides a representation space
with dynamic structure that can modify its own topology and dimensionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DynamicRepresentationSpace")

class DynamicRepresentationSpace:
    """
    Implementation of a representation space with dynamic structure
    that can modify its own topology and dimensionality.
    """
    
    def __init__(
        self, 
        initial_dim: int = 8, 
        min_dim: int = 4, 
        max_dim: int = 24,
        topology_type: str = 'euclidean',
        plasticity_rate: float = 0.1,
        use_torch: bool = False
    ):
        """
        Initialize the dynamic representation space.
        
        Args:
            initial_dim: Initial dimensionality of the space
            min_dim: Minimum allowed dimensionality
            max_dim: Maximum allowed dimensionality
            topology_type: Type of initial topology ('euclidean', 'hyperbolic', 'toroidal')
            plasticity_rate: Rate at which the space can modify its structure
            use_torch: Whether to use PyTorch tensors (for GPU acceleration)
        """
        self.current_dim = initial_dim
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.topology_type = topology_type
        self.plasticity_rate = plasticity_rate
        self.use_torch = use_torch
        
        if use_torch:
            import torch
            self.torch = torch
        
        # Initialize content
        self.points = {}  # Maps point_id to coordinates
        
        # Initialize caches
        self.distance_cache = {}  # Cache for distance computations
        self.nearest_neighbors_cache = {}  # Cache for nearest neighbor computations
        
        # Initialize transformation history for analysis
        self.dim_history = [initial_dim]
        self.metric_history = []
        self.topology_history = [topology_type]
        self.energy_history = [1.0]  # Start with full energy
        
        # Create initial metric tensor based on topology
        self.metric_tensor = self._initialize_metric_tensor()
        self.base_metric = 'euclidean'  # Default base metric
        
        # Adaptation parameters
        self.adaptation_energy = 1.0  # Energy available for adaptation
        self.stability_score = 1.0    # Measure of space stability
    
    def _initialize_metric_tensor(self) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Create the initial metric tensor based on selected topology.
        
        Returns:
            Metric tensor as numpy array or torch tensor
        """
        if self.use_torch:
            eye_func = self.torch.eye
        else:
            eye_func = np.eye
            
        if self.topology_type == 'euclidean':
            # Identity matrix = standard Euclidean metric
            return eye_func(self.current_dim)
        
        elif self.topology_type == 'hyperbolic':
            # Minkowski metric for hyperbolic space
            g = eye_func(self.current_dim)
            if self.use_torch:
                g[0, 0] = -1.0  # Time-like dimension has negative sign
            else:
                g[0, 0] = -1.0
            return g
        
        elif self.topology_type == 'toroidal':
            # Periodic boundary conditions with standard metric
            return eye_func(self.current_dim)
        
        else:
            raise ValueError(f"Unsupported topology type: {self.topology_type}")
    
    def compute_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute distance between points using the current metric.
        
        Args:
            point1: First point coordinates
            point2: Second point coordinates
            
        Returns:
            Distance value according to current metric
        """
        # Check if this distance is in cache
        point1_tuple = tuple(point1.flatten())
        point2_tuple = tuple(point2.flatten())
        cache_key = (point1_tuple, point2_tuple)
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Ensure points match current dimensionality
        p1 = self._project_to_current_dim(point1)
        p2 = self._project_to_current_dim(point2)
        
        if self.use_torch:
            # Convert to torch tensors
            p1 = self.torch.tensor(p1, dtype=self.torch.float32)
            p2 = self.torch.tensor(p2, dtype=self.torch.float32)
            metric = self.metric_tensor
        else:
            metric = self.metric_tensor
            
        # Compute distance based on topology
        if self.topology_type == 'euclidean':
            # Standard Euclidean with metric tensor
            diff = p1 - p2
            
            if self.use_torch:
                distance = self.torch.sqrt(diff @ metric @ diff)
                distance = distance.item()  # Convert to Python float
            else:
                distance = np.sqrt(diff.T @ metric @ diff)
                
        elif self.topology_type == 'hyperbolic':
            # Hyperbolic distance using Minkowski inner product
            
            if self.use_torch:
                inner_product = p1 @ metric @ p2
                # Ensure inner product is valid for hyperbolic distance
                inner_product = max(-1.0001, inner_product.item())
                distance = np.arccosh(-inner_product)
            else:
                inner_product = p1.T @ metric @ p2
                # Ensure inner product is valid
                inner_product = max(-1.0001, inner_product)
                distance = np.arccosh(-inner_product)
                
        elif self.topology_type == 'toroidal':
            # Distance on a torus: minimum distance considering wrapping
            
            if self.use_torch:
                # Compute minimum distance considering wrapping
                diff = self.torch.min(
                    self.torch.abs(p1 - p2), 
                    1.0 - self.torch.abs(p1 - p2)
                )
                distance = self.torch.sqrt(diff @ metric @ diff)
                distance = distance.item()
            else:
                diff = np.minimum(np.abs(p1 - p2), 1.0 - np.abs(p1 - p2))
                distance = np.sqrt(diff.T @ metric @ diff)
        
        else:
            raise ValueError(f"Unsupported topology type: {self.topology_type}")
        
        # Cache the result
        self.distance_cache[cache_key] = distance
        self.distance_cache[(point2_tuple, point1_tuple)] = distance  # Symmetry
        
        return distance
    
    def _project_to_current_dim(self, point: np.ndarray) -> np.ndarray:
        """
        Project a point to the current dimensionality of the space.
        
        Args:
            point: Input point
            
        Returns:
            Projected point
        """
        if len(point) == self.current_dim:
            return point
        elif len(point) < self.current_dim:
            # Pad with zeros
            return np.pad(point, (0, self.current_dim - len(point)))
        else:
            # Truncate extra dimensions
            return point[:self.current_dim]
    
    def add_point(self, point_id: str, coordinates: np.ndarray) -> None:
        """
        Add a point to the representation space.
        
        Args:
            point_id: Unique identifier for the point
            coordinates: Coordinates of the point
        """
        # Project to current dimensionality
        projected_coords = self._project_to_current_dim(coordinates)
        
        # Add to points dictionary
        self.points[point_id] = projected_coords
        
        # Clear caches since space has changed
        self.distance_cache = {}
        self.nearest_neighbors_cache = {}
        
        logger.debug(f"Added point {point_id} with dimensionality {len(projected_coords)}")
    
    def update_point(self, point_id: str, new_coordinates: np.ndarray) -> None:
        """
        Update the coordinates of an existing point.
        
        Args:
            point_id: Unique identifier for the point
            new_coordinates: New coordinates for the point
        """
        if point_id not in self.points:
            raise KeyError(f"Point {point_id} does not exist in the space")
        
        # Project to current dimensionality
        projected_coords = self._project_to_current_dim(new_coordinates)
        
        # Update point
        self.points[point_id] = projected_coords
        
        # Clear caches since space has changed
        self.distance_cache = {}
        self.nearest_neighbors_cache = {}
        
        logger.debug(f"Updated point {point_id} with new coordinates")
    
    def remove_point(self, point_id: str) -> None:
        """
        Remove a point from the representation space.
        
        Args:
            point_id: Unique identifier for the point to remove
        """
        if point_id in self.points:
            del self.points[point_id]
            
            # Clear caches since space has changed
            self.distance_cache = {}
            self.nearest_neighbors_cache = {}
            
            logger.debug(f"Removed point {point_id}")
        else:
            logger.warning(f"Attempted to remove non-existent point {point_id}")
    
    def get_nearest_neighbors(self, query_point: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the k nearest neighbors to a query point.
        
        Args:
            query_point: Query point coordinates
            k: Number of neighbors to return
            
        Returns:
            List of (point_id, distance) tuples for the k nearest neighbors
        """
        # Ensure k is not larger than the number of points
        k = min(k, len(self.points))
        
        if k == 0:
            return []
        
        # Compute distances to all points
        distances = []
        
        for point_id, coords in self.points.items():
            distance = self.compute_distance(query_point, coords)
            distances.append((point_id, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return k nearest
        return distances[:k]
    
    def transform_space(
        self, 
        experiences: List[np.ndarray], 
        internal_state: Dict[str, Any] = None
    ) -> bool:
        """
        Transform the structure of the representation space based on
        new experiences and the system's internal state.
        
        Args:
            experiences: List of experience vectors
            internal_state: Optional dictionary of internal state variables
            
        Returns:
            True if transformation was performed, False otherwise
        """
        if internal_state is None:
            internal_state = {}
        
        # Check if there's sufficient energy for transformation
        energy = internal_state.get('energy', self.adaptation_energy)
        if energy < 0.1:
            logger.info("Insufficient energy for space transformation")
            return False
        
        # Track the original state for measuring the magnitude of change
        original_dim = self.current_dim
        original_metric = self.metric_tensor.copy() if not self.use_torch else self.metric_tensor.clone()
        
        # 1. Analyze experiences to guide transformations
        complexity_estimate = self._estimate_complexity(experiences)
        density_estimate = self._estimate_density(experiences)
        curvature_estimate = self._estimate_curvature(experiences)
        
        # 2. Consider dimension adjustments based on complexity
        if complexity_estimate > 0:
            target_dim = max(
                self.min_dim,
                min(self.max_dim, int(original_dim * complexity_estimate))
            )
            
            # Only change if the difference is significant
            if abs(target_dim - original_dim) >= 2:
                logger.info(f"Adjusting dimensionality from {original_dim} to {target_dim}")
                self._adjust_dimensionality(target_dim)
        
        # 3. Adapt metric tensor based on data distribution
        if len(experiences) >= 5:
            self._adapt_metric_tensor(experiences, energy * 0.4)
        
        # 4. Potentially change topology type based on data characteristics
        topology_scores = {
            'euclidean': 0.5,  # Default score
            'hyperbolic': 0.3,
            'toroidal': 0.2
        }
        
        # Update scores based on data properties
        if curvature_estimate < -0.1:
            # Negative curvature suggests hyperbolic
            topology_scores['hyperbolic'] += 0.3
        elif curvature_estimate > 0.1:
            # Positive curvature suggests spherical-like spaces
            topology_scores['euclidean'] += 0.2
        
        if density_estimate > 0.7:
            # High density suggests benefit from toroidal wrapping
            topology_scores['toroidal'] += 0.2
        
        # Adjust by internal state preferences if present
        for topo in topology_scores.keys():
            if f"{topo}_preference" in internal_state:
                topology_scores[topo] += internal_state[f"{topo}_preference"]
        
        # Only consider changing topology with some probability
        if np.random.random() < self.plasticity_rate * energy:
            # Select topology proportional to scores
            normalized_scores = {k: v / sum(topology_scores.values()) for k, v in topology_scores.items()}
            topologies = list(normalized_scores.keys())
            probs = list(normalized_scores.values())
            
            new_topology = np.random.choice(topologies, p=probs)
            
            if new_topology != self.topology_type:
                logger.info(f"Changing topology from {self.topology_type} to {new_topology}")
                self.topology_type = new_topology
                self.metric_tensor = self._initialize_metric_tensor()
        
        # 5. Consume energy proportional to the magnitude of changes made
        dim_change_ratio = abs(self.current_dim - original_dim) / max(1, original_dim)
        
        if self.use_torch:
            metric_change_ratio = self.torch.norm(self.metric_tensor - original_metric) / self.torch.norm(original_metric)
            metric_change_ratio = metric_change_ratio.item()
        else:
            metric_change_ratio = np.linalg.norm(self.metric_tensor - original_metric) / np.linalg.norm(original_metric)
        
        # Energy consumption is proportional to changes made
        energy_consumed = 0.1 + 0.3 * (dim_change_ratio + metric_change_ratio)
        self.adaptation_energy = max(0.0, self.adaptation_energy - energy_consumed)
        
        # 6. Update internal caches
        self.distance_cache = {}
        self.nearest_neighbors_cache = {}
        
        # 7. Record history
        self.dim_history.append(self.current_dim)
        self.metric_history.append(self.base_metric)
        self.energy_history.append(self.adaptation_energy)
        
        return True
    
    def _estimate_complexity(self, experiences: List[np.ndarray]) -> float:
        """
        Estimate the complexity of the experience data to guide dimension adjustment.
        
        Args:
            experiences: List of experience vectors
            
        Returns:
            Complexity estimate (0-2.0), where 1.0 represents no change needed
        """
        if len(experiences) < 5:
            return 1.0  # Not enough data for reliable estimate
        
        try:
            # Convert experiences to array and standardize
            X = np.vstack([self._project_to_current_dim(e) for e in experiences])
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            
            # Simplified intrinsic dimensionality estimation using PCA
            cov = np.cov(X, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            
            if sum(eigenvalues) == 0:
                return 1.0  # No variance in data
            
            # Normalized eigenvalues
            normalized_eigenvalues = eigenvalues / sum(eigenvalues)
            
            # Estimate complexity using participation ratio
            participation_ratio = 1.0 / sum(normalized_eigenvalues**2)
            
            # Normalize to current dimension
            complexity_ratio = participation_ratio / self.current_dim
            
            # Bound the result
            return max(0.5, min(2.0, complexity_ratio))
            
        except Exception as e:
            logger.warning(f"Error in complexity estimation: {e}")
            return 1.0  # Default to no change
    
    def _estimate_density(self, experiences: List[np.ndarray]) -> float:
        """
        Estimate the density of the experience data to guide topology selection.
        
        Args:
            experiences: List of experience vectors
            
        Returns:
            Density estimate (0-1)
        """
        if len(experiences) < 5:
            return 0.5  # Not enough data
        
        try:
            # Convert to standardized array
            X = np.vstack([self._project_to_current_dim(e) for e in experiences])
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(X, 'euclidean')
            
            if len(distances) == 0:
                return 0.5
            
            # Estimate density using average distance
            avg_distance = np.mean(distances)
            max_distance = np.sqrt(self.current_dim)  # Max distance in normalized space
            
            # Convert to density measure (0-1)
            density = 1.0 - (avg_distance / max_distance)
            
            return density
            
        except Exception as e:
            logger.warning(f"Error in density estimation: {e}")
            return 0.5
    
    def _estimate_curvature(self, experiences: List[np.ndarray]) -> float:
        """
        Estimate the curvature of the manifold formed by experiences.
        
        Args:
            experiences: List of experience vectors
            
        Returns:
            Curvature estimate (-1 to 1)
        """
        if len(experiences) < 10:
            return 0.0  # Not enough data for curvature estimation
        
        try:
            # Use a simple triangle-based curvature estimation
            # Select random triplets and measure sum of internal angles
            
            X = np.vstack([self._project_to_current_dim(e) for e in experiences])
            n_samples = min(30, len(experiences) * (len(experiences) - 1) * (len(experiences) - 2) // 6)
            
            curvature_samples = []
            for _ in range(n_samples):
                # Select 3 random points
                indices = np.random.choice(len(X), size=3, replace=False)
                a, b, c = X[indices]
                
                # Compute sides of the triangle
                side_ab = np.linalg.norm(a - b)
                side_bc = np.linalg.norm(b - c)
                side_ca = np.linalg.norm(c - a)
                
                # Check for degenerate triangles
                if min(side_ab, side_bc, side_ca) < 1e-6:
                    continue
                
                # Compute angles using cosine law
                angle_a = np.arccos(
                    np.clip(
                        (side_ab**2 + side_ca**2 - side_bc**2) / (2 * side_ab * side_ca),
                        -1, 1
                    )
                )
                angle_b = np.arccos(
                    np.clip(
                        (side_ab**2 + side_bc**2 - side_ca**2) / (2 * side_ab * side_bc),
                        -1, 1
                    )
                )
                angle_c = np.arccos(
                    np.clip(
                        (side_bc**2 + side_ca**2 - side_ab**2) / (2 * side_bc * side_ca),
                        -1, 1
                    )
                )
                
                # Sum of angles in radians
                angle_sum = angle_a + angle_b + angle_c
                
                # Compare to flat space (pi radians)
                curvature_sample = (angle_sum - np.pi) / np.pi
                curvature_samples.append(curvature_sample)
            
            if not curvature_samples:
                return 0.0
            
            # Remove outliers
            curvature_samples = np.array(curvature_samples)
            q1, q3 = np.percentile(curvature_samples, [25, 75])
            iqr = q3 - q1
            valid_mask = (curvature_samples >= q1 - 1.5 * iqr) & (curvature_samples <= q3 + 1.5 * iqr)
            valid_samples = curvature_samples[valid_mask]
            
            if len(valid_samples) == 0:
                return 0.0
            
            # Return mean curvature estimate
            return np.mean(valid_samples)
            
        except Exception as e:
            logger.warning(f"Error in curvature estimation: {e}")
            return 0.0
    
    def _adjust_dimensionality(self, target_dim: int) -> None:
        """
        Adjust the dimensionality of the space.
        
        Args:
            target_dim: Target dimensionality
        """
        if target_dim == self.current_dim:
            return  # No change needed
        
        # Store original dimensionality
        original_dim = self.current_dim
        self.current_dim = target_dim
        
        # Update metric tensor
        if target_dim > original_dim:
            # Expand metric tensor
            if self.use_torch:
                expanded = self.torch.eye(target_dim, dtype=self.metric_tensor.dtype,
                                     device=self.metric_tensor.device)
                expanded[:original_dim, :original_dim] = self.metric_tensor
                self.metric_tensor = expanded
            else:
                expanded = np.eye(target_dim)
                expanded[:original_dim, :original_dim] = self.metric_tensor
                self.metric_tensor = expanded
        else:
            # Shrink metric tensor
            if self.use_torch:
                self.metric_tensor = self.metric_tensor[:target_dim, :target_dim]
            else:
                self.metric_tensor = self.metric_tensor[:target_dim, :target_dim]
        
        # Update all points in the space
        for point_id in list(self.points.keys()):
            self.points[point_id] = self._project_to_current_dim(self.points[point_id])
        
        logger.info(f"Dimensionality adjusted from {original_dim} to {target_dim}")
    
    def _adapt_metric_tensor(self, experiences: List[np.ndarray], energy: float) -> None:
        """
        Adapt the metric tensor based on the distribution of experiences.
        
        Args:
            experiences: List of experience vectors
            energy: Available energy for adaptation
        """
        if len(experiences) < 5 or energy < 0.05:
            return  # Insufficient data or energy
        
        try:
            # Project experiences to current dimensionality
            X = np.vstack([self._project_to_current_dim(e) for e in experiences])
            
            # Center data
            X = X - np.mean(X, axis=0)
            
            # Compute covariance
            cov = np.cov(X, rowvar=False)
            
            if np.isnan(cov).any() or np.isinf(cov).any():
                logger.warning("Invalid values in covariance matrix")
                return
            
            # Apply regularization to ensure positive definiteness
            cov += 1e-6 * np.eye(cov.shape[0])
            
            # Compute precision matrix (inverse covariance)
            try:
                if self.use_torch:
                    cov_tensor = self.torch.tensor(cov, dtype=self.metric_tensor.dtype,
                                              device=self.metric_tensor.device)
                    precision = self.torch.inverse(cov_tensor)
                else:
                    precision = np.linalg.inv(cov)
            except Exception as e:
                logger.warning(f"Error inverting covariance matrix: {e}")
                return
            
            # Blend current metric tensor with precision matrix
            # Amount of adaptation depends on energy and plasticity rate
            adaptation_factor = self.plasticity_rate * energy
            
            if self.use_torch:
                self.metric_tensor = (1 - adaptation_factor) * self.metric_tensor + adaptation_factor * precision
            else:
                self.metric_tensor = (1 - adaptation_factor) * self.metric_tensor + adaptation_factor * precision
            
            # Ensure positive definiteness
            self._ensure_positive_definite()
            
            logger.info(f"Metric tensor adapted with factor {adaptation_factor:.3f}")
            
        except Exception as e:
            logger.warning(f"Error in metric tensor adaptation: {e}")
    
    def _ensure_positive_definite(self) -> None:
        """Ensure the metric tensor remains positive definite."""
        try:
            if self.use_torch:
                # Get eigenvalues
                eigenvalues = self.torch.linalg.eigvalsh(self.metric_tensor)
                min_eig = self.torch.min(eigenvalues)
                
                # Add regularization if needed
                if min_eig < 1e-6:
                    self.metric_tensor += (1e-6 - min_eig) * self.torch.eye(
                        self.current_dim, 
                        dtype=self.metric_tensor.dtype,
                        device=self.metric_tensor.device
                    )
            else:
                # Get eigenvalues
                eigenvalues = np.linalg.eigvalsh(self.metric_tensor)
                min_eig = np.min(eigenvalues)
                
                # Add regularization if needed
                if min_eig < 1e-6:
                    self.metric_tensor += (1e-6 - min_eig) * np.eye(self.current_dim)
                    
        except Exception as e:
            logger.warning(f"Error ensuring positive definiteness: {e}")
            # Fallback to identity matrix in case of failure
            if self.use_torch:
                self.metric_tensor = self.torch.eye(
                    self.current_dim,
                    dtype=self.metric_tensor.dtype,
                    device=self.metric_tensor.device
                )
            else:
                self.metric_tensor = np.eye(self.current_dim)
    
    def visualize(self, ax=None, dim_reduction=True, annotate=True):
        """
        Visualize the representation space and its points.
        
        Args:
            ax: Matplotlib axis to plot on
            dim_reduction: Whether to use dimensionality reduction for visualization
            annotate: Whether to annotate points with their IDs
        
        Returns:
            Matplotlib axis with the visualization
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            if self.current_dim > 2 and dim_reduction:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        if not self.points:
            ax.text(0.5, 0.5, "Empty representation space", 
                   horizontalalignment='center', verticalalignment='center')
            return ax
        
        # Get coordinates of all points
        coords = np.array(list(self.points.values()))
        
        # Apply dimensionality reduction if needed
        if self.current_dim > 3 and dim_reduction:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(coords)
        else:
            coords_3d = coords[:, :min(3, self.current_dim)]
            
            # Pad if needed
            if coords_3d.shape[1] < 3:
                coords_3d = np.pad(
                    coords_3d, 
                    ((0, 0), (0, 3 - coords_3d.shape[1])), 
                    mode='constant'
                )
        
        # Plot points
        if coords_3d.shape[1] == 3:
            # 3D plot
            ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], s=100, alpha=0.7)
            
            # Add annotations
            if annotate:
                for i, point_id in enumerate(self.points.keys()):
                    ax.text(
                        coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], 
                        point_id, fontsize=9
                    )
                    
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
        else:
            # 2D plot
            ax.scatter(coords_3d[:, 0], coords_3d[:, 1], s=100, alpha=0.7)
            
            # Add annotations
            if annotate:
                for i, point_id in enumerate(self.points.keys()):
                    ax.text(
                        coords_3d[i, 0], coords_3d[i, 1], 
                        point_id, fontsize=9
                    )
                    
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        # Add information about space
        info_text = (f"Dim: {self.current_dim}, Topology: {self.topology_type}\n"
                     f"Energy: {self.adaptation_energy:.2f}, Stability: {self.stability_score:.2f}")
        ax.set_title(info_text)
        
        return ax


if __name__ == "__main__":
    # Example usage
    # Create a dynamic representation space
    space = DynamicRepresentationSpace(
        initial_dim=8,
        min_dim=4,
        max_dim=16,
        topology_type='euclidean',
        plasticity_rate=0.1
    )
    
    # Add some points
    space.add_point("A", np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    space.add_point("B", np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
    space.add_point("C", np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    
    # Generate some synthetic experiences
    experiences = [
        np.random.rand(8) for _ in range(10)
    ]
    
    # Transform the space
    space.transform_space(experiences, {'energy': 0.9})
    
    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    space.visualize(ax=ax)
    plt.show()
