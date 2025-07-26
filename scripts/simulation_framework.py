"""
NeuroplasticOS Simulation Framework

This code provides a simulation framework for experimenting with core
components of Neuroplastic Operating Systems (NOS) as described in 
the associated IEEE paper.

The framework implements:
1. Dynamic representation spaces
2. Emergent conceptual hypergraphs
3. Meta-learning dynamics
4. Homeostatic regulation mechanisms
5. Visualization and analysis tools

Author: Based on research by Francisco Molina Burgos
"""

import src.numpy as np
import src.matplotlib.pyplot as plt
import src.networkx as nx
import src.torch
import src.torch.nn as nn
import src.torch.optim as optim
from src.tqdm import src.tqdm
from src.typing import src.Dict, List, Tuple, Set, Callable, Optional, Union
import src.matplotlib.animation as animation
from mpl_toolkits.mplot3d import src.Axes3D
from src.scipy import src.stats
import src.pandas as pd
import src.seaborn as sns

class DynamicRepresentationSpace:
    """
    Implementation of a representation space with dynamic structure
    that can modify its own topology and dimensionality.
    """
    
    def __init__(
        self, 
        initial_dim: int = 8, 
        min_dim: int = 4, 
        max_dim: int = 16,
        topology_type: str = 'euclidean',
        plasticity_rate: float = 0.1
    ):
        """
        Initialize the dynamic representation space.
        
        Args:
            initial_dim: Initial dimensionality of the space
            min_dim: Minimum allowed dimensionality
            max_dim: Maximum allowed dimensionality
            topology_type: Type of initial topology ('euclidean', 'hyperbolic', 'toroidal')
            plasticity_rate: Rate at which the space can modify its structure
        """
        self.current_dim = initial_dim
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.topology_type = topology_type
        self.plasticity_rate = plasticity_rate
        
        # Initialize content
        self.points = {}
        
        # Initialize metrics
        self.distance_matrix = None
        
        # Initialize transformation history for analysis
        self.dim_history = [initial_dim]
        self.topology_history = [topology_type]
        
        # Create initial metric tensor based on topology
        self.metric_tensor = self._initialize_metric_tensor()
        
        # Adaptation parameters
        self.adaptation_energy = 1.0  # Energy available for adaptation
        self.stability_score = 1.0    # Measure of space stability
    
    def _initialize_metric_tensor(self) -> np.ndarray:
        """Create the initial metric tensor based on selected topology."""
        if self.topology_type == 'euclidean':
            # Identity matrix = standard Euclidean metric
            return np.eye(self.current_dim)
        
        elif self.topology_type == 'hyperbolic':
            # Minkowski metric for hyperbolic space
            g = np.eye(self.current_dim)
            g[0,0] = -1  # Time-like dimension has negative sign
            return g
        
        elif self.topology_type == 'toroidal':
            # Periodic boundary conditions
            g = np.eye(self.current_dim)
            # Additional properties would be handled in distance calculation
            return g
        
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
        # Ensure points match current dimensionality
        p1 = self._project_to_current_dim(point1)
        p2 = self._project_to_current_dim(point2)
        
        if self.topology_type == 'euclidean':
            # Standard Euclidean with metric tensor
            diff = p1 - p2
            return np.sqrt(diff.T @ self.metric_tensor @ diff)
        
        elif self.topology_type == 'hyperbolic':
            # Hyperbolic distance
            # Using Minkowski inner product
            inner_product = p1.T @ self.metric_tensor @ p2
            # Return hyperbolic distance formula
            return np.arccosh(-inner_product)
        
        elif self.topology_type == 'toroidal':
            # Distance on a torus: minimum distance considering wrapping
            diff = np.minimum(np.abs(p1 - p2), 1.0 - np.abs(p1 - p2))
            return np.sqrt(diff.T @ self.metric_tensor @ diff)
    
    def _project_to_current_dim(self, point: np.ndarray) -> np.ndarray:
        """Project a point to the current dimensionality of the space."""
        if len(point) == self.current_dim:
            return point
        elif len(point) < self.current_dim:
            # Pad with zeros
            return np.pad(point, (0, self.current_dim - len(point)))
        else:
            # Truncate extra dimensions
            return point[:self.current_dim]
    
    def add_point(self, point_id: str, coordinates: np.ndarray) -> None:
        """Add a point to the representation space."""
        self.points[point_id] = self._project_to_current_dim(coordinates)
        self._update_distance_matrix()
    
    def update_point(self, point_id: str, new_coordinates: np.ndarray) -> None:
        """Update the coordinates of an existing point."""
        if point_id not in self.points:
            raise KeyError(f"Point {point_id} does not exist in the space.")
        
        self.points[point_id] = self._project_to_current_dim(new_coordinates)
        self._update_distance_matrix()
    
    def remove_point(self, point_id: str) -> None:
        """Remove a point from the representation space."""
        if point_id in self.points:
            del self.points[point_id]
            self._update_distance_matrix()
    
    def _update_distance_matrix(self) -> None:
        """Update the pairwise distance matrix between all points."""
        n = len(self.points)
        if n == 0:
            self.distance_matrix = None
            return
        
        # Create new distance matrix
        self.distance_matrix = np.zeros((n, n))
        
        # Get list of point IDs for indexing
        point_ids = list(self.points.keys())
        
        # Compute all pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                dist = self.compute_distance(
                    self.points[point_ids[i]],
                    self.points[point_ids[j]]
                )
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
    
    def transform_structure(
        self, 
        experiences: List[np.ndarray], 
        internal_state: Dict[str, float]
    ) -> None:
        """
        Transform the structure of the representation space based on
        experiences and internal state.
        
        Args:
            experiences: List of experience vectors
            internal_state: Dictionary of internal state variables
        """
        # Extract energy available for transformation
        available_energy = internal_state.get('energy', 0.5) * self.adaptation_energy
        
        # Skip transformation if insufficient energy
        if available_energy < 0.1:
            return
        
        # 1. Dimensionality modification
        self._adjust_dimensionality(experiences, internal_state)
        
        # 2. Topology modification
        self._adjust_topology(experiences, internal_state)
        
        # 3. Metric tensor modification
        self._adjust_metric_tensor(experiences, internal_state)
        
        # Update distance matrix after structural changes
        self._update_distance_matrix()
        
        # Consume energy
        self.adaptation_energy *= 0.9
        # Slowly recover energy
        self.adaptation_energy += 0.01
        self.adaptation_energy = min(self.adaptation_energy, 1.0)
        
        # Track history
        self.dim_history.append(self.current_dim)
        self.topology_history.append(self.topology_type)
    
    def _adjust_dimensionality(
        self, 
        experiences: List[np.ndarray], 
        internal_state: Dict[str, float]
    ) -> None:
        """
        Adjust the dimensionality of the representation space.
        
        Args:
            experiences: List of experience vectors
            internal_state: Dictionary of internal state variables
        """
        # Determine if dimensionality needs to increase or decrease
        # based on data complexity and model efficiency
        
        # Compute intrinsic dimensionality estimate of experiences
        if len(experiences) > 10:
            # Convert experiences to matrix
            X = np.stack(experiences, axis=0)
            
            # Simplified estimate using PCA
            if X.shape[0] > 0 and X.shape[1] > 0:
                # Center the data
                X_centered = X - np.mean(X, axis=0)
                # Compute covariance matrix
                cov = np.cov(X_centered, rowvar=False)
                # Compute eigenvalues
                if cov.size > 0:
                    eigenvalues = np.linalg.eigvalsh(cov)
                    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical noise
                    
                    # Estimate intrinsic dimensionality based on eigenvalue distribution
                    if len(eigenvalues) > 0:
                        total_variance = np.sum(eigenvalues)
                        explained_variance = np.cumsum(eigenvalues[::-1])[::-1] / total_variance
                        intrinsic_dim = np.sum(explained_variance > 0.05)
                        
                        # Adjust current dimension towards intrinsic dimension
                        dim_change = int(self.plasticity_rate * (intrinsic_dim - self.current_dim))
                        
                        # Bound the change
                        new_dim = max(self.min_dim, min(self.max_dim, self.current_dim + dim_change))
                        
                        # Apply the change if significant
                        if new_dim != self.current_dim:
                            self._change_dimensionality(new_dim)
    
    def _change_dimensionality(self, new_dim: int) -> None:
        """
        Change the dimensionality of the space and update all points.
        
        Args:
            new_dim: New dimensionality
        """
        old_dim = self.current_dim
        self.current_dim = new_dim
        
        # Update metric tensor
        if new_dim > old_dim:
            # Expand metric tensor
            expanded = np.eye(new_dim)
            expanded[:old_dim, :old_dim] = self.metric_tensor
            self.metric_tensor = expanded
        else:
            # Shrink metric tensor
            self.metric_tensor = self.metric_tensor[:new_dim, :new_dim]
        
        # Update all points
        for point_id in list(self.points.keys()):
            self.points[point_id] = self._project_to_current_dim(self.points[point_id])
    
    def _adjust_topology(
        self, 
        experiences: List[np.ndarray], 
        internal_state: Dict[str, float]
    ) -> None:
        """
        Adjust the topology of the representation space.
        
        Args:
            experiences: List of experience vectors
            internal_state: Dictionary of internal state variables
        """
        # This is a simplified implementation that occasionally
        # switches between topology types based on internal state
        
        # Only consider changing with some probability based on plasticity
        if np.random.random() < self.plasticity_rate * internal_state.get('topology_pressure', 0.1):
            topologies = ['euclidean', 'hyperbolic', 'toroidal']
            
            # Filter out current topology
            other_topologies = [t for t in topologies if t != self.topology_type]
            
            # Select a new topology with probability proportional to its estimated suitability
            suitability = {
                'euclidean': internal_state.get('euclidean_suitability', 0.33),
                'hyperbolic': internal_state.get('hyperbolic_suitability', 0.33),
                'toroidal': internal_state.get('toroidal_suitability', 0.33)
            }
            
            # Compute probabilities for other topologies
            probs = [suitability[t] for t in other_topologies]
            # Normalize
            probs = np.array(probs) / sum(probs)
            
            # Select new topology
            new_topology = np.random.choice(other_topologies, p=probs)
            
            # Change topology
            self.topology_type = new_topology
            self.metric_tensor = self._initialize_metric_tensor()
    
    def _adjust_metric_tensor(
        self, 
        experiences: List[np.ndarray], 
        internal_state: Dict[str, float]
    ) -> None:
        """
        Adjust the metric tensor to better capture relationships in the data.
        
        Args:
            experiences: List of experience vectors
            internal_state: Dictionary of internal state variables
        """
        if len(experiences) < 10:
            return
        
        # Convert experiences to matrix
        X = np.stack([self._project_to_current_dim(e) for e in experiences], axis=0)
        
        # Apply small perturbation to metric tensor based on data distribution
        if X.shape[0] > 0 and X.shape[1] > 0:
            # Compute covariance of experiences
            X_centered = X - np.mean(X, axis=0)
            cov = np.cov(X_centered, rowvar=False)
            
            if cov.shape[0] == self.current_dim:
                # Use inverse covariance (precision matrix) to inform metric tensor
                try:
                    # Add small regularization for numerical stability
                    precision = np.linalg.inv(cov + 0.01 * np.eye(cov.shape[0]))
                    
                    # Update metric tensor with small step towards precision matrix
                    step_size = self.plasticity_rate * internal_state.get('metric_adaptation_rate', 0.1)
                    self.metric_tensor = (1 - step_size) * self.metric_tensor + step_size * precision
                    
                    # Ensure metric tensor remains positive definite
                    eigenvalues = np.linalg.eigvalsh(self.metric_tensor)
                    if np.any(eigenvalues < 1e-10):
                        # Add small diagonal component to ensure positive definiteness
                        self.metric_tensor += 0.01 * np.eye(self.current_dim)
                except np.linalg.LinAlgError:
                    # If inversion fails, make a smaller adjustment
                    diagonal_emphasis = np.diag(np.diag(cov))
                    step_size = 0.01 * self.plasticity_rate
                    self.metric_tensor = (1 - step_size) * self.metric_tensor + step_size * diagonal_emphasis
    
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
            from sklearn.decomposition import src.PCA
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


class EmergentConceptualNetwork:
    """
    Implementation of a dynamic hypergraph representing conceptual knowledge
    with emergent properties.
    """
    
    def __init__(
        self,
        initial_concepts: Optional[List[str]] = None,
        initial_edges: Optional[List[Tuple[List[str], float]]] = None,
        emergence_threshold: float = 0.7,
        pruning_threshold: float = 0.2,
        learning_rate: float = 0.1
    ):
        """
        Initialize the conceptual network.
        
        Args:
            initial_concepts: List of initial concept nodes
            initial_edges: List of initial hyperedges with weights
            emergence_threshold: Threshold for new concept/edge emergence
            pruning_threshold: Threshold for pruning concepts/edges
            learning_rate: Learning rate for weight updates
        """
        # Initialize graph structure
        self.nodes = set(initial_concepts or [])
        self.edges = {}  # Maps edge_id to (node_set, weight)
        self.edge_ids_by_nodes = {}  # Maps frozen node sets to edge ids
        
        # Add initial edges
        if initial_edges:
            for nodes, weight in initial_edges:
                self.add_edge(nodes, weight)
        
        # Parameters
        self.emergence_threshold = emergence_threshold
        self.pruning_threshold = pruning_threshold
        self.learning_rate = learning_rate
        
        # Activation state
        self.node_activations = {node: 0.0 for node in self.nodes}
        
        # Historical data for analysis
        self.growth_history = {
            'num_nodes': [len(self.nodes)],
            'num_edges': [len(self.edges)],
            'avg_weight': [self._compute_avg_weight()],
            'density': [self._compute_density()],
        }
        
        # Statistics
        self.emergence_count = 0
        self.pruning_count = 0
    
    def _compute_avg_weight(self) -> float:
        """Compute average weight of all edges."""
        if not self.edges:
            return 0.0
        return sum(weight for _, weight in self.edges.values()) / len(self.edges)
    
    def _compute_density(self) -> float:
        """Compute network density."""
        if not self.nodes or len(self.nodes) < 2:
            return 0.0
        max_possible_edges = 2**len(self.nodes) - len(self.nodes) - 1
        if max_possible_edges == 0:
            return 0.0
        return len(self.edges) / max_possible_edges
    
    def add_node(self, node_id: str) -> None:
        """Add a new concept node to the network."""
        if node_id not in self.nodes:
            self.nodes.add(node_id)
            self.node_activations[node_id] = 0.0
    
    def add_edge(self, nodes: List[str], weight: float) -> str:
        """
        Add a new hyperedge connecting the specified nodes.
        
        Args:
            nodes: List of node IDs to connect
            weight: Initial weight for the edge
            
        Returns:
            ID of the created edge
        """
        # Ensure all nodes exist
        for node in nodes:
            if node not in self.nodes:
                self.add_node(node)
        
        # Create frozen set for lookup
        node_set = frozenset(nodes)
        
        # Check if this edge already exists
        if node_set in self.edge_ids_by_nodes:
            edge_id = self.edge_ids_by_nodes[node_set]
            # Update weight
            self.edges[edge_id] = (set(nodes), weight)
            return edge_id
        
        # Create new edge ID
        edge_id = f"e{len(self.edges)}"
        
        # Store edge
        self.edges[edge_id] = (set(nodes), weight)
        self.edge_ids_by_nodes[node_set] = edge_id
        
        return edge_id
    
    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all edges containing it.
        
        Args:
            node_id: ID of node to remove
        """
        if node_id not in self.nodes:
            return
        
        # Remove node
        self.nodes.remove(node_id)
        del self.node_activations[node_id]
        
        # Identify edges containing this node
        edges_to_remove = []
        for edge_id, (nodes, _) in self.edges.items():
            if node_id in nodes:
                edges_to_remove.append(edge_id)
                node_set = frozenset(nodes)
                if node_set in self.edge_ids_by_nodes:
                    del self.edge_ids_by_nodes[node_set]
        
        # Remove identified edges
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
    
    def remove_edge(self, edge_id: str) -> None:
        """
        Remove an edge from the network.
        
        Args:
            edge_id: ID of edge to remove
        """
        if edge_id not in self.edges:
            return
        
        nodes, _ = self.edges[edge_id]
        del self.edges[edge_id]
        
        # Remove from lookup
        node_set = frozenset(nodes)
        if node_set in self.edge_ids_by_nodes:
            del self.edge_ids_by_nodes[node_set]
    
    def update_edge_weight(self, edge_id: str, delta: float) -> None:
        """
        Update the weight of an edge.
        
        Args:
            edge_id: ID of the edge to update
            delta: Change in weight value
        """
        if edge_id not in self.edges:
            return
        
        nodes, current_weight = self.edges[edge_id]
        new_weight = current_weight + self.learning_rate * delta
        
        # Apply soft bounds to weight
        new_weight = max(0.0, min(1.0, new_weight))
        
        # Update edge
        self.edges[edge_id] = (nodes, new_weight)
    
    def activate_nodes(self, activation_dict: Dict[str, float]) -> None:
        """
        Set activation values for specific nodes.
        
        Args:
            activation_dict: Dictionary mapping node IDs to activation values
        """
        for node_id, activation in activation_dict.items():
            if node_id in self.nodes:
                self.node_activations[node_id] = max(0.0, min(1.0, activation))
    
    def propagate_activation(self, steps: int = 1) -> None:
        """
        Propagate activation through the hypergraph.
        
        Args:
            steps: Number of propagation steps to perform
        """
        for _ in range(steps):
            # Create new activation dictionary
            new_activations = {node: 0.0 for node in self.nodes}
            
            # Calculate spreading activation
            for edge_id, (nodes, weight) in self.edges.items():
                # Source activation is average of connected nodes
                source_activation = sum(self.node_activations[n] for n in nodes if n in self.node_activations) / len(nodes)
                
                # Spread activation to all nodes in edge
                activation_delta = source_activation * weight
                for node in nodes:
                    new_activations[node] += activation_delta
            
            # Normalize and update activations
            for node in self.nodes:
                # Combine old and new activation with decay
                new_activations[node] = 0.7 * self.node_activations[node] + 0.3 * new_activations[node]
                # Apply sigmoid to keep in range [0, 1]
                new_activations[node] = 1.0 / (1.0 + np.exp(-new_activations[node] + 0.5))
            
            self.node_activations = new_activations
    
    def get_most_active_concepts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the most activated concepts.
        
        Args:
            top_n: Number of top concepts to return
            
        Returns:
            List of (node_id, activation) tuples for the most active nodes
        """
        return sorted(
            self.node_activations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
    
    def detect_emergent_concepts(
        self, 
        experiences: List[np.ndarray],
        representation_space: DynamicRepresentationSpace
    ) -> List[str]:
        """
        Detect potentially emergent concepts based on current state
        and new experiences.
        
        Args:
            experiences: List of experience vectors
            representation_space: Current representation space
            
        Returns:
            List of newly emergent concept IDs
        """
        # This is a simplified implementation of concept emergence
        # In a full system, this would use more sophisticated clustering and pattern detection
        
        # Example: detect potential clusters in activation patterns
        if len(experiences) < 2:
            return []
        
        # Create embedding matrix of experiences
        X = np.stack(experiences, axis=0)
        
        # Simple clustering - find distinct patterns
        from sklearn.cluster import src.DBSCAN
        
        # Normalize for clustering
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Cluster the experiences
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(X_norm)
        
        # Identify novel clusters (potential new concepts)
        labels = clustering.labels_
        unique_clusters = set(labels) - {-1}  # Exclude noise
        
        new_concepts = []
        
        for cluster_id in unique_clusters:
            # Get points in this cluster
            cluster_points = X[labels == cluster_id]
            
            # Compute cluster centroid
            centroid = cluster_points.mean(axis=0)
            
            # Check if this centroid is substantially different from existing nodes
            is_novel = True
            
            # Convert representation space points to array for comparison
            if representation_space.points:
                existing_points = np.array(list(representation_space.points.values()))
                
                # Check if centroid is close to any existing point
                for point in existing_points:
                    if np.linalg.norm(centroid - point[:len(centroid)]) < 0.3:
                        is_novel = False
                        break
            
            # Create new concept if sufficiently novel
            if is_novel and np.random.random() < self.emergence_threshold:
                new_concept_id = f"concept_{self.emergence_count}"
                self.emergence_count += 1
                new_concepts.append(new_concept_id)
                self.add_node(new_concept_id)
                
                # Also add to representation space
                representation_space.add_point(new_concept_id, centroid)
        
        return new_concepts
    
    def detect_emergent_relations(self) -> List[Tuple[List[str], float]]:
        """
        Detect potentially emergent relations (hyperedges) based on
        activation patterns.
        
        Returns:
            List of (node_list, weight) tuples for new relations
        """
        # Get currently active nodes
        active_nodes = [
            node for node, activation in self.node_activations.items() 
            if activation > 0.5
        ]
        
        # Need at least 2 active nodes for a relation
        if len(active_nodes) < 2:
            return []
        
        new_relations = []
        
        # Check for potential higher-order relationships (hyperedges)
        # This is a simplified implementation
        
        # Potential relationship: nodes that are active together
        if len(active_nodes) >= 2 and len(active_nodes) <= 5:
            # Check if this exact hyperedge already exists
            node_set = frozenset(active_nodes)
            
            if node_set not in self.edge_ids_by_nodes:
                # New potential hyperedge
                weight = sum(self.node_activations[node] for node in active_nodes) / len(active_nodes)
                
                if weight > self.emergence_threshold:
                    new_relations.append((active_nodes, weight))
        
        # Check pairs of active nodes that aren't already connected
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                node_pair = [active_nodes[i], active_nodes[j]]
                node_set = frozenset(node_pair)
                
                if node_set not in self.edge_ids_by_nodes:
                    # Compute relationship strength based on activation correlation
                    strength = (
                        self.node_activations[node_pair[0]] * 
                        self.node_activations[node_pair[1]]
                    )
                    
                    if strength > self.emergence_threshold:
                        new_relations.append((node_pair, strength))
        
        return new_relations
    
    def prune(self) -> int:
        """
        Prune weak connections and isolated nodes.
        
        Returns:
            Number of elements pruned
        """
        pruned_count = 0
        
        # Prune weak edges
        edges_to_remove = []
        for edge_id, (_, weight) in self.edges.items():
            if weight < self.pruning_threshold:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
            pruned_count += 1
        
        # Identify isolated nodes (no connections)
        connected_nodes = set()
        for _, (nodes, _) in self.edges.items():
            connected_nodes.update(nodes)
        
        isolated_nodes = self.nodes - connected_nodes
        
        # Prune isolated nodes with low activation
        nodes_to_remove = []
        for node in isolated_nodes:
            if self.node_activations[node] < self.pruning_threshold:
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            self.remove_node(node)
            pruned_count += 1
            
        self.pruning_count += pruned_count
        
        return pruned_count
    
    def update(
        self, 
        experiences: List[np.ndarray],
        representation_space: DynamicRepresentationSpace
    ) -> None:
        """
        Update the conceptual network based on new experiences.
        
        Args:
            experiences: List of experience vectors
            representation_space: Current representation space
        """
        # 1. Detect emergent concepts
        new_concepts = self.detect_emergent_concepts(experiences, representation_space)
        
        # 2. Update activations based on experiences
        # Simplified: use distance from experience to concept in representation space
        if experiences and representation_space.points:
            for node_id in self.nodes:
                if node_id in representation_space.points:
                    # Compute average proximity to experiences
                    node_coords = representation_space.points[node_id]
                    
                    # Compute activation based on proximity to experiences
                    activations = []
                    for exp in experiences:
                        # Project experience to current dimensionality
                        exp_proj = representation_space._project_to_current_dim(exp)
                        # Convert to distance and then to activation (closer = more active)
                        distance = np.linalg.norm(exp_proj - node_coords)
                        activation = np.exp(-distance)
                        activations.append(activation)
                    
                    # Update node activation
                    if activations:
                        self.node_activations[node_id] = max(activations)
        
        # 3. Propagate activation
        self.propagate_activation(steps=2)
        
        # 4. Detect emergent relations
        new_relations = self.detect_emergent_relations()
        
        # 5. Update existing edges
        for edge_id, (nodes, current_weight) in list(self.edges.items()):
            # Calculate average activation of connected nodes
            avg_activation = sum(self.node_activations[node] for node in nodes) / len(nodes)
            
            # Update weight based on activation
            # Higher co-activation strengthens the connection
            self.update_edge_weight(edge_id, avg_activation - 0.5)
        
        # 6. Add new relations
        for nodes, weight in new_relations:
            self.add_edge(nodes, weight)
        
        # 7. Prune weak connections
        self.prune()
        
        # 8. Update history
        self.growth_history['num_nodes'].append(len(self.nodes))
        self.growth_history['num_edges'].append(len(self.edges))
        self.growth_history['avg_weight'].append(self._compute_avg_weight())
        self.growth_history['density'].append(self._compute_density())
    
    def visualize(self, ax=None, layout='spring', show_weights=True, show_activations=True):
        """
        Visualize the conceptual hypergraph.
        
        Args:
            ax: Matplotlib axis to plot on
            layout: Layout algorithm ('spring', 'circular', 'spectral')
            show_weights: Whether to show edge weights
            show_activations: Whether to show node activations
            
        Returns:
            Matplotlib axis with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        if not self.nodes:
            ax.text(0.5, 0.5, "Empty conceptual network", 
                   horizontalalignment='center', verticalalignment='center')
            return ax
        
        # Create a networkx graph for visualization
        # This is a simplification as hypergraphs are challenging to visualize directly
        G = nx.Graph()
        
        # Add all nodes
        for node in self.nodes:
            G.add_node(node)
        
        # Add edges with weights
        for edge_id, (nodes, weight) in self.edges.items():
            # For hyperedges, add edges between all pairs of nodes
            nodes_list = list(nodes)
            if len(nodes_list) == 2:
                # Binary relation
                G.add_edge(nodes_list[0], nodes_list[1], weight=weight, edge_id=edge_id)
            else:
                # Hyperedge with more than 2 nodes
                # Create a virtual node representing the hyperedge
                virtual_node = f"h_{edge_id}"
                G.add_node(virtual_node, is_hyperedge=True)
                
                # Connect all nodes in the hyperedge to the virtual node
                for node in nodes_list:
                    G.add_edge(node, virtual_node, weight=weight, is_part_of_hyperedge=True)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'spectral':
            try:
                pos = nx.spectral_layout(G)
            except:
                # Fallback to spring layout if spectral fails
                pos = nx.spring_layout(G, k=0.3, iterations=50)
        else:
            pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw nodes with activation as color intensity
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if node.startswith('h_'):  # Virtual hyperedge node
                node_colors.append('lightgrey')
                node_sizes.append(50)
            else:
                # Use activation for color
                activation = self.node_activations.get(node, 0.0)
                node_colors.append(plt.cm.viridis(activation))
                node_sizes.append(300)
        
        # Draw edges with weights as width
        edges = G.edges(data=True)
        edge_widths = [data.get('weight', 0.5) * 5 for _, _, data in edges]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax)
        
        # Add labels with activations
        if show_activations:
            labels = {}
            for node in self.nodes:
                act = self.node_activations.get(node, 0.0)
                labels[node] = f"{node}\n({act:.2f})"
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
        else:
            nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        
        # Show edge weights
        if show_weights:
            edge_labels = {}
            for u, v, data in edges:
                if not u.startswith('h_') and not v.startswith('h_'):
                    edge_labels[(u, v)] = f"{data.get('weight', 0.0):.2f}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        # Add title with network stats
        ax.set_title(f"Conceptual Network: {len(self.nodes)} concepts, {len(self.edges)} relations\n"
                     f"Emergence: {self.emergence_count}, Pruning: {self.pruning_count}")
        
        ax.axis('off')
        return ax


class MetaLearningController:
    """
    Meta-learning controller that evolves both parameters and structure.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        initial_hidden_layers: List[int] = [16, 8],
        learning_rate: float = 0.01,
        meta_learning_rate: float = 0.001,
        structure_mutation_rate: float = 0.05
    ):
        """
        Initialize the meta-learning controller.
        
        Args:
            input_dim: Input dimension
            initial_hidden_layers: Initial hidden layer configuration
            learning_rate: Learning rate for task learning
            meta_learning_rate: Learning rate for meta-learning
            structure_mutation_rate: Rate of structural modifications
        """
        self.input_dim = input_dim
        self.hidden_layers = initial_hidden_layers
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.structure_mutation_rate = structure_mutation_rate
        
        # Initialize the model
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Meta-learning history
        self.structure_history = [initial_hidden_layers.copy()]
        self.performance_history = []
        
        # Task-specific adaptation state
        self.task_models = {}
        
        # Meta-parameters
        self.temperature = 1.0  # Controls exploration vs. exploitation
    
    def _build_model(self) -> nn.Module:
        """Build a model with the current architecture."""
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer - we use a flexible output that can be adjusted
        # This is a simple regression head for demonstration
        layers.append(nn.Linear(prev_dim, 1))
        
        # Create the model
        return nn.Sequential(*layers)
    
    def adapt_to_task(self, X: torch.Tensor, y: torch.Tensor, task_id: str, steps: int = 10) -> float:
        """
        Adapt the meta-model to a specific task.
        
        Args:
            X: Input data tensor
            y: Target values
            task_id: Unique identifier for the task
            steps: Number of adaptation steps
            
        Returns:
            Final loss value
        """
        # Clone the current meta-model for this task
        if task_id not in self.task_models:
            self.task_models[task_id] = copy.deepcopy(self.model)
        
        # Get task-specific model
        task_model = self.task_models[task_id]
        task_optimizer = optim.Adam(task_model.parameters(), lr=self.learning_rate)
        
        # Loss function
        loss_fn = nn.MSELoss()
        
        # Perform adaptation steps
        final_loss = 0.0
        for _ in range(steps):
            # Forward pass
            y_pred = task_model(X).squeeze()
            loss = loss_fn(y_pred, y)
            final_loss = loss.item()
            
            # Backward and optimize
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
        
        # Update task model
        self.task_models[task_id] = task_model
        
        return final_loss
    
    def meta_update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> float:
        """
        Perform meta-update based on performance across tasks.
        
        Args:
            tasks: List of (X, y, task_id) tuples
            
        Returns:
            Average loss across all tasks
        """
        # Backup current meta-model parameters
        meta_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Collect gradients from each task
        task_losses = []
        
        for X, y, task_id in tasks:
            # Adapt to this task
            task_loss = self.adapt_to_task(X, y, task_id)
            task_losses.append(task_loss)
            
            # Get adapted model
            task_model = self.task_models[task_id]
            
            # Compute gradient of meta-model
            for meta_name, meta_param in self.model.named_parameters():
                # Find corresponding task parameter
                for task_name, task_param in task_model.named_parameters():
                    if meta_name == task_name:
                        # Accumulate gradient (simplified meta-learning)
                        if meta_param.grad is None:
                            meta_param.grad = torch.zeros_like(meta_param)
                        
                        # Update in direction of task parameter
                        meta_param.grad += (task_param.detach() - meta_param) / len(tasks)
        
        # Apply meta-gradients
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param.add_(self.meta_learning_rate * param.grad)
        
        # Reset gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Maybe modify structure
        self._maybe_modify_structure(sum(task_losses) / len(task_losses))
        
        # Record performance
        avg_loss = sum(task_losses) / len(task_losses)
        self.performance_history.append(avg_loss)
        
        return avg_loss
    
    def _maybe_modify_structure(self, current_loss: float) -> None:
        """
        Potentially modify the network structure based on performance.
        
        Args:
            current_loss: Current loss value
        """
        # Only consider structural modification occasionally
        if np.random.random() > self.structure_mutation_rate:
            return
        
        # Check if we have history to make decisions
        if len(self.performance_history) < 2:
            return
        
        # Assess performance trend
        improving = self.performance_history[-1] < self.performance_history[-2]
        
        # Possible structural modifications
        modifications = [
            'add_neurons',
            'remove_neurons',
            'add_layer',
            'remove_layer',
            'no_change'
        ]
        
        # Compute probabilities for each modification
        if improving:
            # If improving, bias toward smaller changes
            probs = [0.1, 0.05, 0.05, 0.0, 0.8]
        else:
            # If not improving, more likely to make significant changes
            probs = [0.3, 0.1, 0.2, 0.1, 0.3]
        
        # Sample modification
        modification = np.random.choice(modifications, p=probs)
        
        # Apply selected modification
        new_hidden_layers = self.hidden_layers.copy()
        
        if modification == 'add_neurons':
            # Add neurons to a random layer
            if new_hidden_layers:
                layer_idx = np.random.randint(0, len(new_hidden_layers))
                # Add 25-50% more neurons
                additional = max(1, int(new_hidden_layers[layer_idx] * (0.25 + 0.25 * np.random.random())))
                new_hidden_layers[layer_idx] += additional
        
        elif modification == 'remove_neurons':
            # Remove neurons from a random layer
            if new_hidden_layers:
                layer_idx = np.random.randint(0, len(new_hidden_layers))
                # Remove 10-30% of neurons, ensuring at least 2 remain
                reduction = max(1, int(new_hidden_layers[layer_idx] * (0.1 + 0.2 * np.random.random())))
                new_hidden_layers[layer_idx] = max(2, new_hidden_layers[layer_idx] - reduction)
        
        elif modification == 'add_layer':
            # Add a new layer with size between adjacent layers
            if new_hidden_layers:
                insert_idx = np.random.randint(0, len(new_hidden_layers) + 1)
                
                if insert_idx == 0:
                    # Between input and first hidden
                    new_size = int((self.input_dim + new_hidden_layers[0]) / 2)
                elif insert_idx == len(new_hidden_layers):
                    # After last hidden
                    new_size = int(new_hidden_layers[-1] / 2)
                else:
                    # Between two hidden layers
                    new_size = int((new_hidden_layers[insert_idx-1] + new_hidden_layers[insert_idx]) / 2)
                
                new_hidden_layers.insert(insert_idx, new_size)
        
        elif modification == 'remove_layer':
            # Remove a random layer
            if len(new_hidden_layers) > 1:  # Keep at least one hidden layer
                remove_idx = np.random.randint(0, len(new_hidden_layers))
                new_hidden_layers.pop(remove_idx)
        
        # If structure changed, rebuild the model
        if modification != 'no_change' and new_hidden_layers != self.hidden_layers:
            self.hidden_layers = new_hidden_layers
            old_model = self.model
            self.model = self._build_model()
            
            # Transfer parameters where shapes match
            with torch.no_grad():
                for (old_name, old_param), (new_name, new_param) in zip(
                    old_model.named_parameters(), self.model.named_parameters()
                ):
                    if old_param.shape == new_param.shape:
                        new_param.copy_(old_param)
            
            # Update optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Record structure change
            self.structure_history.append(self.hidden_layers.copy())
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the current meta-model."""
        return self.model(X)
    
    def visualize_structure_history(self, ax=None):
        """
        Visualize the history of architectural changes.
        
        Args:
            ax: Matplotlib axis to plot on
            
        Returns:
            Matplotlib axis with the visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a grid to visualize network structure over time
        num_structures = len(self.structure_history)
        max_layers = max(len(struct) for struct in self.structure_history)
        max_neurons = max(max(struct) for struct in self.structure_history)
        
        # Create heatmap data
        heatmap_data = np.zeros((max_layers, num_structures))
        
        # Fill in the heatmap data
        for t, structure in enumerate(self.structure_history):
            for l, layer_size in enumerate(structure):
                heatmap_data[l, t] = layer_size
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Neurons in Layer')
        
        # Add labels
        ax.set_xlabel('Evolution Step')
        ax.set_ylabel('Layer Index')
        ax.set_title('Neural Architecture Evolution')
        
        # Add actual structure annotations
        for t, structure in enumerate(self.structure_history):
            structure_str = ' → '.join([str(s) for s in structure])
            if t % max(1, num_structures // 10) == 0:  # Show only some labels if many
                ax.text(t, -0.5, structure_str, rotation=45, ha='center', fontsize=8)
        
        # Plot performance if available
        if self.performance_history:
            ax2 = ax.twinx()
            ax2.plot(range(len(self.performance_history)), self.performance_history, 'r-', alpha=0.7)
            ax2.set_ylabel('Loss', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        return ax


class NeuroplasticOS:
    """
    Main simulation class for the Neuroplastic Operating System.
    Integrates the dynamic representation space, conceptual network,
    and meta-learning components.
    """
    
    def __init__(
        self,
        initial_dim: int = 8,
        initial_concepts: List[str] = None,
        meta_input_dim: int = 8,
        plasticity_rate: float = 0.1,
        energy_capacity: float = 1.0,
        homeostatic_regulation: bool = True
    ):
        """
        Initialize the NOS simulation.
        
        Args:
            initial_dim: Initial dimensionality of the representation space
            initial_concepts: Initial concept nodes
            meta_input_dim: Input dimension for meta-learning
            plasticity_rate: Overall plasticity rate
            energy_capacity: Maximum energy available for adaptation
            homeostatic_regulation: Whether to use homeostatic regulation
        """
        # Initialize components
        self.representation_space = DynamicRepresentationSpace(
            initial_dim=initial_dim,
            plasticity_rate=plasticity_rate
        )
        
        self.conceptual_network = EmergentConceptualNetwork(
            initial_concepts=initial_concepts or []
        )
        
        self.meta_learning = MetaLearningController(
            input_dim=meta_input_dim,
            structure_mutation_rate=plasticity_rate / 2
        )
        
        # System state
        self.energy = energy_capacity
        self.stability = 1.0
        self.homeostatic_regulation = homeostatic_regulation
        
        # History
        self.energy_history = [energy_capacity]
        self.stability_history = [1.0]
        self.dim_history = [initial_dim]
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 100
        
        # Performance metrics
        self.metrics = {
            'adaptation_rate': [],
            'concept_emergence_rate': [],
            'energy_efficiency': [],
            'stability_index': []
        }
        
        # Step counter
        self.step_count = 0
    
    def step(self, experiences: List[np.ndarray], tasks=None) -> Dict[str, float]:
        """
        Execute one step of the NOS simulation with new experiences.
        
        Args:
            experiences: List of experience vectors
            tasks: Optional list of (X, y, task_id) tuples for meta-learning
            
        Returns:
            Dictionary of metrics for this step
        """
        self.step_count += 1
        
        # 0. Apply homeostatic regulation if enabled
        if self.homeostatic_regulation:
            self._apply_homeostatic_regulation()
        
        # 1. Update experience buffer
        self._update_experience_buffer(experiences)
        
        # 2. Allocate energy for components
        available_energy = self._allocate_energy()
        
        # 3. Update representation space
        self.representation_space.transform_structure(
            experiences,
            {'energy': available_energy['representation']}
        )
        
        # 4. Update conceptual network
        self.conceptual_network.update(
            experiences,
            self.representation_space
        )
        
        # 5. Update meta-learning if tasks provided
        meta_loss = None
        if tasks:
            meta_loss = self.meta_learning.meta_update(tasks)
        
        # 6. Update internal state
        self._update_state()
        
        # 7. Compute and record metrics
        metrics = self._compute_metrics(meta_loss)
        
        return metrics
    
    def _update_experience_buffer(self, new_experiences: List[np.ndarray]) -> None:
        """
        Update the experience buffer with new experiences.
        
        Args:
            new_experiences: List of new experience vectors
        """
        # Add new experiences to buffer
        self.experience_buffer.extend(new_experiences)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            # Keep most recent experiences
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
    
    def _allocate_energy(self) -> Dict[str, float]:
        """
        Allocate available energy among system components.
        
        Returns:
            Dictionary mapping component names to allocated energy
        """
        # In a more complex implementation, this would be adaptive
        # based on current system state and priorities
        
        total_energy = self.energy
        
        # Basic allocation strategy
        allocation = {
            'representation': 0.4 * total_energy,
            'conceptual': 0.3 * total_energy,
            'meta_learning': 0.2 * total_energy,
            'reserve': 0.1 * total_energy
        }
        
        return allocation
    
    def _apply_homeostatic_regulation(self) -> None:
        """Apply homeostatic regulation to maintain system stability."""
        # Check stability level
        if self.stability < 0.3:
            # System becoming unstable, reduce plasticity
            self.representation_space.plasticity_rate *= 0.9
            self.meta_learning.structure_mutation_rate *= 0.9
            
            # Allocate more energy to stabilization
            energy_recovery = min(0.1, 1.0 - self.energy)
            self.energy += energy_recovery
            
        elif self.stability > 0.8 and self.energy > 0.7:
            # System very stable with high energy, can increase plasticity
            self.representation_space.plasticity_rate = min(
                0.5, self.representation_space.plasticity_rate * 1.05
            )
            self.meta_learning.structure_mutation_rate = min(
                0.2, self.meta_learning.structure_mutation_rate * 1.05
            )
        
        # Natural energy recovery
        self.energy = min(1.0, self.energy + 0.01)
    
    def _update_state(self) -> None:
        """Update internal state variables after a step."""
        # Update energy
        # Each operation consumes energy
        energy_consumption = 0.05
        self.energy = max(0.1, self.energy - energy_consumption)
        
        # Update stability based on recent changes
        dim_change_magnitude = 0
        if len(self.dim_history) > 1:
            dim_change_magnitude = abs(self.representation_space.current_dim - self.dim_history[-1]) / 10.0
        
        structure_change_magnitude = 0
        if len(self.meta_learning.structure_history) > 1:
            # Compute difference between current and previous structure
            curr = self.meta_learning.structure_history[-1]
            prev = self.meta_learning.structure_history[-2]
            
            # Simple measure of structural difference
            # More sophisticated measures could be used
            max_len = max(len(curr), len(prev))
            padded_curr = curr + [0] * (max_len - len(curr))
            padded_prev = prev + [0] * (max_len - len(prev))
            
            structure_change_magnitude = sum(abs(c - p) for c, p in zip(padded_curr, padded_prev)) / 20.0
        
        # Concept change magnitude
        concept_change_magnitude = (
            self.conceptual_network.emergence_count + 
            self.conceptual_network.pruning_count
        ) / 100.0
        
        # Update stability based on changes
        stability_impact = dim_change_magnitude + structure_change_magnitude + concept_change_magnitude
        self.stability = max(0.1, min(1.0, self.stability - stability_impact + 0.02))
        
        # Record history
        self.energy_history.append(self.energy)
        self.stability_history.append(self.stability)
        self.dim_history.append(self.representation_space.current_dim)
    
    def _compute_metrics(self, meta_loss: Optional[float] = None) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        Args:
            meta_loss: Optional meta-learning loss
            
        Returns:
            Dictionary of metrics
        """
        # Calculate adaptation rate
        # How quickly system responds to new experiences
        if len(self.dim_history) > 10:
            adaptation_rate = np.std(self.dim_history[-10:]) / np.mean(self.dim_history[-10:])
        else:
            adaptation_rate = 0.0
        
        # Calculate concept emergence rate
        concept_emergence_rate = self.conceptual_network.emergence_count / max(1, self.step_count)
        
        # Energy efficiency
        if meta_loss is not None:
            energy_efficiency = 1.0 / (meta_loss * (1.0 / self.energy))
        else:
            energy_efficiency = self.energy
        
        # Stability index
        stability_index = self.stability
        
        # Record metrics
        metrics = {
            'adaptation_rate': adaptation_rate,
            'concept_emergence_rate': concept_emergence_rate,
            'energy_efficiency': energy_efficiency,
            'stability_index': stability_index
        }
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        return metrics
    
    def visualize_system_state(self, figsize=(18, 14)):
        """
        Visualize the current state of the entire system.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with the visualization
        """
        fig = plt.figure(figsize=figsize)
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3)
        
        # 1. Representation space
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self.representation_space.visualize(ax=ax1)
        
        # 2. Conceptual network
        ax2 = fig.add_subplot(gs[0, 1:])
        self.conceptual_network.visualize(ax=ax2)
        
        # 3. Meta-learning structure
        ax3 = fig.add_subplot(gs[1, 0:2])
        self.meta_learning.visualize_structure_history(ax=ax3)
        
        # 4. System state history
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(self.energy_history, 'r-', label='Energy')
        ax4.plot(self.stability_history, 'b-', label='Stability')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Value')
        ax4.set_title('System State History')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics
        ax5 = fig.add_subplot(gs[2, :])
        metrics_df = pd.DataFrame(self.metrics)
        
        # Normalize metrics for comparison
        for col in metrics_df.columns:
            if metrics_df[col].max() > 0:
                metrics_df[col] = metrics_df[col] / metrics_df[col].max()
        
        # Plot metrics
        metrics_df.plot(ax=ax5)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Normalized Value')
        ax5.set_title('Performance Metrics')
        ax5.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on system state and performance.
        
        Returns:
            Dictionary with report data
        """
        report = {
            'step_count': self.step_count,
            'current_state': {
                'energy': self.energy,
                'stability': self.stability,
                'representation_dim': self.representation_space.current_dim,
                'representation_topology': self.representation_space.topology_type,
                'num_concepts': len(self.conceptual_network.nodes),
                'num_relations': len(self.conceptual_network.edges),
                'meta_learning_structure': self.meta_learning.hidden_layers
            },
            'metrics': {k: (sum(v[-10:]) / min(10, len(v))) for k, v in self.metrics.items()},
            'adaptive_capacity': self._compute_adaptive_capacity(),
            'emergent_properties': self._analyze_emergent_properties()
        }
        
        return report
    
    def _compute_adaptive_capacity(self) -> Dict[str, float]:
        """
        Compute measures of the system's adaptive capacity.
        
        Returns:
            Dictionary with adaptive capacity metrics
        """
        # Calculate structural flexibility
        if len(self.dim_history) > 10:
            dim_changes = sum(1 for i in range(1, len(self.dim_history)) 
                              if self.dim_history[i] != self.dim_history[i-1])
            structural_flexibility = dim_changes / (len(self.dim_history) - 1)
        else:
            structural_flexibility = 0.0
        
        # Calculate knowledge adaptability
        concept_turnover = (self.conceptual_network.emergence_count + 
                            self.conceptual_network.pruning_count)
        knowledge_adaptability = concept_turnover / max(1, len(self.conceptual_network.nodes) * self.step_count)
        
        # Calculate learning efficiency
        if self.metrics['energy_efficiency']:
            learning_efficiency = sum(self.metrics['energy_efficiency'][-10:]) / min(10, len(self.metrics['energy_efficiency']))
        else:
            learning_efficiency = 0.0
        
        return {
            'structural_flexibility': structural_flexibility,
            'knowledge_adaptability': knowledge_adaptability,
            'learning_efficiency': learning_efficiency,
            'composite_score': (structural_flexibility + knowledge_adaptability + learning_efficiency) / 3
        }
    
    def _analyze_emergent_properties(self) -> Dict[str, Any]:
        """
        Analyze emergent properties of the system.
        
        Returns:
            Dictionary with analyses of emergent properties
        """
        # Analyze concept clustering
        if len(self.conceptual_network.nodes) > 5:
            # Create adjacency matrix from hypergraph
            nodes = list(self.conceptual_network.nodes)
            adj_matrix = np.zeros((len(nodes), len(nodes)))
            
            for _, (edge_nodes, weight) in self.conceptual_network.edges.items():
                edge_node_indices = [nodes.index(n) for n in edge_nodes if n in nodes]
                for i in edge_node_indices:
                    for j in edge_node_indices:
                        if i != j:
                            adj_matrix[i, j] += weight
            
            # Detect communities
            try:
                from sklearn.cluster import src.SpectralClustering
                clustering = SpectralClustering(
                    n_clusters=min(5, len(nodes)), 
                    affinity='precomputed',
                    random_state=42
                ).fit(adj_matrix)
                
                clusters = {}
                for i, label in enumerate(clustering.labels_):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(nodes[i])
                
                community_structure = clusters
            except:
                community_structure = {"error": "Clustering failed"}
        else:
            community_structure = {"note": "Too few nodes for meaningful clustering"}
        
        # Analyze knowledge structure
        if self.conceptual_network.edges:
            avg_edge_size = sum(len(nodes) for nodes, _ in self.conceptual_network.edges.values()) / len(self.conceptual_network.edges)
            max_edge_size = max(len(nodes) for nodes, _ in self.conceptual_network.edges.values())
        else:
            avg_edge_size = 0
            max_edge_size = 0
        
        return {
            'community_structure': community_structure,
            'knowledge_structure': {
                'avg_edge_size': avg_edge_size,
                'max_edge_size': max_edge_size,
                'density': self.conceptual_network._compute_density()
            },
            'concept_activations': dict(sorted(
                self.conceptual_network.node_activations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 most active concepts
        }
    
    def run_simulation(
        self, 
        num_steps: int, 
        experience_generator: Callable[[], List[np.ndarray]],
        task_generator: Optional[Callable[[], List[Tuple[torch.Tensor, torch.Tensor, str]]]] = None,
        visualize_every: int = 0
    ):
        """
        Run the simulation for multiple steps.
        
        Args:
            num_steps: Number of steps to run
            experience_generator: Function that generates experiences for each step
            task_generator: Optional function that generates tasks for meta-learning
            visualize_every: If > 0, visualize system every this many steps
        """
        results = []
        
        for step in tqdm(range(num_steps), desc="Simulation Progress"):
            # Generate experiences for this step
            experiences = experience_generator()
            
            # Generate tasks if provided
            tasks = task_generator() if task_generator else None
            
            # Execute step
            metrics = self.step(experiences, tasks)
            results.append(metrics)
            
            # Visualize if needed
            if visualize_every > 0 and (step + 1) % visualize_every == 0:
                fig = self.visualize_system_state()
                plt.suptitle(f"System State at Step {step+1}", fontsize=16)
                plt.tight_layout()
                plt.show()
        
        # Final visualization
        if visualize_every > 0:
            fig = self.visualize_system_state()
            plt.suptitle(f"Final System State after {num_steps} Steps", fontsize=16)
            plt.tight_layout()
            plt.show()
        
        # Return collected results
        return pd.DataFrame(results)


# Helper functions to generate synthetic data for simulation

def generate_dynamic_experiences(
    num_experiences: int = 5,
    base_dim: int = 10,
    complexity: float = 0.5,
    time_step: int = 0,
    drift_rate: float = 0.01
) -> List[np.ndarray]:
    """
    Generate synthetic experiences with evolving patterns.
    
    Args:
        num_experiences: Number of experience vectors to generate
        base_dim: Base dimensionality of vectors
        complexity: Complexity of patterns (0-1)
        time_step: Current time step (for temporal drift)
        drift_rate: Rate of pattern drift over time
        
    Returns:
        List of experience vectors
    """
    # Define basic patterns that evolve over time
    patterns = [
        np.sin(np.linspace(0, 2*np.pi, base_dim) + drift_rate * time_step),
        np.cos(np.linspace(0, 3*np.pi, base_dim) + drift_rate * time_step),
        np.sin(np.linspace(0, 4*np.pi, base_dim) + drift_rate * time_step) * 
        np.cos(np.linspace(0, 3*np.pi, base_dim) + drift_rate * time_step),
        np.exp(-np.linspace(-2, 2, base_dim)**2) * np.sin(np.linspace(0, 4*np.pi, base_dim) + 0.5 * drift_rate * time_step)
    ]
    
    experiences = []
    
    for _ in range(num_experiences):
        # Generate experience as combination of patterns
        weights = np.random.random(len(patterns)) * complexity
        weights /= weights.sum()  # Normalize
        
        experience = sum(w * p for w, p in zip(weights, patterns))
        
        # Add noise inversely proportional to complexity
        noise = np.random.normal(0, 1-complexity, base_dim)
        experience += noise
        
        # Normalize
        experience = (experience - experience.min()) / (experience.max() - experience.min() + 1e-8)
        
        experiences.append(experience)
    
    return experiences


def generate_meta_learning_tasks(
    num_tasks: int = 3,
    samples_per_task: int = 20,
    input_dim: int = 8,
    time_step: int = 0,
    task_drift_rate: float = 0.005
) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    """
    Generate synthetic tasks for meta-learning.
    
    Args:
        num_tasks: Number of tasks to generate
        samples_per_task: Number of samples per task
        input_dim: Dimensionality of input features
        time_step: Current time step
        task_drift_rate: Rate of task drift over time
        
    Returns:
        List of (X, y, task_id) tuples
    """
    tasks = []
    
    for task_idx in range(num_tasks):
        # Create task-specific function (simple regression)
        # This function evolves over time
        def task_function(x):
            phase = task_drift_rate * time_step
            amplitude = 1.0 + 0.1 * np.sin(task_drift_rate * time_step)
            frequency = 1.0 + 0.05 * np.cos(task_drift_rate * time_step)
            
            if task_idx == 0:
                # Linear function with changing slope
                return amplitude * np.sum(x * np.sin(np.linspace(0, np.pi, len(x)) + phase))
            
            elif task_idx == 1:
                # Quadratic function with changing coefficients
                return amplitude * np.sum((x ** 2) * np.cos(np.linspace(0, np.pi, len(x)) + phase))
            
            else:
                # Sinusoidal function with changing frequency
                return amplitude * np.sin(frequency * np.sum(x) + phase)
        
        # Generate input data
        X = np.random.randn(samples_per_task, input_dim)
        
        # Compute outputs
        y = np.array([task_function(x) for x in X])
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create unique task ID
        task_id = f"task_{task_idx}"
        
        tasks.append((X_tensor, y_tensor, task_id))
    
    return tasks


def run_example_simulation():
    """Run an example simulation to demonstrate the framework."""
    # Initialize the NOS with some starter concepts
    nos = NeuroplasticOS(
        initial_dim=8,
        initial_concepts=["concept_A", "concept_B", "concept_C"],
        meta_input_dim=8,
        plasticity_rate=0.1
    )
    
    # Add initial points to representation space
    nos.representation_space.add_point("concept_A", np.array([0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    nos.representation_space.add_point("concept_B", np.array([0.8, 0.2, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]))
    nos.representation_space.add_point("concept_C", np.array([0.4, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    # Add initial relations
    nos.conceptual_network.add_edge(["concept_A", "concept_B"], 0.6)
    nos.conceptual_network.add_edge(["concept_B", "concept_C"], 0.4)
    nos.conceptual_network.add_edge(["concept_A", "concept_B", "concept_C"], 0.3)
    
    # Define experience generator with evolving patterns
    def generate_experiences():
        return generate_dynamic_experiences(
            num_experiences=5,
            base_dim=8,
            complexity=0.7,
            time_step=nos.step_count,
            drift_rate=0.02
        )
    
    # Define task generator for meta-learning
    def generate_tasks():
        return generate_meta_learning_tasks(
            num_tasks=3,
            samples_per_task=20,
            input_dim=8,
            time_step=nos.step_count,
            task_drift_rate=0.01
        )
    
    # Run simulation
    results = nos.run_simulation(
        num_steps=50,
        experience_generator=generate_experiences,
        task_generator=generate_tasks,
        visualize_every=10
    )
    
    # Display final report
    final_report = nos.generate_report()
    print("\nFinal System Report:")
    print("-" * 50)
    print(f"Steps completed: {final_report['step_count']}")
    print("\nCurrent State:")
    for k, v in final_report['current_state'].items():
        print(f"  {k}: {v}")
    
    print("\nPerformance Metrics (10-step average):")
    for k, v in final_report['metrics'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nAdaptive Capacity:")
    for k, v in final_report['adaptive_capacity'].items():
        print(f"  {k}: {v:.4f}")
    
    # Plot summary of results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results['adaptation_rate'], label='Adaptation Rate')
    plt.plot(results['stability_index'], label='Stability')
    plt.title('Adaptation vs Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(results['concept_emergence_rate'], label='Concept Emergence')
    plt.plot(results['energy_efficiency'], label='Energy Efficiency')
    plt.title('Emergence and Efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(nos.energy_history, 'r-', label='Energy')
    plt.plot(nos.stability_history, 'b-', label='Stability')
    plt.title('System State History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(nos.dim_history, 'g-', label='Representation Dimension')
    stacked_layers = []
    for structure in nos.meta_learning.structure_history:
        stacked_layers.append(sum(structure))
    plt.plot(stacked_layers, 'm-', label='Total Neurons')
    plt.title('Structural Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return nos


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run example simulation
    nos = run_example_simulation()
    
    print("\nSimulation complete!")
