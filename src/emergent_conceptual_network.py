"""
Emergent Conceptual Network Implementation

This module implements the emergent conceptual network component of the
Neuroplastic Operating Systems framework. It provides a dynamic hypergraph
for knowledge representation with concept emergence and pruning capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union, Any, FrozenSet
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmergentConceptualNetwork")

class EmergentConceptualNetwork:
    """
    Implementation of a dynamic hypergraph representing conceptual knowledge
    with emergent properties.
    """
    
    def __init__(
        self,
        initial_concepts: Optional[List[str]] = None,
        initial_edges: Optional[List[Tuple[List[str], float]]] = None,
        emergence_threshold: float = 0.6,
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
            logger.debug(f"Added node {node_id}")
    
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
        
        logger.debug(f"Added edge {edge_id} connecting {nodes} with weight {weight:.4f}")
        
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
        
        logger.debug(f"Removed node {node_id} and {len(edges_to_remove)} connected edges")
    
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
        
        logger.debug(f"Removed edge {edge_id}")
    
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
        
        logger.debug(f"Updated edge {edge_id} weight from {current_weight:.4f} to {new_weight:.4f}")
    
    def activate_nodes(self, activation_dict: Dict[str, float]) -> None:
        """
        Set activation values for specific nodes.
        
        Args:
            activation_dict: Dictionary mapping node IDs to activation values
        """
        for node_id, activation in activation_dict.items():
            if node_id in self.nodes:
                self.node_activations[node_id] = max(0.0, min(1.0, activation))
        
        logger.debug(f"Activated {len(activation_dict)} nodes")
    
    def propagate_activation(self, steps: int = 1, decay_rate: float = 0.3) -> None:
        """
        Propagate activation through the hypergraph.
        
        Args:
            steps: Number of propagation steps to perform
            decay_rate: Rate at which activations decay during propagation
        """
        for step in range(steps):
            # Create new activation dictionary
            new_activations = {node: 0.0 for node in self.nodes}
            
            # Calculate spreading activation
            for edge_id, (nodes, weight) in self.edges.items():
                # Skip edges with no activated nodes
                activated_nodes = [n for n in nodes if self.node_activations.get(n, 0.0) > 0.1]
                if not activated_nodes:
                    continue
                
                # Source activation is average of connected nodes
                source_activation = sum(self.node_activations[n] for n in activated_nodes) / len(activated_nodes)
                
                # Spread activation to all nodes in edge, weighted by edge weight
                activation_delta = source_activation * weight
                for node in nodes:
                    new_activations[node] += activation_delta
            
            # Apply decay and update activations
            for node in self.nodes:
                # Combine old and new activation with decay
                new_value = (1 - decay_rate) * self.node_activations[node] + new_activations[node]
                # Apply sigmoid to keep in range [0, 1]
                self.node_activations[node] = 1.0 / (1.0 + np.exp(-5 * (new_value - 0.5)))
            
            logger.debug(f"Completed activation propagation step {step+1}/{steps}")
    
    def get_most_active_concepts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the most activated concepts.
        
        Args:
            top_n: Number of top concepts to return
            
        Returns:
            List of (node_id, activation) tuples for the most active nodes
        """
        # Sort by activation in descending order
        sorted_activations = sorted(
            self.node_activations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top N (or fewer if there aren't enough)
        return sorted_activations[:min(top_n, len(sorted_activations))]
    
    def detect_emergent_concepts(
        self, 
        experiences: List[np.ndarray],
        representation_space: Any
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
        
        try:
            # Create embedding matrix of experiences
            X = np.vstack([e for e in experiences if len(e) >= 2])
            
            if len(X) < 2:
                return []
            
            # Simple clustering - find distinct patterns
            from sklearn.cluster import DBSCAN
            
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
                if hasattr(representation_space, 'points') and representation_space.points:
                    existing_points = np.array(list(representation_space.points.values()))
                    
                    # Check if centroid is close to any existing point
                    for point in existing_points:
                        min_len = min(len(centroid), len(point))
                        if min_len > 0 and np.linalg.norm(centroid[:min_len] - point[:min_len]) < 0.3:
                            is_novel = False
                            break
                
                # Create new concept if sufficiently novel and probability exceeds threshold
                if is_novel and np.random.random() < self.emergence_threshold:
                    new_concept_id = f"concept_{self.emergence_count}"
                    self.emergence_count += 1
                    new_concepts.append(new_concept_id)
                    self.add_node(new_concept_id)
                    
                    # Also add to representation space
                    if hasattr(representation_space, 'add_point'):
                        representation_space.add_point(new_concept_id, centroid)
                    
                    logger.info(f"Emergent concept detected: {new_concept_id}")
            
            return new_concepts
            
        except Exception as e:
            logger.warning(f"Error in concept emergence detection: {e}")
            return []
    
    def detect_emergent_relations(self) -> List[Tuple[List[str], float]]:
        """
        Detect potentially emergent relations (hyperedges) based on
        activation patterns.
        
        Args:
            None
            
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
        
        # Potential relationship: nodes that are active together
        if 2 <= len(active_nodes) <= 5:
            # Check if this exact hyperedge already exists
            node_set = frozenset(active_nodes)
            
            if node_set not in self.edge_ids_by_nodes:
                # New potential hyperedge
                weight = sum(self.node_activations[node] for node in active_nodes) / len(active_nodes)
                
                if weight > self.emergence_threshold:
                    new_relations.append((active_nodes, weight))
                    logger.info(f"Detected emergent relation between {active_nodes} with weight {weight:.4f}")
        
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
                        logger.debug(f"Detected emergent relation between {node_pair} with strength {strength:.4f}")
        
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
        
        logger.info(f"Pruned {len(edges_to_remove)} edges and {len(nodes_to_remove)} isolated nodes")
        
        return pruned_count
    
    def update(
        self, 
        experiences: List[np.ndarray],
        representation_space: Any
    ) -> Dict[str, Any]:
        """
        Update the conceptual network based on new experiences.
        
        Args:
            experiences: List of experience vectors
            representation_space: Current representation space
            
        Returns:
            Dictionary with update statistics
        """
        update_stats = {
            'new_concepts': 0,
            'new_relations': 0,
            'pruned_elements': 0,
            'updated_weights': 0
        }
        
        # 1. Detect emergent concepts
        new_concepts = self.detect_emergent_concepts(experiences, representation_space)
        update_stats['new_concepts'] = len(new_concepts)
        
        # 2. Update activations based on experiences
        # Simplified: use distance from experience to concept in representation space
        if experiences and hasattr(representation_space, 'points') and representation_space.points:
            for node_id in self.nodes:
                if node_id in representation_space.points:
                    # Compute average proximity to experiences
                    node_coords = representation_space.points[node_id]
                    
                    # Compute activation based on proximity to experiences
                    activations = []
                    for exp in experiences:
                        # Ensure dimensions match
                        min_dim = min(len(exp), len(node_coords))
                        if min_dim > 0:
                            # Convert to distance and then to activation (closer = more active)
                            distance = np.linalg.norm(exp[:min_dim] - node_coords[:min_dim])
                            activation = np.exp(-2 * distance)
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
            avg_activation = sum(self.node_activations.get(node, 0) for node in nodes) / len(nodes)
            
            # Update weight based on activation
            # Higher co-activation strengthens the connection
            delta = avg_activation - 0.5  # Positive if above threshold, negative if below
            if abs(delta) > 0.05:  # Only update if change is significant
                self.update_edge_weight(edge_id, delta)
                update_stats['updated_weights'] += 1
        
        # 6. Add new relations
        for nodes, weight in new_relations:
            self.add_edge(nodes, weight)
        
        update_stats['new_relations'] = len(new_relations)
        
        # 7. Prune weak connections
        update_stats['pruned_elements'] = self.prune()
        
        # 8. Update history
        self.growth_history['num_nodes'].append(len(self.nodes))
        self.growth_history['num_edges'].append(len(self.edges))
        self.growth_history['avg_weight'].append(self._compute_avg_weight())
        self.growth_history['density'].append(self._compute_density())
        
        logger.info(f"Network update: {update_stats['new_concepts']} new concepts, "
                   f"{update_stats['new_relations']} new relations, "
                   f"{update_stats['pruned_elements']} pruned elements")
        
        return update_stats
    
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


if __name__ == "__main__":
    # Example usage
    # Create an emergent conceptual network
    network = EmergentConceptualNetwork(
        initial_concepts=["concept_1", "concept_2", "concept_3"],
        emergence_threshold=0.6,
        pruning_threshold=0.2
    )
    
    # Add some initial edges
    network.add_edge(["concept_1", "concept_2"], 0.7)
    network.add_edge(["concept_2", "concept_3"], 0.5)
    network.add_edge(["concept_1", "concept_2", "concept_3"], 0.4)
    
    # Activate some nodes
    network.activate_nodes({
        "concept_1": 0.8,
        "concept_2": 0.6
    })
    
    # Propagate activation
    network.propagate_activation(steps=2)
    
    # Detect emergent relations
    new_relations = network.detect_emergent_relations()
    print(f"Detected {len(new_relations)} new relations")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    network.visualize(ax=ax)
    plt.show()
