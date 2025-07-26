"""
Contextual Resonance Network Implementation

This module implements the contextual resonance network component of the
Neuroplastic Operating Systems framework. It provides a tensor-based approach
to cross-domain knowledge integration and transfer.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContextualResonanceNetwork")

class ContextualResonanceNetwork:
    """
    Implementation of a contextual resonance network that connects
    concepts across domains using the contextual resonance tensor.
    """
    
    def __init__(
        self,
        contextual_dims: int = 5,
        knowledge_domains: int = 3,
        representational_modes: int = 4,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01
    ):
        """
        Initialize the contextual resonance network.
        
        Args:
            contextual_dims: Number of contextual dimensions
            knowledge_domains: Number of knowledge domains
            representational_modes: Number of representational modalities
            learning_rate: Learning rate for tensor updates
            decay_rate: Decay rate for tensor values
        """
        # Dimensions of the contextual resonance tensor
        self.C = contextual_dims
        self.K = knowledge_domains
        self.R = representational_modes
        
        # Initialize tensor
        self.resonance_tensor = np.random.normal(
            0, 0.01, 
            (self.C, self.K, self.R)
        )
        
        # Domain and context labels
        self.domain_labels = [f"domain_{i}" for i in range(self.K)]
        self.context_labels = [f"context_{i}" for i in range(self.C)]
        self.rep_mode_labels = [f"mode_{i}" for i in range(self.R)]
        
        # Update parameters
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # History for analysis
        self.update_history = []
        self.activation_history = []
        
        logger.info(f"Initialized resonance tensor with dimensions: {self.C}x{self.K}x{self.R}")
    
    def update_tensor(self, experiences: List[Dict[str, np.ndarray]]) -> float:
        """
        Update the resonance tensor based on new experiences.
        
        Args:
            experiences: List of experience dictionaries with domain and context info
            
        Returns:
            Magnitude of the update
        """
        if not experiences:
            return 0.0
        
        update_magnitude = 0.0
        
        for exp in experiences:
            # Extract components from the experience
            if 'domain_idx' not in exp or 'context_idx' not in exp or 'content' not in exp:
                continue
            
            domain_idx = exp['domain_idx']
            context_idx = exp['context_idx']
            
            if not (0 <= domain_idx < self.K and 0 <= context_idx < self.C):
                continue
            
            # Extract content vector and ensure it has the right dimensionality
            content = exp['content']
            if len(content) != self.R:
                content = self._project_to_representation_dims(content)
            
            # Calculate update to the tensor
            update = np.zeros((self.C, self.K, self.R))
            
            # Update specific slice of tensor based on context and domain
            update[context_idx, domain_idx, :] = content
            
            # Apply learning rate
            update *= self.learning_rate
            
            # Apply the update
            self.resonance_tensor += update
            
            update_magnitude += np.sum(np.abs(update))
        
        # Apply decay to the entire tensor
        self.resonance_tensor *= (1 - self.decay_rate)
        
        # Record history
        self.update_history.append(update_magnitude)
        
        logger.debug(f"Updated resonance tensor with magnitude {update_magnitude:.4f}")
        
        return update_magnitude
    
    def _project_to_representation_dims(self, vector: np.ndarray) -> np.ndarray:
        """
        Project a vector to match the representation dimensions.
        
        Args:
            vector: Input vector
            
        Returns:
            Projected vector with length R
        """
        if len(vector) == self.R:
            return vector
        elif len(vector) < self.R:
            return np.pad(vector, (0, self.R - len(vector)))
        else:
            return vector[:self.R]
    
    def activate(
        self, 
        context_idx: Optional[int] = None,
        domain_idx: Optional[int] = None,
        input_vector: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Activate the resonance tensor to produce cross-domain resonance.
        
        Args:
            context_idx: Index of the context to activate (if None, use all)
            domain_idx: Index of the domain to activate (if None, use all)
            input_vector: Optional input vector for representation mode
            
        Returns:
            Dictionary of activated resonances across domains
        """
        # Default to full activation if indices not provided
        context_indices = [context_idx] if context_idx is not None else range(self.C)
        domain_indices = [domain_idx] if domain_idx is not None else range(self.K)
        
        # Extract relevant tensor slices
        active_tensor = self.resonance_tensor[context_indices, :, :][:, domain_indices, :]
        
        # If input vector provided, use it to modulate activation
        if input_vector is not None:
            # Ensure correct dimensionality
            input_vector = self._project_to_representation_dims(input_vector)
            
            # Reshape for broadcasting
            input_vector = input_vector.reshape(1, 1, -1)
            
            # Apply input modulation
            active_tensor = active_tensor * input_vector
        
        # Compute activated resonances for each domain
        resonances = {}
        
        for i, d_idx in enumerate(domain_indices):
            domain_name = self.domain_labels[d_idx]
            
            # Sum over contexts
            domain_activation = np.sum(active_tensor[:, i, :], axis=0)
            
            # Store in results
            resonances[domain_name] = domain_activation
        
        # Record activation pattern
        self.activation_history.append({
            'context_indices': context_indices,
            'domain_indices': domain_indices,
            'resonance_magnitude': np.sum(np.abs(list(resonances.values())))
        })
        
        return resonances
    
    def get_cross_domain_associations(
        self, 
        domain_idx: int,
        representation: np.ndarray,
        context_idx: Optional[int] = None,
        threshold: float = 0.5
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Get associations across domains for a specific representation.
        
        Args:
            domain_idx: Source domain index
            representation: Input representation vector
            context_idx: Context index (if None, use all contexts)
            threshold: Similarity threshold for associations
            
        Returns:
            List of (domain, representation, similarity) tuples
        """
        if not (0 <= domain_idx < self.K):
            return []
        
        # Ensure vector has correct dimensionality
        representation = self._project_to_representation_dims(representation)
        
        # Normalize for similarity comparison
        representation_norm = representation / (np.linalg.norm(representation) + 1e-10)
        
        # Contexts to consider
        context_indices = [context_idx] if context_idx is not None else range(self.C)
        
        associations = []
        
        for c_idx in context_indices:
            if not (0 <= c_idx < self.C):
                continue
                
            # Get the source activation
            source_activation = self.resonance_tensor[c_idx, domain_idx, :]
            
            # Skip if source activation is too weak
            if np.linalg.norm(source_activation) < 1e-6:
                continue
            
            # Compute resonance with input representation
            similarity = np.dot(source_activation, representation_norm) / (np.linalg.norm(source_activation) + 1e-10)
            
            # If similarity exceeds threshold, find associations in other domains
            if similarity >= threshold:
                for d_idx in range(self.K):
                    if d_idx != domain_idx:  # Only cross-domain
                        target_domain = self.domain_labels[d_idx]
                        target_representation = self.resonance_tensor[c_idx, d_idx, :]
                        
                        # Skip if target is too weak
                        if np.linalg.norm(target_representation) < 1e-6:
                            continue
                        
                        # Compute similarity between source and target through resonance
                        cross_similarity = np.dot(source_activation, target_representation) / (
                            np.linalg.norm(source_activation) * 
                            np.linalg.norm(target_representation) + 1e-10
                        )
                        
                        if cross_similarity >= threshold:
                            associations.append((
                                target_domain, 
                                target_representation, 
                                cross_similarity * similarity  # Combined similarity
                            ))
        
        # Sort by similarity score
        associations.sort(key=lambda x: x[2], reverse=True)
        
        return associations
    
    def visualize_tensor(self, ax=None):
        """
        Visualize the contextual resonance tensor.
        
        Args:
            ax: Optional matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Compute tensor norm across representation dimension
        tensor_norms = np.linalg.norm(self.resonance_tensor, axis=2)
        
        # Create heatmap
        im = ax.imshow(tensor_norms, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Resonance Strength')
        
        # Add labels
        ax.set_xlabel('Knowledge Domains')
        ax.set_ylabel('Contextual Dimensions')
        ax.set_title('Contextual Resonance Tensor (Strength)')
        
        # Add domain labels on x-axis
        ax.set_xticks(range(self.K))
        ax.set_xticklabels(self.domain_labels, rotation=45, ha='right')
        
        # Add context labels on y-axis
        ax.set_yticks(range(self.C))
        ax.set_yticklabels(self.context_labels)
        
        return ax
    
    def visualize_associations(self, domain_idx: int, representation: np.ndarray, ax=None):
        """
        Visualize cross-domain associations for a specific representation.
        
        Args:
            domain_idx: Source domain index
            representation: Input representation vector
            ax: Optional matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        associations = self.get_cross_domain_associations(
            domain_idx=domain_idx,
            representation=representation,
            threshold=0.3
        )
        
        if not associations:
            ax.text(0.5, 0.5, "No significant cross-domain associations found", 
                   ha='center', va='center')
            ax.set_title(f"Cross-domain associations from {self.domain_labels[domain_idx]}")
            return ax
        
        # Create a network visualization
        import networkx as nx
        G = nx.Graph()
        
        # Add source node
        source_node = self.domain_labels[domain_idx]
        G.add_node(source_node, type='source')
        
        # Add association nodes
        for target_domain, _, similarity in associations:
            G.add_node(target_domain, type='target')
            G.add_edge(source_node, target_domain, weight=similarity)
        
        # Layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[source_node], 
            node_color='blue', 
            node_size=500, 
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[n for n in G.nodes() if n != source_node], 
            node_color='green', 
            node_size=300, 
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges with weights as width
        edge_widths = [d['weight'] * 5 for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, width=edge_widths, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # Add edge labels
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        ax.set_title(f"Cross-domain associations from {source_node}")
        ax.axis('off')
        
        return ax
    
    def visualize_evolution(self, ax=None):
        """
        Visualize the evolution of resonance strength over time.
        
        Args:
            ax: Optional matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if not self.activation_history:
            ax.text(0.5, 0.5, "No activation history available", 
                   ha='center', va='center')
            return ax
        
        # Extract resonance magnitudes
        steps = range(len(self.activation_history))
        magnitudes = [a['resonance_magnitude'] for a in self.activation_history]
        
        # Plot evolution
        ax.plot(steps, magnitudes, 'b-', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Resonance Magnitude')
        ax.set_title('Evolution of Resonance Strength')
        ax.grid(True, alpha=0.3)
        
        return ax


if __name__ == "__main__":
    # Example usage
    # Create a contextual resonance network
    crn = ContextualResonanceNetwork(
        contextual_dims=4,
        knowledge_domains=3,
        representational_modes=8,
        learning_rate=0.1,
        decay_rate=0.01
    )
    
    # Create some experiences
    experiences = [
        {
            'domain_idx': 0,  # Mathematical domain
            'context_idx': 1,  # Abstract context
            'content': np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Binary pattern
        },
        {
            'domain_idx': 1,  # Visual domain
            'context_idx': 1,  # Abstract context
            'content': np.array([0, 0, 1, 1, 1, 1, 0, 0])  # Square-like pattern
        }
    ]
    
    # Update tensor with experiences
    crn.update_tensor(experiences)
    
    # Activate with a test vector
    test_vector = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    resonances = crn.activate(input_vector=test_vector)
    
    # Get cross-domain associations
    associations = crn.get_cross_domain_associations(
        domain_idx=0,
        representation=test_vector,
        threshold=0.3
    )
    
    print(f"Found {len(associations)} cross-domain associations")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    crn.visualize_tensor(ax=ax1)
    crn.visualize_associations(domain_idx=0, representation=test_vector, ax=ax2)
    plt.tight_layout()
    plt.show()
