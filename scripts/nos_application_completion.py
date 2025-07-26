    def get_system_state(self) -> Dict[str, Any]:
        """
        Get the current state of the system.
        
        Returns:
            Dictionary with current system state
        """
        # Gather state from all components
        state = {
            'step': self.step_count,
            'current_dim': self.representation_space.current_dim,
            'min_dim': self.representation_space.min_dim,
            'max_dim': self.representation_space.max_dim,
            'topology_type': self.representation_space.topology_type,
            'num_concepts': len(self.conceptual_network.nodes),
            'num_relations': len(self.conceptual_network.edges)
        }
        
        # Add homeostatic state if enabled
        if self.enable_homeostasis:
            homeostatic_state = self.homeostasis.get_state()
            state.update({
                'energy': homeostatic_state['current_energy'],
                'max_energy': homeostatic_state['max_energy'],
                'stability': homeostatic_state['stability'],
                'temperature': homeostatic_state['operational_temperature']
            })
        else:
            state.update({
                'energy': 1.0,
                'stability': 1.0,
                'temperature': 0.5
            })
        
        # Add metrics
        state['metrics'] = {
            k: v[-1] if v else 0 for k, v in self.metrics.items()
        }
        
        return state
    
    def process_experience(
        self, 
        experience: np.ndarray, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single experience with context.
        
        Args:
            experience: Experience vector
            context: Optional context dictionary
            
        Returns:
            Processing results
        """
        # Create a batch of 1 experience
        return self.step([experience])
    
    def query_knowledge(self, query_vector: np.ndarray) -> Dict[str, Any]:
        """
        Query the system's knowledge using a vector.
        
        Args:
            query_vector: Query vector
            
        Returns:
            Query results
        """
        results = {}
        
        # 1. Find nearest points in representation space
        nearest_points = []
        for point_id, coords in self.representation_space.points.items():
            # Compute distance
            query_proj = self.representation_space._project_to_current_dim(query_vector)
            distance = np.linalg.norm(query_proj - coords)
            nearest_points.append((point_id, distance))
        
        # Sort by distance
        nearest_points.sort(key=lambda x: x[1])
        
        # Return top matches
        results['nearest_concepts'] = nearest_points[:5]
        
        # 2. Activate conceptual network with these points
        activation_dict = {}
        for point_id, distance in nearest_points[:5]:
            # Convert distance to activation (closer = higher activation)
            activation = np.exp(-2 * distance)
            activation_dict[point_id] = min(1.0, max(0.0, activation))
        
        # Activate the network
        self.conceptual_network.activate_nodes(activation_dict)
        self.conceptual_network.propagate_activation(steps=2)
        
        # Get most active concepts after propagation
        results['active_concepts'] = self.conceptual_network.get_most_active_concepts(top_n=10)
        
        # 3. Get conceptual relations
        active_relations = []
        for edge_id, (nodes, weight) in self.conceptual_network.edges.items():
            # Check if edge contains any highly activated nodes
            active_nodes = [node for node in nodes if 
                            node in self.conceptual_network.node_activations and
                            self.conceptual_network.node_activations[node] > 0.4]
            
            if active_nodes and weight > 0.3:
                active_relations.append((edge_id, list(nodes), weight))
        
        # Sort by decreasing weight
        active_relations.sort(key=lambda x: x[2], reverse=True)
        results['active_relations'] = active_relations[:10]
        
        # 4. Check for cross-domain associations via resonance network
        cross_domain = []
        
        # Use the first matching concept as seed for associations
        if nearest_points:
            seed_concept, _ = nearest_points[0]
            
            # Get concept coordinates and find associations
            if seed_concept in self.representation_space.points:
                concept_coords = self.representation_space.points[seed_concept]
                
                # Find cross-domain associations
                for domain_idx in range(self.resonance_network.K):
                    associations = self.resonance_network.get_cross_domain_associations(
                        domain_idx=domain_idx,
                        representation=concept_coords,
                        threshold=0.4
                    )
                    cross_domain.extend(associations)
        
        # Sort by similarity
        cross_domain.sort(key=lambda x: x[2], reverse=True)
        results['cross_domain_associations'] = cross_domain
        
        return results
    
    def generate_output(self, context: Dict[str, Any], output_dim: int) -> np.ndarray:
        """
        Generate an output vector based on context.
        
        Args:
            context: Context dictionary
            output_dim: Dimensionality of output vector
            
        Returns:
            Generated output vector
        """
        # 1. Create context activation
        context_vector = context.get('input_vector', np.random.randn(self.representation_space.current_dim))
        context_vector = self.representation_space._project_to_current_dim(context_vector)
        
        # 2. Query knowledge with context
        query_results = self.query_knowledge(context_vector)
        
        # 3. Construct output from activated concepts
        output = np.zeros(output_dim)
        
        # Use active concepts to influence output
        active_concepts = query_results['active_concepts']
        if active_concepts:
            # Get coordinates of active concepts
            concept_coords = []
            concept_weights = []
            
            for concept_id, activation in active_concepts:
                if concept_id in self.representation_space.points:
                    coords = self.representation_space.points[concept_id]
                    # Project to output dimension
                    if len(coords) > output_dim:
                        coords = coords[:output_dim]
                    elif len(coords) < output_dim:
                        coords = np.pad(coords, (0, output_dim - len(coords)))
                    
                    concept_coords.append(coords)
                    concept_weights.append(activation)
            
            # Combine concepts with weighted average
            if concept_coords:
                concept_coords = np.stack(concept_coords)
                concept_weights = np.array(concept_weights)
                concept_weights = concept_weights / concept_weights.sum()  # Normalize
                
                weighted_output = np.sum(concept_coords * concept_weights[:, np.newaxis], axis=0)
                output += weighted_output
        
        # 4. Add context and noise modulation
        # Convert context modulation factor to range [0.8, 1.2]
        context_modulation = context.get('modulation', 0.5) * 0.4 + 0.8
        
        # Apply modulation
        output *= context_modulation
        
        # Add noise proportional to system temperature
        temperature = self.homeostasis.operational_temperature if self.enable_homeostasis else 0.5
        noise = np.random.normal(0, 0.1 * temperature, output_dim)
        output += noise
        
        # 5. Normalize output
        output_norm = np.linalg.norm(output)
        if output_norm > 0:
            output = output / output_norm
        
        return output
    
    def visualize_system(self, figsize=(20, 16), include_history=True):
        """
        Visualize the current state of the system.
        
        Args:
            figsize: Figure size
            include_history: Whether to include historical plots
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        if include_history:
            gs = fig.add_gridspec(3, 3)
            
            # 1. Representation space
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            self.representation_space.visualize(ax=ax1)
            
            # 2. Conceptual network
            ax2 = fig.add_subplot(gs[0, 1:])
            self.conceptual_network.visualize(ax=ax2)
            
            # 3. Resonance tensor
            ax3 = fig.add_subplot(gs[1, 0])
            self.resonance_network.visualize_tensor(ax=ax3)
            
            # 4. Homeostatic regulation
            if self.enable_homeostasis:
                ax4 = fig.add_subplot(gs[1, 1])
                self.homeostasis.visualize(ax=ax4)
            else:
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.text(0.5, 0.5, "Homeostasis Disabled", 
                      horizontalalignment='center', verticalalignment='center')
            
            # 5. Verification
            if self.enable_verification and self.verification:
                ax5 = fig.add_subplot(gs[1, 2])
                self.verification.visualize_verification_history(ax=ax5)
            else:
                ax5 = fig.add_subplot(gs[1, 2])
                ax5.text(0.5, 0.5, "Verification Disabled", 
                      horizontalalignment='center', verticalalignment='center')
            
            # 6. Performance metrics
            ax6 = fig.add_subplot(gs[2, :])
            self._plot_performance_metrics(ax=ax6)
        else:
            gs = fig.add_gridspec(2, 2)
            
            # 1. Representation space
            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            self.representation_space.visualize(ax=ax1)
            
            # 2. Conceptual network
            ax2 = fig.add_subplot(gs[0, 1])
            self.conceptual_network.visualize(ax=ax2)
            
            # 3. Resonance tensor
            ax3 = fig.add_subplot(gs[1, 0])
            self.resonance_network.visualize_tensor(ax=ax3)
            
            # 4. System state summary
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_system_summary(ax=ax4)
        
        # Add overall title
        fig.suptitle(f"NeuroplasticOS - Step {self.step_count}", fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_metrics(self, ax=None):
        """Plot performance metrics history."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        for metric_name, values in self.metrics.items():
            if values:
                ax.plot(values, label=metric_name)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _plot_system_summary(self, ax=None):
        """Plot a summary of current system state."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get current state
        state = self.get_system_state()
        
        # Create a text summary
        summary = [
            f"Step: {state['step']}",
            f"Representation Dim: {state['current_dim']}",
            f"Topology: {state['topology_type']}",
            f"Concepts: {state['num_concepts']}",
            f"Relations: {state['num_relations']}",
            f"Energy: {state['energy']:.2f}/{state['max_energy']:.2f}",
            f"Stability: {state['stability']:.2f}",
            f"Temperature: {state['temperature']:.2f}",
            "\nMetrics:",
        ]
        
        for k, v in state['metrics'].items():
            summary.append(f"  {k}: {v:.3f}")
        
        summary_text = '\n'.join(summary)
        
        # Display as text
        ax.text(0.05, 0.95, summary_text, 
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title("System Summary")
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics and statistics
        """
        report = {
            'system': {
                'step_count': self.step_count,
                'experience_count': self.cumulative_experience_count,
                'current_dim': self.representation_space.current_dim,
                'topology_type': self.representation_space.topology_type,
                'num_concepts': len(self.conceptual_network.nodes),
                'num_relations': len(self.conceptual_network.edges)
            },
            'metrics': {
                'current': {k: v[-1] if v else 0 for k, v in self.metrics.items()},
                'average': {k: sum(v) / len(v) if v else 0 for k, v in self.metrics.items()},
                'trend': {k: (v[-1] - v[-10]) if len(v) >= 10 else 0 for k, v in self.metrics.items()}
            }
        }
        
        # Add regulation statistics if enabled
        if self.enable_homeostasis:
            report['homeostasis'] = {
                'current_energy': self.homeostasis.current_energy,
                'stability': self.homeostasis.stability,
                'temperature': self.homeostasis.operational_temperature,
                'interventions': len(self.homeostasis.intervention_history)
            }
        
        # Add verification statistics if enabled
        if self.enable_verification and self.verification:
            report['verification'] = self.verification.get_verification_summary()
        
        # Add emergent properties analysis
        report['emergent_properties'] = self._analyze_emergent_properties()
        
        return report
    
    def _analyze_emergent_properties(self) -> Dict[str, Any]:
        """
        Analyze emergent properties of the system.
        
        Returns:
            Dictionary with analysis results
        """
        # Initialize results
        results = {}
        
        # 1. Analyze concept clustering
        if len(self.conceptual_network.nodes) >= 5:
            try:
                # Create adjacency matrix from the conceptual network
                node_list = list(self.conceptual_network.nodes)
                node_indices = {node: i for i, node in enumerate(node_list)}
                
                n = len(node_list)
                adjacency = np.zeros((n, n))
                
                for _, (nodes, weight) in self.conceptual_network.edges.items():
                    # Convert to indices
                    node_indices_list = [node_indices[node] for node in nodes if node in node_indices]
                    
                    # For each pair in the edge, add the weight
                    for i in node_indices_list:
                        for j in node_indices_list:
                            if i != j:
                                adjacency[i, j] += weight
                
                # Detect communities using a simple approach
                from sklearn.cluster import src.SpectralClustering
                n_clusters = min(5, max(2, n // 3))
                
                clustering = SpectralClustering(
                    n_clusters=n_clusters, 
                    affinity='precomputed',
                    assign_labels='discretize'
                ).fit(adjacency)
                
                # Organize concepts by cluster
                clusters = {}
                for i, label in enumerate(clustering.labels_):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(node_list[i])
                
                results['concept_clusters'] = clusters
                
            except Exception as e:
                logger.warning(f"Error in concept clustering: {e}")
                results['concept_clusters'] = {'error': str(e)}
        else:
            results['concept_clusters'] = {'info': 'Too few concepts for clustering'}
        
        # 2. Analyze representation space structure
        if self.representation_space.points:
            try:
                points = np.array(list(self.representation_space.points.values()))
                
                # Calculate basic statistics
                mean_dist = 0
                n_points = len(points)
                
                if n_points > 1:
                    # Calculate average pairwise distance
                    total_dist = 0
                    count = 0
                    
                    for i in range(n_points):
                        for j in range(i+1, n_points):
                            dist = np.linalg.norm(points[i] - points[j])
                            total_dist += dist
                            count += 1
                    
                    mean_dist = total_dist / max(1, count)
                
                results['space_structure'] = {
                    'num_points': n_points,
                    'dimensionality': self.representation_space.current_dim,
                    'mean_distance': mean_dist,
                    'topology': self.representation_space.topology_type
                }
                
            except Exception as e:
                logger.warning(f"Error in space structure analysis: {e}")
                results['space_structure'] = {'error': str(e)}
        else:
            results['space_structure'] = {'info': 'No points in representation space'}
        
        # 3. Analyze resonance patterns
        if self.resonance_network.activation_history:
            try:
                # Get recent activation patterns
                recent_activations = self.resonance_network.activation_history[-10:]
                
                # Calculate average resonance magnitude
                avg_magnitude = sum(a['resonance_magnitude'] for a in recent_activations) / len(recent_activations)
                
                # Count frequency of context and domain activations
                context_counts = {}
                domain_counts = {}
                
                for a in recent_activations:
                    for c in a['context_indices']:
                        context_counts[c] = context_counts.get(c, 0) + 1
                    
                    for d in a['domain_indices']:
                        domain_counts[d] = domain_counts.get(d, 0) + 1
                
                results['resonance_patterns'] = {
                    'average_magnitude': avg_magnitude,
                    'most_active_contexts': sorted(context_counts.items(), key=lambda x: x[1], reverse=True)[:3],
                    'most_active_domains': sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                }
                
            except Exception as e:
                logger.warning(f"Error in resonance pattern analysis: {e}")
                results['resonance_patterns'] = {'error': str(e)}
        else:
            results['resonance_patterns'] = {'info': 'No resonance activations recorded yet'}
        
        return results


def create_sample_experience_generator(
    base_dim: int = 8,
    num_patterns: int = 3,
    noise_level: float = 0.2,
    pattern_drift_rate: float = 0.01
) -> Callable[[int], List[np.ndarray]]:
    """
    Create a function that generates realistic synthetic experiences.
    
    Args:
        base_dim: Base dimensionality of experiences
        num_patterns: Number of underlying patterns
        noise_level: Amount of noise to add
        pattern_drift_rate: Rate at which patterns evolve
        
    Returns:
        Function that generates experiences
    """
    # Create base patterns
    base_patterns = []
    for i in range(num_patterns):
        # Create a distinct pattern
        if i == 0:
            # Sinusoidal pattern
            pattern = np.sin(np.linspace(0, 2 * np.pi, base_dim))
        elif i == 1:
            # Exponential decay pattern
            pattern = np.exp(-np.linspace(0, 2, base_dim))
        else:
            # Random but consistent pattern
            pattern = np.random.randn(base_dim)
            pattern = pattern / np.linalg.norm(pattern)
        
        base_patterns.append(pattern)
    
    def generator(step: int, num_experiences: int = 5) -> List[np.ndarray]:
        """
        Generate synthetic experiences for a specific step.
        
        Args:
            step: Current time step
            num_experiences: Number of experiences to generate
            
        Returns:
            List of experience vectors
        """
        # Evolve patterns slightly based on step
        current_patterns = []
        for pattern in base_patterns:
            # Add time-dependent drift
            drift = pattern_drift_rate * step
            evolved = pattern + drift * np.sin(np.linspace(0, 2 * np.pi, base_dim) + drift)
            current_patterns.append(evolved)
        
        # Generate experiences as combinations of patterns plus noise
        experiences = []
        
        for _ in range(num_experiences):
            # Random weights for combining patterns
            weights = np.random.random(len(current_patterns))
            weights /= weights.sum()  # Normalize
            
            # Combine patterns
            exp = sum(w * p for w, p in zip(weights, current_patterns))
            
            # Add noise
            noise = np.random.normal(0, noise_level, base_dim)
            exp += noise
            
            # Normalize
            exp = exp / (np.linalg.norm(exp) + 1e-10)
            
            experiences.append(exp)
        
        return experiences
    
    return generator


def run_nos_simulation(
    num_steps: int = 100,
    experience_dim: int = 8,
    experiences_per_step: int = 5,
    plot_every: int = 10,
    enable_homeostasis: bool = True,
    enable_verification: bool = True
) -> NeuroplasticOS:
    """
    Run a simulation of the NeuroplasticOS.
    
    Args:
        num_steps: Number of simulation steps to run
        experience_dim: Dimensionality of experience vectors
        experiences_per_step: Number of experiences per step
        plot_every: Interval for plotting system state
        enable_homeostasis: Whether to enable homeostatic regulation
        enable_verification: Whether to enable verification system
        
    Returns:
        The NeuroplasticOS instance after simulation
    """
    # Create a NeuroplasticOS instance
    nos = NeuroplasticOS(
        initial_dim=experience_dim,
        initial_concepts=["concept_1", "concept_2", "concept_3"],
        enable_homeostasis=enable_homeostasis,
        enable_verification=enable_verification
    )
    
    # Create an experience generator
    generator = create_sample_experience_generator(
        base_dim=experience_dim,
        num_patterns=4,
        noise_level=0.3,
        pattern_drift_rate=0.02
    )
    
    # Run simulation
    results = []
    
    print(f"Starting simulation: {num_steps} steps")
    for step in tqdm(range(num_steps), desc="Simulation Progress"):
        # Generate experiences
        experiences = generator(step, experiences_per_step)
        
        # Execute step
        step_result = nos.step(experiences)
        results.append(step_result)
        
        # Plot system state periodically
        if plot_every > 0 and (step + 1) % plot_every == 0:
            fig = nos.visualize_system()
            plt.suptitle(f"NeuroplasticOS - Step {step+1}/{num_steps}", fontsize=16)
            plt.tight_layout()
            plt.show()
    
    # Generate final performance report
    report = nos.get_performance_report()
    
    # Display summary
    print("\n=== Simulation Complete ===")
    print(f"Steps: {num_steps}")
    print(f"Final dimensionality: {report['system']['current_dim']}")
    print(f"Concepts emerged: {report['system']['num_concepts']}")
    print(f"Relations formed: {report['system']['num_relations']}")
    print("\nMetrics:")
    for metric, value in report['metrics']['current'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Final visualization
    fig = nos.visualize_system()
    plt.suptitle(f"NeuroplasticOS - Final State (Step {num_steps})", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return nos


if __name__ == "__main__":
    # Configure logging
    logging.getLogger().setLevel(logging.INFO)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run simulation
    nos = run_nos_simulation(
        num_steps=50,
        experience_dim=8,
        experiences_per_step=5,
        plot_every=10,
        enable_homeostasis=True,
        enable_verification=True
    )
    
    print("NeuroplasticOS simulation complete!")
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
            metric_change_ratio = torch.norm(self.metric_tensor - original_metric) / torch.norm(original_metric)
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
            from scipy.spatial.distance import src.pdist
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
                expanded = torch.eye(target_dim, dtype=self.metric_tensor.dtype,
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
                    cov_tensor = torch.tensor(cov, dtype=self.metric_tensor.dtype,
                                              device=self.metric_tensor.device)
                    precision = torch.inverse(cov_tensor)
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
                eigenvalues = torch.linalg.eigvalsh(self.metric_tensor)
                min_eig = torch.min(eigenvalues)
                
                # Add regularization if needed
                if min_eig < 1e-6:
                    self.metric_tensor += (1e-6 - min_eig) * torch.eye(
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
                self.metric_tensor = torch.eye(
                    self.current_dim,
                    dtype=self.metric_tensor.dtype,
                    device=self.metric_tensor.device
                )
            else:
                self.metric_tensor = np.eye(self.current_dim)


class ConceptualResonanceNetwork:
    """
    Implementation of a conceptual resonance network that connects
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
    
    def update_tensor(self, experiences: List[Dict[str, np.ndarray]]) -> None:
        """
        Update the resonance tensor based on new experiences.
        
        Args:
            experiences: List of experience dictionaries with domain and context info
        """
        if not experiences:
            return
        
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


class HomeostasisRegulator:
    """
    Regulator that maintains system stability while allowing adaptive changes.
    Implements multiple homeostatic mechanisms inspired by biological systems.
    """
    
    def __init__(
        self,
        energy_capacity: float = 1.0,
        stability_threshold: float = 0.3,
        recovery_rate: float = 0.02,
        adaptation_cost_factor: float = 0.05
    ):
        """
        Initialize the homeostasis regulator.
        
        Args:
            energy_capacity: Maximum energy capacity of the system
            stability_threshold: Threshold below which stabilization is prioritized
            recovery_rate: Base rate of energy recovery per step
            adaptation_cost_factor: Cost factor for adaptation operations
        """
        # System energy state
        self.max_energy = energy_capacity
        self.current_energy = energy_capacity
        
        # System stability state
        self.stability = 1.0
        self.stability_threshold = stability_threshold
        
        # Recovery parameters
        self.base_recovery_rate = recovery_rate
        self.adaptation_cost_factor = adaptation_cost_factor
        
        # Operational parameters
        self.operational_temperature = 0.5  # Controls exploration-exploitation balance
        
        # Critical properties that must be maintained
        self.critical_properties = {
            'min_energy': 0.1,
            'min_stability': 0.2
        }
        
        # History for analysis
        self.energy_history = [energy_capacity]
        self.stability_history = [1.0]
        self.temperature_history = [0.5]
        self.intervention_history = []
    
    def regulate(
        self, 
        system_state: Dict[str, Any],
        proposed_adaptations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Regulate the system to maintain stability while allowing adaptation.
        
        Args:
            system_state: Current state of the system
            proposed_adaptations: Dictionary of proposed adaptations with energy costs
            
        Returns:
            Dictionary with regulation decisions and updated state
        """
        # Extract current state
        current_stability = system_state.get('stability', self.stability)
        
        # Calculate total energy required for all proposed adaptations
        total_adaptation_energy = sum(proposed_adaptations.values())
        
        # Check if proposed adaptations would violate critical properties
        critical_violation = False
        projected_energy = self.current_energy - total_adaptation_energy * self.adaptation_cost_factor
        projected_stability = current_stability * (1 - 0.1 * sum(proposed_adaptations.values()) / len(proposed_adaptations))
        
        if projected_energy < self.critical_properties['min_energy']:
            critical_violation = True
            logger.warning("Critical energy level would be violated by proposed adaptations")
        
        if projected_stability < self.critical_properties['min_stability']:
            critical_violation = True
            logger.warning("Critical stability level would be violated by proposed adaptations")
        
        # Make regulation decisions
        allowed_adaptations = {}
        denied_adaptations = {}
        
        # If critical violation, deny all adaptations
        if critical_violation:
            for adaptation, energy in proposed_adaptations.items():
                denied_adaptations[adaptation] = "Critical property violation"
            
            # Record intervention
            self.intervention_history.append({
                'step': len(self.energy_history),
                'type': 'critical_violation',
                'action': 'deny_all'
            })
            
            # Emergency stability recovery
            self.stability = min(1.0, current_stability + 0.1)
            self.current_energy = min(self.max_energy, self.current_energy + 0.1)
        else:
            # Prioritize adaptations if energy is limited
            available_energy = self.current_energy / self.adaptation_cost_factor
            
            if total_adaptation_energy > available_energy:
                # Sort adaptations by importance/energy ratio
                # In a full implementation, this would use more sophisticated prioritization
                sorted_adaptations = sorted(
                    proposed_adaptations.items(),
                    key=lambda x: system_state.get(f'{x[0]}_priority', 0.5) / x[1],
                    reverse=True
                )
                
                # Allow adaptations until energy is exhausted
                remaining_energy = available_energy
                for adaptation, energy in sorted_adaptations:
                    if energy <= remaining_energy:
                        allowed_adaptations[adaptation] = energy
                        remaining_energy -= energy
                    else:
                        denied_adaptations[adaptation] = "Insufficient energy"
                
                # Record intervention
                self.intervention_history.append({
                    'step': len(self.energy_history),
                    'type': 'energy_prioritization',
                    'action': 'prioritize'
                })
            else:
                # All adaptations allowed
                allowed_adaptations = proposed_adaptations.copy()
            
            # Consume energy for allowed adaptations
            energy_consumed = sum(allowed_adaptations.values()) * self.adaptation_cost_factor
            self.current_energy = max(0, self.current_energy - energy_consumed)
            
            # Update stability based on adaptations
            if allowed_adaptations:
                adaptation_magnitude = sum(allowed_adaptations.values()) / len(allowed_adaptations)
                self.stability = max(
                    self.critical_properties['min_stability'],
                    current_stability * (1 - 0.05 * adaptation_magnitude)
                )
            else:
                # No adaptations, stability recovers slightly
                self.stability = min(1.0, current_stability + 0.01)
        
        # Adjust temperature based on system state
        if self.stability < self.stability_threshold:
            # Lower temperature to favor exploitation when stability is low
            self.operational_temperature = max(0.1, self.operational_temperature - 0.05)
        else:
            # Increase temperature to favor exploration when stability is high
            self.operational_temperature = min(1.0, self.operational_temperature + 0.02)
        
        # Natural energy recovery
        recovery_rate = self.base_recovery_rate * (0.5 + 0.5 * self.stability)
        self.current_energy = min(self.max_energy, self.current_energy + recovery_rate)
        
        # Record history
        self.energy_history.append(self.current_energy)
        self.stability_history.append(self.stability)
        self.temperature_history.append(self.operational_temperature)
        
        # Return regulation results
        return {
            'allowed_adaptations': allowed_adaptations,
            'denied_adaptations': denied_adaptations,
            'current_energy': self.current_energy,
            'current_stability': self.stability,
            'operational_temperature': self.operational_temperature
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the regulator.
        
        Returns:
            Dictionary with current state values
        """
        return {
            'current_energy': self.current_energy,
            'max_energy': self.max_energy,
            'stability': self.stability,
            'stability_threshold': self.stability_threshold,
            'operational_temperature': self.operational_temperature
        }
    
    def visualize(self, ax=None):
        """
        Visualize the history of homeostatic regulation.
        
        Args:
            ax: Optional matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot energy and stability history
        x = range(len(self.energy_history))
        ax.plot(x, self.energy_history, 'r-', label='Energy')
        ax.plot(x, self.stability_history, 'b-', label='Stability')
        ax.plot(x, self.temperature_history, 'g-', label='Temperature')
        
        # Add reference lines
        ax.axhline(y=self.stability_threshold, color='b', linestyle='--', alpha=0.5, label='Stability Threshold')
        ax.axhline(y=self.critical_properties['min_energy'], color='r', linestyle='--', alpha=0.5, label='Critical Energy')
        
        # Mark interventions
        for intervention in self.intervention_history:
            step = intervention['step']
            if step < len(self.energy_history):
                if intervention['type'] == 'critical_violation':
                    ax.axvline(x=step, color='red', alpha=0.3)
                elif intervention['type'] == 'energy_prioritization':
                    ax.axvline(x=step, color='orange', alpha=0.3)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Homeostatic Regulation History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


class VerificationMonitor:
    """
    Runtime verification system that monitors invariant properties and
    safety conditions during system operation.
    """
    
    def __init__(self, reversion_buffer_size: int = 5):
        """
        Initialize the verification monitor.
        
        Args:
            reversion_buffer_size: Size of the system state buffer for reversions
        """
        # Define invariant properties that must hold
        self.invariant_properties = {
            'energy_positive': lambda state: state.get('energy', 0) >= 0,
            'stability_in_range': lambda state: 0 <= state.get('stability', 0) <= 1,
            'representation_dim_in_bounds': lambda state: (
                state.get('min_dim', 0) <= state.get('current_dim', 0) <= state.get('max_dim', float('inf'))
            ),
            'metrics_bounded': lambda state: all(
                0 <= metric <= 10 for metric in state.get('metrics', {}).values()
            ),
            'temperature_in_range': lambda state: 0 <= state.get('temperature', 0) <= 1
        }
        
        # Define critical properties that trigger reversion if violated
        self.critical_properties = {
            'min_energy': lambda state: state.get('energy', 0) >= 0.05,
            'min_stability': lambda state: state.get('stability', 0) >= 0.1
        }
        
        # Initialize violation counters
        self.violation_counts = {
            prop: 0 for prop in list(self.invariant_properties.keys()) + list(self.critical_properties.keys())
        }
        
        # Initialize state history for reversions
        self.state_history = []
        self.reversion_buffer_size = reversion_buffer_size
        
        # Monitoring statistics
        self.checks_performed = 0
        self.violations_detected = 0
        self.reversions_performed = 0
        
        # Verification history
        self.verification_history = []
    
    def check_state(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the current system state satisfies all invariant properties.
        
        Args:
            current_state: Current system state dictionary
            
        Returns:
            Dictionary with verification results
        """
        self.checks_performed += 1
        
        # Store state in history buffer
        self.state_history.append(current_state.copy())
        if len(self.state_history) > self.reversion_buffer_size:
            self.state_history.pop(0)
        
        # Check invariant properties
        invariant_results = {}
        for prop_name, prop_fn in self.invariant_properties.items():
            try:
                invariant_results[prop_name] = prop_fn(current_state)
                if not invariant_results[prop_name]:
                    self.violation_counts[prop_name] += 1
            except Exception as e:
                logger.warning(f"Error checking invariant {prop_name}: {e}")
                invariant_results[prop_name] = False
                self.violation_counts[prop_name] += 1
        
        # Check critical properties
        critical_results = {}
        for prop_name, prop_fn in self.critical_properties.items():
            try:
                critical_results[prop_name] = prop_fn(current_state)
                if not critical_results[prop_name]:
                    self.violation_counts[prop_name] += 1
            except Exception as e:
                logger.warning(f"Error checking critical property {prop_name}: {e}")
                critical_results[prop_name] = False
                self.violation_counts[prop_name] += 1
        
        # Calculate verification summary
        all_invariants_satisfied = all(invariant_results.values())
        all_critical_properties_satisfied = all(critical_results.values())
        
        if not all_invariants_satisfied:
            self.violations_detected += 1
        
        # Record verification result
        result = {
            'invariants_satisfied': all_invariants_satisfied,
            'critical_properties_satisfied': all_critical_properties_satisfied,
            'invariant_results': invariant_results,
            'critical_results': critical_results,
            'reversion_needed': not all_critical_properties_satisfied,
            'warnings': []
        }
        
        # Generate warnings for specific issues
        for prop_name, satisfied in {**invariant_results, **critical_results}.items():
            if not satisfied:
                result['warnings'].append(f"Property violation: {prop_name}")
        
        # Record history
        self.verification_history.append({
            'step': self.checks_performed,
            'all_invariants_satisfied': all_invariants_satisfied,
            'all_critical_properties_satisfied': all_critical_properties_satisfied,
            'reversion_needed': result['reversion_needed']
        })
        
        return result
    
    def get_reversion_state(self) -> Optional[Dict[str, Any]]:
        """
        Get a stable state to revert to if needed.
        
        Returns:
            Historical state dictionary, or None if no suitable state is available
        """
        # Find the most recent state where all critical properties were satisfied
        for state in reversed(self.state_history[:-1]):  # Skip current state
            all_critical_satisfied = True
            
            for prop_name, prop_fn in self.critical_properties.items():
                try:
                    if not prop_fn(state):
                        all_critical_satisfied = False
                        break
                except:
                    all_critical_satisfied = False
                    break
            
            if all_critical_satisfied:
                self.reversions_performed += 1
                return state.copy()
        
        # If no suitable state found, return None
        logger.warning("No suitable reversion state found")
        return None
    
    def predict_safety(
        self, 
        current_state: Dict[str, Any],
        proposed_actions: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Predict the safety of proposed actions.
        
        Args:
            current_state: Current system state
            proposed_actions: List of proposed action dictionaries
            
        Returns:
            List of safety scores (0-1) for each action
        """
        safety_scores = []
        
        for action in proposed_actions:
            # Simple simulation of action effects
            projected_state = self._project_action_effects(current_state, action)
            
            # Check invariants on projected state
            invariant_satisfied_count = 0
            for prop_fn in self.invariant_properties.values():
                try:
                    if prop_fn(projected_state):
                        invariant_satisfied_count += 1
                except:
                    pass
            
            # Check critical properties on projected state
            critical_satisfied_count = 0
            for prop_fn in self.critical_properties.values():
                try:
                    if prop_fn(projected_state):
                        critical_satisfied_count += 1
                except:
                    pass
            
            # Calculate safety score
            invariant_ratio = invariant_satisfied_count / len(self.invariant_properties)
            critical_ratio = critical_satisfied_count / len(self.critical_properties)
            
            # Critical properties are weighted more heavily
            safety_score = 0.4 * invariant_ratio + 0.6 * critical_ratio
            safety_scores.append(safety_score)
        
        return safety_scores
    
    def _project_action_effects(
        self, 
        state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Project the effects of an action on the current state.
        
        Args:
            state: Current state
            action: Proposed action
            
        Returns:
            Projected future state
        """
        # Clone the state
        projected = state.copy()
        
        # Simple projection heuristics based on action type
        action_type = action.get('type', '')
        magnitude = action.get('magnitude', 0.5)
        
        if action_type == 'dimension_change':
            projected['current_dim'] = action.get('target_dim', projected.get('current_dim', 0))
            projected['energy'] = max(0, projected.get('energy', 0) - 0.1 * magnitude)
            projected['stability'] = max(0, projected.get('stability', 0) - 0.05 * magnitude)
            
        elif action_type == 'topology_change':
            projected['energy'] = max(0, projected.get('energy', 0) - 0.15 * magnitude)
            projected['stability'] = max(0, projected.get('stability', 0) - 0.1 * magnitude)
            
        elif action_type == 'concept_creation':
            projected['energy'] = max(0, projected.get('energy', 0) - 0.05 * magnitude)
            projected['stability'] = max(0, projected.get('stability', 0) - 0.02 * magnitude)
            
        elif action_type == 'meta_learning':
            projected['energy'] = max(0, projected.get('energy', 0) - 0.2 * magnitude)
            projected['stability'] = max(0, projected.get('stability', 0) - 0.08 * magnitude)
            
        # Generic effects for all actions
        projected['temperature'] = min(1.0, projected.get('temperature', 0.5) + 0.05 * magnitude)
        
        return projected
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the verification system's performance.
        
        Returns:
            Dictionary with verification statistics
        """
        return {
            'checks_performed': self.checks_performed,
            'violations_detected': self.violations_detected,
            'reversions_performed': self.reversions_performed,
            'violation_counts': self.violation_counts,
            'verification_rate': (self.checks_performed - self.violations_detected) / max(1, self.checks_performed),
            'reversion_rate': self.reversions_performed / max(1, self.violations_detected) if self.violations_detected else 0
        }
    
    def visualize_verification_history(self, ax=None):
        """
        Visualize the verification history.
        
        Args:
            ax: Optional matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if not self.verification_history:
            ax.text(0.5, 0.5, "No verification history", 
                   horizontalalignment='center', verticalalignment='center')
            return ax
        
        # Prepare data
        steps = [record['step'] for record in self.verification_history]
        invariants = [int(record['all_invariants_satisfied']) for record in self.verification_history]
        critical = [int(record['all_critical_properties_satisfied']) for record in self.verification_history]
        reversions = [int(record['reversion_needed']) for record in self.verification_history]
        
        # Create plot
        ax.plot(steps, invariants, 'g-', label='Invariants Satisfied')
        ax.plot(steps, critical, 'b-', label='Critical Properties Satisfied')
        ax.plot(steps, reversions, 'r-', label='Reversion Needed')
        
        # Mark reversion points
        reversion_steps = [steps[i] for i, r in enumerate(reversions) if r]
        if reversion_steps:
            ax.scatter(reversion_steps, [0] * len(reversion_steps), color='red', marker='x', s=100, label='Reversion Points')
        
        ax.set_xlabel('Verification Step')
        ax.set_ylabel('Status')
        ax.set_title('System Verification History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


class NeuroplasticOS:
    """
    Main implementation of the Neuroplastic Operating System.
    Integrates all components into a coherent system.
    """
    
    def __init__(
        self,
        initial_dim: int = 8,
        initial_concepts: Optional[List[str]] = None,
        initial_energy: float = 1.0,
        plasticity_rate: float = 0.1,
        enable_verification: bool = True,
        enable_homeostasis: bool = True
    ):
        """
        Initialize the NeuroplasticOS.
        
        Args:
            initial_dim: Initial dimensionality of representation space
            initial_concepts: Optional list of initial concepts
            initial_energy: Initial system energy
            plasticity_rate: Base plasticity rate
            enable_verification: Whether to enable the verification system
            enable_homeostasis: Whether to enable homeostatic regulation
        """
        logger.info("Initializing NeuroplasticOS")
        
        # Initialize components
        self.representation_space = DynamicRepresentationSpace(
            initial_dim=initial_dim,
            min_dim=4,
            max_dim=24,
            plasticity_rate=plasticity_rate
        )
        
        self.conceptual_network = EmergentConceptualNetwork(
            initial_concepts=initial_concepts or ["base_concept_1", "base_concept_2"],
            emergence_threshold=0.6,
            pruning_threshold=0.2,
            learning_rate=0.1
        )
        
        self.resonance_network = ConceptualResonanceNetwork(
            contextual_dims=4,
            knowledge_domains=3,
            representational_modes=initial_dim
        )
        
        # Core regulatory systems
        self.homeostasis = HomeostasisRegulator(
            energy_capacity=initial_energy,
            stability_threshold=0.3,
            recovery_rate=0.03,
            adaptation_cost_factor=0.05
        )
        
        self.verification = VerificationMonitor(
            reversion_buffer_size=10
        ) if enable_verification else None
        
        # System flags
        self.enable_verification = enable_verification
        self.enable_homeostasis = enable_homeostasis
        
        # System state
        self.step_count = 0
        self.cumulative_experience_count = 0
        
        # Experience and activation history
        self.recent_experiences = []
        self.experience_buffer_size = 100
        
        # Performance metrics
        self.metrics = {
            'adaptation_rate': [],
            'representational_efficiency': [],
            'energy_consumption': [],
            'concept_emergence': [],
            'stability_index': []
        }
        
        # Initialize with starter points if concepts provided
        if initial_concepts:
            self._initialize_starter_concepts(initial_concepts)
        
        logger.info(f"NeuroplasticOS initialized with dimension {initial_dim}")
    
    def _initialize_starter_concepts(self, concepts: List[str]) -> None:
        """
        Initialize the system with starter concepts.
        
        Args:
            concepts: List of concept names
        """
        for i, concept in enumerate(concepts):
            # Generate a semi-random embedding for each concept
            coords = np.random.normal(0, 1, self.representation_space.current_dim)
            coords = coords / np.linalg.norm(coords)  # Normalize
            
            # Add to representation space
            self.representation_space.add_point(concept, coords)
            
            # Ensure concept is in the conceptual network
            if concept not in self.conceptual_network.nodes:
                self.conceptual_network.add_node(concept)
        
        # Add simple relations between concepts
        if len(concepts) >= 2:
            for i in range(len(concepts) - 1):
                self.conceptual_network.add_edge(
                    [concepts[i], concepts[i+1]], 
                    0.5  # Initial weight
                )
        
        # If we have 3+ concepts, add one hyperedge
        if len(concepts) >= 3:
            self.conceptual_network.add_edge(
                concepts[:min(5, len(concepts))],
                0.3  # Initial weight
            )
            
        logger.info(f"Initialized with {len(concepts)} starter concepts")
    
    def step(self, experiences: List[np.ndarray]) -> Dict[str, Any]:
        """
        Execute one step of the NOS, processing new experiences.
        
        Args:
            experiences: List of experience vectors
            
        Returns:
            Dictionary with step results
        """
        self.step_count += 1
        self.cumulative_experience_count += len(experiences)
        
        logger.info(f"Step {self.step_count}: Processing {len(experiences)} new experiences")
        
        # Get current system state for regulation
        current_state = self.get_system_state()
        
        # 1. Verify current state if enabled
        if self.enable_verification and self.verification:
            verification_result = self.verification.check_state(current_state)
            
            # Handle critical violations with reversion if needed
            if verification_result['reversion_needed']:
                logger.warning("Critical property violation detected - reverting to safe state")
                reversion_state = self.verification.get_reversion_state()
                
                if reversion_state:
                    # Apply reversion
                    self._apply_reversion(reversion_state)
                    
                    # Return early with reversion info
                    return {
                        'status': 'reverted',
                        'verification': verification_result,
                        'reverted_to_step': reversion_state.get('step', -1),
                        'metrics': {k: v[-1] if v else 0 for k, v in self.metrics.items()}
                    }
        
        # 2. Determine proposed adaptations
        proposed_adaptations = self._plan_adaptations(experiences, current_state)
        
        # 3. Apply homeostatic regulation if enabled
        if self.enable_homeostasis:
            regulation_result = self.homeostasis.regulate(current_state, proposed_adaptations)
            allowed_adaptations = regulation_result['allowed_adaptations']
            current_state.update({
                'energy': regulation_result['current_energy'],
                'stability': regulation_result['current_stability'],
                'temperature': regulation_result['operational_temperature']
            })
        else:
            # If homeostasis disabled, allow all adaptations
            allowed_adaptations = proposed_adaptations
        
        # 4. Update the representation space
        if 'representation_space' in allowed_adaptations:
            self.representation_space.transform_space(
                experiences,
                {'energy': allowed_adaptations['representation_space']}
            )
        
        # 5. Update the conceptual network
        if 'conceptual_network' in allowed_adaptations:
            # Create formatted experiences for concept emergence
            self.conceptual_network.update(experiences, self.representation_space)
        
        # 6. Update the resonance network
        if 'resonance_network' in allowed_adaptations:
            # Format experiences for the resonance network
            formatted_experiences = []
            for i, exp in enumerate(experiences):
                # Assign to random domain and context for demonstration
                formatted_exp = {
                    'domain_idx': i % self.resonance_network.K,
                    'context_idx': (i // self.resonance_network.K) % self.resonance_network.C,
                    'content': exp
                }
                formatted_experiences.append(formatted_exp)
                
            self.resonance_network.update_tensor(formatted_experiences)
        
        # 7. Update experience buffer
        self._update_experience_buffer(experiences)
        
        # 8. Calculate and record metrics
        self._calculate_metrics(current_state)
        
        # 9. Return step results
        result = {
            'status': 'success',
            'step': self.step_count,
            'energy': self.homeostasis.current_energy if self.enable_homeostasis else 1.0,
            'stability': self.homeostasis.stability if self.enable_homeostasis else 1.0,
            'representation_dim': self.representation_space.current_dim,
            'num_concepts': len(self.conceptual_network.nodes),
            'num_relations': len(self.conceptual_network.edges),
            'metrics': {k: v[-1] if v else 0 for k, v in self.metrics.items()},
            'adaptations_allowed': list(allowed_adaptations.keys()),
            'adaptations_denied': [] if not self.enable_homeostasis else 
                                  [k for k in proposed_adaptations if k not in allowed_adaptations]
        }
        
        return result
    
    def _plan_adaptations(
        self, 
        experiences: List[np.ndarray], 
        current_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Plan potential adaptations based on new experiences.
        
        Args:
            experiences: New experience vectors
            current_state: Current system state
            
        Returns:
            Dictionary mapping adaptation types to energy requirements
        """
        adaptations = {}
        
        # Always consider basic updates
        adaptations['conceptual_network'] = 0.1
        adaptations['resonance_network'] = 0.1
        
        # Consider representation space transformation if:
        # 1. We have sufficient experiences
        # 2. Experiences seem different from existing representations
        # 3. We haven't updated recently or temperature is high
        
        if len(experiences) >= 3:
            # Check if experiences are significantly different from existing points
            novelty = self._estimate_experience_novelty(experiences)
            
            if (novelty > 0.6 or 
                self.step_count % 5 == 0 or 
                current_state.get('temperature', 0.5) > 0.7):
                
                # Energy required depends on novelty and current dimension
                energy_required = 0.2 + 0.3 * novelty + 0.01 * self.representation_space.current_dim
                adaptations['representation_space'] = min(0.5, energy_required)
        
        return adaptations
    
    def _estimate_experience_novelty(self, experiences: List[np.ndarray]) -> float:
        """
        Estimate how novel the new experiences are compared to existing representations.
        
        Args:
            experiences: New experience vectors
            
        Returns:
            Novelty score (0-1)
        """
        if not experiences or not self.representation_space.points:
            return 0.5  # Default medium novelty
        
        # Get existing points
        existing_points = list(self.representation_space.points.values())
        
        # Compute minimum distances from each experience to any existing point
        min_distances = []
        
        for exp in experiences:
            exp_proj = self.representation_space._project_to_current_dim(exp)
            
            distances = []
            for point in existing_points:
                dist = np.linalg.norm(exp_proj - point)
                distances.append(dist)
            
            if distances:
                min_distances.append(min(distances))
        
        if not min_distances:
            return 0.5
        
        # Convert distances to novelty score
        avg_min_distance = sum(min_distances) / len(min_distances)
        
        # Normalize by expected distance in the space
        expected_max_distance = np.sqrt(self.representation_space.current_dim)
        novelty = min(1.0, avg_min_distance / (0.5 * expected_max_distance))
        
        return novelty
    
    def _update_experience_buffer(self, new_experiences: List[np.ndarray]) -> None:
        """
        Update the buffer of recent experiences.
        
        Args:
            new_experiences: New experience vectors
        """
        self.recent_experiences.extend(new_experiences)
        
        # Keep only the most recent experiences
        if len(self.recent_experiences) > self.experience_buffer_size:
            self.recent_experiences = self.recent_experiences[-self.experience_buffer_size:]
    
    def _calculate_metrics(self, current_state: Dict[str, Any]) -> None:
        """
        Calculate and record system performance metrics.
        
        Args:
            current_state: Current system state
        """
        # Adaptation rate: measure of how quickly system is changing
        if len(self.representation_space.dim_history) > 5:
            dim_changes = len(set(self.representation_space.dim_history[-5:]))
            adaptation_rate = (dim_changes - 1) / 4  # Range 0-1
        else:
            adaptation_rate = 0.0
        
        # Representational efficiency: how well space represents concepts
        if self.representation_space.points and len(self.conceptual_network.nodes) > 0:
            # Ratio of concepts to dimensions, normalized
            rep_efficiency = min(1.0, len(self.conceptual_network.nodes) / (2 * self.representation_space.current_dim))
        else:
            rep_efficiency = 0.0
        
        # Energy consumption
        energy_consumption = 1.0 - current_state.get('energy', 1.0)
        
        # Concept emergence rate
        if self.step_count > 1:
            concept_emergence = self.conceptual_network.emergence_count / self.step_count
        else:
            concept_emergence = 0.0
        
        # Stability index
        stability_index = current_state.get('stability', 1.0)
        
        # Record metrics
        self.metrics['adaptation_rate'].append(adaptation_rate)
        self.metrics['representational_efficiency'].append(rep_efficiency)
        self.metrics['energy_consumption'].append(energy_consumption)
        self.metrics['concept_emergence'].append(concept_emergence)
        self.metrics['stability_index'].append(stability_index)
    
    def _apply_reversion(self, reversion_state: Dict[str, Any]) -> None:
        """
        Apply a reversion to a previous safe state.
        
        Args:
            reversion_state: State to revert to
        """
        logger.warning(f"Applying reversion to state from step {reversion_state.get('step', 'unknown')}")
        
        # Restore key system parameters
        if 'energy' in reversion_state and self.enable_homeostasis:
            self.homeostasis.current_energy = reversion_state['energy']
        
        if 'stability' in reversion_state and self.enable_homeostasis:
            self.homeostasis.stability = reversion_state['stability']
        
        if 'current_dim' in reversion_state:
            self.representation_space._adjust_dimensionality(reversion_state['current_dim'])
        
        # Record the reversion in metrics
        for metric_name in self.metrics:
            if self.metrics[metric_name]:
                # Add previous value to indicate reversion
                self.metrics[metric_name].append(self.metrics[metric_name][-1])
        
        logger.info("Reversion applied successfully")
