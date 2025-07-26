"""
Verification Monitor Implementation

This module implements the verification monitor component of the Neuroplastic
Operating Systems framework. It provides runtime verification, predictive
verification, and reversion mechanisms to ensure system safety.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
import logging
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerificationMonitor")

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
        
        logger.info(f"Initialized verification monitor with buffer size {reversion_buffer_size}")
    
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
        self.state_history.append(copy.deepcopy(current_state))
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
            logger.warning(f"Invariant violation detected at check {self.checks_performed}")
        
        if not all_critical_properties_satisfied:
            logger.error(f"Critical property violation detected at check {self.checks_performed}")
        
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
                logger.info(f"Found suitable reversion state at step {state.get('step', 'unknown')}")
                return copy.deepcopy(state)
        
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
            
            logger.debug(f"Predicted safety score for action {action.get('type', 'unknown')}: {safety_score:.2f}")
        
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
        projected = copy.deepcopy(state)
        
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
    
    def add_custom_invariant(self, name: str, property_function: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Add a custom invariant property to check.
        
        Args:
            name: Name of the property
            property_function: Function that takes a state dictionary and returns True if the property is satisfied
        """
        self.invariant_properties[name] = property_function
        self.violation_counts[name] = 0
        logger.info(f"Added custom invariant property: {name}")
    
    def add_custom_critical_property(self, name: str, property_function: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Add a custom critical property to check.
        
        Args:
            name: Name of the property
            property_function: Function that takes a state dictionary and returns True if the property is satisfied
        """
        self.critical_properties[name] = property_function
        self.violation_counts[name] = 0
        logger.info(f"Added custom critical property: {name}")
    
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
            'reversion_rate': self.reversions_performed / max(1, self.violations_detected) if self.violations_detected else 0,
            'invariant_properties': list(self.invariant_properties.keys()),
            'critical_properties': list(self.critical_properties.keys())
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
            ax.text(0.5, 0.5, "No verification history available", 
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
    
    def visualize_violation_distribution(self, ax=None):
        """
        Visualize the distribution of property violations.
        
        Args:
            ax: Optional matplotlib axis
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get violation counts
        properties = list(self.violation_counts.keys())
        counts = list(self.violation_counts.values())
        
        # Sort by count for better visualization
        sorted_data = sorted(zip(properties, counts), key=lambda x: x[1], reverse=True)
        properties = [p for p, _ in sorted_data]
        counts = [c for _, c in sorted_data]
        
        # Create bar chart
        bars = ax.bar(properties, counts)
        
        # Color critical properties differently
        for i, prop in enumerate(properties):
            if prop in self.critical_properties:
                bars[i].set_color('red')
        
        ax.set_xlabel('Property')
        ax.set_ylabel('Violation Count')
        ax.set_title('Property Violation Distribution')
        ax.set_xticklabels(properties, rotation=45, ha='right')
        
        # Add a legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Critical Property')
        blue_patch = mpatches.Patch(color='blue', label='Invariant Property')
        ax.legend(handles=[red_patch, blue_patch])
        
        return ax


if __name__ == "__main__":
    # Example usage
    # Create a verification monitor
    monitor = VerificationMonitor(reversion_buffer_size=5)
    
    # Add a custom invariant property
    monitor.add_custom_invariant(
        name='concept_count_reasonable',
        property_function=lambda state: state.get('num_concepts', 0) <= 100
    )
    
    # Add a custom critical property
    monitor.add_custom_critical_property(
        name='network_connected',
        property_function=lambda state: state.get('network_connected', True)
    )
    
    # Simulate a sequence of states
    for step in range(50):
        # Generate a simulated state
        state = {
            'step': step,
            'energy': 1.0 - 0.01 * step if step < 40 else 0.04,  # Energy decreases over time
            'stability': 1.0 - 0.005 * step,
            'current_dim': 8 + step // 10,
            'min_dim': 4,
            'max_dim': 24,
            'temperature': 0.5,
            'num_concepts': 10 + step,
            'metrics': {'metric1': 0.5, 'metric2': 0.8},
            'network_connected': step < 45  # Network becomes disconnected at step 45
        }
        
        # Check state
        result = monitor.check_state(state)
        
        # Handle reversion if needed
        if result['reversion_needed']:
            reversion_state = monitor.get_reversion_state()
            if reversion_state:
                print(f"Step {step}: Reverting to state from step {reversion_state.get('step')}")
    
    # Get verification summary
    summary = monitor.get_verification_summary()
    print("\nVerification Summary:")
    print(f"Checks performed: {summary['checks_performed']}")
    print(f"Violations detected: {summary['violations_detected']}")
    print(f"Reversions performed: {summary['reversions_performed']}")
    print(f"Verification rate: {summary['verification_rate']:.2f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    monitor.visualize_verification_history(ax=ax1)
    monitor.visualize_violation_distribution(ax=ax2)
    plt.tight_layout()
    plt.show()
