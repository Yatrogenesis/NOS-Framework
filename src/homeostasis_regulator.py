"""
Homeostasis Regulator Implementation

This module implements the homeostasis regulator component of the Neuroplastic
Operating Systems framework. It provides mechanisms to maintain system stability
while allowing for adaptive plasticity.
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
logger = logging.getLogger("HomeostasisRegulator")

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
        
        logger.info(f"Initialized homeostasis regulator with energy capacity {energy_capacity}")
    
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
        projected_stability = current_stability * (1 - 0.1 * sum(proposed_adaptations.values()) / max(1, len(proposed_adaptations)))
        
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
            
            logger.info("Critical violation detected - denying all adaptations and initiating recovery")
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
                
                logger.info(f"Energy constraints detected - allowing {len(allowed_adaptations)}/{len(proposed_adaptations)} adaptations")
            else:
                # All adaptations allowed
                allowed_adaptations = proposed_adaptations.copy()
                logger.debug("All proposed adaptations allowed")
            
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
            logger.debug(f"Lowering temperature to {self.operational_temperature:.2f} due to low stability")
        else:
            # Increase temperature to favor exploration when stability is high
            self.operational_temperature = min(1.0, self.operational_temperature + 0.02)
            logger.debug(f"Raising temperature to {self.operational_temperature:.2f} due to high stability")
        
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
    
    def get_regulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the regulation history.
        
        Returns:
            Dictionary with regulation statistics
        """
        critical_violations = sum(1 for i in self.intervention_history if i['type'] == 'critical_violation')
        energy_prioritizations = sum(1 for i in self.intervention_history if i['type'] == 'energy_prioritization')
        
        return {
            'current_energy': self.current_energy,
            'current_stability': self.stability,
            'current_temperature': self.operational_temperature,
            'energy_recovery_rate': self.base_recovery_rate * (0.5 + 0.5 * self.stability),
            'adaptation_capacity': self.current_energy / self.adaptation_cost_factor,
            'total_interventions': len(self.intervention_history),
            'critical_violations': critical_violations,
            'energy_prioritizations': energy_prioritizations,
            'avg_energy': sum(self.energy_history) / len(self.energy_history) if self.energy_history else 0,
            'avg_stability': sum(self.stability_history) / len(self.stability_history) if self.stability_history else 0
        }


if __name__ == "__main__":
    # Example usage
    # Create a homeostasis regulator
    regulator = HomeostasisRegulator(
        energy_capacity=1.0,
        stability_threshold=0.3,
        recovery_rate=0.02,
        adaptation_cost_factor=0.05
    )
    
    # Simulate a sequence of adaptation requests
    for step in range(50):
        # Generate some proposed adaptations
        if step % 10 == 0:
            # Large adaptation request every 10 steps
            proposed_adaptations = {
                'representation_space': 0.4,
                'conceptual_network': 0.3,
                'meta_learning': 0.2
            }
        else:
            # Smaller adaptations otherwise
            proposed_adaptations = {
                'representation_space': 0.1,
                'conceptual_network': 0.05
            }
        
        # Current system state (simplified)
        system_state = {
            'stability': regulator.stability,
            'step': step
        }
        
        # Apply regulation
        result = regulator.regulate(system_state, proposed_adaptations)
        
        # Print decisions for large adaptation steps
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Allowed: {result['allowed_adaptations']}")
            print(f"  Denied: {result['denied_adaptations']}")
            print(f"  Energy: {result['current_energy']:.2f}, Stability: {result['current_stability']:.2f}")
    
    # Visualize regulation history
    plt.figure(figsize=(12, 6))
    regulator.visualize()
    plt.title('Homeostasis Regulation Simulation')
    plt.tight_layout()
    plt.show()
