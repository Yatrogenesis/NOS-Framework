import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dynamic_representation_space import DynamicRepresentationSpace
from emergent_conceptual_network import EmergentConceptualNetwork
from contextual_resonance_network import ContextualResonanceTensor
from homeostasis_regulator import HomeostasisRegulator
from verification_monitor import VerificationMonitor

class TestDynamicRepresentationSpace(unittest.TestCase):
    
    def setUp(self):
        self.space = DynamicRepresentationSpace(
            initial_dim=8,
            max_dim=64,
            plasticity_rate=0.1
        )
    
    def test_initialization(self):
        """Test proper initialization of representation space"""
        self.assertEqual(self.space.current_dim, 8)
        self.assertIsNotNone(self.space.metric_tensor)
        self.assertTrue(torch.is_tensor(self.space.representations))
    
    def test_metric_tensor_evolution(self):
        """Test metric tensor updates"""
        initial_metric = self.space.metric_tensor.clone()
        
        # Simulate experience
        experience = torch.randn(1, 8)
        self.space.update_metric_tensor(experience)
        
        # Metric should have changed
        self.assertFalse(torch.equal(initial_metric, self.space.metric_tensor))
    
    def test_dimensional_adaptation(self):
        """Test adaptive dimensionality changes"""
        initial_dim = self.space.current_dim
        
        # High complexity should increase dimensions
        high_complexity_data = torch.randn(10, 8) * 2.0
        for data in high_complexity_data:
            self.space.adapt_structure(data, complexity=0.9)
        
        # Dimension should increase or stay same
        self.assertGreaterEqual(self.space.current_dim, initial_dim)
    
    def test_stability_constraints(self):
        """Test that space maintains stability constraints"""
        for _ in range(100):
            experience = torch.randn(1, self.space.current_dim)
            self.space.update(experience)
            
            # Dimensions should stay within bounds
            self.assertLessEqual(self.space.current_dim, self.space.max_dim)
            self.assertGreaterEqual(self.space.current_dim, 1)

class TestEmergentConceptualNetwork(unittest.TestCase):
    
    def setUp(self):
        self.network = EmergentConceptualNetwork(
            emergence_threshold=0.3,
            stability_threshold=0.7,
            max_concepts=100
        )
    
    def test_concept_emergence(self):
        """Test emergence of new concepts"""
        initial_concept_count = len(self.network.concepts)
        
        # Add experiences that should trigger concept emergence
        for i in range(10):
            experience = torch.randn(8) + i * 0.5  # Different clusters
            self.network.process_experience(experience, complexity=0.8)
        
        # Should have emerged new concepts
        self.assertGreater(len(self.network.concepts), initial_concept_count)
    
    def test_hypergraph_properties(self):
        """Test hypergraph structure properties"""
        # Add some concepts
        for i in range(5):
            concept = torch.randn(8)
            self.network.add_concept(concept)
        
        # Check graph properties
        self.assertGreaterEqual(self.network.get_network_density(), 0.0)
        self.assertLessEqual(self.network.get_network_density(), 1.0)
    
    def test_concept_pruning(self):
        """Test removal of irrelevant concepts"""
        # Add many low-relevance concepts
        for i in range(50):
            concept = torch.randn(8) * 0.1  # Low magnitude
            concept_id = self.network.add_concept(concept)
            self.network.concept_relevance[concept_id] = 0.1  # Low relevance
        
        initial_count = len(self.network.concepts)
        self.network.prune_concepts()
        
        # Should have pruned some concepts
        self.assertLess(len(self.network.concepts), initial_count)

class TestContextualResonanceTensor(unittest.TestCase):
    
    def setUp(self):
        self.crt = ContextualResonanceTensor(
            context_dims=5,
            knowledge_domains=3,
            representation_modes=4
        )
    
    def test_tensor_initialization(self):
        """Test proper tensor initialization"""
        self.assertEqual(self.crt.tensor.shape, (5, 3, 4))
        self.assertTrue(torch.is_tensor(self.crt.tensor))
    
    def test_resonance_computation(self):
        """Test resonance strength computation"""
        domain1_pattern = torch.randn(5)
        domain2_pattern = torch.randn(5)
        
        resonance = self.crt.compute_resonance(domain1_pattern, domain2_pattern)
        
        self.assertIsInstance(resonance, float)
        self.assertGreaterEqual(resonance, 0.0)
        self.assertLessEqual(resonance, 1.0)
    
    def test_cross_domain_transfer(self):
        """Test knowledge transfer between domains"""
        # Train on domain 1
        for _ in range(10):
            pattern = torch.randn(5)
            self.crt.update_resonance(pattern, domain=0)
        
        # Test transfer to domain 2
        test_pattern = torch.randn(5)
        transfer_strength = self.crt.get_transfer_strength(test_pattern, 
                                                         source_domain=0, 
                                                         target_domain=1)
        
        self.assertIsInstance(transfer_strength, float)

class TestHomeostasisRegulator(unittest.TestCase):
    
    def setUp(self):
        self.regulator = HomeostasisRegulator(
            energy_budget=100.0,
            stability_threshold=0.8
        )
    
    def test_energy_monitoring(self):
        """Test energy consumption monitoring"""
        initial_energy = self.regulator.current_energy
        
        # Consume some energy
        self.regulator.consume_energy(20.0)
        
        self.assertEqual(self.regulator.current_energy, initial_energy - 20.0)
    
    def test_stability_regulation(self):
        """Test stability maintenance"""
        # Set low stability
        self.regulator.current_stability = 0.5
        
        action = self.regulator.regulate()
        
        # Should recommend stability-enhancing action
        self.assertIn(action, ['reduce_plasticity', 'increase_stability', 'maintain'])
    
    def test_energy_budget_enforcement(self):
        """Test energy budget constraints"""
        # Try to exceed budget
        with self.assertRaises(Exception):
            self.regulator.consume_energy(150.0)  # Exceeds budget of 100

class TestVerificationMonitor(unittest.TestCase):
    
    def setUp(self):
        self.monitor = VerificationMonitor()
    
    def test_invariant_checking(self):
        """Test safety invariant verification"""
        # Define test invariants
        def energy_invariant(state):
            return state.get('energy', 0) <= 100
        
        def stability_invariant(state):
            return state.get('stability', 0) >= 0.5
        
        self.monitor.add_invariant('energy', energy_invariant)
        self.monitor.add_invariant('stability', stability_invariant)
        
        # Test valid state
        valid_state = {'energy': 50, 'stability': 0.8}
        self.assertTrue(self.monitor.verify_state(valid_state))
        
        # Test invalid state
        invalid_state = {'energy': 150, 'stability': 0.3}
        self.assertFalse(self.monitor.verify_state(invalid_state))
    
    def test_reversion_mechanism(self):
        """Test state reversion capabilities"""
        state1 = {'energy': 50, 'stability': 0.8}
        state2 = {'energy': 75, 'stability': 0.7}
        state3 = {'energy': 120, 'stability': 0.4}  # Invalid
        
        self.monitor.save_state(state1)
        self.monitor.save_state(state2)
        
        # Revert to last valid state
        reverted_state = self.monitor.revert_to_last_valid()
        self.assertEqual(reverted_state, state2)

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDynamicRepresentationSpace,
        TestEmergentConceptualNetwork,
        TestContextualResonanceTensor,
        TestHomeostasisRegulator,
        TestVerificationMonitor
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)