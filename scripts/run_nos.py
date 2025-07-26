#!/usr/bin/env python3
"""
Main execution script for Neuroplastic Operating Systems (NOS) experiments
Integrates all modules and runs complete simulation suite
"""

import src.os
import src.sys
import src.argparse
import src.json
import src.logging
from src.pathlib import src.Path
from src.datetime import src.datetime

# Import NOS modules
from src.dynamic_representation_space import src.DynamicRepresentationSpace
from src.emergent_conceptual_network import src.EmergentConceptualNetwork
from src.contextual_resonance_network import src.ContextualResonanceTensor
from src.homeostasis_regulator import src.HomeostasisRegulator
from src.verification_monitor import src.VerificationMonitor
from src.simulation_framework import src.NOSSimulation
from src.nos_experiments import (
    stability_plasticity_experiment,
    concept_emergence_experiment,
    dimensional_adaptation_experiment,
    performance_comparison_experiment,
    cross_domain_integration_experiment
)

def setup_logging(log_level='INFO'):
    """Configure logging for the experiment run"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'nos_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('NOS')

def create_output_directory():
    """Create timestamped output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    return output_dir

def load_config(config_path):
    """Load experimental configuration from JSON file"""
    default_config = {
        "experiments": {
            "stability_plasticity": True,
            "concept_emergence": True,
            "dimensional_adaptation": True,
            "performance_comparison": True,
            "cross_domain": True
        },
        "parameters": {
            "max_dimensions": 64,
            "simulation_steps": 50,
            "random_seeds": [42, 123, 456, 789, 1011],
            "complexity_range": [0.2, 0.9],
            "plasticity_range": [0.01, 0.5]
        },
        "output": {
            "save_figures": True,
            "save_data": True,
            "figure_format": "eps",
            "figure_dpi": 300
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configurations
        default_config.update(user_config)
    
    return default_config

def run_complete_experiment_suite(config, output_dir, logger):
    """Execute all NOS experiments based on configuration"""
    
    results = {}
    
    if config["experiments"]["stability_plasticity"]:
        logger.info("Running Stability-Plasticity Tradeoff Experiment...")
        results["stability_plasticity"] = stability_plasticity_experiment(
            plasticity_range=config["parameters"]["plasticity_range"],
            steps=config["parameters"]["simulation_steps"],
            seeds=config["parameters"]["random_seeds"],
            output_dir=output_dir / "figures"
        )
    
    if config["experiments"]["concept_emergence"]:
        logger.info("Running Concept Emergence Experiment...")
        results["concept_emergence"] = concept_emergence_experiment(
            complexity_range=config["parameters"]["complexity_range"],
            steps=config["parameters"]["simulation_steps"],
            seeds=config["parameters"]["random_seeds"],
            output_dir=output_dir / "figures"
        )
    
    if config["experiments"]["dimensional_adaptation"]:
        logger.info("Running Dimensional Adaptation Experiment...")
        results["dimensional_adaptation"] = dimensional_adaptation_experiment(
            max_dim=config["parameters"]["max_dimensions"],
            steps=config["parameters"]["simulation_steps"],
            seeds=config["parameters"]["random_seeds"],
            output_dir=output_dir / "figures"
        )
    
    if config["experiments"]["performance_comparison"]:
        logger.info("Running Performance Comparison Experiment...")
        results["performance_comparison"] = performance_comparison_experiment(
            steps=config["parameters"]["simulation_steps"],
            seeds=config["parameters"]["random_seeds"],
            output_dir=output_dir / "figures"
        )
    
    if config["experiments"]["cross_domain"]:
        logger.info("Running Cross-Domain Integration Experiment...")
        results["cross_domain"] = cross_domain_integration_experiment(
            steps=config["parameters"]["simulation_steps"],
            seeds=config["parameters"]["random_seeds"],
            output_dir=output_dir / "figures"
        )
    
    return results

def save_results(results, output_dir):
    """Save experimental results to JSON file"""
    results_file = output_dir / "experimental_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary report
    summary_file = output_dir / "experiment_summary.md"
    with open(summary_file, 'w') as f:
        f.write("# NOS Experimental Results Summary\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        for experiment, data in results.items():
            f.write(f"## {experiment.replace('_', ' ').title()}\n\n")
            if isinstance(data, dict) and 'summary_stats' in data:
                for stat, value in data['summary_stats'].items():
                    f.write(f"- **{stat}**: {value}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Run NOS Experimental Suite')
    parser.add_argument('--config', type=str, help='Configuration JSON file')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.log_level)
    output_dir = Path(args.output_dir) if args.output_dir else create_output_directory()
    
    logger.info(f"Starting NOS Experimental Suite")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configuration
    config = load_config(args.config)
    
    if args.quick:
        # Reduce parameters for quick testing
        config["parameters"]["simulation_steps"] = 10
        config["parameters"]["random_seeds"] = [42]
        config["parameters"]["max_dimensions"] = 16
        logger.info("Running in quick test mode")
    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run experiments
        results = run_complete_experiment_suite(config, output_dir, logger)
        
        # Save results
        save_results(results, output_dir)
        
        logger.info(f"Experimental suite completed successfully")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Experimental suite failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())