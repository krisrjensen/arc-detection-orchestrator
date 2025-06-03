# src/arc_detection/utils/paths.py

import os
from pathlib import Path
import logging
from typing import Dict, List, Set, Optional, Tuple
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.utils.paths")


def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Absolute path to the created directory
    """
    abs_path = os.path.abspath(os.path.expanduser(directory_path))
    os.makedirs(abs_path, exist_ok=True)
    logger.debug(f"Ensured directory exists: {abs_path}")
    return abs_path

class DatasetScanner:
    """
    Scans and analyzes the directory structure of a dataset.
    Maps folder paths to logical data categories.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the scanner with the base directory.
        
        Args:
            base_dir: Path to the dataset base directory
        """
        self.base_dir = Path(base_dir)
        self.experiment_types: Set[str] = set()
        self.label_mapping: Dict[str, str] = {
            "transient_negative_test": "no_arc_load_transient",
            "arc_matrix_experiment_with_parallel_motor": "arc_motor_transient",
            "arc_matrix_experiment": "arc_transient"
        }
        self.augmented_labels: Dict[str, str] = {
            "transient_negative_test_left": "no_arc_steady_state",
            "transient_negative_test_right": "no_arc_steady_state",
            "arc_matrix_experiment_with_parallel_motor_left": "motor_running_steady_state",
            "arc_matrix_experiment_with_parallel_motor_right": "arc_motor_continuous",
            "arc_matrix_experiment_left": "no_arc_steady_state",
            "arc_matrix_experiment_right": "arc_continuous"
        }
        
    def scan_dataset(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan the dataset directory and classify files.
        
        Returns:
            Dictionary mapping experiment types to files
        """
        if not self.base_dir.exists():
            logger.error(f"Base directory does not exist: {self.base_dir}")
            return {}
            
        dataset_structure = {}
        
        # Scan for experiment types
        for item in self.base_dir.iterdir():
            if item.is_dir():
                self.experiment_types.add(item.name)
                
                experiment_files = list(item.glob("*.mat"))
                dataset_structure[item.name] = {
                    "path": str(item),
                    "files": [str(f) for f in experiment_files],
                    "label": self.label_mapping.get(item.name, "unknown")
                }
                logger.info(f"Found experiment type: {item.name} with {len(experiment_files)} files")
                
        return dataset_structure
        
    def generate_label_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Generate mappings between file paths and data labels.
        
        Returns:
            Dictionary with label mappings
        """
        mappings = {
            "experiment_labels": self.label_mapping.copy(),
            "augmented_labels": self.augmented_labels.copy()
        }
        
        # Generate file-to-label mapping
        file_labels = {}
        
        for exp_type, exp_info in self.scan_dataset().items():
            label = exp_info["label"]
            for file_path in exp_info["files"]:
                file_labels[file_path] = label
                
        mappings["file_labels"] = file_labels
        return mappings
        
    def save_dataset_structure(self, output_path: str = "config/dataset_structure.yaml"):
        """
        Save the dataset structure to a YAML file.
        
        Args:
            output_path: Path to save the structure file
        """
        output_dir = Path(output_path).parent
        os.makedirs(output_dir, exist_ok=True)
        
        structure = {
            "base_directory": str(self.base_dir),
            "experiment_types": list(self.experiment_types),
            "label_mapping": self.label_mapping,
            "augmented_labels": self.augmented_labels,
            "experiments": self.scan_dataset()
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(structure, f, default_flow_style=False)
            
        logger.info(f"Dataset structure saved to {output_path}")
        return output_path