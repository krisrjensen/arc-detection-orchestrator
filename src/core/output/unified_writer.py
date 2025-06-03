"""
Unified Data Writer Module for DC Arc Detection

This module implements a specialized writer for creating unified data files
in the format required for the DC arc detection research project.

Author: Kristophor Jensen
"""

import os
import logging
from pathlib import Path
import numpy as np
import csv
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.output.unified_writer")

class UnifiedDataWriter:
    """
    Generates unified data files centered on detected transients.
    
    This class creates standardized CSV files containing time, source voltage,
    load voltage, and source current data centered on detected transients.
    It also supports generating augmented data files shifted left and right
    from the transient.
    """
    
    def __init__(self, 
                 total_points: int = 165888, 
                 augmentation_shift: int = 16384):
        """
        Initialize the unified data writer.
        
        Args:
            total_points: Total number of points in the output files
            augmentation_shift: Number of points to shift for augmented data
        """
        self.total_points = total_points
        self.augmentation_shift = augmentation_shift
        logger.info(f"Initialized UnifiedDataWriter with {total_points} points per file")
    
    def write_unified_data(self, 
                          aligned_data: Dict[str, Any], 
                          output_path: str,
                          file_prefix: str = "",
                          generate_augmented: bool = True) -> List[str]:
        """
        Write unified data files centered on the transient.
        
        Args:
            aligned_data: Aligned data from the processor
            output_path: Directory to save the files
            file_prefix: Prefix for the output files
            generate_augmented: Whether to generate augmented data files
            
        Returns:
            List of paths to the generated files
        """
        if not aligned_data or 'time_data' not in aligned_data or 'channels' not in aligned_data:
            logger.error("Invalid aligned data structure")
            return []
        
        # Extract required channels
        channels = aligned_data['channels']
        time_data = aligned_data['time_data']
        
        # Validate required channels
        required_channels = ['source_voltage', 'load_voltage', 'source_current']
        missing_channels = [ch for ch in required_channels if ch not in channels]
        if missing_channels:
            logger.error(f"Missing required channels: {missing_channels}")
            return []
        
        # Get transient index
        if 'alignment' not in aligned_data or 'aligned_transient_index' not in aligned_data['alignment']:
            logger.error("Missing alignment information")
            return []
        
        transient_idx = aligned_data['alignment']['aligned_transient_index']
        
        # Create output directory if needed
        os.makedirs(output_path, exist_ok=True)
        
        # Write the main unified file
        main_file = self._write_file(
            time_data, 
            channels['source_voltage'],
            channels['load_voltage'],
            channels['source_current'],
            transient_idx,
            os.path.join(output_path, f"{file_prefix}transient_centered.csv")
        )
        
        generated_files = [main_file] if main_file else []
        
        # Generate augmented data if requested
        if generate_augmented and len(time_data) >= self.total_points:
            # Left-shifted files
            for i in range(1, int((len(time_data) - self.total_points) / self.augmentation_shift) + 1):
                shift = i * self.augmentation_shift
                if transient_idx - shift >= self.total_points // 2:
                    left_file = self._write_file(
                        time_data, 
                        channels['source_voltage'],
                        channels['load_voltage'],
                        channels['source_current'],
                        transient_idx - shift,
                        os.path.join(output_path, f"{file_prefix}L{i:03d}.csv")
                    )
                    if left_file:
                        generated_files.append(left_file)
            
            # Right-shifted files
            for i in range(1, int((len(time_data) - self.total_points) / self.augmentation_shift) + 1):
                shift = i * self.augmentation_shift
                if transient_idx + shift < len(time_data) - self.total_points // 2:
                    right_file = self._write_file(
                        time_data, 
                        channels['source_voltage'],
                        channels['load_voltage'],
                        channels['source_current'],
                        transient_idx + shift,
                        os.path.join(output_path, f"{file_prefix}R{i:03d}.csv")
                    )
                    if right_file:
                        generated_files.append(right_file)
        
        logger.info(f"Generated {len(generated_files)} unified data files")
        return generated_files
    
    def _write_file(self,
                   time_data: np.ndarray,
                   source_voltage: np.ndarray,
                   load_voltage: np.ndarray,
                   source_current: np.ndarray,
                   center_idx: int,
                   filepath: str) -> Optional[str]:
        """
        Write a single unified data file centered at the given index.
        
        Args:
            time_data: Time data array
            source_voltage: Source voltage array
            load_voltage: Load voltage array
            source_current: Source current array
            center_idx: Index to center the data at
            filepath: Path to save the file
            
        Returns:
            Path to the generated file or None if writing failed
        """
        try:
            # Calculate start and end indices
            half_points = self.total_points // 2
            start_idx = max(0, center_idx - half_points)
            end_idx = min(len(time_data), start_idx + self.total_points)
            
            # Adjust start index if end is clipped
            if end_idx < start_idx + self.total_points:
                start_idx = max(0, end_idx - self.total_points)
            
            # Extract data slices
            time_slice = time_data[start_idx:end_idx]
            source_voltage_slice = source_voltage[start_idx:end_idx]
            load_voltage_slice = load_voltage[start_idx:end_idx]
            source_current_slice = source_current[start_idx:end_idx]
            
            # Ensure all slices have the same length
            min_length = min(len(time_slice), len(source_voltage_slice), 
                           len(load_voltage_slice), len(source_current_slice))
            
            time_slice = time_slice[:min_length]
            source_voltage_slice = source_voltage_slice[:min_length]
            load_voltage_slice = load_voltage_slice[:min_length]
            source_current_slice = source_current_slice[:min_length]
            
            # Write CSV file
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'source_voltage', 'load_voltage', 'source_current'])
                
                for i in range(min_length):
                    writer.writerow([
                        time_slice[i],
                        source_voltage_slice[i],
                        load_voltage_slice[i],
                        source_current_slice[i]
                    ])
            
            logger.info(f"Wrote unified data file to {filepath} with {min_length} points")
            return filepath
            
        except Exception as e:
            logger.error(f"Error writing unified data file: {str(e)}")
            return None

def write_unified_dataset(aligned_data: Dict[str, Any], 
                         output_dir: str,
                         exp_type: str = "unknown",
                         file_prefix: str = "",
                         total_points: int = 165888) -> List[str]:
    """
    Convenience function to write a complete unified dataset.
    
    Args:
        aligned_data: Aligned data from the processor
        output_dir: Base output directory
        exp_type: Experiment type for directory organization
        file_prefix: Prefix for output files
        total_points: Total points per file
        
    Returns:
        List of paths to the generated files
    """
    # Create writer
    writer = UnifiedDataWriter(total_points=total_points)
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(output_dir, exp_type)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Write unified files
    return writer.write_unified_data(
        aligned_data,
        exp_output_dir,
        file_prefix=file_prefix,
        generate_augmented=True
    )
