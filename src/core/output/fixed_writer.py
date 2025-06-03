"""
Unified Data Writer Module with fixed left-shift ordering for DC Arc Detection

This module implements a specialized writer for creating unified data files
in various formats required for the DC arc detection research project.
It fixes the left-shifted file numbering to ensure L001 is closest to the transient.

Author: Kristophor Jensen (Modified version)
"""

import os
import logging
from pathlib import Path
import numpy as np
import csv
import scipy.io as sio
from typing import Dict, List, Optional, Any, Union, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.output.fixed_writer")

class FixedUnifiedDataWriter:
    """
    Generates unified data files centered on detected transients with improved left-shift ordering.
    
    This class creates standardized data files containing time, source voltage,
    load voltage, and source current data centered on detected transients.
    It supports multiple output formats including CSV, NPY, NPZ, and MAT.
    
    The left-shifted files are named so that L001 is closest to the transient,
    and higher numbers are farther from the transient (closer to the beginning of the file).
    """
    
    def __init__(self, 
                 total_points: int = 165888, 
                 augmentation_shift: int = 65536,  # Adjusted to balance overlap and file count
                 output_formats: Union[List[str], str, Set[str]] = None):
        """
        Initialize the unified data writer.
        
        Args:
            total_points: Total number of points in the output files
            augmentation_shift: Number of points to shift for augmented data
            output_formats: Output formats ('csv', 'npy', 'npz', 'mat') as list, set or comma-separated string
        """
        self.total_points = total_points
        self.augmentation_shift = augmentation_shift
        
        # Process output_formats parameter (could be list, set, or comma-separated string)
        if output_formats is None:
            self.output_formats = {'npy'}  # Default to NPY format
        elif isinstance(output_formats, str):
            # Split comma-separated string
            self.output_formats = {fmt.strip().lower() for fmt in output_formats.split(',')}
        else:
            # Convert list or set to set of lowercase strings
            self.output_formats = {fmt.strip().lower() for fmt in output_formats}
        
        # Validate output formats
        valid_formats = {'csv', 'npy', 'npz', 'mat'}
        invalid_formats = self.output_formats - valid_formats
        if invalid_formats:
            logger.warning(f"Invalid output formats: {invalid_formats}. Will be ignored.")
            self.output_formats = self.output_formats.intersection(valid_formats)
        
        if not self.output_formats:
            logger.warning("No valid output formats specified. Defaulting to NPY.")
            self.output_formats = {'npy'}
        
        formats_str = ', '.join(sorted(self.output_formats))
        logger.info(f"Initialized FixedUnifiedDataWriter with {total_points} points per file, formats: {formats_str}")
    
    def write_unified_data(self, 
                          data: Dict[str, Any], 
                          transient_result: Dict[str, Any],
                          output_path: str,
                          file_prefix: str = "",
                          generate_augmented: bool = True) -> List[str]:
        """
        Write unified data files centered on the transient.
        
        Args:
            data: Original data dictionary with full time series
            transient_result: Transient detection result
            output_path: Directory to save the files
            file_prefix: Prefix for the output files
            generate_augmented: Whether to generate augmented data files
            
        Returns:
            List of paths to the generated files
        """
        if not data or 'time_data' not in data or 'channels' not in data:
            logger.error("Invalid data structure")
            return []
        
        # Extract required channels
        channels = data['channels']
        time_data = data['time_data']
        
        # Validate required channels
        required_channels = ['source_voltage', 'load_voltage', 'source_current']
        missing_channels = [ch for ch in required_channels if ch not in channels]
        if missing_channels:
            logger.error(f"Missing required channels: {missing_channels}")
            return []
        
        # Get transient index
        if not transient_result or not transient_result.get('success', False):
            logger.error("Missing or invalid transient result")
            return []
        
        transient_idx = transient_result['transient_index']
        logger.info(f"Processing data with transient at index {transient_idx}")
        
        # Create output directory if needed
        output_path = os.path.expanduser(output_path)  # Expand ~ in path
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        # Write the main transient-centered file
        logger.info(f"Writing main transient file with formats: {', '.join(sorted(self.output_formats))}")
        main_files = self._write_files(
            time_data, 
            channels['source_voltage'],
            channels['load_voltage'],
            channels['source_current'],
            transient_idx,
            os.path.join(output_path, f"{file_prefix}transient_centered")
        )
        
        generated_files = main_files
        
        # Generate augmented data if requested
        if generate_augmented:
            logger.info("Generating augmented data files")
            # Calculate maximum possible shift
            max_left_shift = transient_idx - self.total_points//2
            max_right_shift = len(time_data) - transient_idx - self.total_points//2
            
            logger.info(f"Maximum left shift: {max_left_shift}, Maximum right shift: {max_right_shift}")
            
            if max_left_shift > 0 or max_right_shift > 0:
                # Calculate how many left-shifted files will be created
                num_left_files = int(max_left_shift / self.augmentation_shift)
                
                # Left-shifted files - FIXED ORDERING: L001 is closest to transient, LNNN farthest
                for i in range(1, num_left_files + 1):
                    # Calculate position from the transient (reversed from original)
                    # Now i=1 is closest to the transient, i=num_left_files is farthest
                    position = num_left_files - i + 1  # Reverse the position
                    
                    # Calculate the actual shift amount based on position
                    shift = position * self.augmentation_shift
                    
                    if shift <= max_left_shift:
                        logger.info(f"Creating left-shifted file L{i:03d} with shift {shift} (position {position})")
                        left_files = self._write_files(
                            time_data, 
                            channels['source_voltage'],
                            channels['load_voltage'],
                            channels['source_current'],
                            transient_idx - shift,
                            os.path.join(output_path, f"{file_prefix}L{i:03d}")
                        )
                        generated_files.extend(left_files)
                
                # Right-shifted files (same as original)
                for i in range(1, int(max_right_shift / self.augmentation_shift) + 1):
                    shift = i * self.augmentation_shift
                    if shift <= max_right_shift:
                        logger.info(f"Creating right-shifted file R{i:03d} with shift {shift}")
                        right_files = self._write_files(
                            time_data, 
                            channels['source_voltage'],
                            channels['load_voltage'],
                            channels['source_current'],
                            transient_idx + shift,
                            os.path.join(output_path, f"{file_prefix}R{i:03d}")
                        )
                        generated_files.extend(right_files)
            else:
                logger.warning("Data not long enough for augmented files")
        
        # Check file existence after writing
        existing_files = []
        for file_path in generated_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                existing_files.append(file_path)
                logger.debug(f"Verified file {file_path} exists, size: {file_size} bytes")
            else:
                logger.error(f"Output file {file_path} was not created!")
        
        logger.info(f"Successfully generated {len(existing_files)} files out of {len(generated_files)} attempts")
        return existing_files
    
    def _write_files(self,
                    time_data: np.ndarray,
                    source_voltage: np.ndarray,
                    load_voltage: np.ndarray,
                    source_current: np.ndarray,
                    center_idx: int,
                    filepath_base: str) -> List[str]:
        """
        Write data files in all specified formats centered at the given index.
        
        Args:
            time_data: Time data array
            source_voltage: Source voltage array
            load_voltage: Load voltage array
            source_current: Source current array
            center_idx: Index to center the data at
            filepath_base: Base path for output files without extension
            
        Returns:
            List of paths to the generated files
        """
        try:
            # Calculate start and end indices
            half_points = self.total_points // 2
            start_idx = max(0, center_idx - half_points)
            end_idx = min(len(time_data), start_idx + self.total_points)
            
            # Adjust start index if end is clipped
            if end_idx < start_idx + self.total_points:
                start_idx = max(0, end_idx - self.total_points)
            
            logger.debug(f"Extracting data slice from {start_idx} to {end_idx} (center: {center_idx})")
            
            # Extract data slices
            time_slice = time_data[start_idx:end_idx]
            source_voltage_slice = source_voltage[start_idx:end_idx]
            load_voltage_slice = load_voltage[start_idx:end_idx]
            source_current_slice = source_current[start_idx:end_idx]
            
            # Ensure all slices have the same length
            min_length = min(len(time_slice), len(source_voltage_slice), 
                           len(load_voltage_slice), len(source_current_slice))
            
            if min_length < self.total_points:
                logger.warning(f"Data slice length {min_length} is less than requested {self.total_points}")
            
            time_slice = time_slice[:min_length]
            source_voltage_slice = source_voltage_slice[:min_length]
            load_voltage_slice = load_voltage_slice[:min_length]
            source_current_slice = source_current_slice[:min_length]
            
            logger.debug(f"Final data slice length: {min_length}")
            
            # Collect data into a common structure
            data = {
                'time': time_slice,
                'source_voltage': source_voltage_slice,
                'load_voltage': load_voltage_slice,
                'source_current': source_current_slice
            }
            
            generated_files = []
            
            # Write files in each requested format
            if 'csv' in self.output_formats:
                csv_path = f"{filepath_base}.csv"
                if self._write_csv(csv_path, data):
                    generated_files.append(csv_path)
            
            if 'npy' in self.output_formats:
                npy_path = f"{filepath_base}.npy"
                if self._write_npy(npy_path, data):
                    generated_files.append(npy_path)
            
            if 'npz' in self.output_formats:
                npz_path = f"{filepath_base}.npz"
                if self._write_npz(npz_path, data):
                    generated_files.append(npz_path)
            
            if 'mat' in self.output_formats:
                mat_path = f"{filepath_base}.mat"
                if self._write_mat(mat_path, data):
                    generated_files.append(mat_path)
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Error writing unified data files: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    # The following methods are identical to the original UnifiedDataWriter
    def _write_csv(self, filepath: str, data: Dict[str, np.ndarray]) -> bool:
        try:
            logger.debug(f"Writing CSV file: {filepath}")
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'source_voltage', 'load_voltage', 'source_current'])
                
                for i in range(len(data['time'])):
                    writer.writerow([
                        data['time'][i],
                        data['source_voltage'][i],
                        data['load_voltage'][i],
                        data['source_current'][i]
                    ])
            
            # Verify file was created successfully
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Wrote CSV file to {filepath} ({file_size} bytes)")
                return True
            else:
                logger.error(f"Failed to create CSV file {filepath}")
                return False
            
        except Exception as e:
            logger.error(f"Error writing CSV file {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _write_npy(self, filepath: str, data: Dict[str, np.ndarray]) -> bool:
        try:
            logger.debug(f"Writing NPY file: {filepath}")
            # Create a structured array
            dtype = [
                ('time', np.float64),
                ('source_voltage', np.float64),
                ('load_voltage', np.float64),
                ('source_current', np.float64)
            ]
            
            structured_array = np.zeros(len(data['time']), dtype=dtype)
            structured_array['time'] = data['time']
            structured_array['source_voltage'] = data['source_voltage']
            structured_array['load_voltage'] = data['load_voltage']
            structured_array['source_current'] = data['source_current']
            
            # Save to NPY file
            np.save(filepath, structured_array)
            
            # Verify file was created successfully
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Wrote NPY file to {filepath} ({file_size} bytes)")
                return True
            else:
                logger.error(f"Failed to create NPY file {filepath}")
                return False
            
        except Exception as e:
            logger.error(f"Error writing NPY file {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _write_npz(self, filepath: str, data: Dict[str, np.ndarray]) -> bool:
        try:
            logger.debug(f"Writing NPZ file: {filepath}")
            # Save to NPZ file
            np.savez(
                filepath,
                time=data['time'],
                source_voltage=data['source_voltage'],
                load_voltage=data['load_voltage'],
                source_current=data['source_current']
            )
            
            # Verify file was created successfully
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Wrote NPZ file to {filepath} ({file_size} bytes)")
                return True
            else:
                logger.error(f"Failed to create NPZ file {filepath}")
                return False
            
        except Exception as e:
            logger.error(f"Error writing NPZ file {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _write_mat(self, filepath: str, data: Dict[str, np.ndarray]) -> bool:
        try:
            logger.debug(f"Writing MAT file: {filepath}")
            # Save to MAT file
            sio.savemat(
                filepath,
                {
                    'time': data['time'],
                    'source_voltage': data['source_voltage'],
                    'load_voltage': data['load_voltage'],
                    'source_current': data['source_current']
                }
            )
            
            # Verify file was created successfully
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"Wrote MAT file to {filepath} ({file_size} bytes)")
                return True
            else:
                logger.error(f"Failed to create MAT file {filepath}")
                return False
            
        except Exception as e:
            logger.error(f"Error writing MAT file {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def write_fixed_unified_dataset(aligned_data: Dict[str, Any], 
                               raw_data: Dict[str, Any],
                               transient_result: Dict[str, Any],
                               output_dir: str,
                               exp_type: str = "unknown",
                               file_prefix: str = "",
                               total_points: int = 165888,
                               output_formats: Union[List[str], str] = None,
                               augmentation_shift: int = 65536) -> List[str]:
    """
    Convenience function to write a complete unified dataset with fixed left-shift ordering.
    
    Args:
        aligned_data: Aligned data from the processor (used to verify transient location)
        raw_data: Original data with full time series
        transient_result: Transient detection result
        output_dir: Base output directory
        exp_type: Experiment type for directory organization
        file_prefix: Prefix for output files
        total_points: Total points per file
        output_formats: Output formats ('csv', 'npy', 'npz', 'mat') as list or comma-separated string
        augmentation_shift: Number of points to shift for augmented data files (default: 65536)
        
    Returns:
        List of paths to the generated files
    """
    # Create writer
    writer = FixedUnifiedDataWriter(
        total_points=total_points,
        augmentation_shift=augmentation_shift,
        output_formats=output_formats
    )
    
    # Create output directory - note that this should already be the directory-specific output path
    output_dir = os.path.expanduser(output_dir)  # Expand ~ in path
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Writing fixed unified dataset to {output_dir}")
    logger.info(f"Output formats: {output_formats}")
    
    # Write unified files using the full raw data
    return writer.write_unified_data(
        raw_data,
        transient_result,
        output_dir,
        file_prefix=file_prefix,
        generate_augmented=True
    )