"""
Multi-File MATLAB Loader for Arc Detection Dataset

This module provides specialized functionality for loading MATLAB V5 files
organized as separate files per channel with 'time' and 'data' fields.

Author: Kristophor Jensen
"""

import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy.io import loadmat
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.loader")

class MatlabLoader:
    """
    Load oscilloscope data from multiple MATLAB V5 files.
    
    This class handles loading and validation of MATLAB files
    containing oscilloscope data for DC arc detection research.
    
    The expected file structure:
    - Multiple files with suffixes like "_ch1", "_ch2", etc.
    - Each file contains 'time' and 'data' fields
    
    Channel mapping:
    - Channel 1 (_ch1): load_voltage
    - Channel 2 (_ch2): source_voltage
    - Channel 4 (_ch4): source_current
    """
    
    def __init__(self, 
                 time_key: str = 'time', 
                 data_key: str = 'data', 
                 channel_mapping: Dict[str, str] = None):
        """
        Initialize the loader with configurable keys.
        
        Args:
            time_key: Key for time data in MATLAB file (default: 'time')
            data_key: Key for measurement data in MATLAB file (default: 'data')
            channel_mapping: Mapping of channel suffixes to logical names
        """
        self.time_key = time_key
        self.data_key = data_key
        
        # Default channel mapping based on documentation
        self.channel_mapping = channel_mapping or {
            'ch1': 'load_voltage',    # Channel 1
            'ch2': 'source_voltage',  # Channel 2
            'ch4': 'source_current'   # Channel 4
        }
        
        # Pattern for finding channel suffixes
        self.channel_pattern = re.compile(r'_ch(\d+)')
    
    def load_directory(self, directory_path: str, verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load all MATLAB files in a directory, matching by channel suffix.
        
        Args:
            directory_path: Path to directory containing MATLAB files
            verbose: Enable verbose debugging output
            
        Returns:
            Dictionary with loaded data or None if loading fails
        """
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                logger.error(f"Directory not found: {directory_path}")
                return None
                
            # Find all MATLAB files in directory
            matlab_files = list(directory_path.glob("*.mat"))
            if not matlab_files:
                logger.warning(f"No MATLAB files found in {directory_path}")
                return None
                
            logger.info(f"Found {len(matlab_files)} MATLAB files in {directory_path}")
            
            # Group files by base name (without channel suffix)
            file_groups = self._group_files_by_base_name(matlab_files)
            
            if verbose:
                logger.debug(f"Found {len(file_groups)} file groups")
                for base_name, files in file_groups.items():
                    logger.debug(f"  Base name: {base_name}, Files: {[f.name for f in files]}")
            
            # Process each group independently
            results = []
            for base_name, files in file_groups.items():
                group_result = self._process_file_group(base_name, files, verbose)
                if group_result:
                    results.append(group_result)
            
            if not results:
                logger.error(f"No valid data loaded from any file group in {directory_path}")
                return None
                
            # Combine results if multiple groups
            if len(results) == 1:
                return results[0]
            else:
                # Handle multiple file groups - use the first group as base
                combined = results[0].copy()
                combined['metadata']['file_groups'] = len(results)
                
                # Add metadata about other groups
                group_info = []
                for result in results:
                    group_info.append({
                        'base_name': result['metadata'].get('base_name', 'unknown'),
                        'channels': list(result['channels'].keys()),
                        'time_points': len(result['time_data']) if result['time_data'] is not None else 0
                    })
                combined['metadata']['group_info'] = group_info
                
                logger.info(f"Combined {len(results)} file groups")
                return combined
                
        except Exception as e:
            logger.error(f"Error loading directory {directory_path}: {str(e)}")
            traceback.print_exc()
            return None
    
    def _group_files_by_base_name(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by their base name (name without channel suffix)."""
        groups = {}
        
        for file_path in files:
            # Extract channel suffix
            match = self.channel_pattern.search(file_path.stem)
            if match:
                # Remove channel suffix to get base name
                base_name = file_path.stem.replace(match.group(0), '')
                
                if base_name not in groups:
                    groups[base_name] = []
                    
                groups[base_name].append(file_path)
            else:
                # File doesn't have a channel suffix, treat it as its own group
                groups[file_path.stem] = [file_path]
                
        return groups
    
    def _process_file_group(self, base_name: str, files: List[Path], verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Process a group of files with the same base name but different channel suffixes."""
        if not files:
            return None
            
        logger.info(f"Processing file group: {base_name} with {len(files)} files")
        
        # Initialize result structure
        result = {
            'time_data': None,
            'channels': {},
            'metadata': {
                'base_name': base_name,
                'directory': str(files[0].parent),
                'file_count': len(files)
            }
        }
        
        # Process each file in the group
        for file_path in files:
            # Extract channel information
            channel_info = self._extract_channel_info(file_path)
            if not channel_info:
                logger.warning(f"Could not extract channel info from {file_path}")
                continue
                
            channel_id, channel_name = channel_info
            
            # Load MATLAB file
            try:
                mat_data = loadmat(str(file_path), squeeze_me=True)
                
                # Check if required keys exist
                if self.time_key not in mat_data:
                    logger.warning(f"Time key '{self.time_key}' not found in {file_path}")
                    continue
                    
                if self.data_key not in mat_data:
                    logger.warning(f"Data key '{self.data_key}' not found in {file_path}")
                    continue
                
                # Extract time data if not already set
                if result['time_data'] is None:
                    time_data = self._extract_array(mat_data[self.time_key])
                    result['time_data'] = time_data
                    logger.info(f"Extracted time data with {len(time_data)} points from {file_path.name}")
                
                # Extract channel data
                channel_data = self._extract_array(mat_data[self.data_key])
                
                # Check if channel data length matches time data
                if result['time_data'] is not None and len(channel_data) != len(result['time_data']):
                    logger.warning(f"Channel data length ({len(channel_data)}) doesn't match time data length ({len(result['time_data'])}) in {file_path}")
                    # Adjust lengths - truncate to shorter length
                    min_length = min(len(channel_data), len(result['time_data']))
                    channel_data = channel_data[:min_length]
                    if channel_id == 'ch1':  # Only adjust time data once
                        result['time_data'] = result['time_data'][:min_length]
                        logger.warning(f"Adjusted time data length to {min_length}")
                
                # Add channel data to result
                result['channels'][channel_name] = channel_data
                logger.info(f"Added channel '{channel_name}' ({channel_id}) with {len(channel_data)} points")
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Check if any channels were loaded
        if not result['channels']:
            logger.error(f"No valid channels loaded from file group {base_name}")
            return None
            
        # Check if time data was loaded
        if result['time_data'] is None:
            logger.error(f"No time data loaded from file group {base_name}")
            return None
            
        # Add filenames to metadata
        result['metadata']['filenames'] = [f.name for f in files]
        
        logger.info(f"Successfully processed file group {base_name} with {len(result['channels'])} channels")
        return result
    
    def _extract_channel_info(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Extract channel ID and name from filename."""
        match = self.channel_pattern.search(file_path.stem)
        if match:
            channel_id = match.group(0)[1:]  # Remove the underscore
            
            # Map to logical channel name if available
            channel_name = self.channel_mapping.get(channel_id, channel_id)
            return channel_id, channel_name
            
        return None
    
    def _extract_array(self, matlab_var: Any) -> np.ndarray:
        """
        Extract a clean 1D numpy array from a MATLAB variable.
        
        Args:
            matlab_var: MATLAB variable from loadmat
            
        Returns:
            1D numpy array
        """
        # Handle empty case
        if matlab_var is None:
            return np.array([])
            
        # Handle scalar values
        if np.isscalar(matlab_var):
            return np.array([matlab_var])
        
        # Convert to numpy array if it's not already
        if not isinstance(matlab_var, np.ndarray):
            try:
                matlab_var = np.array(matlab_var)
            except:
                logger.error(f"Failed to convert variable to numpy array: {type(matlab_var)}")
                return np.array([])
                
        # Handle empty array
        if matlab_var.size == 0:
            return np.array([])
            
        # Handle n-dimensional arrays
        if matlab_var.ndim > 1:
            # For 2D, check if one dimension is singleton
            if matlab_var.ndim == 2:
                if matlab_var.shape[0] == 1:  # Row vector
                    return matlab_var.flatten()
                elif matlab_var.shape[1] == 1:  # Column vector
                    return matlab_var.flatten()
                else:
                    # Try to determine which axis contains the signal
                    if matlab_var.shape[0] > matlab_var.shape[1]:
                        # More rows than columns, likely columns are channels
                        logger.info(f"Found 2D array with shape {matlab_var.shape}, using first column as data")
                        return matlab_var[:, 0]
                    else:
                        # More columns than rows, likely rows are channels
                        logger.info(f"Found 2D array with shape {matlab_var.shape}, using first row as data")
                        return matlab_var[0, :]
            else:
                # Higher dimensional array - flatten and warn
                logger.warning(f"Found {matlab_var.ndim}D array with shape {matlab_var.shape}, flattening")
                return matlab_var.flatten()
                
        # Return the flattened array
        return matlab_var.flatten()

def discover_dataset_directories(base_path: str, depth: int = 1) -> List[str]:
    """
    Discover dataset directories within a base path.
    
    Args:
        base_path: Base directory to search
        depth: Directory depth to search (default: 1)
        
    Returns:
        List of discovered dataset directories
    """
    base_path = Path(base_path)
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return []
        
    if depth == 0:
        return [str(base_path)]
        
    discovered = []
    
    # Check if current directory has MATLAB files with channel pattern
    has_channel_files = False
    for mat_file in base_path.glob("*.mat"):
        if re.search(r'_ch\d+', mat_file.stem):
            has_channel_files = True
            break
            
    if has_channel_files:
        discovered.append(str(base_path))
        
    # Explore subdirectories
    if depth > 0:
        for item in base_path.iterdir():
            if item.is_dir():
                subdirs = discover_dataset_directories(item, depth - 1)
                discovered.extend(subdirs)
                
    return discovered
