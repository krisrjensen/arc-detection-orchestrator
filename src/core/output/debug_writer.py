"""
Debug version of the unified data writer with additional logging for augmented data.
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.output.debug_writer")

def debug_augmented_data(aligned_data, output_dir):
    """
    Analyze aligned data and check if it meets criteria for augmented data generation.
    
    Args:
        aligned_data: Aligned data from the processor
        output_dir: Output directory (for logging only)
    """
    logger.info("=== DEBUGGING AUGMENTED DATA GENERATION ===")
    
    # Extract time data and get its length
    if 'time_data' not in aligned_data:
        logger.error("No time_data found in aligned_data")
        return
    
    time_data = aligned_data['time_data']
    time_data_length = len(time_data)
    logger.info(f"Time data length: {time_data_length}")
    
    # Get transient index
    if 'alignment' not in aligned_data or 'aligned_transient_index' not in aligned_data['alignment']:
        logger.error("Missing alignment or aligned_transient_index")
        return
    
    transient_idx = aligned_data['alignment']['aligned_transient_index']
    logger.info(f"Transient index: {transient_idx}")
    
    # Calculate potential for augmented data
    total_points = 165888  # Default value from UnifiedDataWriter
    augmentation_shift = 16384  # Default value from UnifiedDataWriter
    
    logger.info(f"Total points per file: {total_points}")
    logger.info(f"Augmentation shift: {augmentation_shift}")
    
    # Check if data is long enough for augmented data
    if time_data_length < total_points:
        logger.error(f"Time data length ({time_data_length}) is less than total points ({total_points})")
        logger.error("Cannot generate augmented data - insufficient data length")
        return
    
    # Calculate potential left-shifted files
    potential_left_shifts = int((time_data_length - total_points) / augmentation_shift)
    logger.info(f"Potential left shifts: {potential_left_shifts}")
    
    left_files = []
    for i in range(1, potential_left_shifts + 1):
        shift = i * augmentation_shift
        if transient_idx - shift >= total_points // 2:
            left_files.append(f"L{i:03d}")
    
    # Calculate potential right-shifted files
    potential_right_shifts = int((time_data_length - total_points) / augmentation_shift)
    logger.info(f"Potential right shifts: {potential_right_shifts}")
    
    right_files = []
    for i in range(1, potential_right_shifts + 1):
        shift = i * augmentation_shift
        if transient_idx + shift < time_data_length - total_points // 2:
            right_files.append(f"R{i:03d}")
    
    logger.info(f"Potential left-shifted files: {len(left_files)} {left_files}")
    logger.info(f"Potential right-shifted files: {len(right_files)} {right_files}")
    
    if not left_files and not right_files:
        logger.error("No augmented data can be generated with current parameters")
        logger.error("Possible causes:")
        logger.error(f"1. Transient index {transient_idx} too close to edge of data")
        logger.error(f"2. Data not long enough for multiple shifts with shift size {augmentation_shift}")
        logger.error(f"3. Half-window size {total_points//2} too large relative to data length")
    else:
        logger.info(f"Expected total augmented files: {len(left_files) + len(right_files)}")
    
    # Summary of results
    logger.info("=== SUMMARY ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data length: {time_data_length}")
    logger.info(f"Minimum required length: {total_points}")
    logger.info(f"Transient index: {transient_idx}")
    logger.info(f"Potential augmented files: {len(left_files) + len(right_files)}")
    logger.info("=== END DEBUGGING ===")
