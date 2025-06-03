"""
Custom Center-Favoring Transient Detection for DC Arc Analysis

This module provides algorithms for detecting transients near the center of oscilloscope data,
specifically focusing on identifying potential arc events in DC power systems with custom parameters.

Based on the original transient_center.py but with exposed parameters for command-line configuration.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.transient_center_custom")

# Import the original CenterFavoringTransientDetector
from src.arc_detection.core.transient_center import CenterFavoringTransientDetector

# Function to create a center detector with custom parameters
def create_center_detector(window_size=1024, center_preference=0.7, diagnostic_mode=False):
    """
    Create a center-favoring transient detector with custom parameters.
    
    Args:
        window_size: Size of windows for processing (default: 1024)
        center_preference: Weight for center preference (0 to 1, default: 0.7)
        diagnostic_mode: Whether to store diagnostic information (default: False)
        
    Returns:
        CenterFavoringTransientDetector instance
    """
    logger.info(f"Creating custom center detector with window_size={window_size}, center_preference={center_preference}")
    return CenterFavoringTransientDetector(
        window_size=window_size,
        overlap=0.5,  # Default overlap
        sigma_threshold=3.0,  # Default sigma threshold
        center_preference=center_preference,
        diagnostic_mode=diagnostic_mode
    )

# Modified detection function that uses custom parameters
def detect_transients_with_custom_center_preference(data, window_size=1024, center_preference=0.7, diagnostic_mode=False):
    """
    Detect transients in data using a custom center-favoring approach.
    
    Args:
        data: Data dictionary from the MATLAB loader
        window_size: Size of windows for processing
        center_preference: Weight for center preference (0 to 1)
        diagnostic_mode: Whether to store diagnostic information
        
    Returns:
        Dictionary with transient detection results
    """
    # Create detector with custom parameters
    detector = create_center_detector(window_size, center_preference, diagnostic_mode)
    
    # Process each channel (similar to the original function)
    if not data or 'channels' not in data:
        logger.error("Invalid data structure for transient detection")
        return {'success': False, 'error': 'Invalid data structure'}
        
    # Track results for each channel
    channel_results = {}
    best_channel = None
    best_metric = -1
    best_result = None

    # Check specified detection channels
    for channel_name, channel_data in data['channels'].items():
        if channel_name in detector.detection_channels:
            logger.info(f"Processing channel: {channel_name}")
            result = detector.find_transient(channel_data)
            
            if result:
                # Calculate detection quality metric (peak height)
                quality = result.get('peak_height', 1.0)
                channel_results[channel_name] = {
                    'success': True,
                    'transient_index': result['transient_index'],
                    'quality': quality
                }
                
                # Update best result
                if quality > best_metric:
                    best_metric = quality
                    best_channel = channel_name
                    best_result = result
            else:
                channel_results[channel_name] = {
                    'success': False,
                    'error': 'No transient found'
                }
    
    # If no transients found in any channel, try other channels
    if best_result is None:
        for channel_name, channel_data in data['channels'].items():
            if channel_name not in channel_results:
                logger.info(f"Trying additional channel: {channel_name}")
                result = detector.find_transient(channel_data)
                
                if result:
                    quality = result.get('peak_height', 0.5)  # Lower default quality for non-primary channels
                    channel_results[channel_name] = {
                        'success': True,
                        'transient_index': result['transient_index'],
                        'quality': quality
                    }
                    
                    if quality > best_metric:
                        best_metric = quality
                        best_channel = channel_name
                        best_result = result
                else:
                    channel_results[channel_name] = {
                        'success': False,
                        'error': 'No transient found'
                    }
    
    # Return the best result
    if best_result:
        logger.info(f"Best transient found in channel {best_channel} at index {best_result['transient_index']}")
        
        return {
            'success': True,
            'best_channel': best_channel,
            'transient_index': best_result['transient_index'],
            'window': (best_result['window_start'], best_result['window_end']),
            'pre_transient': best_result.get('pre_transient', []),
            'transient': best_result.get('transient', []),
            'post_transient': best_result.get('post_transient', []),
            'channel_results': channel_results,
            'center_preference': center_preference
        }
    else:
        logger.warning(f"No transient found in any channel with center_preference={center_preference}")
        return {
            'success': False,
            'error': 'No transient found in any channel',
            'channel_results': channel_results,
            'center_preference': center_preference
        }