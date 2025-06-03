#!/usr/bin/env python3
"""
Distance Metrics Module

This module provides various distance measures for comparing signal data.
Each function calculates a different type of distance between two arrays.

Available distance measures:
- L1 (Manhattan distance)
- L2 (Euclidean distance)
- Cosine distance
- Wave-Hedges distance
- Kumar-Hassebrook distance (also known as Tanimoto coefficient)
- Fidelity
- Additive Symmetric distance
- Kullback-Leibler divergence
- Jensen-Shannon divergence
- Kumar-Johnson divergence

Author: Kristophor Jensen
"""

import numpy as np
import logging
from typing import Union, Optional, Callable

# Configure logging
logger = logging.getLogger("arc_detection.metrics.distance")


def l1_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the L1 (Manhattan) distance between two arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        L1 distance value
    """
    # Ensure arrays are 1D
    x_flat = x.flatten() if hasattr(x, 'flatten') else x
    y_flat = y.flatten() if hasattr(y, 'flatten') else y
    
    # Calculate L1 distance and verify it's not zero for different arrays
    distance = np.sum(np.abs(x_flat - y_flat))
    
    # If distance is zero but arrays are not identical, use a small value
    if distance == 0 and not np.array_equal(x_flat, y_flat):
        # Calculate a small distance based on array values
        distance = 1e-8 * (np.sum(np.abs(x_flat)) + np.sum(np.abs(y_flat)))
        # If still zero, use a tiny constant
        if distance == 0:
            distance = 1e-10
    
    return distance


def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the L2 (Euclidean) distance between two arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        L2 distance value
    """
    return np.sqrt(np.sum((x - y) ** 2))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cosine distance between two arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        Cosine distance value (1 - cosine similarity)
    """
    # Ensure arrays are not zero vectors
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    
    if x_norm == 0 or y_norm == 0:
        return 1.0  # Maximum distance for zero vectors
        
    cosine_sim = np.dot(x, y) / (x_norm * y_norm)
    
    # Clamp to [-1, 1] to handle floating point issues
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    # Cosine distance is 1 - cosine similarity
    return 1.0 - cosine_sim


def wave_hedges_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Wave-Hedges distance between two arrays.
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Wave-Hedges distance value
    """
    # Ensure arrays have no negative values
    x = np.abs(x)
    y = np.abs(y)
    
    # Add epsilon to avoid division by zero
    x = np.maximum(x, epsilon)
    y = np.maximum(y, epsilon)
    
    # Calculate distance
    numerator = np.abs(x - y)
    denominator = np.maximum(x, y)
    return np.sum(numerator / denominator)


def kumar_hassebrook_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Kumar-Hassebrook distance (Tanimoto coefficient).
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Kumar-Hassebrook distance value
    """
    # Ensure arrays have no negative values
    x = np.abs(x)
    y = np.abs(y)
    
    dot_product = np.dot(x, y)
    x_squared_sum = np.sum(x**2)
    y_squared_sum = np.sum(y**2)
    
    # Add epsilon to avoid division by zero
    denominator = x_squared_sum + y_squared_sum - dot_product + epsilon
    
    # Return dissimilarity (1 - similarity)
    return 1.0 - (dot_product / denominator)


def fidelity_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Fidelity distance between two arrays.
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Fidelity distance value
    """
    # Convert to probability distributions
    x = np.abs(x) + epsilon
    y = np.abs(y) + epsilon
    
    x = x / np.sum(x)
    y = y / np.sum(y)
    
    # Calculate fidelity
    fidelity = np.sum(np.sqrt(x * y))
    
    # Return dissimilarity (1 - fidelity)
    return 1.0 - fidelity


def additive_symmetric_distance(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Additive Symmetric divergence between two arrays.
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Additive Symmetric distance value
    """
    # Convert to probability distributions
    x = np.abs(x) + epsilon
    y = np.abs(y) + epsilon
    
    x = x / np.sum(x)
    y = y / np.sum(y)
    
    # Calculate ratio arrays (avoid division by zero)
    ratio_xy = x / y
    ratio_yx = y / x
    
    # Calculate distance
    return np.sum(x * (ratio_xy - 1.0)**2 + y * (ratio_yx - 1.0)**2)


def kullback_leibler_divergence(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Kullback-Leibler divergence between two arrays.
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Kullback-Leibler divergence value
    """
    # Convert to probability distributions
    x = np.abs(x) + epsilon
    y = np.abs(y) + epsilon
    
    x = x / np.sum(x)
    y = y / np.sum(y)
    
    # Calculate KL divergence
    return np.sum(x * np.log(x / y))


def jensen_shannon_divergence(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Jensen-Shannon divergence between two arrays.
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Jensen-Shannon divergence value
    """
    # Convert to probability distributions
    x = np.abs(x) + epsilon
    y = np.abs(y) + epsilon
    
    x = x / np.sum(x)
    y = y / np.sum(y)
    
    # Calculate the midpoint distribution
    m = 0.5 * (x + y)
    
    # Calculate JS divergence
    js_div = 0.5 * kullback_leibler_divergence(x, m) + 0.5 * kullback_leibler_divergence(y, m)
    
    return js_div


def kumar_johnson_divergence(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate the Kumar-Johnson divergence between two arrays.
    
    Args:
        x: First array
        y: Second array
        epsilon: Small value to avoid division by zero
        
    Returns:
        Kumar-Johnson divergence value
    """
    # Convert to probability distributions
    x = np.abs(x) + epsilon
    y = np.abs(y) + epsilon
    
    x = x / np.sum(x)
    y = y / np.sum(y)
    
    # Calculate squared differences of ratios
    ratio_squared = ((x**2 - y**2) / (x * y))**2
    
    return np.sum(ratio_squared)


def get_distance_function(measure: str) -> Callable:
    """
    Get the distance function based on the measure name.
    
    Args:
        measure: Name of the distance measure
        
    Returns:
        Distance function
        
    Raises:
        ValueError: If the measure is not supported
    """
    measure_map = {
        "L1": l1_distance,
        "L2": l2_distance,
        "cosine": cosine_distance,
        "wave-hedges": wave_hedges_distance,
        "kumar-hassebrook": kumar_hassebrook_distance,
        "fidelity": fidelity_distance,
        "additive-symmetric": additive_symmetric_distance,
        "kullback-leibler": kullback_leibler_divergence,
        "jensen-shannon": jensen_shannon_divergence,
        "kumar-johnson": kumar_johnson_divergence
    }
    
    if measure not in measure_map:
        raise ValueError(f"Unsupported distance measure: {measure}")
    
    return measure_map[measure]


def calculate_distance(x: np.ndarray, y: np.ndarray, measure: str, 
                      custom_measure: Optional[Callable] = None) -> float:
    """
    Calculate distance between two arrays using the specified measure.
    
    Args:
        x: First array
        y: Second array
        measure: Name of the distance measure
        custom_measure: Optional custom distance function
        
    Returns:
        Distance value
        
    Raises:
        ValueError: If the measure is not supported and no custom measure is provided
    """
    # Use custom measure if provided
    if custom_measure is not None:
        return custom_measure(x, y)
    
    # Handle structured arrays
    if hasattr(x.dtype, 'names') and x.dtype.names is not None:
        # For structured arrays, extract fields for calculation
        # This should not normally happen as field extraction should be done before calling this function
        logger.warning("Received structured array when a field array was expected. Attempting to use first field.")
        try:
            # Try to use the first field
            field_name = x.dtype.names[0]
            x_values = x[field_name]
            y_values = y[field_name] if hasattr(y.dtype, 'names') and y.dtype.names is not None else y
        except Exception as e:
            raise ValueError(f"Cannot extract field from structured array: {e}")
    else:
        x_values = x
        y_values = y
    
    # Ensure arrays have matching dimensions by flattening if needed
    x_values = x_values.flatten() if hasattr(x_values, 'flatten') else x_values
    y_values = y_values.flatten() if hasattr(y_values, 'flatten') else y_values
    
    # Ensure arrays have the same length
    min_length = min(len(x_values), len(y_values))
    if len(x_values) != len(y_values):
        logger.warning(f"Arrays have different lengths: {len(x_values)} vs {len(y_values)}. Truncating to {min_length}.")
        x_values = x_values[:min_length]
        y_values = y_values[:min_length]
    
    # Get the standard distance function
    distance_func = get_distance_function(measure)
    
    # Calculate and return the distance
    return distance_func(x_values, y_values)