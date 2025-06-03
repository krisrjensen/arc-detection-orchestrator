"""
Transient Detection Engine for DC Arc Analysis

This module provides algorithms for detecting transients in oscilloscope data,
specifically focusing on identifying potential arc events in DC power systems.

Based on the PhD research of Kristophor Jensen at South Dakota School of Mines and Technology.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.transient")

class TransientDetector:
    """
    Detects transients in oscilloscope data using multi-strategy approaches.
    
    This class implements various algorithms for detecting transients in time series data,
    with a focus on identifying arc events in DC electrical systems.
    """
    
    def __init__(self, 
                 window_size: int = 1024, 
                 overlap: float = 0.5,
                 sigma_threshold: float = 3.0,
                 detection_channels: List[str] = None,
                 diagnostic_mode: bool = False):
        """
        Initialize the transient detector with configurable parameters.
        
        Args:
            window_size: Size of windows for processing (default: 1024)
            overlap: Overlap fraction between windows (default: 0.5)
            sigma_threshold: Threshold for outlier detection in standard deviations (default: 3.0)
            detection_channels: List of channel names to use for detection (default: None = all channels)
            diagnostic_mode: Whether to store diagnostic information (default: False)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sigma_threshold = sigma_threshold
        self.detection_channels = detection_channels or ['source_current', 'load_voltage']
        self.diagnostic_mode = diagnostic_mode
        self.diagnostic_data = {}
        
        logger.info(f"Initialized TransientDetector with window_size={window_size}, overlap={overlap}")
    
    def find_transient(self, signal: np.ndarray) -> Optional[Dict[str, Union[int, List[int]]]]:
        """
        Find a single transient in a signal using windowed analysis.
        
        This method implements the core algorithm for detecting a transient in a signal
        using statistical analysis of windowed data.
        
        Args:
            signal: The signal to analyze
            
        Returns:
            Dictionary with transient indices or None if no transient found
        """
        try:
            if signal is None or len(signal) == 0:
                logger.warning("Empty signal provided for transient detection")
                return None
                
            signal_length = len(signal)
            logger.info(f"Analyzing signal with {signal_length} points for transient detection")
            
            # Clear previous diagnostic data if in diagnostic mode
            if self.diagnostic_mode:
                self.diagnostic_data = {
                    'signal': signal.copy(),
                    'windows': [],
                    'window_indices': [],
                    'features': {
                        'mean': [],
                        'var': [],
                        'mean_diff': [],
                        'transient_indicator': []
                    }
                }
            
            # Remove outliers - important for robust detection
            clean_signal = self._remove_outliers(signal, self.sigma_threshold)
            
            # Create windows with overlap
            stride = int(self.window_size * (1 - self.overlap))
            num_windows = 1 + (signal_length - self.window_size) // stride
            
            if num_windows <= 0:
                logger.warning(f"Signal too short ({signal_length} points) for window size {self.window_size}")
                return None
                
            windows = []
            window_indices = []
            
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = min(start_idx + self.window_size, signal_length)
                
                if end_idx - start_idx < self.window_size // 2:
                    # Skip windows that are too small (less than half the desired size)
                    continue
                    
                window = clean_signal[start_idx:end_idx]
                windows.append(window)
                window_indices.append((start_idx, end_idx))
            
            if not windows:
                logger.warning("No valid windows could be created from the signal")
                return None
                
            # Extract features from windows
            means = np.array([np.mean(w) for w in windows])
            variances = np.array([np.var(w) for w in windows])
            
            # Calculate mean differences (first derivative) - zero-pad for alignment
            mean_diffs = np.abs(np.diff(means, prepend=means[0]))
            
            # Combine features into a transient indicator metric
            # Enhanced formula: mean_diff * variance * abs_mean_change
            # This gives more weight to windows with both variability AND changing mean
            abs_mean_changes = np.abs(means - np.median(means))
            transient_indicators = mean_diffs * variances * abs_mean_changes
            
            # Store diagnostic data if in diagnostic mode
            if self.diagnostic_mode:
                self.diagnostic_data['windows'] = windows
                self.diagnostic_data['window_indices'] = window_indices
                self.diagnostic_data['features']['mean'] = means
                self.diagnostic_data['features']['var'] = variances
                self.diagnostic_data['features']['mean_diff'] = mean_diffs
                self.diagnostic_data['features']['transient_indicator'] = transient_indicators
            
            # Find window with maximum transient indicator
            max_transient_idx = np.argmax(transient_indicators)
            logger.info(f"Detected transient at window index: {max_transient_idx}")
            
            # Get window boundaries
            if max_transient_idx < len(window_indices):
                start_idx, end_idx = window_indices[max_transient_idx]
                
                # Define pre-transient, transient, and post-transient regions
                transient_center = (start_idx + end_idx) // 2
                
                # Create result dictionary with indices
                result = {
                    'transient_index': transient_center,
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'pre_transient': list(range(max(0, start_idx - self.window_size), start_idx)),
                    'transient': list(range(start_idx, end_idx)),
                    'post_transient': list(range(end_idx, min(signal_length, end_idx + self.window_size)))
                }
                
                return result
            else:
                logger.warning("Invalid transient window index detected")
                return None
                
        except Exception as e:
            logger.error(f"Error during transient detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_transients_multi_channel(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Find transients using multiple channels for more robust detection.
        
        This method applies transient detection to multiple channels and combines the results
        to identify the most likely transient location.
        
        Args:
            data: Dictionary of channel data
            
        Returns:
            Dictionary with combined transient detection results
        """
        if not data:
            logger.warning("No data provided for multi-channel transient detection")
            return {'success': False, 'error': 'No data provided'}
        
        logger.info(f"Performing multi-channel transient detection")
        
        # Track results for each channel
        channel_results = {}
        best_channel = None
        best_metric = -1
        best_result = None
        
        # Analyze each detection channel
        for channel_name in self.detection_channels:
            if channel_name in data:
                logger.info(f"Analyzing channel {channel_name}")
                
                # Get channel data
                channel_data = data[channel_name]
                
                # Detect transient
                result = self.find_transient(channel_data)
                
                if result:
                    # Extract transient indicator metric
                    if self.diagnostic_mode and 'features' in self.diagnostic_data and 'transient_indicator' in self.diagnostic_data['features']:
                        transient_indicators = self.diagnostic_data['features']['transient_indicator']
                        if len(transient_indicators) > 0:
                            max_indicator = np.max(transient_indicators)
                            
                            # Store result with metric
                            channel_results[channel_name] = {
                                'result': result,
                                'metric': float(max_indicator)
                            }
                            
                            # Update best result if this is better
                            if max_indicator > best_metric:
                                best_metric = max_indicator
                                best_channel = channel_name
                                best_result = result
                    else:
                        # No diagnostic data available, just store the result
                        channel_results[channel_name] = {
                            'result': result,
                            'metric': 1.0  # Default metric
                        }
                        
                        # Update best result if no best result yet
                        if best_result is None:
                            best_channel = channel_name
                            best_result = result
                            best_metric = 1.0
                else:
                    logger.warning(f"No transient found in channel {channel_name}")
            else:
                logger.warning(f"Channel {channel_name} not found in data")
        
        # Compile final result
        if best_result:
            logger.info(f"Best transient detection from channel: {best_channel} with metric: {best_metric:.6f}")
            
            return {
                'success': True,
                'best_channel': best_channel,
                'transient_index': best_result['transient_index'],
                'window': (best_result['window_start'], best_result['window_end']),
                'pre_transient': best_result['pre_transient'],
                'transient': best_result['transient'],
                'post_transient': best_result['post_transient'],
                'channel_results': channel_results
            }
        else:
            logger.warning("No transient found in any channel")
            return {
                'success': False,
                'error': 'No transient found in any channel',
                'channel_results': channel_results
            }
    
    def _remove_outliers(self, signal: np.ndarray, sigma_threshold: float = 3.0) -> np.ndarray:
        """
        Remove outliers from signal for more robust transient detection.
        
        Args:
            signal: Input signal array
            sigma_threshold: Threshold in standard deviations to consider a point an outlier
            
        Returns:
            Signal with outliers removed
        """
        if signal is None or len(signal) <= 4:
            return signal  # Not enough points to detect outliers
            
        # Create a copy to avoid modifying the original
        cleaned_signal = signal.copy()
        
        # Calculate global statistics
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        if signal_std == 0:
            return signal  # No variation in the signal
            
        # Find potential outliers using global statistics
        outlier_mask = np.abs(signal - signal_mean) > sigma_threshold * signal_std
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) == 0:
            return signal  # No outliers found
            
        # Process each outlier
        for idx in outlier_indices:
            # Define a local window around the outlier
            window_start = max(0, idx - 10)
            window_end = min(len(signal), idx + 11)
            
            # Get points excluding the outlier
            window_points = np.concatenate([
                signal[window_start:idx],
                signal[idx+1:window_end]
            ])
            
            if len(window_points) == 0:
                continue  # Skip if no points in window
                
            # Calculate local statistics
            local_mean = np.mean(window_points)
            local_std = np.std(window_points)
            
            # Only replace if it's also an outlier by local statistics
            if local_std > 0 and abs(signal[idx] - local_mean) > sigma_threshold * local_std:
                # Replace with local mean
                cleaned_signal[idx] = local_mean
        
        return cleaned_signal
    
    def plot_detection_results(self, signal: np.ndarray, result: Dict[str, Any], 
                              title: str = None, save_path: str = None):
        """
        Plot the signal with detected transient regions highlighted.
        
        Args:
            signal: The signal data
            result: The detection result from find_transient
            title: Optional title for the plot
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure
        """
        if signal is None or result is None:
            logger.warning("Cannot create plot: signal or result is None")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create time axis (sample indices)
        x = np.arange(len(signal))
        
        # Plot the entire signal
        ax.plot(x, signal, 'k-', alpha=0.5, label='Signal')
        
        # Highlight the regions
        if 'pre_transient' in result and result['pre_transient']:
            pre_indices = result['pre_transient']
            ax.plot(pre_indices, signal[pre_indices], 'g-', linewidth=2, label='Pre-transient')
            
        if 'transient' in result and result['transient']:
            transient_indices = result['transient']
            ax.plot(transient_indices, signal[transient_indices], 'r-', linewidth=2, label='Transient')
            
        if 'post_transient' in result and result['post_transient']:
            post_indices = result['post_transient']
            ax.plot(post_indices, signal[post_indices], 'b-', linewidth=2, label='Post-transient')
        
        # Add transient marker
        if 'transient_index' in result:
            transient_index = result['transient_index']
            ax.axvline(x=transient_index, color='r', linestyle='--', alpha=0.7, label='Transient Center')
        
        # Add labels and legend
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Transient Detection Results')
        ax.legend()
        ax.grid(True)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved detection plot to {save_path}")
        
        return fig
    
    def plot_diagnostic_data(self, title: str = None, save_path: str = None):
        """
        Plot diagnostic data from the last transient detection.
        
        This method visualizes the features used for transient detection, providing insight
        into how the detection algorithm identified the transient.
        
        Args:
            title: Optional title for the plot
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure
        """
        if not self.diagnostic_mode or not self.diagnostic_data:
            logger.warning("No diagnostic data available. Run detection with diagnostic_mode=True first.")
            return None
        
        # Extract diagnostic data
        signal = self.diagnostic_data.get('signal')
        window_indices = self.diagnostic_data.get('window_indices')
        features = self.diagnostic_data.get('features')
        
        if not signal is not None and window_indices and features:
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Create x-axis for windows (center points)
            window_centers = [(start + end) // 2 for start, end in window_indices]
            
            # Plot signal with window centers
            axs[0].plot(signal, 'k-', alpha=0.7)
            axs[0].scatter(window_centers, [signal[i] for i in window_centers], 
                          c='r', alpha=0.5, s=30)
            axs[0].set_ylabel('Signal')
            axs[0].set_title('Signal with Window Centers' if not title else title)
            axs[0].grid(True)
            
            # Plot mean and variance
            ax1 = axs[1]
            ln1 = ax1.plot(window_centers, features['mean'], 'b-', label='Mean')
            ax1.set_ylabel('Mean', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True)
            
            ax1_twin = ax1.twinx()
            ln2 = ax1_twin.plot(window_centers, features['var'], 'r-', label='Variance')
            ax1_twin.set_ylabel('Variance', color='r')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
            # Add combined legend
            lns = ln1 + ln2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper right')
            
            # Plot transient indicator
            axs[2].plot(window_centers, features['transient_indicator'], 'g-')
            axs[2].set_ylabel('Transient Indicator')
            axs[2].set_xlabel('Sample')
            axs[2].grid(True)
            
            # Add marker at max transient indicator
            if len(features['transient_indicator']) > 0:
                max_idx = np.argmax(features['transient_indicator'])
                if max_idx < len(window_centers):
                    max_center = window_centers[max_idx]
                    max_value = features['transient_indicator'][max_idx]
                    
                    axs[2].scatter([max_center], [max_value], c='r', s=100, marker='*')
                    axs[2].annotate('Max', (max_center, max_value),
                                  xytext=(10, 10), textcoords='offset points',
                                  arrowprops=dict(arrowstyle='->'))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved diagnostic plot to {save_path}")
            
            return fig
        else:
            logger.warning("Incomplete diagnostic data")
            return None


# Function to detect transients in loaded data
def detect_transients_in_data(data, window_size=1024, overlap=0.5, diagnostic_mode=False):
    """
    Detect transients in loaded data from the MATLAB loader.
    
    Args:
        data: Data dictionary from the MATLAB loader
        window_size: Size of windows for transient detection
        overlap: Overlap between windows
        diagnostic_mode: Whether to enable diagnostic mode
        
    Returns:
        Dictionary with detection results
    """
    if not data or 'channels' not in data or not data['channels']:
        logger.error("Invalid data structure for transient detection")
        return {'success': False, 'error': 'Invalid data structure'}
    
    # Initialize detector
    detector = TransientDetector(
        window_size=window_size,
        overlap=overlap,
        diagnostic_mode=diagnostic_mode
    )
    
    # Perform multi-channel detection
    result = detector.find_transients_multi_channel(data['channels'])
    
    # Add time information if available
    if result['success'] and 'transient_index' in result and 'time_data' in data and data['time_data'] is not None:
        time_data = data['time_data']
        transient_idx = result['transient_index']
        
        if 0 <= transient_idx < len(time_data):
            result['transient_time'] = float(time_data[transient_idx])
            logger.info(f"Transient detected at time: {result['transient_time']:.6f}s")
    
    return result