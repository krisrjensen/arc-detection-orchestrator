"""
Preprocessing and Feature Extraction Module for DC Arc Detection

This module provides standardized preprocessing functions, feature extraction,
and dimensionality reduction techniques for DC arc detection data.

Based on the PhD research of Kristophor Jensen at South Dakota School of Mines and Technology.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.signal import stft, spectrogram
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.utils.preprocessing")

class SignalPreprocessor:
    """
    Preprocess signals for feature extraction and analysis.
    
    This class provides methods for normalizing, filtering, and
    transforming signal data for DC arc detection.
    """
    
    def __init__(self):
        """Initialize the signal preprocessor."""
        logger.info("Initialized SignalPreprocessor")
    
    def normalize_signal(self, signal: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize a signal using specified method.
        
        Args:
            signal: Input signal array
            method: Normalization method ('standard', 'minmax', 'mean', 'max')
            
        Returns:
            Normalized signal
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for normalization")
            return np.array([])
        
        # Handle NaN or inf values
        signal = np.nan_to_num(signal)
        
        if method == 'standard':
            # Zero mean, unit variance
            scaler = StandardScaler()
            # StandardScaler expects 2D input
            normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        elif method == 'minmax':
            # Scale to [0, 1] range
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        elif method == 'mean':
            # Simple mean normalization
            mean = np.mean(signal)
            normalized = signal - mean
        elif method == 'max':
            # Scale by maximum absolute value
            max_abs = np.max(np.abs(signal))
            if max_abs > 0:
                normalized = signal / max_abs
            else:
                normalized = signal
        else:
            logger.warning(f"Unknown normalization method: {method}, using 'standard'")
            scaler = StandardScaler()
            normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        
        return normalized
    
    def remove_outliers(self, signal: np.ndarray, sigma_threshold: float = 3.0) -> np.ndarray:
        """
        Remove outliers from signal using statistical thresholding.
        
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
    
    def compute_spectrogram(self, 
                           signal: np.ndarray, 
                           fs: float = 5.0e6,  # 5 MHz sampling rate
                           nperseg: int = 256, 
                           noverlap: int = 128,
                           window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram for time-frequency analysis.
        
        Args:
            signal: Input signal array
            fs: Sampling frequency in Hz
            nperseg: Number of points per segment
            noverlap: Number of points overlap between segments
            window: Window function to use
            
        Returns:
            Tuple of (frequencies, time_points, spectrogram_values)
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for spectrogram")
            return np.array([]), np.array([]), np.array([[]])
        
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
        
        return f, t, Sxx
    
    def compute_stft(self, 
                    signal: np.ndarray, 
                    fs: float = 5.0e6,  # 5 MHz sampling rate
                    nperseg: int = 256, 
                    noverlap: int = 128,
                    window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform for time-frequency analysis.
        
        Args:
            signal: Input signal array
            fs: Sampling frequency in Hz
            nperseg: Number of points per segment
            noverlap: Number of points overlap between segments
            window: Window function to use
            
        Returns:
            Tuple of (frequencies, time_points, stft_values)
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for STFT")
            return np.array([]), np.array([]), np.array([[]])
        
        # Compute STFT
        f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
        
        return f, t, Zxx
    
    def plot_spectrogram(self, 
                        f: np.ndarray, 
                        t: np.ndarray, 
                        Sxx: np.ndarray,
                        title: str = 'Spectrogram',
                        log_scale: bool = True,
                        db_range: float = 60.0,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a spectrogram from precomputed data.
        
        Args:
            f: Frequency array
            t: Time array
            Sxx: Spectrogram values
            title: Plot title
            log_scale: Whether to use log scale for power
            db_range: Dynamic range in dB for log scale
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if log_scale:
            # Convert to dB with limited dynamic range
            Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
            vmax = np.max(Sxx_db)
            vmin = vmax - db_range
            pcm = ax.pcolormesh(t, f, Sxx_db, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(pcm, ax=ax, label='Power/Frequency (dB/Hz)')
        else:
            pcm = ax.pcolormesh(t, f, Sxx, cmap='viridis')
            plt.colorbar(pcm, ax=ax, label='Power/Frequency (Hz)')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        ax.set_yscale('log')
        
        # Add grid
        ax.grid(which='major', linewidth=1.5, linestyle='-', alpha=0.5)
        ax.grid(which='minor', linewidth=0.5, linestyle='--', alpha=0.2)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved spectrogram to {save_path}")
        
        return fig


class FeatureExtractor:
    """
    Extract features from signal data for DC arc detection.
    
    This class implements various feature extraction techniques
    for time-domain and frequency-domain analysis.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        logger.info("Initialized FeatureExtractor")
        self.preprocessor = SignalPreprocessor()
    
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain statistical features from a signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Dictionary of extracted features
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for feature extraction")
            return {}
        
        # Basic statistical features
        features = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'range': np.max(signal) - np.min(signal),
            'median': np.median(signal),
            'skewness': self._skewness(signal),
            'kurtosis': self._kurtosis(signal),
            'rms': np.sqrt(np.mean(np.square(signal))),
            'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(np.square(signal))) if np.mean(np.square(signal)) > 0 else 0,
            'energy': np.sum(np.square(signal)),
            'mean_abs_deviation': np.mean(np.abs(signal - np.mean(signal))),
            'median_abs_deviation': np.median(np.abs(signal - np.median(signal))),
        }
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        
        # First-order differences
        diff1 = np.diff(signal)
        if len(diff1) > 0:
            features['diff_mean'] = np.mean(diff1)
            features['diff_std'] = np.std(diff1)
            features['diff_max'] = np.max(np.abs(diff1))
            features['diff_energy'] = np.sum(np.square(diff1))
        
        # Second-order differences
        diff2 = np.diff(signal, n=2)
        if len(diff2) > 0:
            features['diff2_mean'] = np.mean(diff2)
            features['diff2_std'] = np.std(diff2)
            features['diff2_max'] = np.max(np.abs(diff2))
            features['diff2_energy'] = np.sum(np.square(diff2))
        
        return features
    
    def extract_frequency_domain_features(self, 
                                         signal: np.ndarray, 
                                         fs: float = 5.0e6) -> Dict[str, float]:
        """
        Extract frequency-domain features from a signal.
        
        Args:
            signal: Input signal array
            fs: Sampling frequency in Hz
            
        Returns:
            Dictionary of extracted features
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for feature extraction")
            return {}
        
        # Compute FFT
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_mags = np.abs(fft_vals)
        fft_phase = np.angle(fft_vals)
        fft_power = fft_mags**2
        
        # Avoid division by zero
        total_power = np.sum(fft_power)
        if total_power == 0:
            total_power = 1.0
        
        # Basic spectral features
        features = {
            'spectral_mean': np.mean(fft_mags),
            'spectral_std': np.std(fft_mags),
            'spectral_skewness': self._skewness(fft_mags),
            'spectral_kurtosis': self._kurtosis(fft_mags),
            'spectral_max': np.max(fft_mags),
            'spectral_energy': np.sum(fft_power),
        }
        
        # Peak frequency and magnitude
        peak_idx = np.argmax(fft_mags)
        if 0 <= peak_idx < len(fft_freqs):
            features['peak_frequency'] = fft_freqs[peak_idx]
            features['peak_magnitude'] = fft_mags[peak_idx]
        
        # Spectral centroid (weighted average of frequencies)
        if len(fft_freqs) > 0 and len(fft_mags) > 0:
            features['spectral_centroid'] = np.sum(fft_freqs * fft_mags) / np.sum(fft_mags) if np.sum(fft_mags) > 0 else 0
        
        # Spectral bandwidth
        if 'spectral_centroid' in features:
            centroid = features['spectral_centroid']
            bandwidth = np.sqrt(np.sum(((fft_freqs - centroid)**2) * fft_mags) / np.sum(fft_mags)) if np.sum(fft_mags) > 0 else 0
            features['spectral_bandwidth'] = bandwidth
        
        # Spectral flatness (ratio of geometric mean to arithmetic mean)
        if np.all(fft_mags > 0) and len(fft_mags) > 0:
            geo_mean = np.exp(np.mean(np.log(fft_mags)))
            arith_mean = np.mean(fft_mags)
            features['spectral_flatness'] = geo_mean / arith_mean if arith_mean > 0 else 0
        
        # Power in frequency bands
        if len(fft_freqs) > 0 and len(fft_power) > 0:
            # Define frequency bands (adjust based on your specific needs)
            bands = [
                (0, 1e3),       # 0-1 kHz
                (1e3, 10e3),    # 1-10 kHz
                (10e3, 100e3),  # 10-100 kHz
                (100e3, 500e3), # 100-500 kHz
                (500e3, 1e6),   # 500 kHz - 1 MHz
                (1e6, 2.5e6)    # 1-2.5 MHz
            ]
            
            for i, (low, high) in enumerate(bands):
                # Find indices in the frequency array
                indices = np.where((fft_freqs >= low) & (fft_freqs < high))[0]
                if len(indices) > 0:
                    band_power = np.sum(fft_power[indices])
                    features[f'band_{i}_power'] = band_power
                    features[f'band_{i}_ratio'] = band_power / total_power
        
        # Pink noise characteristics (1/f slope in log-log scale)
        if len(fft_freqs) > 1 and len(fft_power) > 1:
            # Skip DC component
            valid_indices = np.where((fft_freqs > 0) & (fft_power > 0))[0]
            if len(valid_indices) > 1:
                log_freqs = np.log(fft_freqs[valid_indices])
                log_power = np.log(fft_power[valid_indices])
                
                # Linear regression to find slope
                try:
                    A = np.vstack([log_freqs, np.ones(len(log_freqs))]).T
                    slope, intercept = np.linalg.lstsq(A, log_power, rcond=None)[0]
                    features['pink_noise_slope'] = slope
                    features['pink_noise_intercept'] = intercept
                    
                    # Calculate R²
                    y_mean = np.mean(log_power)
                    ss_tot = np.sum((log_power - y_mean)**2)
                    y_pred = slope * log_freqs + intercept
                    ss_res = np.sum((log_power - y_pred)**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    features['pink_noise_r_squared'] = r_squared
                except:
                    # Fallback if linear regression fails
                    features['pink_noise_slope'] = 0
                    features['pink_noise_intercept'] = 0
                    features['pink_noise_r_squared'] = 0
        
        return features
    
    def extract_spectrogram_features(self, 
                                    signal: np.ndarray, 
                                    fs: float = 5.0e6,
                                    nperseg: int = 256, 
                                    noverlap: int = 128) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract features from spectrogram analysis.
        
        Args:
            signal: Input signal array
            fs: Sampling frequency in Hz
            nperseg: Number of points per segment
            noverlap: Number of points overlap between segments
            
        Returns:
            Dictionary of extracted features and spectrogram data
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for spectrogram analysis")
            return {}
        
        # Compute spectrogram
        f, t, Sxx = self.preprocessor.compute_spectrogram(
            signal, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        
        if len(f) == 0 or len(t) == 0 or Sxx.size == 0:
            logger.warning("Failed to compute spectrogram")
            return {}
        
        # Extract basic features
        features = {
            'f': f,            # Frequency bins
            't': t,            # Time bins
            'Sxx': Sxx,        # Spectrogram values
            'mean_power': np.mean(Sxx),
            'max_power': np.max(Sxx),
            'std_power': np.std(Sxx),
            'total_energy': np.sum(Sxx),
        }
        
        # Time evolution of spectral properties
        if Sxx.shape[1] > 0:  # If there are multiple time points
            # Spectral centroid over time
            spectral_centroid = np.zeros(Sxx.shape[1])
            for i in range(Sxx.shape[1]):
                if np.sum(Sxx[:, i]) > 0:
                    spectral_centroid[i] = np.sum(f * Sxx[:, i]) / np.sum(Sxx[:, i])
            
            features['spectral_centroid_time'] = spectral_centroid
            features['mean_spectral_centroid'] = np.mean(spectral_centroid)
            features['std_spectral_centroid'] = np.std(spectral_centroid)
            
            # Band power over time
            bands = [
                (0, 1e3),       # 0-1 kHz
                (1e3, 10e3),    # 1-10 kHz
                (10e3, 100e3),  # 10-100 kHz
                (100e3, 500e3), # 100-500 kHz
                (500e3, 1e6),   # 500 kHz - 1 MHz
                (1e6, 2.5e6)    # 1-2.5 MHz
            ]
            
            band_power_time = {}
            for i, (low, high) in enumerate(bands):
                # Find indices in the frequency array
                indices = np.where((f >= low) & (f < high))[0]
                if len(indices) > 0:
                    band_power = np.sum(Sxx[indices, :], axis=0)
                    band_power_time[f'band_{i}_power_time'] = band_power
                    
                    # Average and std of band power over time
                    features[f'mean_band_{i}_power'] = np.mean(band_power)
                    features[f'std_band_{i}_power'] = np.std(band_power)
                    
                    # Maximum band power and its time
                    if len(band_power) > 0:
                        max_idx = np.argmax(band_power)
                        features[f'max_band_{i}_power'] = band_power[max_idx]
                        if 0 <= max_idx < len(t):
                            features[f'max_band_{i}_power_time'] = t[max_idx]
            
            features['band_power_time'] = band_power_time
        
        return features
    
    def extract_stft_features(self, 
                             signal: np.ndarray, 
                             fs: float = 5.0e6,
                             nperseg: int = 256, 
                             noverlap: int = 128) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract features from Short-Time Fourier Transform analysis.
        
        Args:
            signal: Input signal array
            fs: Sampling frequency in Hz
            nperseg: Number of points per segment
            noverlap: Number of points overlap between segments
            
        Returns:
            Dictionary of extracted features and STFT data
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for STFT analysis")
            return {}
        
        # Compute STFT
        f, t, Zxx = self.preprocessor.compute_stft(
            signal, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        
        if len(f) == 0 or len(t) == 0 or Zxx.size == 0:
            logger.warning("Failed to compute STFT")
            return {}
        
        # Extract magnitude and phase
        Zxx_mag = np.abs(Zxx)
        Zxx_phase = np.angle(Zxx)
        Zxx_power = Zxx_mag**2
        
        # Extract basic features
        features = {
            'f': f,            # Frequency bins
            't': t,            # Time bins
            'Zxx': Zxx,        # STFT values (complex)
            'Zxx_mag': Zxx_mag,    # Magnitude
            'Zxx_phase': Zxx_phase,  # Phase
            'Zxx_power': Zxx_power,  # Power
            'mean_magnitude': np.mean(Zxx_mag),
            'max_magnitude': np.max(Zxx_mag),
            'std_magnitude': np.std(Zxx_mag),
            'total_energy': np.sum(Zxx_power),
        }
        
        # Time evolution of spectral properties
        if Zxx_mag.shape[1] > 0:  # If there are multiple time points
            # Spectral centroid over time
            spectral_centroid = np.zeros(Zxx_mag.shape[1])
            for i in range(Zxx_mag.shape[1]):
                if np.sum(Zxx_mag[:, i]) > 0:
                    spectral_centroid[i] = np.sum(f * Zxx_mag[:, i]) / np.sum(Zxx_mag[:, i])
            
            features['spectral_centroid_time'] = spectral_centroid
            features['mean_spectral_centroid'] = np.mean(spectral_centroid)
            features['std_spectral_centroid'] = np.std(spectral_centroid)
            
            # Phase coherence over time (standard deviation of phase)
            phase_std = np.std(Zxx_phase, axis=0)
            features['phase_std_time'] = phase_std
            features['mean_phase_std'] = np.mean(phase_std)
            
            # Time-frequency moments
            if len(f) > 0 and Zxx_mag.shape[1] > 0:
                # Create meshgrid for moment calculation
                F, T = np.meshgrid(f, t, indexing='ij')
                
                # Calculate joint time-frequency moments
                # First moment (mean) in time and frequency
                tf_mean_f = np.sum(F * Zxx_mag) / np.sum(Zxx_mag) if np.sum(Zxx_mag) > 0 else 0
                tf_mean_t = np.sum(T * Zxx_mag) / np.sum(Zxx_mag) if np.sum(Zxx_mag) > 0 else 0
                
                # Second moment (variance) in time and frequency
                tf_var_f = np.sum(((F - tf_mean_f)**2) * Zxx_mag) / np.sum(Zxx_mag) if np.sum(Zxx_mag) > 0 else 0
                tf_var_t = np.sum(((T - tf_mean_t)**2) * Zxx_mag) / np.sum(Zxx_mag) if np.sum(Zxx_mag) > 0 else 0
                
                # Covariance between time and frequency
                tf_cov = np.sum(((F - tf_mean_f) * (T - tf_mean_t)) * Zxx_mag) / np.sum(Zxx_mag) if np.sum(Zxx_mag) > 0 else 0
                
                features['tf_mean_f'] = tf_mean_f
                features['tf_mean_t'] = tf_mean_t
                features['tf_var_f'] = tf_var_f
                features['tf_var_t'] = tf_var_t
                features['tf_cov'] = tf_cov
                
                # If valid covariance, calculate correlation
                if tf_var_f > 0 and tf_var_t > 0:
                    features['tf_correlation'] = tf_cov / np.sqrt(tf_var_f * tf_var_t)
                else:
                    features['tf_correlation'] = 0
        
        return features
    
    def extract_all_features(self, 
                            signal: np.ndarray, 
                            fs: float = 5.0e6,
                            include_spectrogram: bool = True,
                            include_stft: bool = True) -> Dict[str, Any]:
        """
        Extract all available features from a signal.
        
        Args:
            signal: Input signal array
            fs: Sampling frequency in Hz
            include_spectrogram: Whether to include spectrogram features
            include_stft: Whether to include STFT features
            
        Returns:
            Dictionary of all extracted features
        """
        if signal is None or len(signal) == 0:
            logger.warning("Empty signal provided for feature extraction")
            return {}
        
        # Initialize feature dictionary
        features = {}
        
        # Extract time-domain features
        time_features = self.extract_time_domain_features(signal)
        for key, value in time_features.items():
            features[f'time_{key}'] = value
        
        # Extract frequency-domain features
        freq_features = self.extract_frequency_domain_features(signal, fs)
        for key, value in freq_features.items():
            features[f'freq_{key}'] = value
        
        # Extract spectrogram features if requested
        if include_spectrogram:
            spec_features = self.extract_spectrogram_features(signal, fs)
            # Only include scalar features in the main dictionary
            for key, value in spec_features.items():
                if not isinstance(value, np.ndarray):
                    features[f'spec_{key}'] = value
                    
            # Store array features separately
            features['spectrogram_data'] = {
                'f': spec_features.get('f'),
                't': spec_features.get('t'),
                'Sxx': spec_features.get('Sxx')
            }
            if 'band_power_time' in spec_features:
                features['spectrogram_data']['band_power_time'] = spec_features['band_power_time']
            if 'spectral_centroid_time' in spec_features:
                features['spectrogram_data']['spectral_centroid_time'] = spec_features['spectral_centroid_time']
        
        # Extract STFT features if requested
        if include_stft:
            stft_features = self.extract_stft_features(signal, fs)
            # Only include scalar features in the main dictionary
            for key, value in stft_features.items():
                if not isinstance(value, np.ndarray) and not isinstance(value, np.complex128):
                    features[f'stft_{key}'] = value
                    
            # Store array features separately
            features['stft_data'] = {
                'f': stft_features.get('f'),
                't': stft_features.get('t'),
                'Zxx_mag': stft_features.get('Zxx_mag'),
                'Zxx_phase': stft_features.get('Zxx_phase')
            }
            if 'spectral_centroid_time' in stft_features:
                features['stft_data']['spectral_centroid_time'] = stft_features['spectral_centroid_time']
            if 'phase_std_time' in stft_features:
                features['stft_data']['phase_std_time'] = stft_features['phase_std_time']
        
        return features
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate the skewness of a distribution."""
        n = len(x)
        if n <= 1:
            return 0
        
        mean = np.mean(x)
        std = np.std(x)
        
        if std == 0:
            return 0
            
        return np.sum(((x - mean) / std) ** 3) / n
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate the kurtosis of a distribution."""
        n = len(x)
        if n <= 1:
            return 0
        
        mean = np.mean(x)
        std = np.std(x)
        
        if std == 0:
            return 0
            
        return np.sum(((x - mean) / std) ** 4) / n


class DimensionalityReducer:
    """
    Reduce dimensionality of feature sets for more efficient processing.
    
    This class implements various dimensionality reduction techniques
    for DC arc detection data, including PCA, t-SNE, UMAP, and LLE.
    """
    
    def __init__(self):
        """Initialize the dimensionality reducer."""
        logger.info("Initialized DimensionalityReducer")
    
    def apply_pca(self, 
                 data: np.ndarray, 
                 n_components: int = 2, 
                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Principal Component Analysis to data.
        
        Args:
            data: Input data matrix (samples x features)
            n_components: Number of components to keep
            normalize: Whether to normalize data before PCA
            
        Returns:
            Tuple of (transformed_data, explained_variance_ratio, components)
        """
        if data is None or data.size == 0:
            logger.warning("Empty data provided for PCA")
            return np.array([]), np.array([]), np.array([])
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            logger.warning("Expanded 1D data to 2D for PCA")
            
        # Check if we have enough samples
        n_samples, n_features = data.shape
        if n_samples < 2:
            logger.warning("Need at least 2 samples for PCA")
            return np.array([]), np.array([]), np.array([])
        
        # Adjust n_components if needed
        max_components = min(n_samples, n_features)
        if n_components > max_components:
            logger.warning(f"Reducing n_components from {n_components} to {max_components}")
            n_components = max_components
        
        # Apply normalization if requested
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        
        return transformed, pca.explained_variance_ratio_, pca.components_
    
    def apply_tsne(self, 
                  data: np.ndarray, 
                  n_components: int = 2, 
                  perplexity: float = 30.0,
                  normalize: bool = True) -> np.ndarray:
        """
        Apply t-SNE (t-distributed Stochastic Neighbor Embedding) to data.
        
        Args:
            data: Input data matrix (samples x features)
            n_components: Number of components to keep
            perplexity: Perplexity parameter for t-SNE
            normalize: Whether to normalize data before t-SNE
            
        Returns:
            Transformed data
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.error("scikit-learn is required for t-SNE")
            return np.array([])
        
        if data is None or data.size == 0:
            logger.warning("Empty data provided for t-SNE")
            return np.array([])
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            logger.warning("Expanded 1D data to 2D for t-SNE")
            
        # Check if we have enough samples
        n_samples, n_features = data.shape
        if n_samples < 2:
            logger.warning("Need at least 2 samples for t-SNE")
            return np.array([])
        
        # Adjust perplexity if needed
        max_perplexity = (n_samples - 1) / 3
        if perplexity > max_perplexity:
            logger.warning(f"Reducing perplexity from {perplexity} to {max_perplexity}")
            perplexity = max_perplexity
        
        # Apply normalization if requested
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        transformed = tsne.fit_transform(data)
        
        return transformed
    
    def apply_umap(self, 
                  data: np.ndarray, 
                  n_components: int = 2, 
                  n_neighbors: int = 15,
                  min_dist: float = 0.1,
                  normalize: bool = True) -> np.ndarray:
        """
        Apply UMAP (Uniform Manifold Approximation and Projection) to data.
        
        Args:
            data: Input data matrix (samples x features)
            n_components: Number of components to keep
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance parameter for UMAP
            normalize: Whether to normalize data before UMAP
            
        Returns:
            Transformed data
        """
        try:
            import umap
        except ImportError:
            logger.error("umap-learn is required for UMAP")
            return np.array([])
        
        if data is None or data.size == 0:
            logger.warning("Empty data provided for UMAP")
            return np.array([])
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            logger.warning("Expanded 1D data to 2D for UMAP")
            
        # Check if we have enough samples
        n_samples, n_features = data.shape
        if n_samples < 2:
            logger.warning("Need at least 2 samples for UMAP")
            return np.array([])
        
        # Adjust n_neighbors if needed
        if n_neighbors >= n_samples:
            logger.warning(f"Reducing n_neighbors from {n_neighbors} to {n_samples-1}")
            n_neighbors = n_samples - 1
        
        # Apply normalization if requested
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42)
        transformed = reducer.fit_transform(data)
        
        return transformed
    
    def apply_lle(self, 
                 data: np.ndarray, 
                 n_components: int = 2, 
                 n_neighbors: int = 15,
                 normalize: bool = True) -> np.ndarray:
        """
        Apply LLE (Locally Linear Embedding) to data.
        
        Args:
            data: Input data matrix (samples x features)
            n_components: Number of components to keep
            n_neighbors: Number of neighbors for LLE
            normalize: Whether to normalize data before LLE
            
        Returns:
            Transformed data
        """
        try:
            from sklearn.manifold import LocallyLinearEmbedding
        except ImportError:
            logger.error("scikit-learn is required for LLE")
            return np.array([])
        
        if data is None or data.size == 0:
            logger.warning("Empty data provided for LLE")
            return np.array([])
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            logger.warning("Expanded 1D data to 2D for LLE")
            
        # Check if we have enough samples
        n_samples, n_features = data.shape
        if n_samples < 2:
            logger.warning("Need at least 2 samples for LLE")
            return np.array([])
        
        # Adjust n_neighbors if needed
        if n_neighbors >= n_samples:
            logger.warning(f"Reducing n_neighbors from {n_neighbors} to {n_samples-1}")
            n_neighbors = n_samples - 1
        
        # Apply normalization if requested
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        
        # Apply LLE
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, 
                                    random_state=42)
        transformed = lle.fit_transform(data)
        
        return transformed


# Helper functions for feature extraction from processed data

def extract_features_from_chunks(chunked_data: Dict[str, Any], channel_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract features from all chunks for a specific channel.
    
    Args:
        chunked_data: Chunked data from DataProcessor.chunk_aligned_data
        channel_name: Name of the channel to extract features from
        
    Returns:
        Dictionary of chunk features by chunk index
    """
    if not chunked_data or 'chunks' not in chunked_data or not chunked_data['chunks']:
        logger.error("Invalid chunked data for feature extraction")
        return {}
    
    extractor = FeatureExtractor()
    chunk_features = {}
    
    # Process each chunk
    for chunk in chunked_data['chunks']:
        chunk_idx = chunk['index']
        
        if 'channels' in chunk and channel_name in chunk['channels']:
            signal = chunk['channels'][channel_name]
            
            # Extract features
            features = extractor.extract_all_features(signal)
            
            # Add region if available
            if 'region' in chunk:
                features['region'] = chunk['region']
                
            # Add to result
            chunk_features[chunk_idx] = features
        else:
            logger.warning(f"Channel {channel_name} not found in chunk {chunk_idx}")
    
    return chunk_features

def extract_features_from_groups(grouped_data: Dict[str, Any], channel_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract features from all groups for a specific channel.
    
    Args:
        grouped_data: Grouped data from DataProcessor.form_chunk_groups
        channel_name: Name of the channel to extract features from
        
    Returns:
        Dictionary of group features by group index
    """
    if not grouped_data or 'groups' not in grouped_data or not grouped_data['groups']:
        logger.error("Invalid grouped data for feature extraction")
        return {}
    
    extractor = FeatureExtractor()
    group_features = {}
    
    # Process each group
    for group in grouped_data['groups']:
        group_idx = group['index']
        
        if 'channels' in group and channel_name in group['channels']:
            # Group channel data is a list of arrays (one per chunk)
            chunk_signals = group['channels'][channel_name]
            
            # Extract features for each chunk
            chunk_feature_list = []
            for i, signal in enumerate(chunk_signals):
                features = extractor.extract_all_features(signal)
                chunk_feature_list.append(features)
            
            # Compute aggregate features across chunks
            agg_features = {}
            
            # Find all scalar features that appear in all chunks
            if chunk_feature_list:
                first_features = chunk_feature_list[0]
                scalar_keys = [k for k, v in first_features.items() 
                              if not isinstance(v, dict) and not isinstance(v, np.ndarray)]
                
                # Calculate mean and std for each scalar feature
                for key in scalar_keys:
                    values = [features.get(key, 0) for features in chunk_feature_list]
                    agg_features[f'{key}_mean'] = np.mean(values)
                    agg_features[f'{key}_std'] = np.std(values)
                    agg_features[f'{key}_min'] = np.min(values)
                    agg_features[f'{key}_max'] = np.max(values)
            
            # Add region and label if available
            if 'region' in group:
                agg_features['region'] = group['region']
            if 'label' in group:
                agg_features['label'] = group['label']
                
            # Add to result
            group_features[group_idx] = {
                'chunk_features': chunk_feature_list,
                'aggregate_features': agg_features
            }
        else:
            logger.warning(f"Channel {channel_name} not found in group {group_idx}")
    
    return group_features