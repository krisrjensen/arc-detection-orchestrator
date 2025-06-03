"""
Data Processor and Chunking System for DC Arc Detection

This module provides functionality for processing detected transients,
creating time-aligned data chunks, and preparing data for distance metric analysis.

Based on the PhD research of Kristophor Jensen at South Dakota School of Mines and Technology.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arc_detection.core.processor")

class DataProcessor:
    """
    Process data around detected transients for analysis.
    
    This class handles data chunking, time alignment, and preprocessing of data
    around detected transients. It prepares data for distance metric analysis and
    classification.
    """
    
    def __init__(self, 
                 chunk_size: int = 1024,
                 samples_before: int = 105000,
                 samples_after: int = 105000,
                 normalize_chunks: bool = True,
                 required_channels: List[str] = None):
        """
        Initialize the data processor with configurable parameters.
        
        Args:
            chunk_size: Size of data chunks (default: 1024)
            samples_before: Number of samples before transient to include (default: 105000)
            samples_after: Number of samples after transient to include (default: 105000)
            normalize_chunks: Whether to normalize chunks using StandardScaler (default: True)
            required_channels: List of required channels (default: None = all available)
        """
        self.chunk_size = chunk_size
        self.samples_before = samples_before
        self.samples_after = samples_after
        self.normalize_chunks = normalize_chunks
        self.required_channels = required_channels or ['load_voltage', 'source_voltage', 'source_current']
        
        logger.info(f"Initialized DataProcessor with chunk_size={chunk_size}, "
                   f"samples_before={samples_before}, samples_after={samples_after}")
    
    def extract_aligned_data(self, 
                            data: Dict[str, Any], 
                            transient_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract time-aligned data around the detected transient.
        
        Args:
            data: Data dictionary from the MATLAB loader
            transient_result: Transient detection result
            
        Returns:
            Dictionary with aligned data or None if extraction fails
        """
        if not data or 'channels' not in data or not data['channels'] or 'time_data' not in data:
            logger.error("Invalid data structure for alignment")
            return None
            
        if not transient_result or not transient_result.get('success', False) or 'transient_index' not in transient_result:
            logger.error("Invalid transient result for alignment")
            return None
        
        # Get transient index
        transient_index = transient_result['transient_index']
        logger.info(f"Aligning data around transient at index {transient_index}")
        
        # Calculate alignment window
        start_idx = max(0, transient_index - self.samples_before)
        end_idx = min(len(data['time_data']), transient_index + self.samples_after + 1)
        
        # If start or end index would result in fewer than chunk_size samples, adjust
        if end_idx - start_idx < self.chunk_size:
            logger.warning(f"Alignment window too small ({end_idx - start_idx} < {self.chunk_size})")
            # Try to extend the window while maintaining transient position
            extension_needed = self.chunk_size - (end_idx - start_idx)
            # Distribute extension before and after if possible
            extension_before = min(start_idx, extension_needed // 2)
            extension_after = min(len(data['time_data']) - end_idx, extension_needed - extension_before)
            
            start_idx = max(0, start_idx - extension_before)
            end_idx = min(len(data['time_data']), end_idx + extension_after)
            
            logger.info(f"Adjusted alignment window to {end_idx - start_idx} samples")
            
            if end_idx - start_idx < self.chunk_size:
                logger.error(f"Cannot create alignment window of at least {self.chunk_size} samples")
                return None
        
        # Create aligned data structure
        aligned_data = {
            'time_data': data['time_data'][start_idx:end_idx],
            'channels': {},
            'alignment': {
                'original_signal_length': len(data['time_data']),
                'transient_index': transient_index,
                'aligned_transient_index': transient_index - start_idx,
                'original_start_index': start_idx,
                'original_end_index': end_idx
            }
        }
        
        # Add metadata if available
        if 'metadata' in data:
            aligned_data['metadata'] = data['metadata'].copy()
        
        # Add alignment info to metadata
        if 'metadata' not in aligned_data:
            aligned_data['metadata'] = {}
        aligned_data['metadata']['alignment'] = aligned_data['alignment']
        
        # Extract channel data
        for channel_name, channel_data in data['channels'].items():
            if channel_name in self.required_channels:
                if len(channel_data) == len(data['time_data']):
                    aligned_data['channels'][channel_name] = channel_data[start_idx:end_idx]
                else:
                    logger.warning(f"Channel {channel_name} length mismatch: {len(channel_data)} vs {len(data['time_data'])}")
                    # Try to resize or skip based on policy
                    if abs(len(channel_data) - len(data['time_data'])) < 100:
                        # Minor length mismatch, can truncate
                        logger.warning(f"Truncating channel {channel_name} to match time data length")
                        aligned_data['channels'][channel_name] = channel_data[:len(data['time_data'])][start_idx:end_idx]
                    else:
                        logger.error(f"Cannot align channel {channel_name} due to significant length mismatch")
        
        # Check if required channels are present
        missing_channels = [ch for ch in self.required_channels if ch not in aligned_data['channels']]
        if missing_channels:
            logger.warning(f"Missing required channels in aligned data: {missing_channels}")
            if len(missing_channels) == len(self.required_channels):
                logger.error("No required channels available in aligned data")
                return None
        
        # Add aligned transient regions
        transient_window = transient_result.get('window', (0, 0))
        if transient_window[0] >= start_idx and transient_window[1] <= end_idx:
            aligned_transient_window = (transient_window[0] - start_idx, transient_window[1] - start_idx)
            aligned_data['alignment']['aligned_transient_window'] = aligned_transient_window
            
            # Create three regions: pre-transient, transient, post-transient
            aligned_data['regions'] = {
                'pre_transient': list(range(0, aligned_transient_window[0])),
                'transient': list(range(aligned_transient_window[0], aligned_transient_window[1])),
                'post_transient': list(range(aligned_transient_window[1], end_idx - start_idx))
            }
        
        logger.info(f"Successfully aligned data with {len(aligned_data['time_data'])} time points and {len(aligned_data['channels'])} channels")
        return aligned_data
    
    def chunk_aligned_data(self, 
                          aligned_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create overlapping chunks from aligned data.
        
        Args:
            aligned_data: Aligned data from extract_aligned_data
            
        Returns:
            Dictionary with chunked data or None if chunking fails
        """
        if not aligned_data or 'channels' not in aligned_data or not aligned_data['channels']:
            logger.error("Invalid aligned data for chunking")
            return None
        
        logger.info(f"Creating chunks with size {self.chunk_size}")
        
        # Calculate stride with 50% overlap
        stride = self.chunk_size // 2
        
        # Get data length from first channel
        first_channel = next(iter(aligned_data['channels'].values()))
        data_length = len(first_channel)
        
        # Calculate number of chunks
        num_chunks = max(1, (data_length - self.chunk_size) // stride + 1)
        logger.info(f"Creating {num_chunks} chunks from {data_length} samples with stride {stride}")
        
        # Initialize chunk storage
        chunked_data = {
            'chunks': [],
            'chunk_indices': [],
            'metadata': aligned_data.get('metadata', {}).copy()
        }
        
        # Create each chunk
        for i in range(num_chunks):
            start_idx = i * stride
            end_idx = min(start_idx + self.chunk_size, data_length)
            
            # Ensure chunk is full size
            if end_idx - start_idx < self.chunk_size:
                # If at the end, move window back to get full chunk size
                if end_idx == data_length:
                    start_idx = max(0, data_length - self.chunk_size)
                    end_idx = data_length
                else:
                    # Skip incomplete chunks in the middle (shouldn't happen with proper stride)
                    continue
            
            # Create chunk with data from all channels
            chunk = {
                'index': i,
                'start': start_idx,
                'end': end_idx,
                'channels': {}
            }
            
            # Extract chunk data for each channel
            for channel_name, channel_data in aligned_data['channels'].items():
                chunk_data = channel_data[start_idx:end_idx]
                
                # Apply normalization if enabled
                if self.normalize_chunks:
                    scaler = StandardScaler()
                    # Reshape for sklearn (n_samples, n_features)
                    reshaped = chunk_data.reshape(-1, 1)
                    normalized = scaler.fit_transform(reshaped).flatten()
                    chunk['channels'][channel_name] = normalized
                else:
                    chunk['channels'][channel_name] = chunk_data
            
            # Determine chunk region based on alignment info
            if 'regions' in aligned_data:
                # Check if chunk is entirely within a single region
                chunk_range = set(range(start_idx, end_idx))
                region_name = None
                
                for region, indices in aligned_data['regions'].items():
                    region_indices = set(indices)
                    overlap = len(chunk_range.intersection(region_indices))
                    
                    if overlap == len(chunk_range):
                        # Chunk entirely within this region
                        region_name = region
                        break
                    elif overlap > 0 and region_name is None:
                        # Chunk partially in this region (will take the first match if multiple)
                        region_name = f"partial_{region}"
                
                if region_name:
                    chunk['region'] = region_name
            
            # If time data is available, add time information
            if 'time_data' in aligned_data:
                chunk['time_start'] = float(aligned_data['time_data'][start_idx])
                chunk['time_end'] = float(aligned_data['time_data'][end_idx - 1])
                chunk['time_duration'] = chunk['time_end'] - chunk['time_start']
            
            # Add chunk to result
            chunked_data['chunks'].append(chunk)
            chunked_data['chunk_indices'].append((start_idx, end_idx))
        
        # Add chunk metadata
        chunked_data['metadata']['chunk_size'] = self.chunk_size
        chunked_data['metadata']['stride'] = stride
        chunked_data['metadata']['num_chunks'] = len(chunked_data['chunks'])
        chunked_data['metadata']['normalized'] = self.normalize_chunks
        
        logger.info(f"Created {len(chunked_data['chunks'])} chunks from aligned data")
        return chunked_data
    
    def form_chunk_groups(self, 
                          chunked_data: Dict[str, Any], 
                          group_size: int = 3,
                          step: int = 1,
                          labels: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Form groups of consecutive chunks for group-based distance analysis.
        
        Args:
            chunked_data: Chunked data from chunk_aligned_data
            group_size: Number of consecutive chunks in each group
            step: Step size between groups
            labels: Dictionary mapping regions to labels
            
        Returns:
            Dictionary with chunk groups or None if grouping fails
        """
        if not chunked_data or 'chunks' not in chunked_data or not chunked_data['chunks']:
            logger.error("Invalid chunked data for group formation")
            return None
        
        chunks = chunked_data['chunks']
        num_chunks = len(chunks)
        
        if num_chunks < group_size:
            logger.error(f"Not enough chunks for group size {group_size} (have {num_chunks})")
            return None
        
        logger.info(f"Forming chunk groups with size {group_size} and step {step}")
        
        # Initialize group storage
        grouped_data = {
            'groups': [],
            'metadata': chunked_data.get('metadata', {}).copy()
        }
        
        # Create each group
        num_groups = max(1, (num_chunks - group_size) // step + 1)
        for i in range(num_groups):
            start_idx = i * step
            end_idx = min(start_idx + group_size, num_chunks)
            
            # Skip incomplete groups
            if end_idx - start_idx < group_size:
                continue
            
            # Get chunks for this group
            group_chunks = chunks[start_idx:end_idx]
            
            # Determine group region
            group_region = None
            region_counts = {}
            
            for chunk in group_chunks:
                if 'region' in chunk:
                    region = chunk['region']
                    region_counts[region] = region_counts.get(region, 0) + 1
            
            if region_counts:
                # Find most common region
                group_region = max(region_counts.items(), key=lambda x: x[1])[0]
            
            # Create group with combined data from all chunks
            group = {
                'index': i,
                'start_chunk': start_idx,
                'end_chunk': end_idx,
                'chunks': group_chunks,
                'channels': {}
            }
            
            if group_region:
                group['region'] = group_region
                
                # Apply label if available
                if labels and group_region in labels:
                    group['label'] = labels[group_region]
            
            # Combine channel data from all chunks
            for chunk in group_chunks:
                for channel_name, channel_data in chunk['channels'].items():
                    if channel_name not in group['channels']:
                        group['channels'][channel_name] = []
                    
                    group['channels'][channel_name].append(channel_data)
            
            # Add group to result
            grouped_data['groups'].append(group)
        
        # Add group metadata
        grouped_data['metadata']['group_size'] = group_size
        grouped_data['metadata']['step'] = step
        grouped_data['metadata']['num_groups'] = len(grouped_data['groups'])
        
        logger.info(f"Created {len(grouped_data['groups'])} groups from chunked data")
        return grouped_data
    
    def save_processed_data(self, 
                           data: Dict[str, Any], 
                           output_path: str, 
                           file_prefix: str = "") -> Optional[str]:
        """
        Save processed data to files.
        
        Args:
            data: Processed data (aligned, chunked, or grouped)
            output_path: Directory to save files
            file_prefix: Prefix for output files
            
        Returns:
            Path to saved files or None if saving fails
        """
        if not data:
            logger.error("No data to save")
            return None
        
        try:
            # Create output directory if needed
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine data type and create appropriate files
            if 'chunks' in data and isinstance(data['chunks'], list):
                # Chunked data
                data_type = "chunked"
                logger.info(f"Saving chunked data with {len(data['chunks'])} chunks")
                
                # Create CSV file with chunk metadata
                chunk_meta = []
                for i, chunk in enumerate(data['chunks']):
                    row = {
                        'chunk_index': i,
                        'start': chunk.get('start', 0),
                        'end': chunk.get('end', 0),
                        'region': chunk.get('region', 'unknown')
                    }
                    
                    # Add time info if available
                    if 'time_start' in chunk:
                        row['time_start'] = chunk['time_start']
                        row['time_end'] = chunk['time_end']
                        row['time_duration'] = chunk['time_duration']
                    
                    chunk_meta.append(row)
                
                # Save chunk metadata
                meta_df = pd.DataFrame(chunk_meta)
                meta_path = output_dir / f"{file_prefix}chunk_metadata.csv"
                meta_df.to_csv(meta_path, index=False)
                logger.info(f"Saved chunk metadata to {meta_path}")
                
                # Save chunk data for each channel
                for chunk_idx, chunk in enumerate(data['chunks']):
                    for channel_name, channel_data in chunk['channels'].items():
                        # Create directory for channel
                        channel_dir = output_dir / channel_name
                        channel_dir.mkdir(exist_ok=True)
                        
                        # Save chunk data
                        chunk_path = channel_dir / f"{file_prefix}chunk_{chunk_idx:04d}.npy"
                        np.save(chunk_path, channel_data)
                
                logger.info(f"Saved chunk data for {len(data['chunks'])} chunks")
                
            elif 'groups' in data and isinstance(data['groups'], list):
                # Grouped data
                data_type = "grouped"
                logger.info(f"Saving grouped data with {len(data['groups'])} groups")
                
                # Create CSV file with group metadata
                group_meta = []
                for i, group in enumerate(data['groups']):
                    row = {
                        'group_index': i,
                        'start_chunk': group.get('start_chunk', 0),
                        'end_chunk': group.get('end_chunk', 0),
                        'region': group.get('region', 'unknown'),
                        'label': group.get('label', 'unknown')
                    }
                    group_meta.append(row)
                
                # Save group metadata
                meta_df = pd.DataFrame(group_meta)
                meta_path = output_dir / f"{file_prefix}group_metadata.csv"
                meta_df.to_csv(meta_path, index=False)
                logger.info(f"Saved group metadata to {meta_path}")
                
                # Save group data for each channel
                for group_idx, group in enumerate(data['groups']):
                    for channel_name, channel_data_list in group['channels'].items():
                        # Create directory for channel
                        channel_dir = output_dir / channel_name
                        channel_dir.mkdir(exist_ok=True)
                        
                        # Save group data
                        group_path = channel_dir / f"{file_prefix}group_{group_idx:04d}.npy"
                        np.save(group_path, np.array(channel_data_list))
                
                logger.info(f"Saved group data for {len(data['groups'])} groups")
                
            elif 'channels' in data and isinstance(data['channels'], dict):
                # Aligned data
                data_type = "aligned"
                logger.info(f"Saving aligned data with {len(data['channels'])} channels")
                
                # Save each channel as a separate file
                for channel_name, channel_data in data['channels'].items():
                    channel_path = output_dir / f"{file_prefix}{channel_name}.npy"
                    np.save(channel_path, channel_data)
                
                # Save time data if available
                if 'time_data' in data and data['time_data'] is not None:
                    time_path = output_dir / f"{file_prefix}time_data.npy"
                    np.save(time_path, data['time_data'])
                
                logger.info(f"Saved aligned data for {len(data['channels'])} channels")
                
            else:
                logger.error("Unknown data format for saving")
                return None
            
            # Save metadata
            if 'metadata' in data:
                metadata = data['metadata']
                metadata['data_type'] = data_type
                
                meta_path = output_dir / f"{file_prefix}metadata.json"
                
                # Convert numpy values to Python natives for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(i) for i in obj]
                    else:
                        return obj
                
                import json
                with open(meta_path, 'w') as f:
                    json.dump(convert_numpy(metadata), f, indent=2)
                
                logger.info(f"Saved metadata to {meta_path}")
            
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_aligned_data(self, 
                         aligned_data: Dict[str, Any], 
                         title: str = None,
                         save_path: str = None,
                         max_points: int = 10000):
        """
        Create a visualization of aligned data with transient regions highlighted.
        
        Args:
            aligned_data: Aligned data from extract_aligned_data
            title: Optional title for the plot
            save_path: Optional path to save the figure
            max_points: Maximum number of points to plot (will downsample if exceeded)
            
        Returns:
            The matplotlib figure
        """
        if not aligned_data or 'channels' not in aligned_data or not aligned_data['channels']:
            logger.warning("Cannot create plot: invalid aligned data")
            return None
        
        # Get channels and time
        channels = aligned_data['channels']
        time_data = aligned_data.get('time_data')
        
        # Downsample if needed
        if time_data is not None and len(time_data) > max_points:
            step = len(time_data) // max_points
            time_data = time_data[::step]
            downsampled_channels = {}
            for name, channel_data in channels.items():
                downsampled_channels[name] = channel_data[::step]
            channels = downsampled_channels
        else:
            # Use indices if no time data
            if time_data is None:
                first_channel = next(iter(channels.values()))
                if len(first_channel) > max_points:
                    step = len(first_channel) // max_points
                    time_data = np.arange(0, len(first_channel), step)
                    downsampled_channels = {}
                    for name, channel_data in channels.items():
                        downsampled_channels[name] = channel_data[::step]
                    channels = downsampled_channels
                else:
                    time_data = np.arange(len(first_channel))
        
        # Create plot with one subplot per channel
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3*len(channels)), sharex=True)
        
        # Handle case with only one channel
        if len(channels) == 1:
            axes = [axes]
        
        # Get transient information
        transient_idx = None
        if 'alignment' in aligned_data and 'aligned_transient_index' in aligned_data['alignment']:
            transient_idx = aligned_data['alignment']['aligned_transient_index']
        
        # Get regions if available
        pre_transient_indices = []
        transient_indices = []
        post_transient_indices = []
        
        if 'regions' in aligned_data:
            regions = aligned_data['regions']
            if 'pre_transient' in regions:
                pre_transient_indices = regions['pre_transient']
            if 'transient' in regions:
                transient_indices = regions['transient']
            if 'post_transient' in regions:
                post_transient_indices = regions['post_transient']
        
        # Define region colors
        colors = {'pre_transient': 'green', 'transient': 'red', 'post_transient': 'blue'}
        
        # Plot each channel
        for i, (channel_name, channel_data) in enumerate(channels.items()):
            # Use indices instead of full arrays for downsampled plotting
            pre_indices = [idx for idx in pre_transient_indices if idx < len(time_data)]
            transient_indices_subset = [idx for idx in transient_indices if idx < len(time_data)]
            post_indices = [idx for idx in post_transient_indices if idx < len(time_data)]
            
            # Plot pre-transient region in green
            if pre_indices:
                axes[i].plot(time_data[pre_indices], channel_data[pre_indices], 
                           color=colors['pre_transient'], label='Pre-Transient')
            
            # Plot transient region in red
            if transient_indices_subset:
                # If transient indices are provided, use the window centered on transient
                # Ensure we get a window of 1024 samples around the transient
                if len(transient_indices_subset) > 0:
                    # Try to center it on transient_idx if available
                    if transient_idx is not None and transient_idx < len(time_data):
                        # Calculate window start and end indices
                        win_size = 1024
                        win_start = max(0, transient_idx - win_size // 2)
                        win_end = min(len(time_data), win_start + win_size)
                        
                        # Adjust window to ensure it's the right size
                        if win_end - win_start < win_size:
                            win_start = max(0, win_end - win_size)
                        
                        # Use window indices
                        window_indices = list(range(win_start, win_end))
                        axes[i].plot(time_data[window_indices], channel_data[window_indices], 
                                   color=colors['transient'], label='Transient')
                    else:
                        # Use provided transient indices
                        axes[i].plot(time_data[transient_indices_subset], channel_data[transient_indices_subset], 
                                   color=colors['transient'], label='Transient')
            
            # Plot post-transient region in blue
            if post_indices:
                axes[i].plot(time_data[post_indices], channel_data[post_indices], 
                           color=colors['post_transient'], label='Post-Transient')
            
            # Add channel label
            axes[i].set_ylabel(channel_name)
            axes[i].grid(True)
        
        # Add transient marker if available
        if transient_idx is not None and transient_idx < len(time_data):
            for ax in axes:
                ax.axvline(x=time_data[transient_idx], color='black', linestyle='--', 
                          alpha=0.7, label='Transient Center')
        
        # Add labels and title
        if time_data is not None and 'time_data' in aligned_data:
            axes[-1].set_xlabel('Time (s)')
        else:
            axes[-1].set_xlabel('Sample')
            
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle('Aligned Data with Transient Regions')
        
        # Add legend to first subplot
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys())
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved aligned data plot to {save_path}")
        
        return fig
    
    def plot_augmented_data(self,
                           original_data: Dict[str, Any],
                           augmented_files: List[str],
                           channel_name: str = 'source_current',
                           max_points: int = 4000,
                           title: str = None,
                           folder_name: str = None,
                           label: str = None,
                           save_path: str = None):
        """
        Create a visualization of augmented data with color-coded left/right regions.
        
        Args:
            original_data: Original data dictionary from loader
            augmented_files: List of augmented file paths generated
            channel_name: Channel to visualize (default: source_current)
            max_points: Maximum number of points to show
            title: Optional title for the plot
            folder_name: Folder name for the title
            label: Resolved label for the title
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure
        """
        # Prepare figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Track which files we've loaded
        transient_data = None
        left_files = []
        right_files = []
        
        # Sort augmented files into categories
        for file_path in augmented_files:
            basename = os.path.basename(file_path)
            if 'transient_centered' in basename:
                transient_data = file_path
            elif 'L' in basename and any(f'L{i:03d}' in basename for i in range(1, 100)):
                left_files.append(file_path)
            elif 'R' in basename and any(f'R{i:03d}' in basename for i in range(1, 100)):
                right_files.append(file_path)
        
        # Sort by file number
        left_files = sorted(left_files, key=lambda x: int(re.search(r'L(\d+)', os.path.basename(x)).group(1)))
        right_files = sorted(right_files, key=lambda x: int(re.search(r'R(\d+)', os.path.basename(x)).group(1)))
        
        # Load data from files
        all_data = []
        all_colors = []
        
        # Load left files (green)
        for file_path in left_files:
            try:
                # Load data based on file extension
                if file_path.endswith('.npy'):
                    data = np.load(file_path)
                    # Extract the channel data if it's a structured array
                    if hasattr(data, 'dtype') and data.dtype.names is not None:
                        if channel_name in data.dtype.names:
                            channel_data = data[channel_name]
                        else:
                            channel_data = data['source_current']  # Fallback
                    else:
                        channel_data = data  # Assume it's just the channel data
                    all_data.append(channel_data)
                    all_colors.append('green')
            except Exception as e:
                logger.warning(f"Failed to load left file {file_path}: {str(e)}")
        
        # Load transient file (red)
        if transient_data:
            try:
                if transient_data.endswith('.npy'):
                    data = np.load(transient_data)
                    # Extract the channel data if it's a structured array
                    if hasattr(data, 'dtype') and data.dtype.names is not None:
                        if channel_name in data.dtype.names:
                            channel_data = data[channel_name]
                        else:
                            channel_data = data['source_current']  # Fallback
                    else:
                        channel_data = data  # Assume it's just the channel data
                    all_data.append(channel_data)
                    all_colors.append('red')
            except Exception as e:
                logger.warning(f"Failed to load transient file {transient_data}: {str(e)}")
        
        # Load right files (blue)
        for file_path in right_files:
            try:
                if file_path.endswith('.npy'):
                    data = np.load(file_path)
                    # Extract the channel data if it's a structured array
                    if hasattr(data, 'dtype') and data.dtype.names is not None:
                        if channel_name in data.dtype.names:
                            channel_data = data[channel_name]
                        else:
                            channel_data = data['source_current']  # Fallback
                    else:
                        channel_data = data  # Assume it's just the channel data
                    all_data.append(channel_data)
                    all_colors.append('blue')
            except Exception as e:
                logger.warning(f"Failed to load right file {file_path}: {str(e)}")
        
        # Plot the data
        if all_data:
            # Determine how many points to take from each file to get max_points total
            points_per_file = max_points // len(all_data)
            
            # Track total samples for resampled x-axis (bottom)
            total_samples = 0
            
            # Store original indices for global x-axis (top)
            original_indices = []
            resampled_positions = []
            
            for i, (data, color) in enumerate(zip(all_data, all_colors)):
                # Store original file indices before downsampling
                # Check if the file is transient, left or right for the original index calculation
                file_type = all_colors[i]  # 'green' for left, 'red' for transient, 'blue' for right
                
                # The original file length
                original_length = len(data)
                
                # Calculate original file indices before downsampling
                if file_type == 'green':  # Left files
                    # Left files indices depend on their position among left files
                    left_idx = [j for j, c in enumerate(all_colors) if c == 'green'].index(i)
                    start_idx = original_length * left_idx
                    original_file_indices = np.arange(start_idx, start_idx + original_length)
                elif file_type == 'red':  # Transient
                    # Place transient after all left files
                    left_files_count = all_colors.count('green')
                    start_idx = original_length * left_files_count
                    original_file_indices = np.arange(start_idx, start_idx + original_length)
                else:  # Right files (blue)
                    # Place right files after transient
                    right_idx = [j for j, c in enumerate(all_colors) if c == 'blue'].index(i)
                    left_files_count = all_colors.count('green')
                    transient_length = len([d for j, (d, c) in enumerate(zip(all_data, all_colors)) if c == 'red'][0]) if 'red' in all_colors else 0
                    start_idx = (original_length * left_files_count) + transient_length + (original_length * right_idx)
                    original_file_indices = np.arange(start_idx, start_idx + original_length)
                
                # If data is too long, downsample
                if len(data) > points_per_file:
                    # Get indices before downsampling
                    sample_indices = np.arange(0, len(data))
                    
                    # Calculate downsample step
                    step = len(data) // points_per_file
                    
                    # Downsample data
                    data = data[::step]
                    
                    # Downsample original indices to match
                    original_file_indices = original_file_indices[::step]
                
                # Calculate resampled x values for plotting (bottom axis)
                resampled_x = np.arange(len(data)) + total_samples
                
                # Store mapping between resampled positions and original indices for top axis
                for res_pos, orig_idx in zip(resampled_x, original_file_indices):
                    resampled_positions.append(res_pos)
                    original_indices.append(orig_idx)
                
                # Update total samples for next segment
                total_samples += len(data)
                
                # Plot this segment using resampled indices
                ax.plot(resampled_x, data, color=color, label=f"{color.capitalize()}" if i == 0 else "")
            
            # Create second x-axis (top) with original global file indices
            ax2 = ax.twiny()
            
            # Set 10 evenly spaced ticks using original file indices
            num_ticks = 10
            # Get min and max of resampled positions
            min_pos = min(resampled_positions)
            max_pos = max(resampled_positions)
            
            # Create evenly spaced positions for the 10 ticks
            tick_positions = np.linspace(min_pos, max_pos, num_ticks)
            
            # For each tick position, find the nearest resampled position and get its original index
            tick_labels = []
            adjusted_positions = []
            
            for pos in tick_positions:
                # Find nearest resampled position
                nearest_idx = np.argmin(np.abs(np.array(resampled_positions) - pos))
                nearest_position = resampled_positions[nearest_idx]
                original_index = original_indices[nearest_idx]
                
                # Store for plotting
                adjusted_positions.append(nearest_position)
                tick_labels.append(f"{int(original_index)}")
            
            # Set the top x-axis
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(adjusted_positions)
            ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel('Global File Index')
        
        # Set plot labels and title
        ax.set_xlabel('Sample Position')
        ax.set_ylabel(channel_name)
        
        # Create title with channel name, folder name and label
        plot_title = f"{channel_name.replace('_', ' ').title()} Channel"
        if folder_name:
            plot_title = f"{plot_title} - {folder_name}"
        if label:
            plot_title = f"{plot_title} (Label: {label})"
        
        ax.set_title(plot_title)
        
        # Add legend for colors
        handles = [plt.Line2D([0], [0], color='green', label='Left (pre-transient)'),
                  plt.Line2D([0], [0], color='red', label='Transient'),
                  plt.Line2D([0], [0], color='blue', label='Right (post-transient)')]
        ax.legend(handles=handles)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved augmented data plot for {channel_name} to {save_path}")
        
        return fig
        
    def plot_raw_data_subplots(self,
                            original_data: Dict[str, Any],
                            max_points: int = 4000,
                            folder_name: str = None,
                            file_prefix: str = "",
                            label: str = None,
                            save_path: str = None):
        """
        Create a visualization with 3 subplots showing original raw data for source_current, 
        load_voltage, and source_voltage.
        
        Args:
            original_data: Original data dictionary from loader
            max_points: Maximum number of points to show
            folder_name: Folder name for the title
            file_prefix: Prefix for output files
            label: Resolved label for the title
            save_path: Path to save the figure
            
        Returns:
            The matplotlib figure
        """
        # Check if required channels exist
        required_channels = ['source_current', 'load_voltage', 'source_voltage']
        
        if 'channels' not in original_data:
            logger.warning("No channels in original data")
            return None
            
        missing_channels = [ch for ch in required_channels if ch not in original_data['channels']]
        if missing_channels:
            logger.warning(f"Missing required channels: {missing_channels}")
            return None
            
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'hspace': 0.3})
        
        # Plot order: source_current (top), load_voltage (middle), source_voltage (bottom)
        plot_order = ['source_current', 'load_voltage', 'source_voltage']
        
        # For each channel
        for i, channel in enumerate(plot_order):
            ax = axes[i]
            
            # Get channel data
            channel_data = original_data['channels'][channel]
            
            # Downsample if needed
            if len(channel_data) > max_points:
                step = len(channel_data) // max_points
                indices = np.arange(0, len(channel_data), step)[:max_points]
                downsampled_data = channel_data[indices]
            else:
                indices = np.arange(len(channel_data))
                downsampled_data = channel_data
            
            # Plot the data
            ax.plot(indices, downsampled_data, color='blue')
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set y-axis label
            ax.set_ylabel(channel)
            
            # Add title to top subplot
            if i == 0:
                main_title = f"Raw Data - {folder_name}"
                if label:
                    main_title += f" (Label: {label})"
                ax.set_title(main_title)
        
        # Set x-axis label for bottom subplot only
        axes[2].set_xlabel('Sample Index')
        
        # Add 10 evenly spaced tick marks
        total_points = len(original_data['channels'][plot_order[0]])
        tick_positions = np.linspace(0, total_points-1, 10)
        tick_labels = [f"{int(pos)}" for pos in tick_positions]
        
        # Set x ticks
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels(tick_labels)
        
        # Auto-adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved raw data subplots to {save_path}")
        
        return fig
    
    def plot_all_channels(self,
                         original_data: Dict[str, Any],
                         augmented_files: List[str],
                         channels: List[str] = None,
                         max_points: int = 4000,
                         folder_name: str = None,
                         file_prefix: str = "",
                         label: str = None,
                         save_dir: str = None):
        """
        Create visualizations for multiple channels of augmented data.
        
        Args:
            original_data: Original data dictionary from loader
            augmented_files: List of augmented file paths generated
            channels: List of channels to visualize (default: source_current, source_voltage, load_voltage)
            max_points: Maximum number of points to show per visualization
            folder_name: Folder name for the title
            file_prefix: Prefix for output files (e.g., "raw_")
            label: Resolved label for the title
            save_dir: Directory to save the figures
            
        Returns:
            Dictionary of matplotlib figures by channel name
        """
        # Use default channels if none specified
        if not channels:
            channels = ['source_current', 'source_voltage', 'load_voltage']
        
        # Create visualizations for each channel
        figures = {}
        for channel in channels:
            logger.info(f"Generating visualization for channel: {channel}")
            
            # Create file path if save_dir is provided
            save_path = None
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # Construct filename according to 'raw_<folder name>_<channel>.png' format
                filename = f"{file_prefix}{folder_name}_{channel}.png"
                save_path = os.path.join(save_dir, filename)
            
            # Generate visualization for this channel
            fig = self.plot_augmented_data(
                original_data,
                augmented_files,
                channel_name=channel,
                max_points=max_points,
                title=None,  # Let plot_augmented_data create the title with channel name
                folder_name=folder_name,
                label=label,
                save_path=save_path
            )
            
            figures[channel] = fig
        
        return figures
        
    def plot_chunk_boundaries(self, 
                             chunked_data: Dict[str, Any],
                             aligned_data: Optional[Dict[str, Any]] = None,
                             channel_name: Optional[str] = None,
                             title: str = None,
                             save_path: str = None):
        """
        Plot chunk boundaries over aligned data for visualization.
        
        Args:
            chunked_data: Chunked data from chunk_aligned_data
            aligned_data: Optional aligned data for background visualization
            channel_name: Channel to plot (default: first channel)
            title: Optional title for the plot
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib figure
        """
        if not chunked_data or 'chunks' not in chunked_data or not chunked_data['chunks']:
            logger.warning("Cannot create plot: invalid chunked data")
            return None
        
        # Get chunks and indices
        chunks = chunked_data['chunks']
        chunk_indices = chunked_data.get('chunk_indices', [])
        
        # If chunk_indices not available, extract from chunks
        if not chunk_indices and chunks:
            chunk_indices = [(chunk.get('start', 0), chunk.get('end', 0)) for chunk in chunks]
        
        # Prepare figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot aligned data if available
        if aligned_data and 'channels' in aligned_data and aligned_data['channels']:
            channels = aligned_data['channels']
            
            # Select channel to plot
            if not channel_name or channel_name not in channels:
                channel_name = next(iter(channels.keys()))
                
            channel_data = channels[channel_name]
            time_data = aligned_data.get('time_data', np.arange(len(channel_data)))
            
            # Plot channel data
            ax.plot(time_data, channel_data, 'k-', alpha=0.5, label=channel_name)
            
            # Add transient marker if available
            if 'alignment' in aligned_data and 'aligned_transient_index' in aligned_data['alignment']:
                transient_idx = aligned_data['alignment']['aligned_transient_index']
                if 0 <= transient_idx < len(time_data):
                    ax.axvline(x=time_data[transient_idx], color='r', linestyle='--', linewidth=2, 
                              alpha=0.7, label='Transient')
        
        # Plot chunk boundaries
        colors = plt.cm.jet(np.linspace(0, 1, len(chunk_indices)))
        
        for i, (start, end) in enumerate(chunk_indices):
            color = colors[i]
            
            # Plot vertical lines at chunk boundaries
            ax.axvspan(start, end, alpha=0.2, color=color)
            
            # Add chunk number
            chunk_center = (start + end) / 2
            ax.text(chunk_center, ax.get_ylim()[1] * 0.9, f"{i}", 
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Add labels and title
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Chunk Boundaries')
            
        ax.grid(True)
        ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved chunk boundary plot to {save_path}")
        
        return fig


# Convenience functions for full pipeline processing

def process_detected_transient(data, transient_result, 
                              chunk_size=1024, samples_before=105000, samples_after=105000,
                              group_size=3, group_step=1, normalize_chunks=True,
                              required_channels=None, labels=None):
    """
    Process detected transient through the full pipeline.
    
    Args:
        data: Data dictionary from the MATLAB loader
        transient_result: Transient detection result
        chunk_size: Size of data chunks
        samples_before: Number of samples before transient to include
        samples_after: Number of samples after transient to include
        group_size: Number of consecutive chunks in each group
        group_step: Step size between groups
        normalize_chunks: Whether to normalize chunks
        required_channels: List of required channels
        labels: Dictionary mapping regions to labels
        
    Returns:
        Dictionary with processed data through all stages
    """
    # Initialize processor
    processor = DataProcessor(
        chunk_size=chunk_size,
        samples_before=samples_before,
        samples_after=samples_after,
        normalize_chunks=normalize_chunks,
        required_channels=required_channels
    )
    
    # Extract aligned data
    aligned_data = processor.extract_aligned_data(data, transient_result)
    if not aligned_data:
        logger.error("Failed to extract aligned data")
        return None
    
    # Chunk aligned data
    chunked_data = processor.chunk_aligned_data(aligned_data)
    if not chunked_data:
        logger.error("Failed to chunk aligned data")
        return None
    
    # Form chunk groups
    grouped_data = processor.form_chunk_groups(chunked_data, group_size, group_step, labels)
    if not grouped_data:
        logger.error("Failed to form chunk groups")
        return None
    
    # Return all stages
    return {
        'aligned': aligned_data,
        'chunked': chunked_data,
        'grouped': grouped_data
    }