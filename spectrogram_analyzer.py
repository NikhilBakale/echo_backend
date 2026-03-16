import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
import librosa
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EnhancedCallParameters:
    """Enhanced call parameters matching bat bioacoustics standards"""
    start_frequency: float      # Frequency at call start (kHz)
    end_frequency: float        # Frequency at call end (kHz)
    minimum_frequency: float    # Lowest frequency in call (kHz)
    maximum_frequency: float    # Highest frequency in call (kHz)
    peak_frequency: float       # Frequency with maximum energy (kHz)
    bandwidth: float            # Max - Min frequency (kHz)
    call_length: float          # Duration of single call (ms)
    call_distance: float        # Time between consecutive calls (ms)
    pulse_count: int            # Number of pulses detected
    intensity: float            # Mean amplitude (dB)
    sonotype: str              # Call classification (cf-e, cf-n, fm-l, fm-d, fm-a)
    frequency_modulation_rate: float  # Rate of frequency change (kHz/ms)
    
    # Additional metrics
    characteristic_frequency: float  # For CF calls
    knee_frequency: Optional[float]  # Transition point in QCF calls
    slope: float                     # Linear slope for FM calls


class SpectrogramParameterExtractor:
    """
    Advanced spectrogram analysis using image processing and signal analysis
    Extracts bat call parameters according to bioacoustic standards
    """
    
    def __init__(self, 
                 intensity_threshold_db: float = -40,
                 min_call_duration_ms: float = 0.5,
                 max_call_gap_ms: float = 50):
        """
        Args:
            intensity_threshold_db: Threshold for detecting call presence
            min_call_duration_ms: Minimum duration to consider as valid call
            max_call_gap_ms: Maximum gap to merge into single call
        """
        self.intensity_threshold_db = intensity_threshold_db
        self.min_call_duration_ms = min_call_duration_ms
        self.max_call_gap_ms = max_call_gap_ms
    
    def extract_parameters(self, audio_path: Path) -> EnhancedCallParameters:
        """Main extraction pipeline"""
        # Load audio and generate spectrogram
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate high-resolution spectrogram
        n_fft = 2048
        hop_length = 256  # Higher temporal resolution
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft) / 1000.0  # Convert to kHz
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
        
        # Apply bandpass filter (bat call range)
        freq_mask = (freqs >= 10) & (freqs <= 250)
        S_db_filtered = S_db[freq_mask, :]
        freqs_filtered = freqs[freq_mask]
        
        # Detect call segments
        call_segments = self._detect_call_segments(S_db_filtered, times)
        
        if not call_segments:
            return self._get_default_parameters()
        
        # Analyze primary call (longest or strongest)
        primary_segment = self._select_primary_segment(call_segments, S_db_filtered)
        
        # Extract frequency parameters
        freq_params = self._extract_frequency_parameters(
            S_db_filtered, freqs_filtered, times, primary_segment
        )
        
        # Extract temporal parameters
        temporal_params = self._extract_temporal_parameters(
            call_segments, times, sr, hop_length
        )
        
        # Classify sonotype
        sonotype = self._classify_sonotype(
            freq_params, temporal_params, S_db_filtered, primary_segment
        )
        
        # Calculate intensity
        intensity = self._calculate_intensity(S_db_filtered, primary_segment)
        
        # Combine all parameters
        return EnhancedCallParameters(
            start_frequency=freq_params['start_freq'],
            end_frequency=freq_params['end_freq'],
            minimum_frequency=freq_params['min_freq'],
            maximum_frequency=freq_params['max_freq'],
            peak_frequency=freq_params['peak_freq'],
            bandwidth=freq_params['bandwidth'],
            call_length=temporal_params['call_length'],
            call_distance=temporal_params['call_distance'],
            pulse_count=temporal_params['pulse_count'],
            intensity=intensity,
            sonotype=sonotype,
            frequency_modulation_rate=freq_params['fm_rate'],
            characteristic_frequency=freq_params.get('cf_freq', 0.0),
            knee_frequency=freq_params.get('knee_freq'),
            slope=freq_params.get('slope', 0.0)
        )
    
    def _detect_call_segments(self, S_db: np.ndarray, times: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect individual call segments using image processing
        Returns list of (start_idx, end_idx) tuples
        """
        # Create binary mask of call presence
        binary_mask = S_db > self.intensity_threshold_db
        
        # Collapse frequency dimension to get temporal energy profile
        temporal_energy = np.sum(binary_mask, axis=0)
        
        # Normalize
        if temporal_energy.max() > 0:
            temporal_energy = temporal_energy / temporal_energy.max()
        
        # Threshold to find call regions
        call_presence = temporal_energy > 0.1
        
        # Find connected components (call segments)
        labeled, num_features = ndimage.label(call_presence)
        
        segments = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) > 0:
                start_idx = indices[0]
                end_idx = indices[-1]
                
                # Filter by minimum duration
                duration_ms = (times[end_idx] - times[start_idx]) * 1000
                if duration_ms >= self.min_call_duration_ms:
                    segments.append((start_idx, end_idx))
        
        # Merge nearby segments (likely same call)
        segments = self._merge_close_segments(segments, times)
        
        return segments
    
    def _merge_close_segments(self, segments: List[Tuple[int, int]], 
                              times: np.ndarray) -> List[Tuple[int, int]]:
        """Merge segments that are close together"""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            gap_ms = (times[current[0]] - times[last[1]]) * 1000
            
            if gap_ms <= self.max_call_gap_ms:
                # Merge with previous
                merged[-1] = (last[0], current[1])
            else:
                merged.append(current)
        
        return merged
    
    def _select_primary_segment(self, segments: List[Tuple[int, int]], 
                                S_db: np.ndarray) -> Tuple[int, int]:
        """Select the strongest or longest segment as primary"""
        if len(segments) == 1:
            return segments[0]
        
        # Score each segment by total energy
        scores = []
        for start, end in segments:
            segment_data = S_db[:, start:end+1]
            energy = np.sum(segment_data[segment_data > self.intensity_threshold_db])
            scores.append(energy)
        
        return segments[np.argmax(scores)]
    
    def _extract_frequency_parameters(self, S_db: np.ndarray, freqs: np.ndarray,
                                     times: np.ndarray, segment: Tuple[int, int]) -> Dict:
        """Extract all frequency-related parameters"""
        start_idx, end_idx = segment
        call_data = S_db[:, start_idx:end_idx+1]
        call_freqs = freqs
        call_times = times[start_idx:end_idx+1]
        
        # Get frequency contour (peak frequency at each time point)
        freq_contour = []
        for t_idx in range(call_data.shape[1]):
            time_slice = call_data[:, t_idx]
            peak_idx = np.argmax(time_slice)
            freq_contour.append(call_freqs[peak_idx])
        
        freq_contour = np.array(freq_contour)
        
        # Smooth the contour to reduce noise
        if len(freq_contour) > 5:
            from scipy.ndimage import gaussian_filter1d
            freq_contour = gaussian_filter1d(freq_contour, sigma=2)
        
        # Start frequency (first 10% of call)
        start_portion = int(len(freq_contour) * 0.1) or 1
        start_freq = np.median(freq_contour[:start_portion])
        
        # End frequency (last 10% of call)
        end_portion = int(len(freq_contour) * 0.1) or 1
        end_freq = np.median(freq_contour[-end_portion:])
        
        # Min and Max frequencies across entire call
        # Consider only frequencies with significant energy
        energy_mask = call_data > (self.intensity_threshold_db + 10)
        freq_indices_with_energy = np.where(np.any(energy_mask, axis=1))[0]
        
        if len(freq_indices_with_energy) > 0:
            min_freq = call_freqs[freq_indices_with_energy[0]]
            max_freq = call_freqs[freq_indices_with_energy[-1]]
        else:
            min_freq = freq_contour.min()
            max_freq = freq_contour.max()
        
        # Peak frequency (maximum energy across all time)
        max_energy_per_freq = np.max(call_data, axis=1)
        peak_freq_idx = np.argmax(max_energy_per_freq)
        peak_freq = call_freqs[peak_freq_idx]
        
        # Bandwidth
        bandwidth = max_freq - min_freq
        
        # Frequency modulation rate
        if len(call_times) > 1:
            freq_change = abs(end_freq - start_freq)
            time_duration = (call_times[-1] - call_times[0]) * 1000  # ms
            fm_rate = freq_change / time_duration if time_duration > 0 else 0
        else:
            fm_rate = 0
        
        # For CF calls, find characteristic frequency
        cf_freq = peak_freq if bandwidth < 5 else 0
        
        # Calculate slope for linear FM
        if len(freq_contour) > 2:
            slope, _ = np.polyfit(np.arange(len(freq_contour)), freq_contour, 1)
        else:
            slope = 0
        
        # Detect knee frequency (for QCF/CF-FM calls)
        knee_freq = self._detect_knee_frequency(freq_contour, call_freqs) if bandwidth > 2 else None
        
        return {
            'start_freq': round(start_freq, 2),
            'end_freq': round(end_freq, 2),
            'min_freq': round(min_freq, 2),
            'max_freq': round(max_freq, 2),
            'peak_freq': round(peak_freq, 2),
            'bandwidth': round(bandwidth, 2),
            'fm_rate': round(fm_rate, 3),
            'cf_freq': round(cf_freq, 2),
            'knee_freq': round(knee_freq, 2) if knee_freq else None,
            'slope': round(slope, 3)
        }
    
    def _detect_knee_frequency(self, freq_contour: np.ndarray, freqs: np.ndarray) -> Optional[float]:
        """Detect knee point in QCF calls where frequency changes slope"""
        if len(freq_contour) < 10:
            return None
        
        # Calculate second derivative to find inflection point
        gradient = np.gradient(freq_contour)
        second_derivative = np.gradient(gradient)
        
        # Find point of maximum curvature
        abs_curvature = np.abs(second_derivative)
        if abs_curvature.max() > 0.1:  # Significant curvature
            knee_idx = np.argmax(abs_curvature)
            return freq_contour[knee_idx]
        
        return None
    
    def _extract_temporal_parameters(self, segments: List[Tuple[int, int]], 
                                    times: np.ndarray, sr: int, 
                                    hop_length: int) -> Dict:
        """Extract temporal parameters"""
        pulse_count = len(segments)
        
        if pulse_count == 0:
            return {
                'call_length': 0.0,
                'call_distance': 0.0,
                'pulse_count': 0
            }
        
        # Call length (duration of primary call)
        primary = segments[0]
        call_length = (times[primary[1]] - times[primary[0]]) * 1000  # ms
        
        # Call distance (inter-pulse interval)
        if pulse_count > 1:
            distances = []
            for i in range(len(segments) - 1):
                end_current = times[segments[i][1]]
                start_next = times[segments[i+1][0]]
                distance = (start_next - end_current) * 1000  # ms
                distances.append(distance)
            call_distance = np.mean(distances)
        else:
            call_distance = 0.0
        
        return {
            'call_length': round(call_length, 2),
            'call_distance': round(call_distance, 2),
            'pulse_count': pulse_count
        }
    
    def _classify_sonotype(self, freq_params: Dict, temporal_params: Dict,
                          S_db: np.ndarray, segment: Tuple[int, int]) -> str:
        """
        Classify call into sonotype categories based on frequency characteristics
        Categories: cf-e, cf-n, fm-l, fm-d, fm-a
        """
        bandwidth = freq_params['bandwidth']
        start_freq = freq_params['start_freq']
        end_freq = freq_params['end_freq']
        fm_rate = freq_params['fm_rate']
        
        freq_change = abs(end_freq - start_freq)
        
        # Constant Frequency (CF) calls
        if bandwidth < 2:
            return "cf-e"  # Exactly constant
        elif bandwidth < 5 and freq_change < 3:
            return "cf-n"  # Nearly constant (QCF)
        
        # Frequency Modulated (FM) calls
        else:
            # Determine if ascending or descending
            if end_freq > start_freq + 5:
                return "fm-a"  # Ascending
            elif start_freq > end_freq + 5:
                # Distinguish between linear and descending curved
                # Linear if slope is relatively constant
                if abs(freq_params['slope']) > 0.5 and bandwidth > 20:
                    return "fm-l"  # Linear (steep)
                else:
                    return "fm-d"  # Descending (curved)
            else:
                # Small change - classify as QCF or shallow FM
                if bandwidth < 10:
                    return "cf-n"  # QCF
                else:
                    return "fm-d"  # Shallow FM descent
    
    def _calculate_intensity(self, S_db: np.ndarray, segment: Tuple[int, int]) -> float:
        """Calculate mean intensity of call"""
        start_idx, end_idx = segment
        call_data = S_db[:, start_idx:end_idx+1]
        
        # Only consider values above threshold
        valid_data = call_data[call_data > self.intensity_threshold_db]
        
        if len(valid_data) > 0:
            return round(float(np.mean(valid_data)), 2)
        return 0.0
    
    def _get_default_parameters(self) -> EnhancedCallParameters:
        """Return default parameters when no call detected"""
        return EnhancedCallParameters(
            start_frequency=0.0,
            end_frequency=0.0,
            minimum_frequency=0.0,
            maximum_frequency=0.0,
            peak_frequency=0.0,
            bandwidth=0.0,
            call_length=0.0,
            call_distance=0.0,
            pulse_count=0,
            intensity=0.0,
            sonotype="unknown",
            frequency_modulation_rate=0.0,
            characteristic_frequency=0.0,
            knee_frequency=None,
            slope=0.0
        )


# Example usage function
def extract_enhanced_call_parameters(audio_path: Path) -> EnhancedCallParameters:
    """
    Convenience function to extract parameters from audio file
    
    Args:
        audio_path: Path to WAV file
        
    Returns:
        EnhancedCallParameters object with all extracted metrics
    """
    extractor = SpectrogramParameterExtractor(
        intensity_threshold_db=-40,
        min_call_duration_ms=0.5,
        max_call_gap_ms=50
    )
    
    return extractor.extract_parameters(audio_path)


# Integration function for your app.py
def get_enhanced_parameters_dict(audio_path: Path) -> dict:
    """
    Returns parameters in format compatible with existing CallParameters model
    Can be used as drop-in replacement for extract_call_parameters()
    """
    params = extract_enhanced_call_parameters(audio_path)
    
    return {
        'start_frequency': params.start_frequency,
        'end_frequency': params.end_frequency,
        'minimum_frequency': params.minimum_frequency,
        'maximum_frequency': params.maximum_frequency,
        'peak_frequency': params.peak_frequency,
        'bandwidth': params.bandwidth,
        'intensity': params.intensity,
        'pulse_duration': params.call_length,  # Renamed for compatibility
        'total_length': params.call_length * params.pulse_count,
        'shape': params.sonotype,
        'call_distance': params.call_distance,
        'pulse_count': params.pulse_count,
        'fm_rate': params.frequency_modulation_rate
    }