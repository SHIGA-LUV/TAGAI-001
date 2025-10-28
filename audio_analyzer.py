#!/usr/bin/env python3
"""
Advanced Audio Analysis using librosa
Phase 2: Deep audio feature extraction for AI tagging
"""

import librosa
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AudioAnalyzer:
    def __init__(self):
        """Initialize the audio analyzer with default parameters"""
        self.sample_rate = 22050  # Standard sample rate for music analysis
        self.hop_length = 512     # Standard hop length
        
        # Key mappings (Camelot wheel style)
        self.key_mappings = {
            0: 'C', 1: 'C#/Db', 2: 'D', 3: 'D#/Eb', 4: 'E', 5: 'F',
            6: 'F#/Gb', 7: 'G', 8: 'G#/Ab', 9: 'A', 10: 'A#/Bb', 11: 'B'
        }
        
        # Mode mappings
        self.mode_mappings = {0: 'minor', 1: 'major'}
        
        # Audio feature to tag mappings
        self.feature_tag_mappings = {
            'tempo': {
                'slow': ['5-After Hours', 'Dreamy', 'Emotional'],
                'medium': ['4-Cool Down', 'Deep House'],
                'fast': ['2-Build up', '3-Peak Time', 'Energetic']
            },
            'spectral_centroid': {
                'low': ['Dark', 'Deep House'],
                'medium': ['Progressive House'],
                'high': ['Synth Lead', 'Energetic']
            },
            'zero_crossing_rate': {
                'low': ['Piano', 'Strings'],
                'high': ['Percussion', 'Tribal']
            },
            'mfcc_variance': {
                'low': ['Dreamy', 'Emotional'],
                'high': ['Complex', 'Tribal']
            }
        }
    
    def load_audio(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load audio file using librosa
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) if error
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f'✅ Loaded audio: {len(y)} samples at {sr}Hz')
            return y, sr
        except Exception as e:
            print(f'❌ Error loading audio file {file_path}: {e}')
            return None, None
    
    def analyze_tempo_and_beats(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analyze tempo and beat information
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with tempo and beat analysis
        """
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Beat intervals
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Tempo stability (variance in beat intervals)
            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                tempo_stability = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                tempo_stability = 0.0
            
            return {
                'tempo': float(tempo),
                'beat_count': len(beats),
                'tempo_stability': float(tempo_stability),
                'beat_times': beat_times.tolist()[:10]  # First 10 beats
            }
        except Exception as e:
            print(f'❌ Tempo analysis error: {e}')
            return {'tempo': 0, 'beat_count': 0, 'tempo_stability': 0}
    
    def analyze_key_and_mode(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analyze musical key and mode
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with key and mode analysis
        """
        try:
            # Chromagram for key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Key detection using chroma features
            # This is a simplified approach - more sophisticated methods exist
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = np.argmax(chroma_mean)
            
            # Mode detection (major/minor) using harmonic analysis
            harmonic = librosa.effects.harmonic(y)
            chroma_harmonic = librosa.feature.chroma_stft(y=harmonic, sr=sr)
            
            # Simple major/minor classification based on chord patterns
            major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            chroma_norm = chroma_mean / np.sum(chroma_mean)
            major_corr = np.corrcoef(chroma_norm, major_template)[0, 1]
            minor_corr = np.corrcoef(chroma_norm, minor_template)[0, 1]
            
            mode_idx = 1 if major_corr > minor_corr else 0
            confidence = abs(major_corr - minor_corr)
            
            return {
                'key': self.key_mappings.get(key_idx, 'Unknown'),
                'mode': self.mode_mappings.get(mode_idx, 'Unknown'),
                'key_confidence': float(confidence),
                'chroma_vector': chroma_mean.tolist()
            }
        except Exception as e:
            print(f'❌ Key analysis error: {e}')
            return {'key': 'Unknown', 'mode': 'Unknown', 'key_confidence': 0}
    
    def analyze_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analyze spectral characteristics
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with spectral analysis
        """
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Zero crossing rate (percussiveness)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # MFCCs (timbral characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr)),
                'mfcc_means': np.mean(mfccs, axis=1).tolist(),
                'mfcc_stds': np.std(mfccs, axis=1).tolist(),
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms))
            }
        except Exception as e:
            print(f'❌ Spectral analysis error: {e}')
            return {}
    
    def analyze_harmonic_percussive(self, y: np.ndarray, sr: int) -> Dict:
        """
        Separate and analyze harmonic vs percussive components
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary with harmonic/percussive analysis
        """
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Energy ratios
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0
            percussive_ratio = percussive_energy / total_energy if total_energy > 0 else 0
            
            return {
                'harmonic_ratio': float(harmonic_ratio),
                'percussive_ratio': float(percussive_ratio),
                'harmonic_energy': float(harmonic_energy),
                'percussive_energy': float(percussive_energy)
            }
        except Exception as e:
            print(f'❌ Harmonic-percussive analysis error: {e}')
            return {}
    
    def comprehensive_audio_analysis(self, file_path: str) -> Dict:
        """
        Perform comprehensive audio analysis
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with all analysis results
        """
        print(f'🎵 Analyzing audio file: {os.path.basename(file_path)}')
        
        # Load audio
        y, sr = self.load_audio(file_path)
        if y is None:
            return {'error': 'Could not load audio file'}
        
        analysis = {
            'file_path': file_path,
            'duration': float(len(y) / sr),
            'sample_rate': sr
        }
        
        # Perform all analyses
        print('  🎵 Analyzing tempo and beats...')
        analysis.update(self.analyze_tempo_and_beats(y, sr))
        
        print('  🎹 Analyzing key and mode...')
        analysis.update(self.analyze_key_and_mode(y, sr))
        
        print('  🌊 Analyzing spectral features...')
        spectral_features = self.analyze_spectral_features(y, sr)
        analysis['spectral_features'] = spectral_features
        
        print('  🎶 Analyzing harmonic/percussive...')
        hp_features = self.analyze_harmonic_percussive(y, sr)
        analysis['harmonic_percussive'] = hp_features
        
        # Generate AI suggestions based on audio analysis
        print('  🤖 Generating AI suggestions...')
        ai_suggestions = self.generate_ai_suggestions_from_audio(analysis)
        analysis['librosa_ai_suggestions'] = ai_suggestions
        
        print('✅ Audio analysis complete!')
        return analysis
    
    def generate_ai_suggestions_from_audio(self, analysis: Dict) -> List[str]:
        """
        Generate AI tag suggestions based on audio analysis
        
        Args:
            analysis: Audio analysis results
            
        Returns:
            List of suggested tags
        """
        suggestions = []
        
        # Tempo-based suggestions
        tempo = analysis.get('tempo', 0)
        if tempo > 0:
            if tempo < 100:
                suggestions.extend(self.feature_tag_mappings['tempo']['slow'])
            elif tempo < 130:
                suggestions.extend(self.feature_tag_mappings['tempo']['medium'])
            else:
                suggestions.extend(self.feature_tag_mappings['tempo']['fast'])
        
        # Spectral centroid (brightness) suggestions
        spectral_features = analysis.get('spectral_features', {})
        centroid_mean = spectral_features.get('spectral_centroid_mean', 0)
        if centroid_mean > 0:
            if centroid_mean < 2000:
                suggestions.extend(self.feature_tag_mappings['spectral_centroid']['low'])
            elif centroid_mean < 4000:
                suggestions.extend(self.feature_tag_mappings['spectral_centroid']['medium'])
            else:
                suggestions.extend(self.feature_tag_mappings['spectral_centroid']['high'])
        
        # Zero crossing rate (percussiveness) suggestions
        zcr_mean = spectral_features.get('zero_crossing_rate_mean', 0)
        if zcr_mean > 0:
            if zcr_mean < 0.1:
                suggestions.extend(self.feature_tag_mappings['zero_crossing_rate']['low'])
            else:
                suggestions.extend(self.feature_tag_mappings['zero_crossing_rate']['high'])
        
        # Harmonic vs percussive suggestions
        hp_features = analysis.get('harmonic_percussive', {})
        percussive_ratio = hp_features.get('percussive_ratio', 0)
        if percussive_ratio > 0.3:
            suggestions.extend(['Tribal', 'Percussion', 'Energetic'])
        
        harmonic_ratio = hp_features.get('harmonic_ratio', 0)
        if harmonic_ratio > 0.7:
            suggestions.extend(['Piano', 'Strings', 'Melodic Techno'])
        
        # Key-based suggestions
        key = analysis.get('key', '')
        mode = analysis.get('mode', '')
        if mode == 'minor':
            suggestions.extend(['Dark', 'Emotional'])
        elif mode == 'major':
            suggestions.extend(['Uplifting', 'Energetic'])
        
        # Remove duplicates and return
        return list(set(suggestions))
    
    def batch_analyze_tracks(self, track_paths: List[str], max_tracks: int = 10) -> Dict:
        """
        Analyze multiple audio tracks
        
        Args:
            track_paths: List of audio file paths
            max_tracks: Maximum number of tracks to analyze
            
        Returns:
            Dictionary of analysis results
        """
        results = {}
        
        print(f'🎵 Batch analyzing {min(len(track_paths), max_tracks)} tracks...')
        
        for i, track_path in enumerate(track_paths[:max_tracks]):
            print(f'
Track {i+1}/{min(len(track_paths), max_tracks)}')
            analysis = self.comprehensive_audio_analysis(track_path)
            results[track_path] = analysis
        
        print(f'
✅ Batch analysis complete! Analyzed {len(results)} tracks.')
        return results

# Example usage and testing
if __name__ == '__main__':
    print('🎵 Audio Analyzer - Test Mode')
    print('=' * 40)
    
    analyzer = AudioAnalyzer()
    
    print('
📋 Available Analysis Features:')
    print('- Tempo and beat tracking')
    print('- Key and mode detection')
    print('- Spectral analysis (brightness, timbre)')
    print('- Harmonic vs percussive separation')
    print('- AI tag suggestions from audio features')
    
    print('
🔧 Feature to Tag Mappings:')
    for feature, mappings in analyzer.feature_tag_mappings.items():
        print(f'   {feature}: {list(mappings.keys())}')
    
    print('
⚠️ To analyze audio files:')
    print('1. Install librosa: pip install librosa')
    print('2. Use: analyzer.comprehensive_audio_analysis(file_path)')
    print('3. Supported formats: WAV, MP3, FLAC, etc.')
