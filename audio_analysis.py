#!/usr/bin/env python3
"""
AI MyTag DJ Assistant - Audio Analysis with librosa
Phase 2: Direct audio file analysis for enhanced tagging
"""

import os
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Audio analysis libraries
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa not available. Install with: pip install librosa")
    LIBROSA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import the original tagger
from rekordbox_ai_tagger import RekordboxAITagger

class AudioAnalyzer:
    def __init__(self):
        """Initialize audio analyzer"""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio analysis. Install with: pip install librosa")
            
        self.tagger = RekordboxAITagger()
        
        # Audio analysis parameters
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        
        # Feature thresholds for tag mapping
        self.thresholds = {
            'energy': {'high': 0.7, 'medium': 0.4},
            'tempo': {'slow': 100, 'medium': 120, 'fast': 130},
            'spectral_centroid': {'bright': 3000, 'warm': 1500},
            'zero_crossing_rate': {'percussive': 0.1, 'smooth': 0.05},
            'mfcc_variance': {'complex': 50, 'simple': 20},
            'harmonic_ratio': {'harmonic': 0.7, 'percussive': 0.3}
        }
        
        # Tag mappings based on audio features
        self.feature_tag_mappings = {
            'energy_high': ['3-Peak Time', 'Energetic'],
            'energy_medium': ['2-Build up'],
            'energy_low': ['4-Cool Down', '5-After Hours', 'Dreamy'],
            'tempo_slow': ['5-After Hours', 'Dreamy'],
            'tempo_medium': ['1-Opener', '2-Build up'],
            'tempo_fast': ['3-Peak Time', 'Energetic'],
            'bright_sound': ['Synth Lead'],
            'warm_sound': ['Piano', 'Strings'],
            'percussive': ['Darbuka', 'Percussion'],
            'harmonic': ['Piano', 'Strings'],
            'complex_timbre': ['Ethnic House'],
            'vocal_detected': ['Female Vocal', 'Male Vocal'],
            'minor_key': ['Dark', 'Emotional'],
            'major_key': ['Uplifting']
        }
        
    def load_audio(self, file_path: str, duration: Optional[float] = None, offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration, offset=offset)
            return y, sr
        except Exception as e:
            raise Exception(f"Failed to load audio file {file_path}: {e}")
            
    def extract_basic_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract basic audio features"""
        features = {}
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        features['tempo'] = float(tempo)
        features['beat_count'] = len(beats)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        features['rms_energy_mean'] = float(np.mean(rms))
        features['rms_energy_std'] = float(np.std(rms))
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        return features
        
    def extract_advanced_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract advanced audio features"""
        features = {}
        
        # Harmonic and percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic-to-percussive ratio
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        
        if total_energy > 0:
            features['harmonic_ratio'] = float(harmonic_energy / total_energy)
            features['percussive_ratio'] = float(percussive_energy / total_energy)
        else:
            features['harmonic_ratio'] = 0.0
            features['percussive_ratio'] = 0.0
        
        # Chroma features (key detection)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        features['chroma_mean'] = [float(np.mean(chroma[i])) for i in range(12)]
        
        # Dominant key estimation
        chroma_mean = np.mean(chroma, axis=1)
        dominant_key = np.argmax(chroma_mean)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features['estimated_key'] = key_names[dominant_key]
        
        # Key strength (how pronounced the key is)
        features['key_strength'] = float(np.max(chroma_mean) / np.mean(chroma_mean))
        
        # Tonnetz (harmonic network)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = [float(np.mean(tonnetz[i])) for i in range(6)]
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = float(len(onset_frames) / (len(y) / sr))  # onsets per second
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        features['spectral_contrast_mean'] = [float(np.mean(contrast[i])) for i in range(7)]
        
        return features
        
    def detect_vocals(self, y: np.ndarray, sr: int) -> Dict:
        """Detect presence of vocals using spectral analysis"""
        # Separate vocals using harmonic-percussive separation and spectral subtraction
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Vocal frequency range analysis (roughly 80Hz - 1100Hz for human voice)
        stft = librosa.stft(y, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Frequency bins corresponding to vocal range
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
        vocal_bins = np.where((freqs >= 80) & (freqs <= 1100))[0]
        
        # Energy in vocal frequency range
        vocal_energy = np.sum(magnitude[vocal_bins, :], axis=0)
        total_energy = np.sum(magnitude, axis=0)
        
        vocal_ratio = np.mean(vocal_energy / (total_energy + 1e-8))
        
        # Spectral centroid in vocal range (indicator of vocal presence)
        vocal_spectral_centroid = np.mean([
            np.sum(freqs[vocal_bins] * magnitude[vocal_bins, i]) / (np.sum(magnitude[vocal_bins, i]) + 1e-8)
            for i in range(magnitude.shape[1])
        ])
        
        return {
            'vocal_ratio': float(vocal_ratio),
            'vocal_spectral_centroid': float(vocal_spectral_centroid),
            'has_vocals': vocal_ratio > 0.15  # Threshold for vocal detection
        }
        
    def analyze_structure(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze track structure (intro, verse, chorus, etc.)"""
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # Segment the track using beat-synchronous features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        chroma_sync = librosa.util.sync(chroma, beats)
        
        # Compute recurrence matrix
        R = librosa.segment.recurrence_matrix(chroma_sync, width=3, mode='affinity', sym=True)
        
        # Detect segments
        boundaries = librosa.segment.agglomerative(chroma_sync, k=8)
        bound_times = librosa.frames_to_time(boundaries, sr=sr, hop_length=self.hop_length)
        
        # Estimate track sections
        track_duration = len(y) / sr
        sections = []
        
        for i, boundary in enumerate(bound_times):
            if i == 0:
                section_type = 'intro'
            elif i == len(bound_times) - 1:
                section_type = 'outro'
            elif boundary < track_duration * 0.3:
                section_type = 'verse'
            elif boundary > track_duration * 0.7:
                section_type = 'breakdown'
            else:
                section_type = 'chorus'
                
            sections.append({
                'start_time': float(boundary),
                'type': section_type
            })
        
        return {
            'estimated_sections': sections,
            'num_sections': len(sections),
            'track_duration': float(track_duration)
        }
        
    def generate_tags_from_features(self, features: Dict) -> Tuple[List[str], float]:
        """Generate tags based on extracted audio features"""
        suggested_tags = []
        confidence_scores = []
        
        # Energy-based tags
        rms_energy = features.get('rms_energy_mean', 0)
        if rms_energy > self.thresholds['energy']['high']:
            suggested_tags.extend(self.feature_tag_mappings['energy_high'])
            confidence_scores.append(0.8)
        elif rms_energy > self.thresholds['energy']['medium']:
            suggested_tags.extend(self.feature_tag_mappings['energy_medium'])
            confidence_scores.append(0.7)
        else:
            suggested_tags.extend(self.feature_tag_mappings['energy_low'])
            confidence_scores.append(0.6)
        
        # Tempo-based tags
        tempo = features.get('tempo', 0)
        if tempo > 0:
            if tempo < self.thresholds['tempo']['slow']:
                suggested_tags.extend(self.feature_tag_mappings['tempo_slow'])
                confidence_scores.append(0.8)
            elif tempo < self.thresholds['tempo']['medium']:
                suggested_tags.extend(self.feature_tag_mappings['tempo_medium'])
                confidence_scores.append(0.7)
            else:
                suggested_tags.extend(self.feature_tag_mappings['tempo_fast'])
                confidence_scores.append(0.8)
        
        # Spectral characteristics
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        if spectral_centroid > self.thresholds['spectral_centroid']['bright']:
            suggested_tags.extend(self.feature_tag_mappings['bright_sound'])
            confidence_scores.append(0.6)
        elif spectral_centroid > self.thresholds['spectral_centroid']['warm']:
            suggested_tags.extend(self.feature_tag_mappings['warm_sound'])
            confidence_scores.append(0.7)
        
        # Harmonic vs percussive content
        harmonic_ratio = features.get('harmonic_ratio', 0.5)
        if harmonic_ratio > self.thresholds['harmonic_ratio']['harmonic']:
            suggested_tags.extend(self.feature_tag_mappings['harmonic'])
            confidence_scores.append(0.7)
        elif harmonic_ratio < self.thresholds['harmonic_ratio']['percussive']:
            suggested_tags.extend(self.feature_tag_mappings['percussive'])
            confidence_scores.append(0.8)
        
        # Vocal detection
        if features.get('has_vocals', False):
            suggested_tags.extend(self.feature_tag_mappings['vocal_detected'])
            confidence_scores.append(0.6)
        
        # Key-based mood tags
        estimated_key = features.get('estimated_key', '')
        if estimated_key:
            # Simple major/minor detection based on key
            minor_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Simplified
            if any(key in estimated_key for key in ['m', 'minor']):
                suggested_tags.extend(self.feature_tag_mappings['minor_key'])
                confidence_scores.append(0.5)
            else:
                suggested_tags.extend(self.feature_tag_mappings['major_key'])
                confidence_scores.append(0.5)
        
        # Timbral complexity
        mfcc_variance = np.mean([features.get(f'mfcc_{i}_std', 0) for i in range(13)])
        if mfcc_variance > self.thresholds['mfcc_variance']['complex']:
            suggested_tags.extend(self.feature_tag_mappings['complex_timbre'])
            confidence_scores.append(0.6)
        
        # Remove duplicates and sort by hierarchy
        unique_tags = list(set(suggested_tags))
        sorted_tags = self.tagger.sort_tags_by_hierarchy(unique_tags)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return sorted_tags, overall_confidence
        
    def analyze_audio_file(self, file_path: str, duration: Optional[float] = 30.0) -> Dict:
        """Analyze an audio file and return comprehensive analysis"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        print(f"🎵 Analyzing audio file: {Path(file_path).name}")
        
        # Load audio (analyze first 30 seconds by default for efficiency)
        y, sr = self.load_audio(file_path, duration=duration)
        
        analysis_result = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_rate': sr,
            'duration_analyzed': float(len(y) / sr)
        }
        
        try:
            # Extract features
            print("  📊 Extracting basic features...")
            basic_features = self.extract_basic_features(y, sr)
            
            print("  🔬 Extracting advanced features...")
            advanced_features = self.extract_advanced_features(y, sr)
            
            print("  🎤 Detecting vocals...")
            vocal_features = self.detect_vocals(y, sr)
            
            print("  🏗️ Analyzing structure...")
            structure_features = self.analyze_structure(y, sr)
            
            # Combine all features
            all_features = {**basic_features, **advanced_features, **vocal_features, **structure_features}
            analysis_result['audio_features'] = all_features
            
            # Generate tags
            print("  🏷️ Generating tags...")
            suggested_tags, confidence = self.generate_tags_from_features(all_features)
            
            analysis_result['suggested_tags'] = suggested_tags
            analysis_result['confidence'] = confidence
            
            print(f"  ✅ Analysis complete. Suggested tags: {suggested_tags}")
            print(f"     Confidence: {confidence:.1%}")
            
        except Exception as e:
            analysis_result['error'] = str(e)
            print(f"  ❌ Analysis failed: {e}")
        
        return analysis_result
        
    def batch_analyze_audio_files(self, file_paths: List[str], duration: Optional[float] = 30.0) -> Dict[str, Dict]:
        """Analyze multiple audio files in batch"""
        results = {}
        total_files = len(file_paths)
        
        print(f"🎵 Starting batch audio analysis for {total_files} files...")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"
[{i}/{total_files}] Processing: {Path(file_path).name}")
            
            try:
                result = self.analyze_audio_file(file_path, duration)
                results[file_path] = result
            except Exception as e:
                print(f"  ❌ Failed to analyze {file_path}: {e}")
                results[file_path] = {
                    'error': str(e),
                    'analysis_timestamp': datetime.now().isoformat()
                }
        
        successful_analyses = len([r for r in results.values() if 'suggested_tags' in r])
        print(f"
✅ Batch analysis complete. Successfully analyzed {successful_analyses}/{total_files} files.")
        
        return results
        
    def save_analysis_results(self, results: Dict, output_file: str = 'audio_analysis_results.json'):
        """Save analysis results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Analysis results saved to {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def demo_audio_analysis():
    """Demo function to test audio analysis"""
    print("🎵 AI MyTag DJ Assistant - Audio Analysis Demo")
    print("=" * 50)
    
    if not LIBROSA_AVAILABLE:
        print("❌ librosa not available. Install with: pip install librosa")
        return
    
    try:
        analyzer = AudioAnalyzer()
        
        # Look for audio files in current directory
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
        current_dir = Path('.')
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(current_dir.glob(f'*{ext}'))
            audio_files.extend(current_dir.glob(f'*{ext.upper()}'))
        
        if not audio_files:
            print("❌ No audio files found in current directory")
            print("   Supported formats: .mp3, .wav, .flac, .m4a, .aac")
            return
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Analyze first few files as demo
        demo_files = [str(f) for f in audio_files[:3]]
        results = analyzer.batch_analyze_audio_files(demo_files, duration=15.0)
        
        # Save results
        analyzer.save_analysis_results(results)
        
        # Print summary
        print("
📊 Analysis Summary:")
        for file_path, result in results.items():
            if 'suggested_tags' in result:
                print(f"  {Path(file_path).name}:")
                print(f"    Tags: {result['suggested_tags']}")
                print(f"    Confidence: {result['confidence']:.1%}")
                print(f"    Tempo: {result['audio_features'].get('tempo', 'N/A'):.0f} BPM")
                print(f"    Key: {result['audio_features'].get('estimated_key', 'N/A')}")
            else:
                print(f"  {Path(file_path).name}: Analysis failed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_audio_analysis()
