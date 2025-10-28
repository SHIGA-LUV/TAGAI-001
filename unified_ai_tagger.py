#!/usr/bin/env python3
"""
Unified AI Tagger System
Integrates all Phase 1 & 2 components into a cohesive system
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
sys.path.append('/Users/shiraazoulay')

# Import all our components
try:
    from rekordbox_ai_tagger import RekordboxAITagger
    CORE_TAGGER_AVAILABLE = True
except ImportError:
    CORE_TAGGER_AVAILABLE = False
    print('⚠️ Core tagger not available')

try:
    from realtime_vy_tagger import RealTimeVyTagger
    REALTIME_GUI_AVAILABLE = True
except ImportError:
    REALTIME_GUI_AVAILABLE = False
    print('⚠️ Real-time GUI not available')

try:
    from spotify_enhancer import SpotifyEnhancer
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    print('⚠️ Spotify enhancer not available')

try:
    from audio_analyzer import AudioAnalyzer
    AUDIO_ANALYZER_AVAILABLE = True
except ImportError:
    AUDIO_ANALYZER_AVAILABLE = False
    print('⚠️ Audio analyzer not available')

try:
    from ml_pattern_learner import MLPatternLearner
    ML_LEARNER_AVAILABLE = True
except ImportError:
    ML_LEARNER_AVAILABLE = False
    print('⚠️ ML pattern learner not available')

class UnifiedAITagger:
    def __init__(self, config: Dict = None):
        """
        Initialize the unified AI tagger system
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.components = {}
        self.capabilities = []
        
        # Initialize all available components
        self.initialize_components()
        
        # Track system state
        self.session_stats = {
            'tracks_processed': 0,
            'tags_suggested': 0,
            'user_interactions': 0,
            'session_start': datetime.now().isoformat()
        }
        
        print(f'🎆 Unified AI Tagger initialized with {len(self.capabilities)} capabilities')
    
    def initialize_components(self):
        """Initialize all available components"""
        print('🚀 Initializing AI Tagger Components...')
        
        # Core XML processor
        if CORE_TAGGER_AVAILABLE:
            self.components['core_tagger'] = RekordboxAITagger()
            self.capabilities.append('XML Processing')
            print('✅ Core XML processor initialized')
        
        # Real-time GUI
        if REALTIME_GUI_AVAILABLE:
            self.components['realtime_gui'] = RealTimeVyTagger()
            self.capabilities.append('Real-time GUI')
            print('✅ Real-time GUI initialized')
        
        # Spotify integration
        if SPOTIFY_AVAILABLE:
            spotify_config = self.config.get('spotify', {})
            self.components['spotify'] = SpotifyEnhancer(
                client_id=spotify_config.get('client_id'),
                client_secret=spotify_config.get('client_secret')
            )
            self.capabilities.append('Spotify Integration')
            print('✅ Spotify enhancer initialized')
        
        # Audio analysis
        if AUDIO_ANALYZER_AVAILABLE:
            self.components['audio_analyzer'] = AudioAnalyzer()
            self.capabilities.append('Audio Analysis')
            print('✅ Audio analyzer initialized')
        
        # ML pattern learning
        if ML_LEARNER_AVAILABLE:
            self.components['ml_learner'] = MLPatternLearner()
            self.capabilities.append('ML Pattern Learning')
            print('✅ ML pattern learner initialized')
    
    def process_rekordbox_xml(self, xml_file_path: str, enhance_with_ai: bool = True) -> Dict:
        """
        Process Rekordbox XML with full AI enhancement
        
        Args:
            xml_file_path: Path to Rekordbox XML file
            enhance_with_ai: Whether to enhance with AI features
            
        Returns:
            Dictionary of processed and enhanced tracks
        """
        print(f'🎵 Processing Rekordbox XML: {os.path.basename(xml_file_path)}')
        
        if not CORE_TAGGER_AVAILABLE:
            print('❌ Core tagger not available')
            return {}
        
        # Step 1: Parse XML
        core_tagger = self.components['core_tagger']
        tracks = core_tagger.parse_xml(xml_file_path)
        
        if not tracks:
            print('❌ No tracks found in XML')
            return {}
        
        print(f'✅ Parsed {len(tracks)} tracks from XML')
        
        if not enhance_with_ai:
            return tracks
        
        # Step 2: Enhance with AI features
        enhanced_tracks = self.enhance_tracks_with_ai(tracks)
        
        self.session_stats['tracks_processed'] += len(enhanced_tracks)
        
        return enhanced_tracks
    
    def enhance_tracks_with_ai(self, tracks: Dict, max_tracks: int = 50) -> Dict:
        """
        Enhance tracks with all available AI features
        
        Args:
            tracks: Dictionary of tracks to enhance
            max_tracks: Maximum number of tracks to enhance (for performance)
            
        Returns:
            Dictionary of enhanced tracks
        """
        print(f'🤖 Enhancing tracks with AI features...')
        
        enhanced_tracks = {}
        processed_count = 0
        
        for track_id, track_info in tracks.items():
            if processed_count >= max_tracks:
                print(f'⚠️ Reached processing limit ({max_tracks} tracks)')
                break
            
            enhanced_track = track_info.copy()
            
            # Step 1: Spotify enhancement
            if 'spotify' in self.components:
                try:
                    spotify_enhanced = self.components['spotify'].enhance_track_with_spotify(track_info)
                    enhanced_track.update(spotify_enhanced)
                    if 'spotify_ai_suggestions' in spotify_enhanced:
                        print(f'   🎵 Added Spotify AI suggestions for {track_info.get("title", "Unknown")}')
                except Exception as e:
                    print(f'   ⚠️ Spotify enhancement failed: {e}')
            
            # Step 2: Audio analysis (if audio file path available)
            if 'audio_analyzer' in self.components and 'location' in track_info:
                try:
                    # Convert Rekordbox file URL to local path
                    audio_path = self.convert_rekordbox_location_to_path(track_info['location'])
                    if audio_path and os.path.exists(audio_path):
                        audio_analysis = self.components['audio_analyzer'].comprehensive_audio_analysis(audio_path)
                        enhanced_track['audio_analysis'] = audio_analysis
                        if 'librosa_ai_suggestions' in audio_analysis:
                            print(f'   🌊 Added audio AI suggestions for {track_info.get("title", "Unknown")}')
                except Exception as e:
                    print(f'   ⚠️ Audio analysis failed: {e}')
            
            # Step 3: ML pattern predictions
            if 'ml_learner' in self.components:
                try:
                    ml_predictions = self.components['ml_learner'].predict_tags_for_track(enhanced_track)
                    enhanced_track['ml_predictions'] = ml_predictions
                    print(f'   🤖 Added ML predictions for {track_info.get("title", "Unknown")}')
                except Exception as e:
                    print(f'   ⚠️ ML prediction failed: {e}')
            
            # Step 4: Generate unified AI suggestions
            unified_suggestions = self.generate_unified_ai_suggestions(enhanced_track)
            enhanced_track['unified_ai_suggestions'] = unified_suggestions
            
            enhanced_tracks[track_id] = enhanced_track
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f'   Processed {processed_count} tracks...')
        
        print(f'✅ Enhanced {len(enhanced_tracks)} tracks with AI features')
        return enhanced_tracks
    
    def generate_unified_ai_suggestions(self, track_info: Dict) -> Dict:
        """
        Generate unified AI suggestions from all available sources
        
        Args:
            track_info: Enhanced track information
            
        Returns:
            Dictionary with unified suggestions and confidence scores
        """
        all_suggestions = []
        suggestion_sources = {}
        
        # Collect suggestions from all sources
        if 'spotify_ai_suggestions' in track_info:
            spotify_suggestions = track_info['spotify_ai_suggestions']
            all_suggestions.extend(spotify_suggestions)
            for tag in spotify_suggestions:
                suggestion_sources[tag] = suggestion_sources.get(tag, []) + ['spotify']
        
        if 'audio_analysis' in track_info and 'librosa_ai_suggestions' in track_info['audio_analysis']:
            audio_suggestions = track_info['audio_analysis']['librosa_ai_suggestions']
            all_suggestions.extend(audio_suggestions)
            for tag in audio_suggestions:
                suggestion_sources[tag] = suggestion_sources.get(tag, []) + ['audio_analysis']
        
        if 'ml_predictions' in track_info and 'pattern_based' in track_info['ml_predictions']:
            ml_suggestions = track_info['ml_predictions']['pattern_based']
            all_suggestions.extend(ml_suggestions)
            for tag in ml_suggestions:
                suggestion_sources[tag] = suggestion_sources.get(tag, []) + ['ml_patterns']
        
        # Calculate confidence scores based on source agreement
        tag_confidence = {}
        for tag, sources in suggestion_sources.items():
            # Higher confidence for tags suggested by multiple sources
            confidence = len(sources) / 3.0  # Max 3 sources
            tag_confidence[tag] = min(confidence, 1.0)
        
        # Remove duplicates and sort by confidence
        unique_suggestions = list(set(all_suggestions))
        sorted_suggestions = sorted(unique_suggestions, 
                                  key=lambda x: tag_confidence.get(x, 0), 
                                  reverse=True)
        
        self.session_stats['tags_suggested'] += len(sorted_suggestions)
        
        return {
            'suggested_tags': sorted_suggestions,
            'confidence_scores': tag_confidence,
            'sources': suggestion_sources,
            'total_sources': len([s for s in [
                'spotify_ai_suggestions' in track_info,
                'audio_analysis' in track_info,
                'ml_predictions' in track_info
            ] if s])
        }
    
    def record_user_feedback(self, track_info: Dict, suggested_tags: List[str], 
                           selected_tags: List[str]):
        """
        Record user feedback for learning
        
        Args:
            track_info: Track information
            suggested_tags: Tags suggested by AI
            selected_tags: Tags selected by user
        """
        if 'ml_learner' in self.components:
            self.components['ml_learner'].record_user_interaction(
                track_info, suggested_tags, selected_tags
            )
            self.session_stats['user_interactions'] += 1
            print(f'✅ Recorded user feedback for learning')
    
    def launch_realtime_gui(self):
        """Launch the real-time tagging GUI"""
        if not REALTIME_GUI_AVAILABLE:
            print('❌ Real-time GUI not available')
            return
        
        print('🚀 Launching real-time tagging GUI...')
        
        # Enhance the GUI with our unified system
        gui = self.components['realtime_gui']
        
        # Override the GUI's suggestion method with our unified approach
        original_analyze = gui.analyze_track_for_suggestions
        
        def enhanced_analyze(track_info):
            # Use our unified AI suggestions
            enhanced_track = {'title': track_info.get('title', ''),
                            'artist': track_info.get('artist', ''),
                            'bpm': track_info.get('bpm', 0),
                            'key': track_info.get('key', ''),
                            'genre': track_info.get('genre', '')}
            
            # Get suggestions from all available sources
            if 'spotify' in self.components:
                try:
                    enhanced_track = self.components['spotify'].enhance_track_with_spotify(enhanced_track)
                except:
                    pass
            
            if 'ml_learner' in self.components:
                try:
                    ml_predictions = self.components['ml_learner'].predict_tags_for_track(enhanced_track)
                    enhanced_track['ml_predictions'] = ml_predictions
                except:
                    pass
            
            unified_suggestions = self.generate_unified_ai_suggestions(enhanced_track)
            return unified_suggestions.get('suggested_tags', [])
        
        gui.analyze_track_for_suggestions = enhanced_analyze
        
        # Launch the enhanced GUI
        from realtime_vy_tagger import main
        main()
    
    def convert_rekordbox_location_to_path(self, location: str) -> Optional[str]:
        """
        Convert Rekordbox file location URL to local file path
        
        Args:
            location: Rekordbox file location URL
            
        Returns:
            Local file path or None if conversion fails
        """
        if not location:
            return None
        
        try:
            # Remove file:// prefix and decode URL encoding
            import urllib.parse
            if location.startswith('file://localhost'):
                path = location.replace('file://localhost', '')
            elif location.startswith('file://'):
                path = location.replace('file://', '')
            else:
                path = location
            
            # Decode URL encoding
            decoded_path = urllib.parse.unquote(path)
            
            return decoded_path if os.path.exists(decoded_path) else None
            
        except Exception:
            return None
    
    def get_system_status(self) -> Dict:
        """
        Get current system status and capabilities
        
        Returns:
            Dictionary with system status
        """
        return {
            'capabilities': self.capabilities,
            'components_loaded': list(self.components.keys()),
            'session_stats': self.session_stats,
            'system_health': {
                'core_tagger': CORE_TAGGER_AVAILABLE,
                'realtime_gui': REALTIME_GUI_AVAILABLE,
                'spotify': SPOTIFY_AVAILABLE,
                'audio_analyzer': AUDIO_ANALYZER_AVAILABLE,
                'ml_learner': ML_LEARNER_AVAILABLE
            }
        }
    
    def save_session_data(self, output_path: str = None):
        """
        Save session data and learned patterns
        
        Args:
            output_path: Path to save session data
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'/Users/shiraazoulay/session_data_{timestamp}.json'
        
        session_data = {
            'session_info': self.session_stats,
            'system_status': self.get_system_status(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save ML models if available
        if 'ml_learner' in self.components:
            self.components['ml_learner'].save_models()
        
        print(f'✅ Session data saved to: {output_path}')

# Example usage and testing
def main():
    print('🎆 Unified AI Tagger System')
    print('=' * 40)
    
    # Initialize system
    config = {
        'spotify': {
            # Add your Spotify API credentials here
            'client_id': 'your_spotify_client_id',
            'client_secret': 'your_spotify_client_secret'
        }
    }
    
    tagger = UnifiedAITagger(config)
    
    # Show system status
    status = tagger.get_system_status()
    print(f'
📊 System Status:')
    print(f'Capabilities: {status["capabilities"]}')
    print(f'Components: {status["components_loaded"]}')
    
    # Example workflow
    print('
📋 Example Workflow:')
    print('1. Process Rekordbox XML with AI enhancement')
    print('2. Launch real-time GUI for live tagging')
    print('3. Record user feedback for learning')
    print('4. Save session data and models')
    
    # Test with sample XML if available
    xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
    if os.path.exists(xml_file):
        print(f'
🎵 Found XML file: {os.path.basename(xml_file)}')
        print('Ready to process with full AI enhancement!')
    else:
        print('
⚠️ No XML file found for testing')
    
    print('
🚀 System ready for use!')

if __name__ == '__main__':
    main()
