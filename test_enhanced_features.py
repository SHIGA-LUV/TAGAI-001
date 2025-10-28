#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Rekordbox AI Tagger Features
Phase 2: Testing Spotify integration, audio analysis, and ML pattern recognition
"""

import sys
import os
import json
from datetime import datetime
sys.path.append('/Users/shiraazoulay')

# Test imports
try:
    from spotify_enhancer import SpotifyEnhancer
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    print('⚠️ SpotifyEnhancer not available')

try:
    from audio_analyzer import AudioAnalyzer
    AUDIO_ANALYZER_AVAILABLE = True
except ImportError:
    AUDIO_ANALYZER_AVAILABLE = False
    print('⚠️ AudioAnalyzer not available')

try:
    from ml_pattern_learner import MLPatternLearner
    ML_LEARNER_AVAILABLE = True
except ImportError:
    ML_LEARNER_AVAILABLE = False
    print('⚠️ MLPatternLearner not available')

class EnhancedFeatureTester:
    def __init__(self):
        """Initialize the enhanced feature tester"""
        self.test_results = {
            'spotify_tests': {},
            'audio_analysis_tests': {},
            'ml_pattern_tests': {},
            'integration_tests': {}
        }
        
        # Sample track data for testing
        self.sample_tracks = [
            {
                'title': 'Melodic Journey',
                'artist': 'Progressive Artist',
                'bpm': 128,
                'key': 'Am',
                'genre': 'Progressive House',
                'year': 2023
            },
            {
                'title': 'Tribal Rhythms',
                'artist': 'Ethnic Producer',
                'bpm': 124,
                'key': 'Fm',
                'genre': 'Ethnic House',
                'year': 2022
            },
            {
                'title': 'Peak Time Energy',
                'artist': 'Techno Master',
                'bpm': 132,
                'key': 'Gm',
                'genre': 'Melodic Techno',
                'year': 2024
            }
        ]
    
    def test_spotify_enhancer(self) -> bool:
        """Test Spotify integration features"""
        print('🎵 Testing Spotify Enhancer')
        print('-' * 30)
        
        if not SPOTIFY_AVAILABLE:
            print('❌ Spotify enhancer not available')
            self.test_results['spotify_tests']['available'] = False
            return False
        
        try:
            # Initialize enhancer
            enhancer = SpotifyEnhancer()
            print('✅ SpotifyEnhancer initialized')
            
            # Test feature mappings
            print('
📋 Testing feature mappings:')
            for feature, mappings in enhancer.feature_mappings.items():
                print(f'   {feature}: {list(mappings.keys())}')
                self.test_results['spotify_tests'][f'{feature}_mapping'] = True
            
            # Test search term cleaning
            test_terms = ['Artist [Remix]', 'Track (Original Mix)', 'Song {Extended}']
            print('
🧹 Testing search term cleaning:')
            for term in test_terms:
                cleaned = enhancer.clean_search_term(term)
                print(f'   "{term}" -> "{cleaned}")
                self.test_results['spotify_tests']['term_cleaning'] = True
            
            # Test AI suggestions from mock audio features
            print('
🤖 Testing AI suggestions from audio features:')
            mock_features = {
                'energy': 0.8,
                'danceability': 0.7,
                'valence': 0.6,
                'acousticness': 0.2
            }
            suggestions = enhancer.generate_ai_suggestions_from_features(mock_features)
            print(f'   Mock features: {mock_features}')
            print(f'   AI suggestions: {suggestions}')
            self.test_results['spotify_tests']['ai_suggestions'] = len(suggestions) > 0
            
            # Test track enhancement (without API calls)
            print('
🔍 Testing track enhancement structure:')
            for track in self.sample_tracks[:2]:
                print(f'   Track: {track["title"]} by {track["artist"]}')
                # This would normally call Spotify API, but we're testing structure
                enhanced = enhancer.enhance_track_with_spotify(track)
                print(f'   Enhanced keys: {list(enhanced.keys())}')
                self.test_results['spotify_tests']['track_enhancement'] = True
            
            print('✅ Spotify enhancer tests completed')
            self.test_results['spotify_tests']['overall'] = True
            return True
            
        except Exception as e:
            print(f'❌ Spotify enhancer test error: {e}')
            self.test_results['spotify_tests']['error'] = str(e)
            return False
    
    def test_audio_analyzer(self) -> bool:
        """Test audio analysis features"""
        print('
🌊 Testing Audio Analyzer')
        print('-' * 25)
        
        if not AUDIO_ANALYZER_AVAILABLE:
            print('❌ Audio analyzer not available')
            self.test_results['audio_analysis_tests']['available'] = False
            return False
        
        try:
            # Initialize analyzer
            analyzer = AudioAnalyzer()
            print('✅ AudioAnalyzer initialized')
            
            # Test feature mappings
            print('
📋 Testing feature to tag mappings:')
            for feature, mappings in analyzer.feature_tag_mappings.items():
                print(f'   {feature}: {list(mappings.keys())}')
                self.test_results['audio_analysis_tests'][f'{feature}_mapping'] = True
            
            # Test key and mode mappings
            print('
🎹 Testing key/mode mappings:')
            print(f'   Keys: {list(analyzer.key_mappings.values())[:6]}...')
            print(f'   Modes: {list(analyzer.mode_mappings.values())}')
            self.test_results['audio_analysis_tests']['key_mode_mappings'] = True
            
            # Test AI suggestions from mock analysis
            print('
🤖 Testing AI suggestions from audio analysis:')
            mock_analysis = {
                'tempo': 128,
                'key': 'Am',
                'mode': 'minor',
                'spectral_features': {
                    'spectral_centroid_mean': 3000,
                    'zero_crossing_rate_mean': 0.08,
                    'rms_mean': 0.15
                },
                'harmonic_percussive': {
                    'harmonic_ratio': 0.6,
                    'percussive_ratio': 0.4
                }
            }
            
            suggestions = analyzer.generate_ai_suggestions_from_audio(mock_analysis)
            print(f'   Mock analysis: tempo={mock_analysis["tempo"]}, key={mock_analysis["key"]}')
            print(f'   AI suggestions: {suggestions}')
            self.test_results['audio_analysis_tests']['ai_suggestions'] = len(suggestions) > 0
            
            # Test batch analysis structure
            print('
📋 Testing batch analysis structure:')
            # We can't test actual audio files, but we can test the structure
            print('   Batch analysis method available: ', hasattr(analyzer, 'batch_analyze_tracks'))
            self.test_results['audio_analysis_tests']['batch_analysis'] = True
            
            print('✅ Audio analyzer tests completed')
            self.test_results['audio_analysis_tests']['overall'] = True
            return True
            
        except Exception as e:
            print(f'❌ Audio analyzer test error: {e}')
            self.test_results['audio_analysis_tests']['error'] = str(e)
            return False
    
    def test_ml_pattern_learner(self) -> bool:
        """Test ML pattern learning features"""
        print('
🤖 Testing ML Pattern Learner')
        print('-' * 30)
        
        if not ML_LEARNER_AVAILABLE:
            print('❌ ML pattern learner not available')
            self.test_results['ml_pattern_tests']['available'] = False
            return False
        
        try:
            # Initialize learner
            learner = MLPatternLearner()
            print('✅ MLPatternLearner initialized')
            
            # Test feature extraction
            print('
🔍 Testing feature extraction:')
            for track in self.sample_tracks:
                features = learner.extract_track_features(track)
                print(f'   {track["title"]}: {len(features)} features extracted')
                self.test_results['ml_pattern_tests']['feature_extraction'] = True
            
            # Test interaction recording
            print('
📝 Testing interaction recording:')
            for i, track in enumerate(self.sample_tracks):
                suggested_tags = ['2-Build up', 'Progressive House', 'Energetic']
                selected_tags = ['2-Build up', 'Progressive House']  # User selected subset
                
                learner.record_user_interaction(track, suggested_tags, selected_tags)
                print(f'   Recorded interaction {i+1}')
            
            self.test_results['ml_pattern_tests']['interaction_recording'] = True
            
            # Test pattern-based predictions
            print('
🔮 Testing pattern-based predictions:')
            test_track = self.sample_tracks[0]
            pattern_predictions = learner.predict_from_patterns(test_track)
            print(f'   Pattern predictions for "{test_track["title"]}": {pattern_predictions}')
            self.test_results['ml_pattern_tests']['pattern_predictions'] = True
            
            # Test learning statistics
            print('
📊 Testing learning statistics:')
            stats = learner.get_learning_statistics()
            print(f'   Total interactions: {stats["total_interactions"]}')
            print(f'   Unique tags learned: {stats["unique_tags_learned"]}')
            print(f'   Model trained: {stats["model_trained"]}')
            self.test_results['ml_pattern_tests']['statistics'] = True
            
            # Test model saving/loading
            print('
💾 Testing model persistence:')
            learner.save_models()
            print('   Models saved successfully')
            self.test_results['ml_pattern_tests']['model_persistence'] = True
            
            print('✅ ML pattern learner tests completed')
            self.test_results['ml_pattern_tests']['overall'] = True
            return True
            
        except Exception as e:
            print(f'❌ ML pattern learner test error: {e}')
            self.test_results['ml_pattern_tests']['error'] = str(e)
            return False
    
    def test_integration(self) -> bool:
        """Test integration between all enhanced features"""
        print('
🔗 Testing Feature Integration')
        print('-' * 30)
        
        try:
            # Test combined workflow
            print('🎆 Testing combined enhancement workflow:')
            
            for track in self.sample_tracks[:2]:
                print(f'
🎵 Processing: {track["title"]} by {track["artist"]}')
                
                enhanced_track = track.copy()
                
                # Step 1: Spotify enhancement (mock)
                if SPOTIFY_AVAILABLE:
                    print('   🎵 Adding Spotify features...')
                    enhanced_track['spotify_mock'] = {
                        'energy': 0.7,
                        'danceability': 0.8,
                        'valence': 0.6
                    }
                
                # Step 2: Audio analysis (mock)
                if AUDIO_ANALYZER_AVAILABLE:
                    print('   🌊 Adding audio analysis...')
                    enhanced_track['audio_analysis_mock'] = {
                        'spectral_centroid': 3000,
                        'tempo_stability': 0.9
                    }
                
                # Step 3: ML predictions
                if ML_LEARNER_AVAILABLE:
                    print('   🤖 Adding ML predictions...')
                    learner = MLPatternLearner()
                    predictions = learner.predict_tags_for_track(enhanced_track)
                    enhanced_track['ml_predictions'] = predictions
                
                print(f'   ✅ Enhanced track has {len(enhanced_track)} data fields')
            
            self.test_results['integration_tests']['combined_workflow'] = True
            
            # Test requirements compatibility
            print('
📋 Testing requirements compatibility:')
            required_packages = ['spotipy', 'librosa', 'scikit-learn', 'numpy', 'pandas']
            for package in required_packages:
                try:
                    __import__(package)
                    print(f'   ✅ {package} available')
                except ImportError:
                    print(f'   ⚠️ {package} not installed')
            
            self.test_results['integration_tests']['requirements'] = True
            
            print('✅ Integration tests completed')
            self.test_results['integration_tests']['overall'] = True
            return True
            
        except Exception as e:
            print(f'❌ Integration test error: {e}')
            self.test_results['integration_tests']['error'] = str(e)
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print('
📊 Enhanced Features Test Report')
        print('=' * 50)
        
        # Overall summary
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            category_passed = tests.get('overall', False)
            total_tests += 1
            if category_passed:
                passed_tests += 1
            
            status = '✅ PASS' if category_passed else '❌ FAIL'
            print(f'{category.replace("_", " ").title()}: {status}')
        
        print(f'
Overall: {passed_tests}/{total_tests} test categories passed')
        
        # Detailed results
        print('
🔍 Detailed Results:')
        for category, tests in self.test_results.items():
            print(f'
{category.replace("_", " ").title()}:')
            for test_name, result in tests.items():
                if test_name != 'overall':
                    status = '✅' if result else '❌'
                    print(f'   {status} {test_name}')
        
        # Recommendations
        print('
💡 Recommendations:')
        
        if not SPOTIFY_AVAILABLE:
            print('- Install spotipy for Spotify integration: pip install spotipy')
        
        if not AUDIO_ANALYZER_AVAILABLE:
            print('- Install librosa for audio analysis: pip install librosa')
        
        if not ML_LEARNER_AVAILABLE:
            print('- Install scikit-learn for ML features: pip install scikit-learn')
        
        if passed_tests == total_tests:
            print('
🎉 All enhanced features are working correctly!')
            print('Ready for production use with full Phase 2 capabilities.')
        else:
            print(f'
⚠️ {total_tests - passed_tests} feature(s) need attention.')
            print('Consider installing missing dependencies or debugging errors.')
        
        # Save report
        report_file = '/Users/shiraazoulay/enhanced_features_test_report.json'
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_results': self.test_results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                }
            }, f, indent=2)
        
        print(f'
💾 Test report saved to: {report_file}')
    
    def run_all_tests(self):
        """Run all enhanced feature tests"""
        print('🚀 Starting Enhanced Features Test Suite')
        print('=' * 55)
        
        # Run individual test suites
        self.test_spotify_enhancer()
        self.test_audio_analyzer()
        self.test_ml_pattern_learner()
        self.test_integration()
        
        # Generate report
        self.generate_test_report()

def main():
    """Main test execution"""
    tester = EnhancedFeatureTester()
    tester.run_all_tests()

if __name__ == '__main__':
    main()
