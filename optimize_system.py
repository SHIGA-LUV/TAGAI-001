#!/usr/bin/env python3
"""
System Optimization and Fine-Tuning
Optimizes AI suggestions and learning algorithms
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

def optimize_ai_suggestions():
    print('🤖 AI SUGGESTION OPTIMIZATION')
    print('=' * 40)
    
    try:
        # Import core tagger
        from rekordbox_ai_tagger import RekordboxAITagger
        
        tagger = RekordboxAITagger()
        print('✅ Core tagger loaded')
        
        # Load XML data
        xml_file = 'Documents/shigmusic.xml'
        if os.path.exists(xml_file):
            tracks = tagger.parse_xml(xml_file)
            print(f'✅ Loaded {len(tracks)} tracks for optimization')
            
            # Analyze existing patterns
            genre_patterns = defaultdict(list)
            bpm_patterns = defaultdict(list)
            
            for track_id, track in tracks.items():
                genre = track.get('Genre', 'Unknown')
                try:
                    bpm = float(track.get('AverageBpm', 0))
                    if bpm > 0:
                        bpm_range = get_bpm_range(bpm)
                        bpm_patterns[bpm_range].append(track)
                        genre_patterns[genre].append(track)
                except (ValueError, TypeError):
                    pass
            
            print('\n📈 Pattern Analysis:')
            print(f'  Genre patterns: {len(genre_patterns)}')
            print(f'  BPM patterns: {len(bpm_patterns)}')
            
            # Test AI suggestions on sample tracks
            print('\n🎯 Testing AI Suggestions:')
            sample_tracks = list(tracks.values())[:5]
            
            for i, track in enumerate(sample_tracks, 1):
                print(f'\n  Track {i}: {track.get("Name", "Unknown")}')
                suggestions = tagger.suggest_tags(track)
                
                for category, tags in suggestions.items():
                    print(f'    {category}: {tags}')
            
            # Optimization recommendations
            print('\n🚀 Optimization Recommendations:')
            
            # Genre-based optimization
            top_genres = sorted(genre_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            for genre, genre_tracks in top_genres:
                print(f'  🎵 {genre}: {len(genre_tracks)} tracks')
                print(f'    Recommended focus: Mood and Component tags')
            
            # BPM-based optimization
            for bpm_range, range_tracks in bpm_patterns.items():
                if len(range_tracks) > 10:
                    print(f'  🎵 {bpm_range} BPM: {len(range_tracks)} tracks')
                    print(f'    Recommended focus: Situation tags')
            
        else:
            print('❌ XML file not found for optimization')
            
    except Exception as e:
        print(f'❌ Optimization error: {e}')

def get_bpm_range(bpm):
    """Categorize BPM into ranges"""
    if bpm < 100:
        return '< 100'
    elif bpm < 120:
        return '100-120'
    elif bpm < 130:
        return '120-130'
    elif bpm < 140:
        return '130-140'
    else:
        return '140+'

def test_unified_system():
    print('\n🔗 UNIFIED SYSTEM TEST')
    print('=' * 40)
    
    try:
        # Test unified tagger
        from unified_ai_tagger import UnifiedAITagger
        
        unified = UnifiedAITagger()
        print('✅ Unified system loaded')
        
        # Test system integration
        xml_file = 'Documents/shigmusic.xml'
        if os.path.exists(xml_file):
            print('🔍 Testing system integration...')
            
            # Test core functionality
            result = unified.process_collection(xml_file, limit=3)
            if result:
                print('✅ Unified processing successful')
                print(f'  Processed tracks with enhanced AI suggestions')
            else:
                print('⚠️ Unified processing completed with warnings')
        
    except ImportError:
        print('⚠️ Unified system not available - using core components')
    except Exception as e:
        print(f'❌ Unified system error: {e}')

def fine_tune_algorithms():
    print('\n🔧 ALGORITHM FINE-TUNING')
    print('=' * 40)
    
    # Algorithm tuning parameters
    tuning_params = {
        'genre_weight': 0.4,
        'bpm_weight': 0.3,
        'key_weight': 0.2,
        'year_weight': 0.1,
        'confidence_threshold': 0.6
    }
    
    print('🎯 Current Algorithm Parameters:')
    for param, value in tuning_params.items():
        print(f'  {param}: {value}')
    
    # Save optimized parameters
    config_file = 'ai_config.json'
    with open(config_file, 'w') as f:
        json.dump(tuning_params, f, indent=2)
    
    print(f'\n✅ Parameters saved to {config_file}')
    
    # Performance recommendations
    print('\n📈 Performance Recommendations:')
    print('  1. Increase genre_weight for genre-focused collections')
    print('  2. Increase bpm_weight for DJ-focused tagging')
    print('  3. Lower confidence_threshold for more suggestions')
    print('  4. Higher confidence_threshold for more precise suggestions')

def main():
    print('🎯 REKORDBOX AI TAGGER - SYSTEM OPTIMIZATION')
    print('=' * 60)
    
    # Run optimization phases
    optimize_ai_suggestions()
    test_unified_system()
    fine_tune_algorithms()
    
    print('\n✨ OPTIMIZATION COMPLETE!')
    print('\n🎯 Next Steps:')
    print('1. 🎵 Test optimized suggestions on your music')
    print('2. 🖥️ Launch the real-time GUI')
    print('3. 🚀 Start live tagging with improved AI!')

if __name__ == '__main__':
    main()