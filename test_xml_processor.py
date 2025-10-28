#!/usr/bin/env python3
"""
Test script for the Rekordbox AI Tagger XML processor
"""

import sys
import os
from rekordbox_ai_tagger import RekordboxAITagger

def test_xml_processing():
    print('🎵 Testing Rekordbox AI Tagger XML Processor')
    print('=' * 50)
    
    # Initialize the tagger
    tagger = RekordboxAITagger()
    print('✅ Tagger initialized successfully')
    
    # Test with the XML file
    xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
    
    if not os.path.exists(xml_file):
        print(f'❌ XML file not found: {xml_file}')
        return
    
    print(f'📁 Found XML file: {xml_file}')
    
    try:
        # Parse the XML
        print('🔄 Parsing XML file...')
        tracks = tagger.parse_xml(xml_file)
        
        print(f'✅ Successfully parsed {len(tracks)} tracks')
        
        # Show some sample tracks
        print('
📋 Sample tracks:')
        count = 0
        for track_id, track_info in tracks.items():
            if count >= 5:  # Show first 5 tracks
                break
            print(f'
🎵 Track {track_id}:')
            print(f'   Name: {track_info.get("name", "Unknown")}')
            print(f'   Artist: {track_info.get("artist", "Unknown")}')
            print(f'   BPM: {track_info.get("bpm", "Unknown")}')
            print(f'   Genre: {track_info.get("genre", "Unknown")}')
            count += 1
        
        # Test AI suggestions
        print('
🤖 Testing AI tag suggestions...')
        sample_track_id = list(tracks.keys())[0]
        sample_track = tracks[sample_track_id]
        
        suggestions = tagger.suggest_tags(sample_track)
        print(f'
💡 AI Suggestions for "{sample_track.get("name", "Unknown")}":')
        
        for category, tags in suggestions.items():
            print(f'   {category}: {tags}')
        
        print('
🎉 XML processing test completed successfully!')
        
    except Exception as e:
        print(f'❌ Error during XML processing: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_xml_processing()
