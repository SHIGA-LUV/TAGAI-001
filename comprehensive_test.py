#!/usr/bin/env python3
# Comprehensive test for Rekordbox AI Tagger

import sys
import os
sys.path.append('/Users/shiraazoulay')

try:
    from rekordbox_ai_tagger import RekordboxAITagger
    TAGGER_AVAILABLE = True
except ImportError:
    TAGGER_AVAILABLE = False
    print('⚠️  Could not import RekordboxAITagger, running basic XML test only')

import xml.etree.ElementTree as ET

def test_basic_xml_parsing():
    """Test basic XML parsing without the tagger class"""
    print('📁 Testing Basic XML Parsing')
    print('-' * 30)
    
    xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
    
    if not os.path.exists(xml_file):
        print(f'❌ XML file not found: {xml_file}')
        return False
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tracks = root.findall('.//TRACK')
        
        print(f'✅ Successfully parsed XML with {len(tracks)} tracks')
        
        # Show sample tracks
        for i, track in enumerate(tracks[:3]):
            print(f'
🎵 Sample Track {i+1}:')
            print(f'   ID: {track.get("TrackID")}')
            print(f'   Name: {track.get("Name")}')
            print(f'   Artist: {track.get("Artist")}')
            print(f'   BPM: {track.get("AverageBpm")}')
            print(f'   Genre: {track.get("Genre")}')
            print(f'   Key: {track.get("Tonality")}')
        
        return True
        
    except Exception as e:
        print(f'❌ XML parsing error: {e}')
        return False

def test_tagger_functionality():
    """Test the full tagger functionality"""
    if not TAGGER_AVAILABLE:
        print('⚠️  Skipping tagger test - module not available')
        return False
        
    print('
🤖 Testing AI Tagger Functionality')
    print('-' * 35)
    
    try:
        tagger = RekordboxAITagger()
        print('✅ Tagger initialized successfully')
        
        # Test XML parsing with tagger
        xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
        tracks = tagger.parse_xml(xml_file)
        
        if not tracks:
            print('❌ No tracks parsed by tagger')
            return False
            
        print(f'✅ Tagger parsed {len(tracks)} tracks')
        
        # Test tag suggestions
        sample_track_id = list(tracks.keys())[0]
        sample_track = tracks[sample_track_id]
        
        print(f'
💡 Testing AI suggestions for: {sample_track.get("title", "Unknown")}')
        
        # Check if suggest_tags method exists
        if hasattr(tagger, 'suggest_tags'):
            suggestions = tagger.suggest_tags(sample_track)
            print('✅ AI suggestions generated:')
            for category, tags in suggestions.items():
                print(f'   {category}: {tags}')
        else:
            print('⚠️  suggest_tags method not found')
            
        return True
        
    except Exception as e:
        print(f'❌ Tagger test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    print('🎵 Rekordbox AI Tagger - Comprehensive Test')
    print('=' * 50)
    
    # Test 1: Basic XML parsing
    xml_success = test_basic_xml_parsing()
    
    # Test 2: Full tagger functionality
    tagger_success = test_tagger_functionality()
    
    # Summary
    print('
📊 Test Summary')
    print('-' * 15)
    print(f'XML Parsing: {"✅ PASS" if xml_success else "❌ FAIL"}')
    print(f'AI Tagger: {"✅ PASS" if tagger_success else "❌ FAIL"}')
    
    if xml_success and tagger_success:
        print('
🎉 All tests passed! System is ready.')
    elif xml_success:
        print('
⚠️  XML parsing works, but tagger needs attention.')
    else:
        print('
❌ System needs debugging.')

if __name__ == '__main__':
    main()
