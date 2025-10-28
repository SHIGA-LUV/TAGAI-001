#!/usr/bin/env python3
# Updated test script
# Quick test to verify XML parsing works

import xml.etree.ElementTree as ET
import os

def quick_xml_test():
    xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
    
    print('🎵 Quick XML Structure Test')
    print('=' * 40)
    
    if not os.path.exists(xml_file):
        print(f'❌ XML file not found: {xml_file}')
        return
    
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find tracks
        tracks = root.findall('.//TRACK')
        print(f'✅ Found {len(tracks)} tracks in XML')
        
        # Show first 3 tracks
        for i, track in enumerate(tracks[:3]):
            print(f'
🎵 Track {i+1}:')
            print(f'   ID: {track.get("TrackID")}')
            print(f'   Name: {track.get("Name")}')
            print(f'   Artist: {track.get("Artist")}')
            print(f'   BPM: {track.get("AverageBpm")}')
            print(f'   Genre: {track.get("Genre")}')
        
        print('
✅ XML parsing test successful!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == '__main__':
    quick_xml_test()
