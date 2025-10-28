#!/usr/bin/env python3
"""
Rekordbox AI Tagger - System Status Check
Quick verification of all components
"""

import os
import sys
from pathlib import Path

def check_system_status():
    print('🎯 REKORDBOX AI TAGGER - SYSTEM STATUS')
    print('=' * 50)
    
    # Core system files
    core_files = {
        'rekordbox_ai_tagger.py': 'Core XML Processor',
        'realtime_vy_tagger.py': 'Real-Time GUI',
        'unified_ai_tagger.py': 'Unified System',
        'spotify_enhancer.py': 'Spotify Integration',
        'audio_analyzer.py': 'Audio Analysis',
        'ml_pattern_learner.py': 'ML Pattern Learning',
        'Documents/shigmusic.xml': 'Rekordbox XML Data'
    }
    
    print('\n📁 Core Components:')
    available_components = 0
    total_components = len(core_files)
    
    for filename, description in core_files.items():
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f'✅ {description}: {filename} ({size:,} bytes)')
            available_components += 1
        else:
            print(f'❌ {description}: {filename} - Missing')
    
    # Test XML parsing capability
    print('\n🎵 XML Processing Test:')
    xml_file = Path('Documents/shigmusic.xml')
    if xml_file.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            tracks = root.findall('.//TRACK')
            print(f'✅ XML parsing successful: {len(tracks)} tracks detected')
            
            if tracks:
                sample = tracks[0]
                name = sample.get('Name', 'Unknown')
                artist = sample.get('Artist', 'Unknown')
                print(f'  Sample: "{name}" by {artist}')
        except Exception as e:
            print(f'❌ XML parsing failed: {e}')
    else:
        print('❌ No XML file found for testing')
    
    # System readiness assessment
    print('\n🏁 System Readiness:')
    readiness_score = (available_components / total_components) * 100
    
    if readiness_score >= 90:
        status = '🚀 FULLY OPERATIONAL'
        recommendation = 'Ready for live use!'
    elif readiness_score >= 70:
        status = '⚠️ MOSTLY READY'
        recommendation = 'Minor components missing, core functionality available'
    elif readiness_score >= 50:
        status = '🔧 PARTIAL SYSTEM'
        recommendation = 'Core components available, enhanced features may be limited'
    else:
        status = '❌ NEEDS SETUP'
        recommendation = 'Major components missing, system setup required'
    
    print(f'Status: {status}')
    print(f'Components: {available_components}/{total_components} ({readiness_score:.0f}%)')
    print(f'Recommendation: {recommendation}')
    
    # Next steps
    print('\n🎯 Next Steps:')
    if readiness_score >= 90:
        print('1. 🎵 Test with your music collection')
        print('2. 🖥️ Launch real-time GUI')
        print('3. 🚀 Start live tagging!')
    elif readiness_score >= 70:
        print('1. 🔍 Check missing components')
        print('2. 🎵 Test core functionality')
        print('3. 🖥️ Try the GUI interface')
    else:
        print('1. 📝 Review system requirements')
        print('2. 🔧 Install missing components')
        print('3. 🔄 Re-run system check')
    
    return readiness_score

if __name__ == '__main__':
    score = check_system_status()
    print(f'\n✨ System check complete! Score: {score:.0f}%')