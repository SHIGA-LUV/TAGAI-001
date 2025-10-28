#!/usr/bin/env python3
"""
Rekordbox AI Tagger - Master Control Panel
Central hub for all system operations
"""

import os
import sys
import subprocess
from pathlib import Path

def show_menu():
    print('🎯 REKORDBOX AI TAGGER - MASTER CONTROL')
    print('=' * 50)
    print()
    print('📁 SYSTEM OPERATIONS:')
    print('  1. 🔍 System Status Check')
    print('  2. 🎵 Test Core XML Processor')
    print('  3. 🖥️ Launch Real-Time GUI')
    print('  4. 📈 Run Collection Analytics')
    print('  5. 🚀 Optimize AI System')
    print('  6. 🔗 Test Unified System')
    print()
    print('🎵 QUICK ACTIONS:')
    print('  7. 🎧 Start Live Tagging Session')
    print('  8. 📊 Generate Music Report')
    print('  9. 🔧 System Maintenance')
    print()
    print('  0. 🚪 Exit')
    print()

def run_system_status():
    print('🔍 Running system status check...')
    try:
        result = subprocess.run(['python3', 'system_status.py'], 
                              capture_output=True, text=True, cwd='/Users/shiraazoulay')
        print(result.stdout)
        if result.stderr:
            print('Errors:', result.stderr)
    except Exception as e:
        print(f'Error running system status: {e}')

def run_analytics():
    print('📈 Running collection analytics...')
    try:
        result = subprocess.run(['python3', 'analytics_runner.py'], 
                              capture_output=True, text=True, cwd='/Users/shiraazoulay')
        print(result.stdout)
        if result.stderr:
            print('Errors:', result.stderr)
    except Exception as e:
        print(f'Error running analytics: {e}')

def launch_gui():
    print('🖥️ Launching real-time GUI...')
    try:
        # Run GUI in background
        subprocess.Popen(['python3', 'launch_gui.py'], cwd='/Users/shiraazoulay')
        print('✅ GUI launched successfully!')
        print('Check for the GUI window on your screen.')
    except Exception as e:
        print(f'Error launching GUI: {e}')

def run_optimization():
    print('🚀 Running system optimization...')
    try:
        result = subprocess.run(['python3', 'optimize_system.py'], 
                              capture_output=True, text=True, cwd='/Users/shiraazoulay')
        print(result.stdout)
        if result.stderr:
            print('Errors:', result.stderr)
    except Exception as e:
        print(f'Error running optimization: {e}')

def test_core_processor():
    print('🎵 Testing core XML processor...')
    try:
        # Import and test core functionality
        sys.path.append('/Users/shiraazoulay')
        from rekordbox_ai_tagger import RekordboxAITagger
        
        tagger = RekordboxAITagger()
        print('✅ Core tagger initialized')
        
        xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
        if os.path.exists(xml_file):
            tracks = tagger.parse_xml(xml_file)
            print(f'✅ Successfully parsed {len(tracks)} tracks')
            
            # Test AI suggestions on first track
            if tracks:
                sample_track = list(tracks.values())[0]
                suggestions = tagger.suggest_tags(sample_track)
                print(f'✅ Generated AI suggestions: {len(suggestions)} categories')
                for category, tags in suggestions.items():
                    print(f'  {category}: {tags}')
        else:
            print('❌ XML file not found')
            
    except Exception as e:
        print(f'Error testing core processor: {e}')

def start_live_session():
    print('🎧 Starting live tagging session...')
    print('This will:')
    print('1. Launch the real-time GUI')
    print('2. Load your music collection')
    print('3. Enable live AI tagging')
    
    try:
        # Launch GUI for live session
        launch_gui()
        print('\n✅ Live session ready!')
        print('Use the GUI to tag tracks in real-time while DJing.')
    except Exception as e:
        print(f'Error starting live session: {e}')

def generate_report():
    print('📊 Generating comprehensive music report...')
    
    # Run both analytics and optimization
    run_analytics()
    print('\n' + '='*50)
    run_optimization()
    
    print('\n✅ Comprehensive report generated!')

def system_maintenance():
    print('🔧 Running system maintenance...')
    
    # Check all components
    components = [
        'rekordbox_ai_tagger.py',
        'realtime_vy_tagger.py',
        'unified_ai_tagger.py',
        'system_status.py',
        'analytics_runner.py',
        'optimize_system.py'
    ]
    
    print('\n📁 Component Status:')
    for component in components:
        if os.path.exists(f'/Users/shiraazoulay/{component}'):
            size = os.path.getsize(f'/Users/shiraazoulay/{component}')
            print(f'✅ {component} ({size:,} bytes)')
        else:
            print(f'❌ {component} - Missing')
    
    # Check XML data
    xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
    if os.path.exists(xml_file):
        size = os.path.getsize(xml_file)
        print(f'✅ shigmusic.xml ({size:,} bytes)')
    else:
        print('❌ shigmusic.xml - Missing')
    
    print('\n✅ Maintenance check complete!')

def main():
    while True:
        show_menu()
        
        try:
            choice = input('🎯 Select option (0-9): ').strip()
            
            if choice == '0':
                print('\n👋 Goodbye! Happy DJing!')
                break
            elif choice == '1':
                run_system_status()
            elif choice == '2':
                test_core_processor()
            elif choice == '3':
                launch_gui()
            elif choice == '4':
                run_analytics()
            elif choice == '5':
                run_optimization()
            elif choice == '6':
                print('🔗 Testing unified system...')
                run_optimization()  # Includes unified system test
            elif choice == '7':
                start_live_session()
            elif choice == '8':
                generate_report()
            elif choice == '9':
                system_maintenance()
            else:
                print('❌ Invalid option. Please try again.')
            
            input('\n⏸️ Press Enter to continue...')
            
        except KeyboardInterrupt:
            print('\n\n👋 Goodbye!')
            break
        except Exception as e:
            print(f'\n❌ Error: {e}')
            input('Press Enter to continue...')

if __name__ == '__main__':
    main()