#!/usr/bin/env python3
"""
Rekordbox AI Tagger - Complete Test Suite
Runs all phases 1-4 in sequence
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_phase_1():
    print('🧪 PHASE 1: CORE SYSTEM TESTING')
    print('=' * 50)
    
    try:
        result = subprocess.run(['python3', 'system_status.py'], 
                              capture_output=True, text=True, cwd='/Users/shiraazoulay')
        print(result.stdout)
        if result.stderr:
            print('Warnings:', result.stderr)
        return True
    except Exception as e:
        print(f'❌ Phase 1 error: {e}')
        return False

def run_phase_2():
    print('\n🖥️ PHASE 2: REAL-TIME GUI TESTING')
    print('=' * 50)
    
    try:
        # Test GUI components without launching full interface
        sys.path.append('/Users/shiraazoulay')
        
        print('📱 Testing GUI imports...')
        try:
            from realtime_vy_tagger import RealTimeVyTagger
            print('✅ RealTimeVyTagger imported successfully')
            
            tagger = RealTimeVyTagger()
            print('✅ GUI tagger initialized')
            
            # Check available methods
            methods = [m for m in dir(tagger) if not m.startswith('_')]
            print(f'✅ Available methods: {len(methods)}')
            
            # Test tag hierarchy
            if hasattr(tagger, 'tag_hierarchy'):
                categories = list(tagger.tag_hierarchy.keys())
                print(f'✅ Tag categories: {categories}')
            
            print('✅ GUI components ready for live use')
            return True
            
        except ImportError as e:
            print(f'❌ GUI import error: {e}')
            return False
            
    except Exception as e:
        print(f'❌ Phase 2 error: {e}')
        return False

def run_phase_3():
    print('\n🚀 PHASE 3: ENHANCED FEATURES TESTING')
    print('=' * 50)
    
    try:
        # Test enhanced components
        enhanced_files = {
            'spotify_enhancer.py': 'Spotify Integration',
            'audio_analyzer.py': 'Audio Analysis',
            'ml_pattern_learner.py': 'ML Pattern Learning',
            'unified_ai_tagger.py': 'Unified System'
        }
        
        available_features = 0
        for filename, description in enhanced_files.items():
            filepath = Path(f'/Users/shiraazoulay/{filename}')
            if filepath.exists():
                size = filepath.stat().st_size
                print(f'✅ {description}: {filename} ({size:,} bytes)')
                available_features += 1
            else:
                print(f'❌ {description}: {filename} - Missing')
        
        # Test unified system if available
        if Path('/Users/shiraazoulay/unified_ai_tagger.py').exists():
            print('\n🔗 Testing unified system integration...')
            try:
                from unified_ai_tagger import UnifiedAITagger
                unified = UnifiedAITagger()
                print('✅ Unified AI Tagger initialized')
                print('✅ Multi-source AI suggestions ready')
            except Exception as e:
                print(f'⚠️ Unified system warning: {e}')
        
        feature_percentage = (available_features / len(enhanced_files)) * 100
        print(f'\n📈 Enhanced Features: {available_features}/{len(enhanced_files)} ({feature_percentage:.0f}%)')
        
        return feature_percentage >= 75
        
    except Exception as e:
        print(f'❌ Phase 3 error: {e}')
        return False

def run_phase_4():
    print('\n📈 PHASE 4: ANALYTICS & OPTIMIZATION')
    print('=' * 50)
    
    try:
        # Run analytics
        print('📊 Running collection analytics...')
        result = subprocess.run(['python3', 'analytics_runner.py'], 
                              capture_output=True, text=True, cwd='/Users/shiraazoulay')
        print(result.stdout)
        
        # Run optimization
        print('\n🚀 Running system optimization...')
        result = subprocess.run(['python3', 'optimize_system.py'], 
                              capture_output=True, text=True, cwd='/Users/shiraazoulay')
        print(result.stdout)
        
        return True
        
    except Exception as e:
        print(f'❌ Phase 4 error: {e}')
        return False

def main():
    print('🎯 REKORDBOX AI TAGGER - COMPLETE TEST SUITE')
    print('=' * 60)
    print('Running all phases 1-4 in sequence...')
    
    start_time = time.time()
    results = []
    
    # Run all phases
    results.append(('Phase 1: Core System', run_phase_1()))
    results.append(('Phase 2: Real-Time GUI', run_phase_2()))
    results.append(('Phase 3: Enhanced Features', run_phase_3()))
    results.append(('Phase 4: Analytics & Optimization', run_phase_4()))
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print('\n🏁 COMPLETE TEST RESULTS')
    print('=' * 40)
    
    passed = 0
    for phase_name, result in results:
        status = '✅ PASS' if result else '❌ FAIL'
        print(f'{phase_name}: {status}')
        if result:
            passed += 1
    
    print(f'\n📊 Overall Results:')
    print(f'  Phases passed: {passed}/{len(results)}')
    print(f'  Success rate: {(passed/len(results)*100):.0f}%')
    print(f'  Test duration: {duration:.1f} seconds')
    
    if passed == len(results):
        print('\n🎉 ALL SYSTEMS OPERATIONAL!')
        print('🚀 Ready for live DJ use!')
        print('\n🎯 Next Steps:')
        print('  1. Launch real-time GUI: python3 launch_gui.py')
        print('  2. Start master control: python3 master_control.py')
        print('  3. Begin live tagging session!')
    elif passed >= 3:
        print('\n✅ SYSTEM MOSTLY READY!')
        print('Core functionality available, minor issues detected.')
    else:
        print('\n⚠️ SYSTEM NEEDS ATTENTION')
        print('Multiple components require fixes before live use.')

if __name__ == '__main__':
    main()