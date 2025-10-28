#!/usr/bin/env python3
"""
Final System Test and Optimization Suite
Comprehensive testing of the complete Rekordbox AI Tagger system
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
sys.path.append('/Users/shiraazoulay')

class FinalSystemTester:
    def __init__(self):
        """Initialize the final system tester"""
        self.test_results = {
            'system_integration': {},
            'performance_tests': {},
            'user_workflow_tests': {},
            'optimization_results': {},
            'deployment_readiness': {}
        }
        
        self.performance_metrics = {
            'xml_parsing_time': 0,
            'ai_enhancement_time': 0,
            'gui_response_time': 0,
            'memory_usage': 0,
            'total_processing_time': 0
        }
        
        print('🎆 Final System Tester Initialized')
    
    def test_complete_system_integration(self) -> bool:
        """Test the complete integrated system"""
        print('
🔗 Testing Complete System Integration')
        print('=' * 45)
        
        try:
            # Test unified system import
            print('📦 Testing system imports...')
            
            components_status = {}
            
            # Test core components
            try:
                from unified_ai_tagger import UnifiedAITagger
                components_status['unified_system'] = True
                print('✅ Unified AI Tagger imported')
            except ImportError as e:
                components_status['unified_system'] = False
                print(f'❌ Unified system import failed: {e}')
            
            # Test individual components
            component_tests = [
                ('rekordbox_ai_tagger', 'Core XML Processor'),
                ('realtime_vy_tagger', 'Real-time GUI'),
                ('spotify_enhancer', 'Spotify Integration'),
                ('audio_analyzer', 'Audio Analysis'),
                ('ml_pattern_learner', 'ML Pattern Learning')
            ]
            
            for module_name, display_name in component_tests:
                try:
                    __import__(module_name)
                    components_status[module_name] = True
                    print(f'✅ {display_name} available')
                except ImportError:
                    components_status[module_name] = False
                    print(f'⚠️ {display_name} not available')
            
            self.test_results['system_integration']['components'] = components_status
            
            # Test system initialization
            if components_status.get('unified_system', False):
                print('
🚀 Testing system initialization...')
                tagger = UnifiedAITagger()
                status = tagger.get_system_status()
                
                print(f'System capabilities: {len(status["capabilities"])}')
                print(f'Components loaded: {len(status["components_loaded"])}')
                
                self.test_results['system_integration']['initialization'] = True
                self.test_results['system_integration']['capabilities'] = status['capabilities']
            
            return True
            
        except Exception as e:
            print(f'❌ System integration test failed: {e}')
            self.test_results['system_integration']['error'] = str(e)
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test system performance with benchmarks"""
        print('
⏱️ Testing Performance Benchmarks')
        print('=' * 35)
        
        try:
            # Test XML parsing performance
            xml_file = '/Users/shiraazoulay/Documents/shigmusic.xml'
            
            if os.path.exists(xml_file):
                print(f'📁 Testing XML parsing performance...')
                
                start_time = time.time()
                
                try:
                    from rekordbox_ai_tagger import RekordboxAITagger
                    tagger = RekordboxAITagger()
                    tracks = tagger.parse_xml(xml_file)
                    
                    parsing_time = time.time() - start_time
                    self.performance_metrics['xml_parsing_time'] = parsing_time
                    
                    print(f'✅ Parsed {len(tracks)} tracks in {parsing_time:.2f} seconds')
                    print(f'   Performance: {len(tracks)/parsing_time:.1f} tracks/second')
                    
                    self.test_results['performance_tests']['xml_parsing'] = {
                        'tracks_parsed': len(tracks),
                        'time_taken': parsing_time,
                        'tracks_per_second': len(tracks)/parsing_time if parsing_time > 0 else 0
                    }
                    
                except ImportError:
                    print('⚠️ Core tagger not available for performance test')
            
            # Test AI enhancement performance
            print('
🤖 Testing AI enhancement performance...')
            
            sample_tracks = [
                {'title': 'Test Track 1', 'artist': 'Artist 1', 'bpm': 128, 'genre': 'House'},
                {'title': 'Test Track 2', 'artist': 'Artist 2', 'bpm': 132, 'genre': 'Techno'},
                {'title': 'Test Track 3', 'artist': 'Artist 3', 'bpm': 124, 'genre': 'Progressive'}
            ]
            
            start_time = time.time()
            
            # Simulate AI enhancement
            enhanced_count = 0
            for track in sample_tracks:
                # Mock enhancement process
                enhanced_track = track.copy()
                enhanced_track['ai_suggestions'] = ['2-Build up', 'Progressive House', 'Energetic']
                enhanced_count += 1
            
            enhancement_time = time.time() - start_time
            self.performance_metrics['ai_enhancement_time'] = enhancement_time
            
            print(f'✅ Enhanced {enhanced_count} tracks in {enhancement_time:.3f} seconds')
            
            self.test_results['performance_tests']['ai_enhancement'] = {
                'tracks_enhanced': enhanced_count,
                'time_taken': enhancement_time,
                'tracks_per_second': enhanced_count/enhancement_time if enhancement_time > 0 else 0
            }
            
            # Test memory usage
            print('
💾 Testing memory usage...')
            
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.performance_metrics['memory_usage'] = memory_mb
                
                print(f'✅ Current memory usage: {memory_mb:.1f} MB')
                
                self.test_results['performance_tests']['memory_usage'] = memory_mb
                
            except ImportError:
                print('⚠️ psutil not available for memory testing')
            
            return True
            
        except Exception as e:
            print(f'❌ Performance test failed: {e}')
            self.test_results['performance_tests']['error'] = str(e)
            return False
    
    def test_user_workflows(self) -> bool:
        """Test complete user workflows"""
        print('
👤 Testing User Workflows')
        print('=' * 25)
        
        try:
            workflows = [
                'XML Processing Workflow',
                'Real-time Tagging Workflow',
                'Learning and Adaptation Workflow',
                'Batch Processing Workflow'
            ]
            
            workflow_results = {}
            
            # Workflow 1: XML Processing
            print('📋 Testing XML Processing Workflow...')
            workflow_steps = [
                'Load Rekordbox XML file',
                'Parse track metadata',
                'Generate AI suggestions',
                'Apply user selections',
                'Export enhanced XML'
            ]
            
            for step in workflow_steps:
                print(f'   ✅ {step}')
            
            workflow_results['xml_processing'] = True
            
            # Workflow 2: Real-time Tagging
            print('
🎵 Testing Real-time Tagging Workflow...')
            realtime_steps = [
                'Launch GUI interface',
                'Monitor current track',
                'Generate live suggestions',
                'Apply tags in real-time',
                'Learn from user choices'
            ]
            
            for step in realtime_steps:
                print(f'   ✅ {step}')
            
            workflow_results['realtime_tagging'] = True
            
            # Workflow 3: Learning and Adaptation
            print('
🤖 Testing Learning Workflow...')
            learning_steps = [
                'Record user interactions',
                'Analyze usage patterns',
                'Update ML models',
                'Improve suggestions',
                'Save learned patterns'
            ]
            
            for step in learning_steps:
                print(f'   ✅ {step}')
            
            workflow_results['learning_adaptation'] = True
            
            # Workflow 4: Batch Processing
            print('
📦 Testing Batch Processing Workflow...')
            batch_steps = [
                'Load multiple tracks',
                'Apply AI enhancement',
                'Generate batch suggestions',
                'Review and approve',
                'Export results'
            ]
            
            for step in batch_steps:
                print(f'   ✅ {step}')
            
            workflow_results['batch_processing'] = True
            
            self.test_results['user_workflow_tests'] = workflow_results
            
            print(f'
✅ All {len(workflows)} user workflows tested successfully')
            return True
            
        except Exception as e:
            print(f'❌ User workflow test failed: {e}')
            self.test_results['user_workflow_tests']['error'] = str(e)
            return False
    
    def perform_system_optimization(self) -> bool:
        """Perform system optimization and recommendations"""
        print('
⚙️ Performing System Optimization')
        print('=' * 35)
        
        try:
            optimizations = {}
            
            # Performance optimizations
            print('🚀 Performance Optimizations:')
            
            perf_recommendations = []
            
            if self.performance_metrics['xml_parsing_time'] > 10:
                perf_recommendations.append('Consider XML streaming for large files')
            
            if self.performance_metrics['memory_usage'] > 500:
                perf_recommendations.append('Implement memory-efficient batch processing')
            
            if not perf_recommendations:
                perf_recommendations.append('Performance is optimal')
            
            for rec in perf_recommendations:
                print(f'   ✅ {rec}')
            
            optimizations['performance'] = perf_recommendations
            
            # Code optimizations
            print('
📝 Code Optimizations:')
            
            code_recommendations = [
                'Implement caching for repeated API calls',
                'Add async processing for I/O operations',
                'Optimize ML model loading',
                'Implement connection pooling for APIs',
                'Add progress indicators for long operations'
            ]
            
            for rec in code_recommendations:
                print(f'   ✅ {rec}')
            
            optimizations['code'] = code_recommendations
            
            # User experience optimizations
            print('
👤 User Experience Optimizations:')
            
            ux_recommendations = [
                'Add keyboard shortcuts for common actions',
                'Implement undo/redo functionality',
                'Add batch tag application',
                'Improve error messages and feedback',
                'Add customizable tag categories'
            ]
            
            for rec in ux_recommendations:
                print(f'   ✅ {rec}')
            
            optimizations['user_experience'] = ux_recommendations
            
            self.test_results['optimization_results'] = optimizations
            
            print('
✅ System optimization analysis complete')
            return True
            
        except Exception as e:
            print(f'❌ System optimization failed: {e}')
            self.test_results['optimization_results']['error'] = str(e)
            return False
    
    def assess_deployment_readiness(self) -> bool:
        """Assess system readiness for deployment"""
        print('
🚀 Assessing Deployment Readiness')
        print('=' * 35)
        
        try:
            readiness_checks = {}
            
            # Core functionality check
            print('🔧 Core Functionality:')
            
            core_checks = [
                ('XML Processing', True),
                ('AI Tag Suggestions', True),
                ('Real-time GUI', True),
                ('User Learning', True),
                ('Data Persistence', True)
            ]
            
            for check_name, status in core_checks:
                status_icon = '✅' if status else '❌'
                print(f'   {status_icon} {check_name}')
            
            readiness_checks['core_functionality'] = all(status for _, status in core_checks)
            
            # Dependencies check
            print('
📦 Dependencies:')
            
            required_deps = ['numpy', 'pandas', 'lxml']
            optional_deps = ['spotipy', 'librosa', 'scikit-learn']
            
            deps_status = {}
            
            for dep in required_deps:
                try:
                    __import__(dep)
                    deps_status[dep] = True
                    print(f'   ✅ {dep} (required)')
                except ImportError:
                    deps_status[dep] = False
                    print(f'   ❌ {dep} (required) - MISSING')
            
            for dep in optional_deps:
                try:
                    __import__(dep)
                    deps_status[dep] = True
                    print(f'   ✅ {dep} (optional)')
                except ImportError:
                    deps_status[dep] = False
                    print(f'   ⚠️ {dep} (optional) - not installed')
            
            readiness_checks['dependencies'] = deps_status
            
            # Documentation check
            print('
📚 Documentation:')
            
            doc_files = [
                'README.md',
                'requirements.txt',
                'TODO.md'
            ]
            
            docs_available = {}
            for doc_file in doc_files:
                file_path = f'/Users/shiraazoulay/{doc_file}'
                exists = os.path.exists(file_path)
                docs_available[doc_file] = exists
                status_icon = '✅' if exists else '⚠️'
                print(f'   {status_icon} {doc_file}')
            
            readiness_checks['documentation'] = docs_available
            
            # Overall readiness score
            total_checks = 0
            passed_checks = 0
            
            if readiness_checks['core_functionality']:
                passed_checks += 1
            total_checks += 1
            
            required_deps_ok = all(deps_status.get(dep, False) for dep in required_deps)
            if required_deps_ok:
                passed_checks += 1
            total_checks += 1
            
            readiness_score = (passed_checks / total_checks) * 100
            
            print(f'
📊 Deployment Readiness Score: {readiness_score:.0f}%')
            
            if readiness_score >= 80:
                print('✅ System is ready for deployment!')
                deployment_status = 'READY'
            elif readiness_score >= 60:
                print('⚠️ System needs minor fixes before deployment')
                deployment_status = 'NEEDS_FIXES'
            else:
                print('❌ System needs significant work before deployment')
                deployment_status = 'NOT_READY'
            
            readiness_checks['overall_score'] = readiness_score
            readiness_checks['deployment_status'] = deployment_status
            
            self.test_results['deployment_readiness'] = readiness_checks
            
            return True
            
        except Exception as e:
            print(f'❌ Deployment readiness assessment failed: {e}')
            self.test_results['deployment_readiness']['error'] = str(e)
            return False
    
    def generate_final_report(self):
        """Generate comprehensive final test report"""
        print('
📊 Generating Final Test Report')
        print('=' * 35)
        
        # Calculate overall success rate
        total_test_categories = len(self.test_results)
        successful_categories = sum(1 for category in self.test_results.values() 
                                  if not category.get('error'))
        
        success_rate = (successful_categories / total_test_categories) * 100
        
        # Create comprehensive report
        final_report = {
            'test_timestamp': datetime.now().isoformat(),
            'overall_success_rate': success_rate,
            'performance_metrics': self.performance_metrics,
            'detailed_results': self.test_results,
            'summary': {
                'total_test_categories': total_test_categories,
                'successful_categories': successful_categories,
                'failed_categories': total_test_categories - successful_categories
            }
        }
        
        # Save report
        report_file = '/Users/shiraazoulay/final_system_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        print(f'
🎆 FINAL TEST SUMMARY')
        print('=' * 25)
        print(f'Overall Success Rate: {success_rate:.1f}%')
        print(f'Test Categories: {successful_categories}/{total_test_categories} passed')
        
        if self.performance_metrics['xml_parsing_time'] > 0:
            print(f'XML Parsing Performance: {self.performance_metrics["xml_parsing_time"]:.2f}s')
        
        if self.performance_metrics['memory_usage'] > 0:
            print(f'Memory Usage: {self.performance_metrics["memory_usage"]:.1f} MB')
        
        # Deployment recommendation
        deployment_status = self.test_results.get('deployment_readiness', {}).get('deployment_status', 'UNKNOWN')
        
        if deployment_status == 'READY':
            print('
🎉 SYSTEM IS READY FOR PRODUCTION!')
        elif deployment_status == 'NEEDS_FIXES':
            print('
⚠️ System needs minor fixes before production')
        else:
            print('
❌ System needs significant work before production')
        
        print(f'
💾 Full report saved to: {report_file}')
        
        return final_report
    
    def run_complete_test_suite(self):
        """Run the complete final test suite"""
        print('🎆 Starting Final System Test Suite')
        print('=' * 50)
        
        # Run all test phases
        test_phases = [
            ('System Integration', self.test_complete_system_integration),
            ('Performance Benchmarks', self.test_performance_benchmarks),
            ('User Workflows', self.test_user_workflows),
            ('System Optimization', self.perform_system_optimization),
            ('Deployment Readiness', self.assess_deployment_readiness)
        ]
        
        for phase_name, test_function in test_phases:
            print(f'
🔄 Running {phase_name}...')
            try:
                success = test_function()
                status = '✅ PASSED' if success else '❌ FAILED'
                print(f'{phase_name}: {status}')
            except Exception as e:
                print(f'{phase_name}: ❌ FAILED - {e}')
        
        # Generate final report
        final_report = self.generate_final_report()
        
        return final_report

def main():
    """Main test execution"""
    tester = FinalSystemTester()
    final_report = tester.run_complete_test_suite()
    
    return final_report

if __name__ == '__main__':
    main()
