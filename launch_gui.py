#!/usr/bin/env python3
# Launch script for the Real-Time Vy Tagger GUI

import sys
import os
sys.path.append('/Users/shiraazoulay')

try:
    from realtime_vy_tagger import RealTimeVyTagger
    print('✅ Successfully imported RealTimeVyTagger')
    
    # Launch the main GUI (based on the main() function we found)
    print('🎆 Starting Real-Time Tagger GUI...')
    
    # Import and run the main function directly
    from realtime_vy_tagger import main
    main()
            
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Checking if realtime_vy_tagger.py exists...')
    
    if os.path.exists('/Users/shiraazoulay/realtime_vy_tagger.py'):
        print('✅ File exists, but import failed')
        print('This might be due to missing dependencies')
    else:
        print('❌ File not found')
        
except Exception as e:
    print(f'❌ Error launching GUI: {e}')
    import traceback
    traceback.print_exc()
