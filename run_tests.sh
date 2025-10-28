#!/bin/bash

echo "🎯 REKORDBOX AI TAGGER - SYSTEM TEST"
echo "===================================="

cd /Users/shiraazoulay

echo ""
echo "🔍 Checking system components..."

# Check if core files exist
if [ -f "rekordbox_ai_tagger.py" ]; then
    echo "✅ Core XML processor found"
else
    echo "❌ Core XML processor missing"
fi

if [ -f "realtime_vy_tagger.py" ]; then
    echo "✅ Real-time GUI found"
else
    echo "❌ Real-time GUI missing"
fi

if [ -f "Documents/shigmusic.xml" ]; then
    echo "✅ Rekordbox XML file found"
else
    echo "❌ Rekordbox XML file missing"
fi

echo ""
echo "🐍 Running Python system test..."
python3 test_complete_system.py

echo ""
echo "✨ Test complete!"