#!/usr/bin/env python3
"""
Rekordbox AppleScript Integration for AI MyTag DJ Assistant
Phase 3: Advanced Features - Direct Rekordbox Control
"""

import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class RekordboxAppleScriptIntegrator:
    def __init__(self):
        self.rekordbox_bundle_id = "com.pioneer.rekordbox"
        self.applescript_templates = self.load_applescript_templates()
        
    def load_applescript_templates(self) -> Dict[str, str]:
        """Load AppleScript templates for Rekordbox automation"""
        return {
            'check_rekordbox_running': '''
                tell application "System Events"
                    return (name of processes) contains "rekordbox"
                end tell
            ''',
            
            'get_current_track': '''
                tell application "rekordbox"
                    try
                        set currentTrack to current track
                        set trackTitle to title of currentTrack
                        set trackArtist to artist of currentTrack
                        set trackBPM to bpm of currentTrack
                        set trackKey to key of currentTrack
                        set trackGenre to genre of currentTrack
                        
                        return trackTitle & "|" & trackArtist & "|" & trackBPM & "|" & trackKey & "|" & trackGenre
                    on error
                        return "ERROR: No track playing or Rekordbox not accessible"
                    end try
                end tell
            ''',
            
            'open_mytag_panel': '''
                tell application "rekordbox"
                    activate
                    delay 0.5
                end tell
                
                tell application "System Events"
                    tell process "rekordbox"
                        -- Try to open MyTag panel (this may vary by Rekordbox version)
                        try
                            -- Look for MyTag in menu
                            click menu item "MyTag" of menu "View" of menu bar 1
                        on error
                            -- Alternative: use keyboard shortcut if available
                            key code 17 using {command down} -- Cmd+T (example)
                        end try
                        delay 1
                    end tell
                end tell
            ''',
            
            'add_tag_to_track': '''
                on addTagToTrack(tagName)
                    tell application "rekordbox"
                        activate
                        delay 0.5
                    end tell
                    
                    tell application "System Events"
                        tell process "rekordbox"
                            try
                                -- Navigate to MyTag panel
                                -- This is a simplified example - actual implementation depends on UI structure
                                
                                -- Look for tag in hierarchy
                                set tagFound to false
                                
                                -- Try to find and click the tag
                                repeat with tagElement in (every UI element whose name contains tagName)
                                    if exists tagElement then
                                        click tagElement
                                        set tagFound to true
                                        exit repeat
                                    end if
                                end repeat
                                
                                if not tagFound then
                                    return "ERROR: Tag '" & tagName & "' not found in MyTag panel"
                                else
                                    return "SUCCESS: Tag '" & tagName & "' applied"
                                end if
                                
                            on error errMsg
                                return "ERROR: " & errMsg
                            end try
                        end tell
                    end tell
                end addTagToTrack
            ''',
            
            'create_tag_hierarchy': '''
                on createTagHierarchy(categoryName, tagList)
                    tell application "rekordbox"
                        activate
                        delay 0.5
                    end tell
                    
                    tell application "System Events"
                        tell process "rekordbox"
                            try
                                -- Right-click in MyTag panel to create new category
                                -- This is a simplified example
                                
                                -- Create category folder
                                right click at {100, 200} -- Approximate position
                                delay 0.5
                                
                                -- Look for "New Folder" or similar option
                                try
                                    click menu item "New Folder" of menu 1
                                    delay 0.5
                                    
                                    -- Type category name
                                    keystroke categoryName
                                    key code 36 -- Enter
                                    
                                    return "SUCCESS: Category '" & categoryName & "' created"
                                on error
                                    return "ERROR: Could not create category"
                                end try
                                
                            on error errMsg
                                return "ERROR: " & errMsg
                            end try
                        end tell
                    end tell
                end createTagHierarchy
            ''',
            
            'get_track_comments': '''
                tell application "rekordbox"
                    try
                        set currentTrack to current track
                        set trackComments to comment of currentTrack
                        return trackComments
                    on error
                        return "ERROR: Could not get track comments"
                    end try
                end tell
            ''',
            
            'set_track_comments': '''
                on setTrackComments(newComments)
                    tell application "rekordbox"
                        try
                            set currentTrack to current track
                            set comment of currentTrack to newComments
                            return "SUCCESS: Comments updated"
                        on error
                            return "ERROR: Could not update comments"
                        end try
                    end tell
                end setTrackComments
            '''
        }
    
    def run_applescript(self, script: str) -> Tuple[bool, str]:
        """Execute AppleScript and return result"""
        try:
            # Clean up the script (remove extra whitespace)
            clean_script = '\n'.join(line.strip() for line in script.strip().split('\n') if line.strip())
            
            # Run the AppleScript
            result = subprocess.run(
                ['osascript', '-e', clean_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "AppleScript execution timed out"
        except Exception as e:
            return False, f"AppleScript execution failed: {str(e)}"
    
    def is_rekordbox_running(self) -> bool:
        """Check if Rekordbox is currently running"""
        success, result = self.run_applescript(self.applescript_templates['check_rekordbox_running'])
        return success and 'true' in result.lower()
    
    def get_current_track_info(self) -> Optional[Dict]:
        """Get information about the currently playing track"""
        if not self.is_rekordbox_running():
            print("⚠️  Rekordbox is not running")
            return None
        
        success, result = self.run_applescript(self.applescript_templates['get_current_track'])
        
        if success and not result.startswith('ERROR'):
            try:
                # Parse the result (title|artist|bpm|key|genre)
                parts = result.split('|')
                if len(parts) >= 5:
                    return {
                        'title': parts[0],
                        'artist': parts[1],
                        'bpm': parts[2],
                        'key': parts[3],
                        'genre': parts[4]
                    }
            except Exception as e:
                print(f"❌ Error parsing track info: {e}")
        
        print(f"❌ Could not get current track info: {result}")
        return None
    
    def open_mytag_panel(self) -> bool:
        """Open the MyTag panel in Rekordbox"""
        if not self.is_rekordbox_running():
            print("⚠️  Rekordbox is not running")
            return False
        
        print("🏷️  Opening MyTag panel...")
        success, result = self.run_applescript(self.applescript_templates['open_mytag_panel'])
        
        if success:
            print("✅ MyTag panel opened")
            return True
        else:
            print(f"❌ Failed to open MyTag panel: {result}")
            return False
    
    def apply_tag_to_current_track(self, tag_name: str) -> bool:
        """Apply a tag to the currently playing track"""
        if not self.is_rekordbox_running():
            print("⚠️  Rekordbox is not running")
            return False
        
        print(f"🏷️  Applying tag: {tag_name}")
        
        # First ensure MyTag panel is open
        self.open_mytag_panel()
        time.sleep(1)
        
        # Create the script with the tag name
        script = self.applescript_templates['add_tag_to_track'].replace('addTagToTrack', f'addTagToTrack("{tag_name}")')
        
        success, result = self.run_applescript(script)
        
        if success and 'SUCCESS' in result:
            print(f"✅ Tag '{tag_name}' applied successfully")
            return True
        else:
            print(f"❌ Failed to apply tag '{tag_name}': {result}")
            return False
    
    def apply_multiple_tags(self, tags: List[str]) -> Dict[str, bool]:
        """Apply multiple tags to the current track"""
        results = {}
        
        print(f"🚀 Applying {len(tags)} tags to current track...")
        
        # Open MyTag panel once
        if not self.open_mytag_panel():
            return {tag: False for tag in tags}
        
        for tag in tags:
            time.sleep(0.5)  # Small delay between tags
            results[tag] = self.apply_tag_to_current_track(tag)
        
        successful_tags = [tag for tag, success in results.items() if success]
        print(f"✅ Successfully applied {len(successful_tags)}/{len(tags)} tags")
        
        return results
    
    def get_track_comments(self) -> Optional[str]:
        """Get comments from the current track"""
        if not self.is_rekordbox_running():
            return None
        
        success, result = self.run_applescript(self.applescript_templates['get_track_comments'])
        
        if success and not result.startswith('ERROR'):
            return result
        return None
    
    def update_track_comments(self, new_comments: str) -> bool:
        """Update comments for the current track"""
        if not self.is_rekordbox_running():
            return False
        
        script = self.applescript_templates['set_track_comments'].replace('setTrackComments', f'setTrackComments("{new_comments}")')
        success, result = self.run_applescript(script)
        
        return success and 'SUCCESS' in result
    
    def create_tag_category(self, category_name: str, tags: List[str]) -> bool:
        """Create a new tag category with specified tags"""
        if not self.is_rekordbox_running():
            return False
        
        print(f"📁 Creating tag category: {category_name}")
        
        # This is a simplified implementation
        # Real implementation would need to interact with Rekordbox UI elements
        script = self.applescript_templates['create_tag_hierarchy']
        success, result = self.run_applescript(script)
        
        if success and 'SUCCESS' in result:
            print(f"✅ Category '{category_name}' created")
            return True
        else:
            print(f"❌ Failed to create category: {result}")
            return False
    
    def setup_ai_tag_hierarchy(self) -> bool:
        """Set up the AI tag hierarchy in Rekordbox"""
        print("🤖 Setting up AI tag hierarchy in Rekordbox...")
        
        tag_categories = {
            'AI-SITUATION': ['1-Opener', '2-Build up', '3-Peak Time', '4-Cool Down', '5-After Hours'],
            'AI-GENRE': ['Melodic Techno', 'Ethnic House', 'Progressive House', 'Deep House', 'Afro House'],
            'AI-COMPONENTS': ['Piano', 'Darbuka', 'Female Vocal', 'Male Vocal', 'Strings', 'Synth Lead', 'Percussion'],
            'AI-MOOD': ['Tribal', 'Dreamy', 'Sexy', 'Energetic', 'Emotional', 'Dark', 'Uplifting']
        }
        
        success_count = 0
        for category, tags in tag_categories.items():
            if self.create_tag_category(category, tags):
                success_count += 1
            time.sleep(1)  # Delay between categories
        
        print(f"✅ Set up {success_count}/{len(tag_categories)} tag categories")
        return success_count == len(tag_categories)
    
    def demo_integration(self):
        """Demo the Rekordbox integration"""
        print("🎵 REKORDBOX APPLESCRIPT INTEGRATION DEMO")
        print("=" * 50)
        
        # Check if Rekordbox is running
        if self.is_rekordbox_running():
            print("✅ Rekordbox is running")
            
            # Get current track info
            track_info = self.get_current_track_info()
            if track_info:
                print(f"\n🎵 Current Track:")
                print(f"   Title: {track_info['title']}")
                print(f"   Artist: {track_info['artist']}")
                print(f"   BPM: {track_info['bpm']}")
                print(f"   Key: {track_info['key']}")
                print(f"   Genre: {track_info['genre']}")
                
                # Demo tag application
                demo_tags = ['Progressive House', 'Energetic', 'Synth Lead']
                print(f"\n🎯 Demo: Applying tags {demo_tags}")
                
                results = self.apply_multiple_tags(demo_tags)
                
                print(f"\n📈 Results:")
                for tag, success in results.items():
                    status = "✅" if success else "❌"
                    print(f"   {status} {tag}")
            else:
                print("⚠️  No track currently playing or could not access track info")
        else:
            print("❌ Rekordbox is not running")
            print("\n📊 Simulating integration demo...")
            
            # Simulate the integration
            print(f"\n🎵 Simulated Current Track:")
            print(f"   Title: Progressive Journey")
            print(f"   Artist: Demo Artist")
            print(f"   BPM: 128")
            print(f"   Key: Am")
            print(f"   Genre: Progressive House")
            
            print(f"\n🎯 Simulated Tag Application:")
            demo_tags = ['2-Build up', 'Progressive House', 'Emotional', 'Synth Lead']
            
            for i, tag in enumerate(demo_tags, 1):
                print(f"   {i}. 🏷️  Applying '{tag}'...")
                time.sleep(0.5)
                print(f"      ✅ Success: Tag '{tag}' applied")
            
            print(f"\n✅ Successfully applied {len(demo_tags)} tags!")
        
        print(f"\n{'=' * 50}")
        print("✅ REKORDBOX INTEGRATION DEMO COMPLETE!")
        print("\n🚀 Key Features:")
        print("   • Direct Rekordbox control via AppleScript")
        print("   • Real-time track information retrieval")
        print("   • Automated MyTag panel interaction")
        print("   • Batch tag application")
        print("   • Tag hierarchy creation")
        print("   • Comment field integration")
        print("\n⚠️  Requirements:")
        print("   • macOS with AppleScript support")
        print("   • Rekordbox running and accessible")
        print("   • Accessibility permissions for automation")
        print("=" * 50)

def main():
    """Main function to demo the integration"""
    integrator = RekordboxAppleScriptIntegrator()
    integrator.demo_integration()

if __name__ == "__main__":
    main()
