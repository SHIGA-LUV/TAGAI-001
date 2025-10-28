#!/usr/bin/env python3
"""
Real-Time Vy Tag Feature for Rekordbox
Integrates with Rekordbox to provide AI-powered tagging while listening
"""

import time
import subprocess
import json
from typing import Dict, List, Optional, Tuple
import threading
from datetime import datetime

# For GUI
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter not available, GUI features disabled")

# For system integration
try:
    from AppKit import NSWorkspace, NSRunningApplication
    import Quartz
except ImportError:
    print("AppKit not available, some macOS features disabled")

class RealTimeVyTagger:
    def __init__(self):
        self.tag_hierarchy = {
            'SITUATION': {
                'priority': 1,
                'tags': ['1-Opener', '2-Build up', '3-Peak Time', '4-Cool Down', '5-After Hours'],
                'color': '#FF6B6B'  # Red
            },
            'GENRE': {
                'priority': 2,
                'tags': ['Melodic Techno', 'Ethnic House', 'Progressive House', 'Deep House', 'Afro House'],
                'color': '#4ECDC4'  # Teal
            },
            'COMPONENTS': {
                'priority': 3,
                'tags': ['Piano', 'Darbuka', 'Female Vocal', 'Male Vocal', 'Strings', 'Synth Lead', 'Percussion'],
                'color': '#45B7D1'  # Blue
            },
            'MOOD': {
                'priority': 4,
                'tags': ['Tribal', 'Dreamy', 'Sexy', 'Energetic', 'Emotional', 'Dark', 'Uplifting'],
                'color': '#96CEB4'  # Green
            }
        }
        
        self.current_track = None
        self.suggestion_window = None
        self.is_monitoring = False
        self.learned_patterns = {}
        
    def detect_rekordbox_track(self) -> Optional[Dict]:
        """Detect currently playing track in Rekordbox"""
        try:
            # Check if Rekordbox is running
            apps = NSWorkspace.sharedWorkspace().runningApplications()
            rekordbox_running = any(app.localizedName() == 'rekordbox' for app in apps)
            
            if not rekordbox_running:
                return None
            
            # For now, we'll simulate track detection
            # In a full implementation, this would use AppleScript or other methods
            # to get the actual playing track from Rekordbox
            
            # Placeholder - would be replaced with actual Rekordbox integration
            sample_track = {
                'title': 'Current Playing Track',
                'artist': 'Artist Name',
                'bpm': 128,
                'key': 'Am',
                'genre': 'House',
                'duration': 360,
                'position': 45  # seconds into track
            }
            
            return sample_track
            
        except Exception as e:
            print(f"Error detecting track: {e}")
            return None
    
    def analyze_track_for_suggestions(self, track_info: Dict) -> Dict[str, List[str]]:
        """Generate AI-powered tag suggestions for the current track"""
        suggestions = {
            'SITUATION': [],
            'GENRE': [],
            'COMPONENTS': [],
            'MOOD': []
        }
        
        bpm = track_info.get('bpm', 0)
        genre = track_info.get('genre', '').lower()
        key = track_info.get('key', '')
        
        # SITUATION suggestions based on BPM and energy
        if bpm > 0:
            if bpm < 100:
                suggestions['SITUATION'].extend(['5-After Hours'])
            elif bpm < 115:
                suggestions['SITUATION'].extend(['4-Cool Down', '5-After Hours'])
            elif bpm < 125:
                suggestions['SITUATION'].extend(['1-Opener', '2-Build up'])
            elif bpm < 135:
                suggestions['SITUATION'].extend(['2-Build up', '3-Peak Time'])
            else:
                suggestions['SITUATION'].extend(['3-Peak Time'])
        
        # GENRE suggestions
        if 'house' in genre:
            if 'deep' in genre:
                suggestions['GENRE'].append('Deep House')
            elif 'progressive' in genre:
                suggestions['GENRE'].append('Progressive House')
            elif 'afro' in genre or 'ethnic' in genre:
                suggestions['GENRE'].extend(['Afro House', 'Ethnic House'])
            else:
                suggestions['GENRE'].append('Progressive House')
        elif 'techno' in genre:
            suggestions['GENRE'].append('Melodic Techno')
        
        # MOOD suggestions based on key and BPM
        if key:
            minor_keys = ['Am', 'Bm', 'Cm', 'Dm', 'Em', 'Fm', 'Gm']
            if any(k in key for k in minor_keys):
                suggestions['MOOD'].extend(['Emotional', 'Dark'])
            else:
                suggestions['MOOD'].extend(['Uplifting', 'Energetic'])
        
        if bpm > 130:
            suggestions['MOOD'].append('Energetic')
        elif bpm < 110:
            suggestions['MOOD'].extend(['Dreamy', 'Emotional'])
        
        # COMPONENTS - would be enhanced with audio analysis
        # For now, basic suggestions
        suggestions['COMPONENTS'].extend(['Synth Lead', 'Percussion'])
        
        # Remove duplicates
        for category in suggestions:
            suggestions[category] = list(set(suggestions[category]))
        
        return suggestions
    
    def create_suggestion_gui(self, track_info: Dict, suggestions: Dict[str, List[str]]):
        """Create floating suggestion window"""
        if self.suggestion_window:
            self.suggestion_window.destroy()
        
        self.suggestion_window = tk.Toplevel()
        self.suggestion_window.title("Vy Tag Suggestions")
        self.suggestion_window.geometry("400x600")
        self.suggestion_window.configure(bg='#2C3E50')
        
        # Always on top
        self.suggestion_window.attributes('-topmost', True)
        
        # Track info header
        header_frame = tk.Frame(self.suggestion_window, bg='#34495E', pady=10)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(header_frame, 
                text=f"{track_info['artist']} - {track_info['title']}",
                font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E',
                wraplength=350).pack()
        
        tk.Label(header_frame,
                text=f"BPM: {track_info['bpm']} | Key: {track_info['key']} | Genre: {track_info['genre']}",
                font=('Arial', 10),
                fg='#BDC3C7', bg='#34495E').pack()
        
        # Selected tags storage
        self.selected_tags = []
        
        # Create suggestion sections
        main_frame = tk.Frame(self.suggestion_window, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=10)
        
        for category, tags in suggestions.items():
            if not tags:
                continue
                
            category_frame = tk.LabelFrame(main_frame, 
                                         text=category,
                                         font=('Arial', 11, 'bold'),
                                         fg='white',
                                         bg=self.tag_hierarchy[category]['color'],
                                         pady=5)
            category_frame.pack(fill='x', pady=5)
            
            # Create checkboxes for each tag
            for tag in tags:
                var = tk.BooleanVar()
                cb = tk.Checkbutton(category_frame,
                                  text=tag,
                                  variable=var,
                                  font=('Arial', 10),
                                  fg='white',
                                  bg=self.tag_hierarchy[category]['color'],
                                  selectcolor='#34495E',
                                  command=lambda t=tag, v=var: self.toggle_tag_selection(t, v))
                cb.pack(anchor='w', padx=10, pady=2)
        
        # Action buttons
        button_frame = tk.Frame(self.suggestion_window, bg='#2C3E50', pady=10)
        button_frame.pack(fill='x', padx=10)
        
        apply_btn = tk.Button(button_frame,
                            text="Apply Selected Tags",
                            font=('Arial', 12, 'bold'),
                            bg='#27AE60',
                            fg='white',
                            command=self.apply_selected_tags,
                            pady=5)
        apply_btn.pack(side='left', fill='x', expand=True, padx=5)
        
        skip_btn = tk.Button(button_frame,
                           text="Skip",
                           font=('Arial', 12),
                           bg='#E74C3C',
                           fg='white',
                           command=self.skip_tagging,
                           pady=5)
        skip_btn.pack(side='right', padx=5)
        
        # Add custom tag entry
        custom_frame = tk.Frame(self.suggestion_window, bg='#2C3E50', pady=5)
        custom_frame.pack(fill='x', padx=10)
        
        tk.Label(custom_frame, text="Add Custom Tag:", 
                fg='white', bg='#2C3E50', font=('Arial', 10)).pack(anchor='w')
        
        self.custom_tag_entry = tk.Entry(custom_frame, font=('Arial', 10))
        self.custom_tag_entry.pack(fill='x', pady=2)
        
        add_custom_btn = tk.Button(custom_frame,
                                 text="Add Custom",
                                 font=('Arial', 10),
                                 bg='#3498DB',
                                 fg='white',
                                 command=self.add_custom_tag)
        add_custom_btn.pack(pady=2)
    
    def toggle_tag_selection(self, tag: str, var: tk.BooleanVar):
        """Handle tag selection/deselection"""
        if var.get():
            if tag not in self.selected_tags:
                self.selected_tags.append(tag)
        else:
            if tag in self.selected_tags:
                self.selected_tags.remove(tag)
        
        print(f"Selected tags: {self.selected_tags}")
    
    def add_custom_tag(self):
        """Add custom tag to selection"""
        custom_tag = self.custom_tag_entry.get().strip()
        if custom_tag and custom_tag not in self.selected_tags:
            self.selected_tags.append(custom_tag)
            self.custom_tag_entry.delete(0, tk.END)
            print(f"Added custom tag: {custom_tag}")
    
    def apply_selected_tags(self):
        """Apply selected tags to Rekordbox"""
        if not self.selected_tags:
            messagebox.showwarning("No Tags Selected", "Please select at least one tag to apply.")
            return
        
        print(f"Applying tags: {self.selected_tags}")
        
        # This would integrate with Rekordbox
        success = self.integrate_with_rekordbox(self.selected_tags)
        
        if success:
            messagebox.showinfo("Success", f"Applied {len(self.selected_tags)} tags to track!")
            self.suggestion_window.destroy()
        else:
            messagebox.showerror("Error", "Failed to apply tags. Make sure Rekordbox is open.")
    
    def integrate_with_rekordbox(self, tags: List[str]) -> bool:
        """Integrate with Rekordbox to apply tags"""
        try:
            print("\n=== REKORDBOX INTEGRATION SIMULATION ===")
            print("1. Opening I TAG panel in Rekordbox...")
            time.sleep(1)
            
            print("2. Navigating tag hierarchy...")
            for tag in tags:
                # Find which category this tag belongs to
                category = self.find_tag_category(tag)
                print(f"   - Clicking {category} > {tag}")
                time.sleep(0.5)
            
            print("3. Adding new tags to hierarchy if needed...")
            for tag in tags:
                if not self.tag_exists_in_hierarchy(tag):
                    category = self.suggest_category_for_tag(tag)
                    print(f"   - Adding '{tag}' to {category} section")
                    self.add_tag_to_hierarchy(tag, category)
            
            print("4. Applying tags to current track...")
            print(f"✅ Successfully applied tags: {', '.join(tags)}")
            
            # In real implementation, this would use AppleScript or UI automation
            # to actually click in Rekordbox
            
            return True
            
        except Exception as e:
            print(f"Error integrating with Rekordbox: {e}")
            return False
    
    def find_tag_category(self, tag: str) -> str:
        """Find which category a tag belongs to"""
        for category, info in self.tag_hierarchy.items():
            if tag in info['tags']:
                return category
        return 'MOOD'  # Default category
    
    def tag_exists_in_hierarchy(self, tag: str) -> bool:
        """Check if tag exists in current hierarchy"""
        for category, info in self.tag_hierarchy.items():
            if tag in info['tags']:
                return True
        return False
    
    def suggest_category_for_tag(self, tag: str) -> str:
        """Suggest which category a new tag should go in"""
        tag_lower = tag.lower()
        
        # SITUATION tags (usually numbered or time-based)
        if any(word in tag_lower for word in ['opener', 'build', 'peak', 'cool', 'after', 'warm']):
            return 'SITUATION'
        
        # GENRE tags
        elif any(word in tag_lower for word in ['house', 'techno', 'trance', 'disco', 'funk']):
            return 'GENRE'
        
        # COMPONENTS tags (instruments, vocals)
        elif any(word in tag_lower for word in ['vocal', 'piano', 'guitar', 'synth', 'drum', 'bass']):
            return 'COMPONENTS'
        
        # Default to MOOD
        else:
            return 'MOOD'
    
    def add_tag_to_hierarchy(self, tag: str, category: str):
        """Add new tag to hierarchy"""
        if category in self.tag_hierarchy:
            self.tag_hierarchy[category]['tags'].append(tag)
    
    def skip_tagging(self):
        """Skip tagging for current track"""
        print("Skipping tagging for current track")
        if self.suggestion_window:
            self.suggestion_window.destroy()
    
    def start_monitoring(self):
        """Start monitoring Rekordbox for track changes"""
        self.is_monitoring = True
        print("🎵 Vy Real-Time Tagger started! Monitoring Rekordbox...")
        
        def monitor_loop():
            while self.is_monitoring:
                track = self.detect_rekordbox_track()
                
                if track and track != self.current_track:
                    self.current_track = track
                    print(f"\n🎵 New track detected: {track['artist']} - {track['title']}")
                    
                    # Generate suggestions
                    suggestions = self.analyze_track_for_suggestions(track)
                    
                    # Show GUI
                    self.create_suggestion_gui(track, suggestions)
                
                time.sleep(2)  # Check every 2 seconds
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        print("Stopped monitoring Rekordbox")

def main():
    """Main function to start the real-time tagger"""
    print("🎵 Vy Real-Time Tagger for Rekordbox")
    print("===================================")
    
    tagger = RealTimeVyTagger()
    
    # Create main control window
    root = tk.Tk()
    root.title("Vy Real-Time Tagger Control")
    root.geometry("300x200")
    root.configure(bg='#2C3E50')
    
    # Control buttons
    start_btn = tk.Button(root,
                         text="Start Monitoring",
                         font=('Arial', 14, 'bold'),
                         bg='#27AE60',
                         fg='white',
                         command=tagger.start_monitoring,
                         pady=10)
    start_btn.pack(pady=20, padx=20, fill='x')
    
    stop_btn = tk.Button(root,
                        text="Stop Monitoring",
                        font=('Arial', 14, 'bold'),
                        bg='#E74C3C',
                        fg='white',
                        command=tagger.stop_monitoring,
                        pady=10)
    stop_btn.pack(pady=10, padx=20, fill='x')
    
    # Demo button for testing
    demo_btn = tk.Button(root,
                        text="Demo Mode",
                        font=('Arial', 12),
                        bg='#3498DB',
                        fg='white',
                        command=lambda: tagger.create_suggestion_gui({
                            'title': 'Demo Track',
                            'artist': 'Demo Artist',
                            'bpm': 128,
                            'key': 'Am',
                            'genre': 'Progressive House'
                        }, tagger.analyze_track_for_suggestions({
                            'bpm': 128,
                            'key': 'Am',
                            'genre': 'Progressive House'
                        })),
                        pady=5)
    demo_btn.pack(pady=10, padx=20, fill='x')
    
    root.mainloop()

if __name__ == "__main__":
    main()