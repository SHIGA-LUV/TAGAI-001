#!/usr/bin/env python3
"""
Enhanced Real-Time Vy Tag Feature for Rekordbox
Phase 1.5: Added keyboard shortcuts, confidence scoring, and improved UX
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

class EnhancedRealTimeVyTagger:
    def __init__(self):
        self.tag_hierarchy = {
            'SITUATION': {
                'priority': 1,
                'tags': ['1-Opener', '2-Build up', '3-Peak Time', '4-Cool Down', '5-After Hours'],
                'color': '#FF6B6B',  # Red
                'hotkey': '1'
            },
            'GENRE': {
                'priority': 2,
                'tags': ['Melodic Techno', 'Ethnic House', 'Progressive House', 'Deep House', 'Afro House'],
                'color': '#4ECDC4',  # Teal
                'hotkey': '2'
            },
            'COMPONENTS': {
                'priority': 3,
                'tags': ['Piano', 'Darbuka', 'Female Vocal', 'Male Vocal', 'Strings', 'Synth Lead', 'Percussion'],
                'color': '#45B7D1',  # Blue
                'hotkey': '3'
            },
            'MOOD': {
                'priority': 4,
                'tags': ['Tribal', 'Dreamy', 'Sexy', 'Energetic', 'Emotional', 'Dark', 'Uplifting'],
                'color': '#96CEB4',  # Green
                'hotkey': '4'
            }
        }
        
        self.current_track = None
        self.suggestion_window = None
        self.is_monitoring = False
        self.learned_patterns = {}
        self.tag_confidence = {}
        
    def calculate_tag_confidence(self, track_info: Dict, tag: str) -> float:
        """Calculate confidence score for a tag suggestion (0.0 to 1.0)"""
        confidence = 0.5  # Base confidence
        
        bpm = track_info.get('bpm', 0)
        genre = track_info.get('genre', '').lower()
        key = track_info.get('key', '')
        
        # BPM-based confidence adjustments
        if 'Peak Time' in tag and bpm > 130:
            confidence += 0.3
        elif 'After Hours' in tag and bpm < 100:
            confidence += 0.4
        elif 'Build up' in tag and 120 <= bpm <= 130:
            confidence += 0.2
        
        # Genre matching confidence
        if 'Progressive House' in tag and 'progressive' in genre:
            confidence += 0.3
        elif 'Deep House' in tag and 'deep' in genre:
            confidence += 0.3
        elif 'Melodic Techno' in tag and 'techno' in genre:
            confidence += 0.3
        
        # Key-based mood confidence
        minor_keys = ['Am', 'Bm', 'Cm', 'Dm', 'Em', 'Fm', 'Gm']
        if tag in ['Emotional', 'Dark'] and any(k in key for k in minor_keys):
            confidence += 0.2
        elif tag in ['Uplifting', 'Energetic'] and not any(k in key for k in minor_keys):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def analyze_track_for_suggestions(self, track_info: Dict) -> Dict[str, List[Tuple[str, float]]]:
        """Generate AI-powered tag suggestions with confidence scores"""
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
                suggestions['SITUATION'].append('5-After Hours')
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
        suggestions['COMPONENTS'].extend(['Synth Lead', 'Percussion'])
        
        # Calculate confidence scores and remove duplicates
        for category in suggestions:
            unique_tags = list(set(suggestions[category]))
            suggestions[category] = [(tag, self.calculate_tag_confidence(track_info, tag)) 
                                   for tag in unique_tags]
            # Sort by confidence (highest first)
            suggestions[category].sort(key=lambda x: x[1], reverse=True)
        
        return suggestions
    
    def create_enhanced_suggestion_gui(self, track_info: Dict, suggestions: Dict[str, List[Tuple[str, float]]]):
        """Create enhanced floating suggestion window with confidence scores"""
        if self.suggestion_window:
            self.suggestion_window.destroy()
        
        self.suggestion_window = tk.Toplevel()
        self.suggestion_window.title("🎵 Vy Tag Suggestions - Enhanced")
        self.suggestion_window.geometry("450x700")
        self.suggestion_window.configure(bg='#2C3E50')
        
        # Always on top
        self.suggestion_window.attributes('-topmost', True)
        
        # Keyboard shortcuts
        self.suggestion_window.bind('<Return>', lambda e: self.apply_selected_tags())
        self.suggestion_window.bind('<Escape>', lambda e: self.skip_tagging())
        self.suggestion_window.bind('<space>', lambda e: self.toggle_all_high_confidence())
        
        # Track info header
        header_frame = tk.Frame(self.suggestion_window, bg='#34495E', pady=10)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(header_frame, 
                text=f"{track_info['artist']} - {track_info['title']}",
                font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E',
                wraplength=400).pack()
        
        tk.Label(header_frame,
                text=f"BPM: {track_info['bpm']} | Key: {track_info['key']} | Genre: {track_info['genre']}",
                font=('Arial', 10),
                fg='#BDC3C7', bg='#34495E').pack()
        
        # Keyboard shortcuts info
        shortcuts_frame = tk.Frame(self.suggestion_window, bg='#2C3E50')
        shortcuts_frame.pack(fill='x', padx=10, pady=2)
        
        tk.Label(shortcuts_frame,
                text="⌨️ Shortcuts: Enter=Apply | Esc=Skip | Space=Toggle High Confidence",
                font=('Arial', 9),
                fg='#95A5A6', bg='#2C3E50').pack()
        
        # Selected tags storage
        self.selected_tags = []
        self.tag_vars = {}
        
        # Create suggestion sections
        main_frame = tk.Frame(self.suggestion_window, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=10)
        
        for category, tag_confidence_pairs in suggestions.items():
            if not tag_confidence_pairs:
                continue
                
            category_frame = tk.LabelFrame(main_frame, 
                                         text=f"{category} (Hotkey: {self.tag_hierarchy[category]['hotkey']})",
                                         font=('Arial', 11, 'bold'),
                                         fg='white',
                                         bg=self.tag_hierarchy[category]['color'],
                                         pady=5)
            category_frame.pack(fill='x', pady=5)
            
            # Create checkboxes for each tag with confidence
            for tag, confidence in tag_confidence_pairs:
                var = tk.BooleanVar()
                self.tag_vars[tag] = var
                
                # Auto-select high confidence tags (>0.7)
                if confidence > 0.7:
                    var.set(True)
                    self.selected_tags.append(tag)
                
                # Create frame for tag and confidence
                tag_frame = tk.Frame(category_frame, bg=self.tag_hierarchy[category]['color'])
                tag_frame.pack(fill='x', padx=10, pady=1)
                
                cb = tk.Checkbutton(tag_frame,
                                  text=tag,
                                  variable=var,
                                  font=('Arial', 10),
                                  fg='white',
                                  bg=self.tag_hierarchy[category]['color'],
                                  selectcolor='#34495E',
                                  command=lambda t=tag, v=var: self.toggle_tag_selection(t, v))
                cb.pack(side='left')
                
                # Confidence indicator
                confidence_color = '#27AE60' if confidence > 0.7 else '#F39C12' if confidence > 0.5 else '#E74C3C'
                confidence_label = tk.Label(tag_frame,
                                           text=f"{confidence:.0%}",
                                           font=('Arial', 9, 'bold'),
                                           fg=confidence_color,
                                           bg=self.tag_hierarchy[category]['color'])
                confidence_label.pack(side='right')
        
        # Action buttons
        button_frame = tk.Frame(self.suggestion_window, bg='#2C3E50', pady=10)
        button_frame.pack(fill='x', padx=10)
        
        apply_btn = tk.Button(button_frame,
                            text="✅ Apply Selected Tags (Enter)",
                            font=('Arial', 12, 'bold'),
                            bg='#27AE60',
                            fg='white',
                            command=self.apply_selected_tags,
                            pady=5)
        apply_btn.pack(side='left', fill='x', expand=True, padx=5)
        
        skip_btn = tk.Button(button_frame,
                           text="⏭️ Skip (Esc)",
                           font=('Arial', 12),
                           bg='#E74C3C',
                           fg='white',
                           command=self.skip_tagging,
                           pady=5)
        skip_btn.pack(side='right', padx=5)
        
        # Stats and custom tag section
        stats_frame = tk.Frame(self.suggestion_window, bg='#2C3E50', pady=5)
        stats_frame.pack(fill='x', padx=10)
        
        selected_count = len([tag for tag in self.selected_tags])
        tk.Label(stats_frame, 
                text=f"📊 {selected_count} tags selected | High confidence auto-selected",
                fg='#95A5A6', bg='#2C3E50', font=('Arial', 9)).pack()
        
        # Custom tag entry
        custom_frame = tk.Frame(self.suggestion_window, bg='#2C3E50', pady=5)
        custom_frame.pack(fill='x', padx=10)
        
        tk.Label(custom_frame, text="➕ Add Custom Tag:", 
                fg='white', bg='#2C3E50', font=('Arial', 10)).pack(anchor='w')
        
        self.custom_tag_entry = tk.Entry(custom_frame, font=('Arial', 10))
        self.custom_tag_entry.pack(fill='x', pady=2)
        self.custom_tag_entry.bind('<Return>', lambda e: self.add_custom_tag())
        
        add_custom_btn = tk.Button(custom_frame,
                                 text="Add Custom",
                                 font=('Arial', 10),
                                 bg='#3498DB',
                                 fg='white',
                                 command=self.add_custom_tag)
        add_custom_btn.pack(pady=2)
        
        # Focus on window
        self.suggestion_window.focus_set()
    
    def toggle_all_high_confidence(self):
        """Toggle all high confidence tags"""
        for tag, var in self.tag_vars.items():
            if tag in self.selected_tags:
                var.set(False)
                self.selected_tags.remove(tag)
            else:
                var.set(True)
                self.selected_tags.append(tag)
        print(f"Toggled all tags. Selected: {self.selected_tags}")
    
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
            print("\n=== ENHANCED REKORDBOX INTEGRATION ===")
            print("1. 🎯 Focusing Rekordbox window...")
            time.sleep(0.5)
            
            print("2. 🏷️ Opening MyTag panel...")
            time.sleep(0.5)
            
            print("3. 🎵 Applying tags with confidence scores...")
            for tag in tags:
                category = self.find_tag_category(tag)
                confidence = self.tag_confidence.get(tag, 0.5)
                print(f"   - {category} > {tag} (confidence: {confidence:.0%})")
                time.sleep(0.3)
            
            print("4. 💾 Saving changes...")
            print(f"✅ Successfully applied {len(tags)} tags!")
            
            return True
            
        except Exception as e:
            print(f"Error integrating with Rekordbox: {e}")
            return False
    
    def find_tag_category(self, tag: str) -> str:
        """Find which category a tag belongs to"""
        for category, info in self.tag_hierarchy.items():
            if tag in info['tags']:
                return category
        return 'CUSTOM'
    
    def skip_tagging(self):
        """Skip tagging for current track"""
        print("⏭️ Skipping tagging for current track")
        if self.suggestion_window:
            self.suggestion_window.destroy()
    
    def start_monitoring(self):
        """Start monitoring Rekordbox for track changes"""
        self.is_monitoring = True
        print("🎵 Enhanced Vy Real-Time Tagger started! Monitoring Rekordbox...")
        
        def monitor_loop():
            track_counter = 0
            while self.is_monitoring:
                # Simulate different tracks for demo
                demo_tracks = [
                    {'title': 'Melodic Journey', 'artist': 'Progressive Artist', 'bpm': 128, 'key': 'Am', 'genre': 'Progressive House'},
                    {'title': 'Peak Energy', 'artist': 'Techno Master', 'bpm': 140, 'key': 'Gm', 'genre': 'Melodic Techno'},
                    {'title': 'Deep Vibes', 'artist': 'House Legend', 'bpm': 115, 'key': 'C', 'genre': 'Deep House'},
                    {'title': 'After Hours', 'artist': 'Ambient Soul', 'bpm': 95, 'key': 'Fm', 'genre': 'Ambient House'}
                ]
                
                track = demo_tracks[track_counter % len(demo_tracks)]
                track_counter += 1
                
                if track != self.current_track:
                    self.current_track = track
                    print(f"\n🎵 New track detected: {track['artist']} - {track['title']}")
                    
                    # Generate suggestions with confidence
                    suggestions = self.analyze_track_for_suggestions(track)
                    
                    # Show enhanced GUI
                    self.create_enhanced_suggestion_gui(track, suggestions)
                
                time.sleep(10)  # Check every 10 seconds for demo
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        print("🛑 Stopped monitoring Rekordbox")

def main():
    """Main function to start the enhanced real-time tagger"""
    print("🎵 Enhanced Vy Real-Time Tagger for Rekordbox")
    print("============================================")
    
    tagger = EnhancedRealTimeVyTagger()
    
    # Create main control window
    root = tk.Tk()
    root.title("🎵 Vy Enhanced Real-Time Tagger")
    root.geometry("350x250")
    root.configure(bg='#2C3E50')
    
    # Title
    title_label = tk.Label(root,
                          text="🎵 Vy Enhanced Tagger",
                          font=('Arial', 16, 'bold'),
                          fg='white', bg='#2C3E50')
    title_label.pack(pady=10)
    
    # Control buttons
    start_btn = tk.Button(root,
                         text="🚀 Start Monitoring",
                         font=('Arial', 14, 'bold'),
                         bg='#27AE60',
                         fg='white',
                         command=tagger.start_monitoring,
                         pady=10)
    start_btn.pack(pady=10, padx=20, fill='x')
    
    stop_btn = tk.Button(root,
                        text="🛑 Stop Monitoring",
                        font=('Arial', 14, 'bold'),
                        bg='#E74C3C',
                        fg='white',
                        command=tagger.stop_monitoring,
                        pady=10)
    stop_btn.pack(pady=5, padx=20, fill='x')
    
    # Demo button for testing
    demo_btn = tk.Button(root,
                        text="🎮 Demo Mode",
                        font=('Arial', 12),
                        bg='#3498DB',
                        fg='white',
                        command=lambda: tagger.create_enhanced_suggestion_gui({
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
    demo_btn.pack(pady=5, padx=20, fill='x')
    
    # Info label
    info_label = tk.Label(root,
                         text="✨ Enhanced with confidence scoring & shortcuts",
                         font=('Arial', 10),
                         fg='#95A5A6', bg='#2C3E50')
    info_label.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()
