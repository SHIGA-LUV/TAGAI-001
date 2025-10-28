#!/usr/bin/env python3
"""
Rekordbox AI Tagger - Chat Style Interface
Provides AI tag suggestions in a modern chat-style window
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading
import json

class ChatStyleTagger:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 Vy AI Tagger Chat")
        self.root.geometry("450x600")
        self.root.configure(bg='#1a1a1a')
        
        # Keep window on top
        self.root.attributes('-topmost', True)
        
        self.selected_tags = []
        self.current_track = None
        
        self.setup_chat_interface()
        self.add_welcome_message()
        
    def setup_chat_interface(self):
        """Create the chat-style interface"""
        # Header
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🎵 Vy AI Tagger",
                              font=('Arial', 16, 'bold'),
                              fg='#00ff88',
                              bg='#2d2d2d')
        title_label.pack(pady=15)
        
        # Chat area
        self.chat_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.chat_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollable chat area
        self.chat_canvas = tk.Canvas(self.chat_frame, bg='#1a1a1a', highlightthickness=0)
        self.chat_scrollbar = ttk.Scrollbar(self.chat_frame, orient='vertical', command=self.chat_canvas.yview)
        self.scrollable_frame = tk.Frame(self.chat_canvas, bg='#1a1a1a')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        
        self.chat_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_scrollbar.pack(side="right", fill="y")
        
        # Input area
        input_frame = tk.Frame(self.root, bg='#2d2d2d', height=80)
        input_frame.pack(fill='x', padx=10, pady=5)
        input_frame.pack_propagate(False)
        
        # Custom tag input
        tk.Label(input_frame, text="Add custom tag:", 
                fg='#ffffff', bg='#2d2d2d', font=('Arial', 10)).pack(anchor='w', pady=2)
        
        entry_frame = tk.Frame(input_frame, bg='#2d2d2d')
        entry_frame.pack(fill='x', pady=2)
        
        self.custom_entry = tk.Entry(entry_frame, 
                                   font=('Arial', 12),
                                   bg='#3d3d3d',
                                   fg='#ffffff',
                                   insertbackground='#ffffff',
                                   relief='flat',
                                   bd=5)
        self.custom_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        send_btn = tk.Button(entry_frame,
                           text="Add",
                           font=('Arial', 10, 'bold'),
                           bg='#00ff88',
                           fg='#000000',
                           relief='flat',
                           command=self.add_custom_tag,
                           padx=15)
        send_btn.pack(side='right')
        
        # Bind Enter key
        self.custom_entry.bind('<Return>', lambda e: self.add_custom_tag())
        
    def add_message(self, sender: str, message: str, msg_type: str = "normal", tags: List[str] = None):
        """Add a message to the chat"""
        timestamp = datetime.now().strftime("%H:%M")
        
        # Message container
        msg_container = tk.Frame(self.scrollable_frame, bg='#1a1a1a')
        msg_container.pack(fill='x', padx=5, pady=3)
        
        if sender == "Vy AI":
            # AI message (left side)
            msg_frame = tk.Frame(msg_container, bg='#2d2d2d', relief='solid', bd=1)
            msg_frame.pack(anchor='w', fill='x', padx=(0, 50))
            
            # Header
            header_frame = tk.Frame(msg_frame, bg='#2d2d2d')
            header_frame.pack(fill='x', padx=10, pady=5)
            
            tk.Label(header_frame, text=f"🤖 {sender}", 
                    font=('Arial', 10, 'bold'), 
                    fg='#00ff88', bg='#2d2d2d').pack(side='left')
            
            tk.Label(header_frame, text=timestamp, 
                    font=('Arial', 8), 
                    fg='#888888', bg='#2d2d2d').pack(side='right')
            
            # Message content
            content_frame = tk.Frame(msg_frame, bg='#2d2d2d')
            content_frame.pack(fill='x', padx=10, pady=(0, 10))
            
            tk.Label(content_frame, text=message, 
                    font=('Arial', 11), 
                    fg='#ffffff', bg='#2d2d2d',
                    wraplength=350, justify='left').pack(anchor='w')
            
            # Add tag buttons if provided
            if tags:
                self.add_tag_buttons(content_frame, tags)
                
        else:
            # User message (right side)
            msg_frame = tk.Frame(msg_container, bg='#00ff88', relief='solid', bd=1)
            msg_frame.pack(anchor='e', padx=(50, 0))
            
            # Header
            header_frame = tk.Frame(msg_frame, bg='#00ff88')
            header_frame.pack(fill='x', padx=10, pady=5)
            
            tk.Label(header_frame, text=timestamp, 
                    font=('Arial', 8), 
                    fg='#000000', bg='#00ff88').pack(side='left')
            
            tk.Label(header_frame, text=f"{sender} 🎧", 
                    font=('Arial', 10, 'bold'), 
                    fg='#000000', bg='#00ff88').pack(side='right')
            
            # Message content
            tk.Label(msg_frame, text=message, 
                    font=('Arial', 11), 
                    fg='#000000', bg='#00ff88',
                    wraplength=300, justify='right',
                    padx=10, pady=(0, 10)).pack(anchor='e')
        
        # Auto-scroll to bottom
        self.root.after(100, self._scroll_to_bottom)
        
    def add_tag_buttons(self, parent: tk.Frame, tags: List[str]):
        """Add clickable tag buttons"""
        tags_frame = tk.Frame(parent, bg='#2d2d2d')
        tags_frame.pack(fill='x', pady=5)
        
        tk.Label(tags_frame, text="💡 Suggested tags:", 
                font=('Arial', 9, 'bold'), 
                fg='#ffaa00', bg='#2d2d2d').pack(anchor='w')
        
        # Create tag buttons in rows
        button_frame = tk.Frame(tags_frame, bg='#2d2d2d')
        button_frame.pack(fill='x', pady=2)
        
        for i, tag in enumerate(tags):
            color = self.get_tag_color(tag)
            
            btn = tk.Button(button_frame,
                          text=f"+ {tag}",
                          font=('Arial', 9),
                          bg=color,
                          fg='#ffffff',
                          relief='flat',
                          padx=8, pady=2,
                          command=lambda t=tag: self.select_tag(t))
            btn.pack(side='left', padx=2, pady=1)
            
            # New row every 3 tags
            if (i + 1) % 3 == 0:
                button_frame = tk.Frame(tags_frame, bg='#2d2d2d')
                button_frame.pack(fill='x', pady=1)
        
        # Apply button
        apply_frame = tk.Frame(tags_frame, bg='#2d2d2d')
        apply_frame.pack(fill='x', pady=5)
        
        apply_btn = tk.Button(apply_frame,
                            text="✅ Apply Selected Tags",
                            font=('Arial', 10, 'bold'),
                            bg='#ff6b6b',
                            fg='#ffffff',
                            relief='flat',
                            command=self.apply_tags,
                            padx=15, pady=5)
        apply_btn.pack(side='right')
        
    def get_tag_color(self, tag: str) -> str:
        """Get color for tag based on category"""
        situation_tags = ['1-Opener', '2-Build up', '3-Peak Time', '4-Cool Down', '5-After Hours']
        genre_tags = ['Melodic Techno', 'Ethnic House', 'Progressive House', 'Deep House', 'Afro House']
        component_tags = ['Piano', 'Darbuka', 'Female Vocal', 'Male Vocal', 'Strings', 'Synth Lead']
        mood_tags = ['Tribal', 'Dreamy', 'Sexy', 'Energetic', 'Emotional', 'Dark', 'Uplifting']
        
        if tag in situation_tags:
            return '#ff6b6b'  # Red
        elif tag in genre_tags:
            return '#4ecdc4'  # Teal
        elif tag in component_tags:
            return '#45b7d1'  # Blue
        elif tag in mood_tags:
            return '#96ceb4'  # Green
        else:
            return '#9b59b6'  # Purple for custom
    
    def select_tag(self, tag: str):
        """Handle tag selection"""
        if tag not in self.selected_tags:
            self.selected_tags.append(tag)
            self.add_message("You", f"Selected: {tag}")
            
    def add_custom_tag(self):
        """Add custom tag from input"""
        custom_tag = self.custom_entry.get().strip()
        if custom_tag and custom_tag not in self.selected_tags:
            self.selected_tags.append(custom_tag)
            self.custom_entry.delete(0, tk.END)
            self.add_message("You", f"Added custom tag: {custom_tag}")
            
    def apply_tags(self):
        """Apply selected tags"""
        if self.selected_tags:
            tags_text = ", ".join(self.selected_tags)
            self.add_message("Vy AI", f"✅ Applied {len(self.selected_tags)} tags to your track: {tags_text}")
            self.selected_tags = []
        else:
            self.add_message("Vy AI", "❌ No tags selected. Please select some tags first!")
    
    def add_welcome_message(self):
        """Add welcome message"""
        welcome_msg = "👋 Hey! I'm your AI tagging assistant. I'll help you tag your tracks with smart suggestions based on BPM, key, and musical characteristics."
        self.add_message("Vy AI", welcome_msg)
        
    def analyze_track(self, track_info: Dict):
        """Analyze track and provide suggestions"""
        self.current_track = track_info
        
        # Simulate AI analysis
        track_name = track_info.get('title', 'Unknown Track')
        artist = track_info.get('artist', 'Unknown Artist')
        bpm = track_info.get('bpm', 128)
        key = track_info.get('key', 'Unknown')
        
        analysis_msg = f"🎵 Analyzing: {track_name} by {artist}\nBPM: {bpm} | Key: {key}"
        self.add_message("Vy AI", analysis_msg)
        
        # Generate suggestions based on BPM and characteristics
        suggestions = self.generate_suggestions(track_info)
        
        suggestion_msg = "Here are my AI-powered tag suggestions for this track:"
        self.add_message("Vy AI", suggestion_msg, tags=suggestions)
        
    def generate_suggestions(self, track_info: Dict) -> List[str]:
        """Generate AI tag suggestions"""
        bpm = track_info.get('bpm', 128)
        key = track_info.get('key', '')
        genre = track_info.get('genre', '')
        
        suggestions = []
        
        # BPM-based suggestions
        if bpm < 100:
            suggestions.extend(['5-After Hours', 'Dreamy', 'Deep House'])
        elif bpm < 120:
            suggestions.extend(['4-Cool Down', 'Emotional', 'Progressive House'])
        elif bpm < 130:
            suggestions.extend(['2-Build up', 'Energetic', 'Melodic Techno'])
        else:
            suggestions.extend(['3-Peak Time', 'Dark', 'Ethnic House'])
        
        # Key-based suggestions
        if 'A' in key:
            suggestions.append('Uplifting')
        if 'm' in key.lower():
            suggestions.append('Emotional')
        
        # Add some component suggestions
        suggestions.extend(['Female Vocal', 'Synth Lead', 'Piano'])
        
        return list(set(suggestions))[:8]  # Limit to 8 suggestions
    
    def _scroll_to_bottom(self):
        """Scroll chat to bottom"""
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)
        
    def demo_mode(self):
        """Run demo with sample tracks"""
        demo_tracks = [
            {
                'title': 'Boy Oh Boy',
                'artist': 'Paons',
                'bpm': 128,
                'key': '7A',
                'genre': 'Progressive House'
            },
            {
                'title': 'Digital Mess',
                'artist': 'Unknown',
                'bpm': 132,
                'key': '6B',
                'genre': 'Melodic Techno'
            }
        ]
        
        def analyze_next_track(index=0):
            if index < len(demo_tracks):
                self.analyze_track(demo_tracks[index])
                # Schedule next track analysis
                self.root.after(5000, lambda: analyze_next_track(index + 1))
        
        analyze_next_track()
        
    def run(self):
        """Start the chat interface"""
        # Start demo mode after 2 seconds
        self.root.after(2000, self.demo_mode)
        self.root.mainloop()

def main():
    app = ChatStyleTagger()
    app.run()

if __name__ == "__main__":
    main()
