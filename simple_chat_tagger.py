#!/usr/bin/env python3
"""
Simple Chat-Style AI Tagger for Rekordbox
Easy to see and use chat interface for AI tag suggestions
"""

import tkinter as tk
from tkinter import ttk
import time
from datetime import datetime
from typing import List

class SimpleChatTagger:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 Vy AI Tagger Chat")
        self.root.geometry("500x700")
        self.root.configure(bg='#1e1e1e')
        
        # Make window stay on top and visible
        self.root.attributes('-topmost', True)
        self.root.lift()
        self.root.focus_force()
        
        self.selected_tags = []
        self.setup_ui()
        self.add_welcome_messages()
        
    def setup_ui(self):
        """Setup the chat interface"""
        # Title bar
        title_frame = tk.Frame(self.root, bg='#00ff88', height=50)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="🎵 Vy AI Tagger - Chat Mode",
                              font=('Arial', 16, 'bold'),
                              fg='#000000',
                              bg='#00ff88')
        title_label.pack(expand=True)
        
        # Chat messages area
        self.messages_frame = tk.Frame(self.root, bg='#1e1e1e')
        self.messages_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Input area
        input_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        input_frame.pack(fill='x', padx=10, pady=5)
        input_frame.pack_propagate(False)
        
        tk.Label(input_frame, text="Type custom tag:", 
                fg='#ffffff', bg='#2d2d2d', font=('Arial', 10)).pack(anchor='w', pady=2)
        
        entry_frame = tk.Frame(input_frame, bg='#2d2d2d')
        entry_frame.pack(fill='x')
        
        self.entry = tk.Entry(entry_frame, font=('Arial', 12), bg='#3d3d3d', fg='#ffffff')
        self.entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        send_btn = tk.Button(entry_frame, text="Add Tag", 
                           font=('Arial', 10, 'bold'),
                           bg='#00ff88', fg='#000000',
                           command=self.add_custom_tag)
        send_btn.pack(side='right')
        
        self.entry.bind('<Return>', lambda e: self.add_custom_tag())
        
    def add_message(self, sender: str, message: str, is_ai: bool = True, tags: List[str] = None):
        """Add a message to the chat"""
        timestamp = datetime.now().strftime("%H:%M")
        
        # Message container
        msg_container = tk.Frame(self.messages_frame, bg='#1e1e1e')
        msg_container.pack(fill='x', pady=5)
        
        if is_ai:
            # AI message
            msg_frame = tk.Frame(msg_container, bg='#2d2d2d', relief='solid', bd=2)
            msg_frame.pack(anchor='w', fill='x', padx=(0, 100))
            
            # Header
            header = tk.Label(msg_frame, 
                            text=f"🤖 {sender} - {timestamp}",
                            font=('Arial', 10, 'bold'),
                            fg='#00ff88', bg='#2d2d2d')
            header.pack(anchor='w', padx=10, pady=5)
            
            # Message
            msg_label = tk.Label(msg_frame, text=message,
                               font=('Arial', 11),
                               fg='#ffffff', bg='#2d2d2d',
                               wraplength=350, justify='left')
            msg_label.pack(anchor='w', padx=10, pady=(0, 10))
            
            # Add tag buttons
            if tags:
                self.add_tag_buttons(msg_frame, tags)
        else:
            # User message
            msg_frame = tk.Frame(msg_container, bg='#00ff88', relief='solid', bd=2)
            msg_frame.pack(anchor='e', padx=(100, 0))
            
            header = tk.Label(msg_frame, 
                            text=f"{sender} - {timestamp} 🎧",
                            font=('Arial', 10, 'bold'),
                            fg='#000000', bg='#00ff88')
            header.pack(anchor='e', padx=10, pady=5)
            
            msg_label = tk.Label(msg_frame, text=message,
                               font=('Arial', 11),
                               fg='#000000', bg='#00ff88',
                               wraplength=300, justify='right')
            msg_label.pack(anchor='e', padx=10, pady=(0, 10))
        
        # Auto scroll
        self.root.update_idletasks()
        
    def add_tag_buttons(self, parent: tk.Frame, tags: List[str]):
        """Add clickable tag buttons"""
        tags_frame = tk.Frame(parent, bg='#2d2d2d')
        tags_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(tags_frame, text="💡 Click to select tags:",
                font=('Arial', 9, 'bold'),
                fg='#ffaa00', bg='#2d2d2d').pack(anchor='w')
        
        # Tag buttons
        buttons_frame = tk.Frame(tags_frame, bg='#2d2d2d')
        buttons_frame.pack(fill='x', pady=5)
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#9b59b6']
        
        for i, tag in enumerate(tags):
            color = colors[i % len(colors)]
            
            btn = tk.Button(buttons_frame,
                          text=f"+ {tag}",
                          font=('Arial', 9, 'bold'),
                          bg=color, fg='#ffffff',
                          relief='flat', padx=10, pady=3,
                          command=lambda t=tag: self.select_tag(t))
            btn.pack(side='left', padx=2, pady=2)
            
            # New row every 3 buttons
            if (i + 1) % 3 == 0:
                buttons_frame = tk.Frame(tags_frame, bg='#2d2d2d')
                buttons_frame.pack(fill='x', pady=2)
        
        # Apply button
        apply_frame = tk.Frame(tags_frame, bg='#2d2d2d')
        apply_frame.pack(fill='x', pady=10)
        
        apply_btn = tk.Button(apply_frame,
                            text="✅ APPLY SELECTED TAGS TO REKORDBOX",
                            font=('Arial', 12, 'bold'),
                            bg='#ff4757', fg='#ffffff',
                            relief='flat', padx=20, pady=8,
                            command=self.apply_tags)
        apply_btn.pack()
        
    def select_tag(self, tag: str):
        """Select a tag"""
        if tag not in self.selected_tags:
            self.selected_tags.append(tag)
            self.add_message("You", f"Selected: {tag}", is_ai=False)
        
    def add_custom_tag(self):
        """Add custom tag"""
        custom_tag = self.entry.get().strip()
        if custom_tag and custom_tag not in self.selected_tags:
            self.selected_tags.append(custom_tag)
            self.entry.delete(0, tk.END)
            self.add_message("You", f"Added custom: {custom_tag}", is_ai=False)
        
    def apply_tags(self):
        """Apply selected tags"""
        if self.selected_tags:
            tags_text = ", ".join(self.selected_tags)
            self.add_message("Vy AI", f"✅ SUCCESS! Applied {len(self.selected_tags)} tags to Rekordbox:\n{tags_text}")
            self.selected_tags = []
        else:
            self.add_message("Vy AI", "❌ Please select some tags first!")
    
    def add_welcome_messages(self):
        """Add welcome messages"""
        self.add_message("Vy AI", "👋 Welcome to your AI Tagging Assistant!")
        
        # Simulate analyzing current track
        self.root.after(2000, self.analyze_current_track)
        
    def analyze_current_track(self):
        """Analyze the current track"""
        self.add_message("Vy AI", "🎵 I can see you're working with 'Der Dritte Raum - Artificial Hallucination'\nBPM: 130 | Key: 5B\n\nBased on the track characteristics, here are my AI suggestions:")
        
        # AI suggestions based on the track
        suggestions = [
            "3-Peak Time",
            "Melodic Techno", 
            "Energetic",
            "Synth Lead",
            "Dark",
            "Progressive House"
        ]
        
        self.root.after(1000, lambda: self.add_message("Vy AI", "Select the tags that fit your style:", tags=suggestions))
        
        # Schedule next track analysis
        self.root.after(10000, self.analyze_next_track)
        
    def analyze_next_track(self):
        """Analyze next track"""
        self.add_message("Vy AI", "🎵 Analyzing next track: 'Boy Oh Boy, Paons'\nBPM: 128 | Key: 7A\n\nThis track has different characteristics:")
        
        suggestions = [
            "2-Build up",
            "Progressive House",
            "Dreamy",
            "Female Vocal",
            "Uplifting",
            "Piano"
        ]
        
        self.root.after(1000, lambda: self.add_message("Vy AI", "Here are suggestions for this track:", tags=suggestions))
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    app = SimpleChatTagger()
    app.run()

if __name__ == "__main__":
    main()
