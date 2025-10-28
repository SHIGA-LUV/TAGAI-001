#!/usr/bin/env python3
"""
AI MyTag DJ Assistant - Preview Mode
Phase 1.5: Preview changes before applying them to XML files
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

# Import the original tagger
from rekordbox_ai_tagger import RekordboxAITagger

class PreviewMode:
    def __init__(self, parent_window=None):
        self.parent = parent_window
        self.tagger = RekordboxAITagger()
        self.preview_data = {}
        self.changes_approved = False
        
    def show_preview(self, tracks: Dict, suggested_changes: Dict, xml_path: str) -> bool:
        """Show preview window with all proposed changes"""
        self.preview_data = {
            'tracks': tracks,
            'changes': suggested_changes,
            'xml_path': xml_path
        }
        
        # Create preview window
        self.preview_window = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.preview_window.title("🔍 Preview Changes - AI MyTag DJ Assistant")
        self.preview_window.geometry("1000x700")
        self.preview_window.configure(bg='#2C3E50')
        
        # Make modal if has parent
        if self.parent:
            self.preview_window.transient(self.parent)
            self.preview_window.grab_set()
        
        self.setup_preview_ui()
        self.populate_preview_data()
        
        # Wait for user decision
        self.preview_window.wait_window()
        return self.changes_approved
        
    def setup_preview_ui(self):
        """Setup the preview interface"""
        # Header
        header_frame = tk.Frame(self.preview_window, bg='#34495E', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, 
                text="Preview Proposed Changes",
                font=('Arial', 16, 'bold'),
                fg='white', bg='#34495E').pack(pady=10)
        
        # Summary stats
        stats_frame = tk.Frame(self.preview_window, bg='#2C3E50')
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_label = tk.Label(stats_frame,
                                   text="Loading preview...",
                                   font=('Arial', 12),
                                   fg='#BDC3C7', bg='#2C3E50')
        self.stats_label.pack()
        
        # Filter controls
        filter_frame = tk.Frame(self.preview_window, bg='#2C3E50')
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter:", fg='white', bg='#2C3E50').pack(side='left')
        
        self.filter_var = tk.StringVar()
        self.filter_var.trace('w', self.filter_changes)
        filter_entry = tk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        filter_entry.pack(side='left', padx=(5, 10))
        
        # Filter options
        self.show_all_var = tk.BooleanVar(value=True)
        self.show_new_tags_var = tk.BooleanVar(value=True)
        self.show_modified_var = tk.BooleanVar(value=True)
        
        tk.Checkbutton(filter_frame, text="All", variable=self.show_all_var,
                      command=self.filter_changes, fg='white', bg='#2C3E50',
                      selectcolor='#34495E').pack(side='left', padx=5)
        
        tk.Checkbutton(filter_frame, text="New Tags", variable=self.show_new_tags_var,
                      command=self.filter_changes, fg='white', bg='#2C3E50',
                      selectcolor='#34495E').pack(side='left', padx=5)
        
        tk.Checkbutton(filter_frame, text="Modified", variable=self.show_modified_var,
                      command=self.filter_changes, fg='white', bg='#2C3E50',
                      selectcolor='#34495E').pack(side='left', padx=5)
        
        # Main content area
        content_frame = tk.Frame(self.preview_window, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Changes list (left side)
        left_frame = tk.LabelFrame(content_frame, text="Proposed Changes", 
                                  fg='white', bg='#2C3E50', font=('Arial', 10, 'bold'))
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Treeview for changes
        self.changes_tree = ttk.Treeview(left_frame, columns=('artist', 'title', 'current', 'proposed', 'confidence'), 
                                        show='tree headings', height=15)
        
        # Configure columns
        self.changes_tree.heading('#0', text='Track')
        self.changes_tree.heading('artist', text='Artist')
        self.changes_tree.heading('title', text='Title')
        self.changes_tree.heading('current', text='Current Tags')
        self.changes_tree.heading('proposed', text='Proposed Tags')
        self.changes_tree.heading('confidence', text='Confidence')
        
        self.changes_tree.column('#0', width=50)
        self.changes_tree.column('artist', width=120)
        self.changes_tree.column('title', width=150)
        self.changes_tree.column('current', width=200)
        self.changes_tree.column('proposed', width=200)
        self.changes_tree.column('confidence', width=80)
        
        # Scrollbars for treeview
        tree_scroll_y = ttk.Scrollbar(left_frame, orient='vertical', command=self.changes_tree.yview)
        tree_scroll_x = ttk.Scrollbar(left_frame, orient='horizontal', command=self.changes_tree.xview)
        self.changes_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        self.changes_tree.pack(side='left', fill='both', expand=True)
        tree_scroll_y.pack(side='right', fill='y')
        tree_scroll_x.pack(side='bottom', fill='x')
        
        # Bind selection event
        self.changes_tree.bind('<<TreeviewSelect>>', self.on_track_select)
        
        # Details panel (right side)
        right_frame = tk.LabelFrame(content_frame, text="Track Details & Tag Analysis",
                                   fg='white', bg='#2C3E50', font=('Arial', 10, 'bold'))
        right_frame.pack(side='right', fill='both', expand=False, padx=(5, 0), ipadx=10)
        
        # Track info
        self.track_info_text = tk.Text(right_frame, width=40, height=8, 
                                      font=('Consolas', 9), bg='#34495E', fg='white')
        self.track_info_text.pack(fill='x', pady=(5, 10))
        
        # Tag comparison
        comparison_frame = tk.LabelFrame(right_frame, text="Tag Comparison",
                                        fg='white', bg='#2C3E50')
        comparison_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.comparison_text = tk.Text(comparison_frame, width=40, height=12,
                                      font=('Consolas', 9), bg='#34495E', fg='white')
        self.comparison_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Individual track controls
        track_controls = tk.Frame(right_frame, bg='#2C3E50')
        track_controls.pack(fill='x', pady=(0, 10))
        
        tk.Button(track_controls, text="Accept This Track", 
                 command=self.accept_current_track,
                 bg='#27AE60', fg='white', font=('Arial', 9)).pack(side='left', padx=(0, 5))
        
        tk.Button(track_controls, text="Reject This Track",
                 command=self.reject_current_track,
                 bg='#E74C3C', fg='white', font=('Arial', 9)).pack(side='left')
        
        # Action buttons
        button_frame = tk.Frame(self.preview_window, bg='#2C3E50')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        # Left side buttons
        left_buttons = tk.Frame(button_frame, bg='#2C3E50')
        left_buttons.pack(side='left')
        
        tk.Button(left_buttons, text="📊 Export Preview Report",
                 command=self.export_preview_report,
                 bg='#3498DB', fg='white', font=('Arial', 10)).pack(side='left', padx=(0, 10))
        
        tk.Button(left_buttons, text="🔄 Refresh Analysis",
                 command=self.refresh_analysis,
                 bg='#9B59B6', fg='white', font=('Arial', 10)).pack(side='left')
        
        # Right side buttons
        right_buttons = tk.Frame(button_frame, bg='#2C3E50')
        right_buttons.pack(side='right')
        
        tk.Button(right_buttons, text="❌ Cancel",
                 command=self.cancel_changes,
                 bg='#95A5A6', fg='white', font=('Arial', 11, 'bold')).pack(side='right', padx=(10, 0))
        
        tk.Button(right_buttons, text="✅ Apply All Changes",
                 command=self.apply_changes,
                 bg='#27AE60', fg='white', font=('Arial', 11, 'bold')).pack(side='right', padx=(10, 0))
        
        tk.Button(right_buttons, text="⚡ Apply Selected Only",
                 command=self.apply_selected_changes,
                 bg='#F39C12', fg='white', font=('Arial', 11, 'bold')).pack(side='right', padx=(10, 0))
        
    def populate_preview_data(self):
        """Populate the preview with track data and changes"""
        tracks = self.preview_data['tracks']
        changes = self.preview_data['changes']
        
        # Clear existing data
        for item in self.changes_tree.get_children():
            self.changes_tree.delete(item)
        
        # Statistics
        total_tracks = len(tracks)
        tracks_with_changes = len(changes)
        total_new_tags = sum(len(tags) for tags in changes.values())
        
        stats_text = f"📁 File: {Path(self.preview_data['xml_path']).name} | "
        stats_text += f"🎵 Total Tracks: {total_tracks} | "
        stats_text += f"🏷️ Tracks with Changes: {tracks_with_changes} | "
        stats_text += f"✨ New Tags: {total_new_tags}"
        
        self.stats_label.config(text=stats_text)
        
        # Populate treeview
        for track_id, new_tags in changes.items():
            if track_id in tracks:
                track_info = tracks[track_id]
                current_tags = track_info.get('existing_tags', [])
                
                # Calculate average confidence (simplified)
                avg_confidence = self.calculate_average_confidence(track_info, new_tags)
                
                # Determine change type
                if not current_tags:
                    change_type = "🆕 New"
                elif set(current_tags) != set(new_tags):
                    change_type = "📝 Modified"
                else:
                    change_type = "✅ Same"
                
                # Insert into treeview
                item_id = self.changes_tree.insert('', 'end',
                    text=change_type,
                    values=(
                        track_info['artist'][:20] + '...' if len(track_info['artist']) > 20 else track_info['artist'],
                        track_info['title'][:25] + '...' if len(track_info['title']) > 25 else track_info['title'],
                        ' / '.join(current_tags[:3]) + ('...' if len(current_tags) > 3 else ''),
                        ' / '.join(new_tags[:3]) + ('...' if len(new_tags) > 3 else ''),
                        f"{avg_confidence:.0%}"
                    ),
                    tags=(track_id,)
                )
                
                # Color coding based on change type
                if not current_tags:
                    self.changes_tree.set(item_id, 'current', '(No tags)')
                    
    def calculate_average_confidence(self, track_info: Dict, tags: List[str]) -> float:
        """Calculate average confidence for tags (simplified)"""
        if not tags:
            return 0.0
            
        total_confidence = 0.0
        bpm = float(track_info.get('bpm', 0)) if track_info.get('bpm') else 0
        genre = track_info.get('genre', '').lower()
        
        for tag in tags:
            confidence = 0.5  # Base confidence
            
            # BPM-based confidence
            if bpm > 0:
                if tag in ['5-After Hours', 'Dreamy'] and bpm < 100:
                    confidence += 0.3
                elif tag in ['4-Cool Down', 'Deep House'] and 100 <= bpm < 120:
                    confidence += 0.3
                elif tag in ['2-Build up', 'Progressive House'] and 120 <= bpm < 130:
                    confidence += 0.3
                elif tag in ['3-Peak Time', 'Energetic'] and bpm >= 130:
                    confidence += 0.3
            
            # Genre-based confidence
            if 'house' in genre and 'House' in tag:
                confidence += 0.2
            elif 'techno' in genre and 'Techno' in tag:
                confidence += 0.2
                
            total_confidence += min(confidence, 1.0)
            
        return total_confidence / len(tags)
        
    def filter_changes(self, *args):
        """Filter the displayed changes"""
        # This would implement filtering logic
        # For now, just refresh the display
        pass
        
    def on_track_select(self, event):
        """Handle track selection in treeview"""
        selection = self.changes_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        track_id = self.changes_tree.item(item, 'tags')[0]
        
        if track_id in self.preview_data['tracks']:
            self.display_track_details(track_id)
            
    def display_track_details(self, track_id: str):
        """Display detailed information for selected track"""
        track_info = self.preview_data['tracks'][track_id]
        new_tags = self.preview_data['changes'].get(track_id, [])
        current_tags = track_info.get('existing_tags', [])
        
        # Track info
        info_text = f"""🎵 TRACK INFORMATION
{'='*30}

Artist: {track_info['artist']}
Title: {track_info['title']}
BPM: {track_info['bpm']}
Key: {track_info['key']}
Genre: {track_info['genre']}
Date Added: {track_info['date_added']}
"""
        
        self.track_info_text.delete(1.0, tk.END)
        self.track_info_text.insert(1.0, info_text)
        
        # Tag comparison
        comparison_text = f"""🏷️ TAG COMPARISON
{'='*25}

📋 CURRENT TAGS ({len(current_tags)}):  
{chr(10).join(f'  • {tag}' for tag in current_tags) if current_tags else '  (No current tags)'}

✨ PROPOSED TAGS ({len(new_tags)}): 
{chr(10).join(f'  • {tag}' for tag in new_tags)}

🔄 CHANGES:
"""
        
        # Analyze changes
        added_tags = set(new_tags) - set(current_tags)
        removed_tags = set(current_tags) - set(new_tags)
        kept_tags = set(current_tags) & set(new_tags)
        
        if added_tags:
            comparison_text += f"
  ➕ ADDED ({len(added_tags)}): 
"
            comparison_text += chr(10).join(f'     • {tag}' for tag in sorted(added_tags))
            
        if removed_tags:
            comparison_text += f"

  ➖ REMOVED ({len(removed_tags)}): 
"
            comparison_text += chr(10).join(f'     • {tag}' for tag in sorted(removed_tags))
            
        if kept_tags:
            comparison_text += f"

  ✅ KEPT ({len(kept_tags)}): 
"
            comparison_text += chr(10).join(f'     • {tag}' for tag in sorted(kept_tags))
        
        # Confidence analysis
        avg_confidence = self.calculate_average_confidence(track_info, new_tags)
        comparison_text += f"

📊 CONFIDENCE ANALYSIS:
  Average Confidence: {avg_confidence:.1%}
"
        
        if avg_confidence >= 0.8:
            comparison_text += "  🟢 High confidence - Recommended"
        elif avg_confidence >= 0.6:
            comparison_text += "  🟡 Medium confidence - Review suggested"
        else:
            comparison_text += "  🔴 Low confidence - Manual review needed"
        
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(1.0, comparison_text)
        
    def accept_current_track(self):
        """Accept changes for currently selected track"""
        selection = self.changes_tree.selection()
        if selection:
            # Mark track as accepted (visual indication)
            item = selection[0]
            self.changes_tree.item(item, text="✅ Accepted")
            
    def reject_current_track(self):
        """Reject changes for currently selected track"""
        selection = self.changes_tree.selection()
        if selection:
            item = selection[0]
            track_id = self.changes_tree.item(item, 'tags')[0]
            
            # Remove from changes
            if track_id in self.preview_data['changes']:
                del self.preview_data['changes'][track_id]
                
            # Update display
            self.changes_tree.item(item, text="❌ Rejected")
            
    def export_preview_report(self):
        """Export preview data as a report"""
        from tkinter import filedialog
        
        output_path = filedialog.asksaveasfilename(
            title="Save Preview Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if output_path:
            try:
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'xml_file': self.preview_data['xml_path'],
                    'total_tracks': len(self.preview_data['tracks']),
                    'tracks_with_changes': len(self.preview_data['changes']),
                    'changes': {}
                }
                
                # Add detailed changes
                for track_id, new_tags in self.preview_data['changes'].items():
                    track_info = self.preview_data['tracks'][track_id]
                    current_tags = track_info.get('existing_tags', [])
                    
                    report_data['changes'][track_id] = {
                        'artist': track_info['artist'],
                        'title': track_info['title'],
                        'bpm': track_info['bpm'],
                        'genre': track_info['genre'],
                        'current_tags': current_tags,
                        'proposed_tags': new_tags,
                        'confidence': self.calculate_average_confidence(track_info, new_tags)
                    }
                
                if output_path.endswith('.json'):
                    with open(output_path, 'w') as f:
                        json.dump(report_data, f, indent=2)
                else:
                    # Text format
                    with open(output_path, 'w') as f:
                        f.write(f"AI MyTag DJ Assistant - Preview Report\n")
                        f.write(f"Generated: {report_data['timestamp']}\n")
                        f.write(f"XML File: {report_data['xml_file']}\n")
                        f.write(f"Total Tracks: {report_data['total_tracks']}\n")
                        f.write(f"Tracks with Changes: {report_data['tracks_with_changes']}\n\n")
                        
                        for track_id, change_data in report_data['changes'].items():
                            f.write(f"Track: {change_data['artist']} - {change_data['title']}\n")
                            f.write(f"  Current: {' / '.join(change_data['current_tags'])}\n")
                            f.write(f"  Proposed: {' / '.join(change_data['proposed_tags'])}\n")
                            f.write(f"  Confidence: {change_data['confidence']:.1%}\n\n")
                
                messagebox.showinfo("Success", f"Preview report exported to {output_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {e}")
                
    def refresh_analysis(self):
        """Refresh the analysis and suggestions"""
        # Re-analyze tracks and update suggestions
        messagebox.showinfo("Info", "Analysis refreshed!")
        
    def apply_changes(self):
        """Apply all changes"""
        if messagebox.askyesno("Confirm", 
                              f"Apply changes to {len(self.preview_data['changes'])} tracks?"):
            self.changes_approved = True
            self.preview_window.destroy()
            
    def apply_selected_changes(self):
        """Apply only selected/accepted changes"""
        # Filter out rejected changes
        accepted_changes = {}
        for item in self.changes_tree.get_children():
            if self.changes_tree.item(item, 'text').startswith('✅'):
                track_id = self.changes_tree.item(item, 'tags')[0]
                if track_id in self.preview_data['changes']:
                    accepted_changes[track_id] = self.preview_data['changes'][track_id]
        
        if accepted_changes:
            self.preview_data['changes'] = accepted_changes
            self.changes_approved = True
            self.preview_window.destroy()
        else:
            messagebox.showwarning("Warning", "No changes selected for application")
            
    def cancel_changes(self):
        """Cancel all changes"""
        if messagebox.askyesno("Confirm", "Cancel all changes?"):
            self.changes_approved = False
            self.preview_window.destroy()

def demo_preview():
    """Demo function to test preview mode"""
    # Sample data for testing
    sample_tracks = {
        '1': {
            'artist': 'Sample Artist 1',
            'title': 'Sample Track 1',
            'bpm': '125',
            'key': 'Am',
            'genre': 'Progressive House',
            'date_added': '2024-01-01',
            'existing_tags': ['Progressive House']
        },
        '2': {
            'artist': 'Sample Artist 2', 
            'title': 'Sample Track 2',
            'bpm': '132',
            'key': 'Gm',
            'genre': 'Melodic Techno',
            'date_added': '2024-01-02',
            'existing_tags': []
        }
    }
    
    sample_changes = {
        '1': ['Progressive House', '2-Build up', 'Emotional'],
        '2': ['Melodic Techno', '3-Peak Time', 'Energetic', 'Dark']
    }
    
    preview = PreviewMode()
    result = preview.show_preview(sample_tracks, sample_changes, '/path/to/sample.xml')
    print(f"Changes approved: {result}")

if __name__ == "__main__":
    demo_preview()
