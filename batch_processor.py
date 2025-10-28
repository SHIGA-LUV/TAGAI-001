#!/usr/bin/env python3
"""
Batch Processing Interface for AI MyTag DJ Assistant
Processes multiple Rekordbox XML files and applies AI tagging
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import our core tagger
from rekordbox_ai_tagger import RekordboxAITagger

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
except ImportError:
    print("tkinter not available, GUI features disabled")

class BatchProcessor:
    def __init__(self):
        self.tagger = RekordboxAITagger()
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_tracks': 0,
            'tagged_tracks': 0,
            'errors': []
        }
        
    def process_single_file(self, xml_path: str, output_dir: str = None, 
                           auto_apply: bool = False, confidence_threshold: float = 0.7) -> Dict:
        """Process a single XML file"""
        try:
            print(f"\n📁 Processing: {os.path.basename(xml_path)}")
            
            # Parse XML
            tracks = self.tagger.parse_xml(xml_path)
            if not tracks:
                return {'success': False, 'error': 'No tracks found in XML'}
            
            # Analyze existing tags
            analysis = self.tagger.analyze_existing_tags(tracks)
            
            # Find untagged tracks
            untagged_tracks = {tid: info for tid, info in tracks.items() 
                              if not info.get('existing_tags')}
            
            print(f"📊 Found {len(tracks)} total tracks, {len(untagged_tracks)} untagged")
            
            # Generate suggestions for untagged tracks
            track_updates = {}
            suggestion_stats = {'high_confidence': 0, 'medium_confidence': 0, 'low_confidence': 0}
            
            for track_id, track_info in untagged_tracks.items():
                suggestions = self.tagger.suggest_tags_for_track(track_info, analysis)
                
                if auto_apply:
                    # Calculate confidence and auto-apply high confidence tags
                    confident_tags = []
                    for tag in suggestions:
                        confidence = self.calculate_confidence(track_info, tag)
                        if confidence >= confidence_threshold:
                            confident_tags.append(tag)
                            suggestion_stats['high_confidence'] += 1
                        elif confidence >= 0.5:
                            suggestion_stats['medium_confidence'] += 1
                        else:
                            suggestion_stats['low_confidence'] += 1
                    
                    if confident_tags:
                        track_updates[track_id] = confident_tags
                else:
                    # Manual review mode - suggest all
                    track_updates[track_id] = suggestions
            
            # Generate output
            if output_dir:
                output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(xml_path))[0]}_tagged.xml")
            else:
                output_path = xml_path.replace('.xml', '_batch_tagged.xml')
            
            if track_updates:
                self.tagger.generate_updated_xml(xml_path, track_updates, output_path)
            
            result = {
                'success': True,
                'input_file': xml_path,
                'output_file': output_path,
                'total_tracks': len(tracks),
                'untagged_tracks': len(untagged_tracks),
                'tagged_tracks': len(track_updates),
                'suggestion_stats': suggestion_stats
            }
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'input_file': xml_path}
    
    def calculate_confidence(self, track_info: Dict, tag: str) -> float:
        """Calculate confidence score for a tag"""
        confidence = 0.5
        bpm = float(track_info.get('bpm', 0)) if track_info.get('bpm') else 0
        genre = track_info.get('genre', '').lower()
        
        # BPM-based confidence
        if 'Peak Time' in tag and bpm > 130:
            confidence += 0.3
        elif 'After Hours' in tag and bpm < 100:
            confidence += 0.4
        elif 'Build up' in tag and 120 <= bpm <= 130:
            confidence += 0.2
        
        # Genre matching
        if 'Progressive House' in tag and 'progressive' in genre:
            confidence += 0.3
        elif 'Deep House' in tag and 'deep' in genre:
            confidence += 0.3
        elif 'Melodic Techno' in tag and 'techno' in genre:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def process_batch(self, xml_files: List[str], output_dir: str = None, 
                     auto_apply: bool = False, confidence_threshold: float = 0.7) -> Dict:
        """Process multiple XML files"""
        print(f"\n🚀 Starting batch processing of {len(xml_files)} files...")
        print(f"📁 Output directory: {output_dir or 'Same as input files'}")
        print(f"🤖 Auto-apply mode: {'ON' if auto_apply else 'OFF'}")
        print(f"🎯 Confidence threshold: {confidence_threshold:.0%}")
        
        results = []
        start_time = time.time()
        
        for i, xml_file in enumerate(xml_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(xml_files)}")
            print(f"{'='*60}")
            
            result = self.process_single_file(xml_file, output_dir, auto_apply, confidence_threshold)
            results.append(result)
            
            if result['success']:
                self.processing_stats['processed_files'] += 1
                self.processing_stats['total_tracks'] += result['total_tracks']
                self.processing_stats['tagged_tracks'] += result['tagged_tracks']
            else:
                self.processing_stats['errors'].append(result)
        
        self.processing_stats['total_files'] = len(xml_files)
        processing_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'processing_time': processing_time,
            'stats': self.processing_stats,
            'results': results
        }
        
        self.print_batch_summary(summary)
        return summary
    
    def print_batch_summary(self, summary: Dict):
        """Print batch processing summary"""
        stats = summary['stats']
        
        print(f"\n{'='*70}")
        print("🎉 BATCH PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"⏱️  Processing time: {summary['processing_time']:.1f} seconds")
        print(f"📁 Files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"🎵 Total tracks: {stats['total_tracks']}")
        print(f"🏷️  Tracks tagged: {stats['tagged_tracks']}")
        
        if stats['errors']:
            print(f"❌ Errors: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"   • {error.get('input_file', 'Unknown')}: {error.get('error', 'Unknown error')}")
        
        success_rate = (stats['processed_files'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if stats['tagged_tracks'] > 0:
            avg_tags_per_file = stats['tagged_tracks'] / stats['processed_files'] if stats['processed_files'] > 0 else 0
            print(f"🎯 Average tags per file: {avg_tags_per_file:.1f}")
        
        print(f"{'='*70}")

class BatchProcessorGUI:
    def __init__(self):
        self.processor = BatchProcessor()
        self.selected_files = []
        self.output_directory = None
        
        self.root = tk.Tk()
        self.root.title("🎵 AI MyTag Batch Processor")
        self.root.geometry("800x600")
        self.root.configure(bg='#2C3E50')
        
        self.create_gui()
    
    def create_gui(self):
        """Create the GUI interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2C3E50', pady=10)
        title_frame.pack(fill='x')
        
        tk.Label(title_frame, 
                text="🎵 AI MyTag Batch Processor",
                font=('Arial', 18, 'bold'),
                fg='white', bg='#2C3E50').pack()
        
        tk.Label(title_frame,
                text="Process multiple Rekordbox XML files with AI tagging",
                font=('Arial', 12),
                fg='#BDC3C7', bg='#2C3E50').pack()
        
        # File selection
        file_frame = tk.LabelFrame(self.root, text="📁 Input Files", 
                                  font=('Arial', 12, 'bold'),
                                  fg='white', bg='#34495E', pady=10)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Button(file_frame,
                 text="Select XML Files",
                 font=('Arial', 12),
                 bg='#3498DB', fg='white',
                 command=self.select_files,
                 pady=5).pack(pady=5)
        
        self.files_listbox = tk.Listbox(file_frame, height=6, 
                                       font=('Arial', 10),
                                       bg='#ECF0F1', fg='#2C3E50')
        self.files_listbox.pack(fill='x', padx=10, pady=5)
        
        # Output directory
        output_frame = tk.LabelFrame(self.root, text="📂 Output Directory",
                                    font=('Arial', 12, 'bold'),
                                    fg='white', bg='#34495E', pady=10)
        output_frame.pack(fill='x', padx=20, pady=10)
        
        output_btn_frame = tk.Frame(output_frame, bg='#34495E')
        output_btn_frame.pack(fill='x', padx=10)
        
        tk.Button(output_btn_frame,
                 text="Select Output Directory",
                 font=('Arial', 12),
                 bg='#27AE60', fg='white',
                 command=self.select_output_dir,
                 pady=5).pack(side='left', padx=5)
        
        tk.Button(output_btn_frame,
                 text="Use Same as Input",
                 font=('Arial', 12),
                 bg='#95A5A6', fg='white',
                 command=self.use_input_dir,
                 pady=5).pack(side='left', padx=5)
        
        self.output_label = tk.Label(output_frame,
                                    text="Output: Same as input files",
                                    font=('Arial', 10),
                                    fg='#BDC3C7', bg='#34495E')
        self.output_label.pack(pady=5)
        
        # Processing options
        options_frame = tk.LabelFrame(self.root, text="⚙️ Processing Options",
                                     font=('Arial', 12, 'bold'),
                                     fg='white', bg='#34495E', pady=10)
        options_frame.pack(fill='x', padx=20, pady=10)
        
        # Auto-apply checkbox
        self.auto_apply_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame,
                      text="Auto-apply high confidence tags",
                      variable=self.auto_apply_var,
                      font=('Arial', 11),
                      fg='white', bg='#34495E',
                      selectcolor='#2C3E50').pack(anchor='w', padx=10)
        
        # Confidence threshold
        threshold_frame = tk.Frame(options_frame, bg='#34495E')
        threshold_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(threshold_frame,
                text="Confidence threshold:",
                font=('Arial', 11),
                fg='white', bg='#34495E').pack(side='left')
        
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = tk.Scale(threshold_frame,
                                  from_=0.5, to=1.0, resolution=0.1,
                                  variable=self.threshold_var,
                                  orient='horizontal',
                                  font=('Arial', 10),
                                  bg='#34495E', fg='white',
                                  highlightbackground='#34495E')
        threshold_scale.pack(side='right', padx=10)
        
        # Process button
        process_frame = tk.Frame(self.root, bg='#2C3E50', pady=20)
        process_frame.pack(fill='x')
        
        self.process_btn = tk.Button(process_frame,
                                    text="🚀 Start Batch Processing",
                                    font=('Arial', 14, 'bold'),
                                    bg='#E74C3C', fg='white',
                                    command=self.start_processing,
                                    pady=10)
        self.process_btn.pack(pady=10)
        
        # Progress and log
        log_frame = tk.LabelFrame(self.root, text="📊 Processing Log",
                                 font=('Arial', 12, 'bold'),
                                 fg='white', bg='#34495E')
        log_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                 height=8,
                                                 font=('Courier', 10),
                                                 bg='#ECF0F1', fg='#2C3E50')
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def select_files(self):
        """Select XML files for processing"""
        files = filedialog.askopenfilenames(
            title="Select Rekordbox XML Files",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
        )
        
        if files:
            self.selected_files = list(files)
            self.files_listbox.delete(0, tk.END)
            for file in self.selected_files:
                self.files_listbox.insert(tk.END, os.path.basename(file))
            
            self.log(f"Selected {len(files)} XML files")
    
    def select_output_dir(self):
        """Select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_directory = directory
            self.output_label.config(text=f"Output: {directory}")
            self.log(f"Output directory: {directory}")
    
    def use_input_dir(self):
        """Use same directory as input files"""
        self.output_directory = None
        self.output_label.config(text="Output: Same as input files")
        self.log("Using input file directories for output")
    
    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_processing(self):
        """Start batch processing"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select XML files to process.")
            return
        
        self.process_btn.config(state='disabled', text="Processing...")
        self.log_text.delete(1.0, tk.END)
        
        try:
            self.log("Starting batch processing...")
            
            # Process files
            summary = self.processor.process_batch(
                self.selected_files,
                self.output_directory,
                self.auto_apply_var.get(),
                self.threshold_var.get()
            )
            
            # Show results
            stats = summary['stats']
            self.log(f"\n✅ Processing complete!")
            self.log(f"Files processed: {stats['processed_files']}/{stats['total_files']}")
            self.log(f"Tracks tagged: {stats['tagged_tracks']}")
            
            if stats['errors']:
                self.log(f"❌ Errors: {len(stats['errors'])}")
            
            messagebox.showinfo("Complete", 
                              f"Batch processing complete!\n\n"
                              f"Files processed: {stats['processed_files']}/{stats['total_files']}\n"
                              f"Tracks tagged: {stats['tagged_tracks']}")
        
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        
        finally:
            self.process_btn.config(state='normal', text="🚀 Start Batch Processing")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🎵 AI MyTag Batch Processor")
    print("==========================")
    
    # Check if GUI is available
    try:
        import tkinter
        print("🖥️  Starting GUI interface...")
        gui = BatchProcessorGUI()
        gui.run()
    except ImportError:
        print("❌ GUI not available, running command line demo...")
        
        # Command line demo
        processor = BatchProcessor()
        
        # Demo with sample data
        print("\n📁 Demo: Processing sample XML files...")
        sample_files = ["sample1.xml", "sample2.xml"]  # Would be real files
        
        # This would process real files:
        # summary = processor.process_batch(sample_files, auto_apply=True, confidence_threshold=0.7)
        
        print("\n✅ Batch processor ready!")
        print("To use: python3 batch_processor.py")

if __name__ == "__main__":
    main()
