#!/usr/bin/env python3
"""
AI MyTag DJ Assistant - Rekordbox XML Processor
Phase 1: Core XML Processing and Tag Hierarchy System
"""

import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import re

class RekordboxAITagger:
    def __init__(self):
        self.tag_hierarchy = {
            'SITUATION': {
                'priority': 1,
                'tags': ['1-Opener', '2-Build up', '3-Peak Time', '4-Cool Down', '5-After Hours']
            },
            'GENRE': {
                'priority': 2,
                'tags': ['Melodic Techno', 'Ethnic House', 'Progressive House', 'Deep House']
            },
            'COMPONENTS': {
                'priority': 3,
                'tags': ['Piano', 'Darbuka', 'Female Vocal', 'Male Vocal', 'Strings', 'Synth Lead']
            },
            'MOOD': {
                'priority': 4,
                'tags': ['Tribal', 'Dreamy', 'Sexy', 'Energetic', 'Emotional', 'Dark']
            }
        }
        
        self.learned_patterns = {}
        self.track_database = {}
        
    def parse_xml(self, xml_file_path: str) -> Dict:
        """Parse Rekordbox XML file and extract track information"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            tracks = {}
            
            # Find all track elements
            for track in root.findall('.//TRACK'):
                track_id = track.get('TrackID')
                if track_id:
                    track_info = {
                        'title': track.get('Name', ''),
                        'artist': track.get('Artist', ''),
                        'bpm': track.get('AverageBpm', ''),
                        'key': track.get('Tonality', ''),
                        'genre': track.get('Genre', ''),
                        'comment': track.get('Comments', ''),
                        'location': track.get('Location', ''),
                        'date_added': track.get('DateAdded', '')
                    }
                    
                    # Extract existing tags from comment field
                    existing_tags = self.extract_tags_from_comment(track_info['comment'])
                    track_info['existing_tags'] = existing_tags
                    
                    tracks[track_id] = track_info
            
            print(f"Parsed {len(tracks)} tracks from XML")
            return tracks
            
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return {}
    
    def extract_tags_from_comment(self, comment: str) -> List[str]:
        """Extract tags from comment field using /* tag / tag / tag */ format"""
        if not comment:
            return []
            
        # Look for tags in /* ... */ format
        tag_pattern = r'/\*\s*(.+?)\s*\*/'
        matches = re.findall(tag_pattern, comment)
        
        if matches:
            # Split tags by ' / ' separator
            tags = [tag.strip() for tag in matches[0].split(' / ') if tag.strip()]
            return tags
        
        return []
    
    def analyze_existing_tags(self, tracks: Dict) -> Dict:
        """Analyze existing tags to learn DNA patterns"""
        tag_frequency = {}
        tag_combinations = {}
        tag_positions = {}
        
        for track_id, track_info in tracks.items():
            tags = track_info.get('existing_tags', [])
            
            if not tags:
                continue
                
            # Analyze tag frequency
            for tag in tags:
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
            
            # Analyze tag positions
            for i, tag in enumerate(tags):
                if tag not in tag_positions:
                    tag_positions[tag] = []
                tag_positions[tag].append(i)
            
            # Analyze tag combinations
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    combo = tuple(sorted([tags[i], tags[j]]))
                    tag_combinations[combo] = tag_combinations.get(combo, 0) + 1
        
        analysis = {
            'tag_frequency': tag_frequency,
            'tag_combinations': tag_combinations,
            'tag_positions': tag_positions,
            'total_tagged_tracks': len([t for t in tracks.values() if t.get('existing_tags')])
        }
        
        print(f"DNA Analysis Complete:")
        print(f"- Most frequent tags: {sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:5]}")
        print(f"- Total unique tags: {len(tag_frequency)}")
        print(f"- Tagged tracks: {analysis['total_tagged_tracks']}")
        
        return analysis
    
    def suggest_tags_for_track(self, track_info: Dict, learned_patterns: Dict) -> List[str]:
        """Suggest tags for a track based on learned patterns"""
        suggestions = []
        
        # This is a basic implementation - will be enhanced with AI in later phases
        bpm = float(track_info.get('bpm', 0)) if track_info.get('bpm') else 0
        genre = track_info.get('genre', '').lower()
        
        # Basic BPM-based suggestions
        if bpm > 0:
            if bpm < 100:
                suggestions.extend(['5-After Hours', 'Dreamy'])
            elif bpm < 120:
                suggestions.extend(['4-Cool Down', 'Deep House'])
            elif bpm < 130:
                suggestions.extend(['2-Build up', 'Progressive House'])
            else:
                suggestions.extend(['3-Peak Time', 'Energetic'])
        
        # Genre-based suggestions
        if 'house' in genre:
            suggestions.append('Progressive House')
        elif 'techno' in genre:
            suggestions.append('Melodic Techno')
        
        # Remove duplicates and sort by hierarchy
        suggestions = list(set(suggestions))
        return self.sort_tags_by_hierarchy(suggestions)
    
    def sort_tags_by_hierarchy(self, tags: List[str]) -> List[str]:
        """Sort tags according to hierarchy rules"""
        sorted_tags = []
        
        # Sort by category priority
        for category, info in sorted(self.tag_hierarchy.items(), key=lambda x: x[1]['priority']):
            category_tags = [tag for tag in tags if tag in info['tags']]
            
            # Special handling for SITUATION tags (numerical order)
            if category == 'SITUATION':
                category_tags.sort(key=lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else 999)
            else:
                category_tags.sort()
                
            sorted_tags.extend(category_tags)
        
        # Add any tags not in hierarchy at the end
        remaining_tags = [tag for tag in tags if tag not in sorted_tags]
        sorted_tags.extend(sorted(remaining_tags))
        
        return sorted_tags
    
    def format_tags_for_comment(self, tags: List[str]) -> str:
        """Format tags in the /* tag / tag / tag */ format"""
        if not tags:
            return ''
        return f"/* {' / '.join(tags)} */"
    
    def generate_updated_xml(self, original_xml_path: str, track_updates: Dict, output_path: str):
        """Generate updated XML file with new tags"""
        try:
            tree = ET.parse(original_xml_path)
            root = tree.getroot()
            
            updated_count = 0
            
            for track in root.findall('.//TRACK'):
                track_id = track.get('TrackID')
                if track_id in track_updates:
                    new_tags = track_updates[track_id]
                    formatted_comment = self.format_tags_for_comment(new_tags)
                    track.set('Comments', formatted_comment)
                    updated_count += 1
            
            # Write updated XML
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            print(f"Updated XML saved to {output_path}")
            print(f"Updated {updated_count} tracks")
            
        except Exception as e:
            print(f"Error generating updated XML: {e}")


def main():
    """Main function to demonstrate the system"""
    tagger = RekordboxAITagger()
    
    print("AI MyTag DJ Assistant - Phase 1 Demo")
    print("====================================")
    
    # Example usage (you'll need to provide your XML file path)
    xml_file_path = input("Enter path to your Rekordbox XML file: ").strip()
    
    if not os.path.exists(xml_file_path):
        print("File not found. Please check the path.")
        return
    
    # Parse XML
    tracks = tagger.parse_xml(xml_file_path)
    
    if not tracks:
        print("No tracks found or error parsing XML")
        return
    
    # Analyze existing tags to learn DNA
    analysis = tagger.analyze_existing_tags(tracks)
    
    # Find tracks without tags and suggest tags
    untagged_tracks = {tid: info for tid, info in tracks.items() 
                      if not info.get('existing_tags')}
    
    print(f"\nFound {len(untagged_tracks)} untagged tracks")
    
    if untagged_tracks:
        print("\nSuggesting tags for first 5 untagged tracks:")
        track_updates = {}
        
        for i, (track_id, track_info) in enumerate(list(untagged_tracks.items())[:5]):
            suggestions = tagger.suggest_tags_for_track(track_info, analysis)
            print(f"\n{i+1}. {track_info['artist']} - {track_info['title']}")
            print(f"   BPM: {track_info['bpm']}, Genre: {track_info['genre']}")
            print(f"   Suggested tags: {suggestions}")
            
            # For demo, auto-accept suggestions
            track_updates[track_id] = suggestions
        
        # Generate updated XML
        output_path = xml_file_path.replace('.xml', '_updated.xml')
        tagger.generate_updated_xml(xml_file_path, track_updates, output_path)

if __name__ == "__main__":
    main()
