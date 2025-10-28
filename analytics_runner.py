#!/usr/bin/env python3
"""
Music Collection Analytics Runner
Analyzes your Rekordbox collection and provides insights
"""

import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

def analyze_collection():
    print('📈 MUSIC COLLECTION ANALYTICS')
    print('=' * 40)
    
    xml_file = Path('Documents/shigmusic.xml')
    if not xml_file.exists():
        print('❌ Rekordbox XML file not found!')
        return
    
    try:
        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tracks = root.findall('.//TRACK')
        
        print(f'✅ Loaded {len(tracks)} tracks from Rekordbox')
        
        # Collect data
        genres = Counter()
        artists = Counter()
        bpms = []
        keys = Counter()
        years = Counter()
        
        for track in tracks:
            # Genre analysis
            genre = track.get('Genre', 'Unknown')
            if genre and genre != 'Unknown':
                genres[genre] += 1
            
            # Artist analysis
            artist = track.get('Artist', 'Unknown')
            if artist and artist != 'Unknown':
                artists[artist] += 1
            
            # BPM analysis
            try:
                bpm = float(track.get('AverageBpm', 0))
                if bpm > 0:
                    bpms.append(bpm)
            except (ValueError, TypeError):
                pass
            
            # Key analysis
            key = track.get('Tonality', '')
            if key:
                keys[key] += 1
            
            # Year analysis
            year = track.get('Year', '')
            if year:
                try:
                    year_int = int(year)
                    if 1900 <= year_int <= 2030:
                        years[year_int] += 1
                except (ValueError, TypeError):
                    pass
        
        # Display analytics
        print('\n🎵 COLLECTION OVERVIEW:')
        print(f'  Total tracks: {len(tracks):,}')
        print(f'  Unique artists: {len(artists):,}')
        print(f'  Unique genres: {len(genres):,}')
        
        if bpms:
            avg_bpm = sum(bpms) / len(bpms)
            print(f'  Average BPM: {avg_bpm:.1f}')
            print(f'  BPM range: {min(bpms):.0f} - {max(bpms):.0f}')
        
        # Top genres
        print('\n🎶 TOP GENRES:')
        for genre, count in genres.most_common(10):
            percentage = (count / len(tracks)) * 100
            print(f'  {genre}: {count} tracks ({percentage:.1f}%)')
        
        # Top artists
        print('\n🎤 TOP ARTISTS:')
        for artist, count in artists.most_common(10):
            print(f'  {artist}: {count} tracks')
        
        # BPM distribution
        if bpms:
            print('\n🎵 BPM DISTRIBUTION:')
            bpm_ranges = {
                '< 100': len([b for b in bpms if b < 100]),
                '100-120': len([b for b in bpms if 100 <= b < 120]),
                '120-130': len([b for b in bpms if 120 <= b < 130]),
                '130-140': len([b for b in bpms if 130 <= b < 140]),
                '140+': len([b for b in bpms if b >= 140])
            }
            for range_name, count in bpm_ranges.items():
                if count > 0:
                    percentage = (count / len(bpms)) * 100
                    print(f'  {range_name} BPM: {count} tracks ({percentage:.1f}%)')
        
        # Key distribution
        if keys:
            print('\n🎹 KEY DISTRIBUTION:')
            for key, count in keys.most_common(10):
                percentage = (count / len(tracks)) * 100
                print(f'  {key}: {count} tracks ({percentage:.1f}%)')
        
        # Year distribution
        if years:
            print('\n📅 YEAR DISTRIBUTION:')
            sorted_years = sorted(years.items(), key=lambda x: x[0], reverse=True)
            for year, count in sorted_years[:10]:
                percentage = (count / len(tracks)) * 100
                print(f'  {year}: {count} tracks ({percentage:.1f}%)')
        
        # AI Tagging Recommendations
        print('\n🤖 AI TAGGING RECOMMENDATIONS:')
        
        # Analyze current tagging patterns
        tagged_tracks = 0
        for track in tracks:
            # Check if track has meaningful tags (not just basic metadata)
            has_tags = any([
                track.get('Comments', ''),
                track.get('Label', ''),
                track.get('Remixer', '')
            ])
            if has_tags:
                tagged_tracks += 1
        
        tag_percentage = (tagged_tracks / len(tracks)) * 100
        print(f'  Currently tagged tracks: {tagged_tracks}/{len(tracks)} ({tag_percentage:.1f}%)')
        
        if tag_percentage < 50:
            print('  🎯 High priority: Most tracks need AI tagging')
        elif tag_percentage < 80:
            print('  🟡 Medium priority: Some tracks could benefit from AI tagging')
        else:
            print('  ✅ Good coverage: AI can help refine existing tags')
        
        # Genre-based recommendations
        if genres:
            top_genre = genres.most_common(1)[0][0]
            print(f'  Primary genre focus: {top_genre}')
            print(f'  Recommended AI tags: Situation, Mood, Components for {top_genre}')
        
    except Exception as e:
        print(f'❌ Analytics error: {e}')

if __name__ == '__main__':
    analyze_collection()
    print('\n✨ Analytics complete!')