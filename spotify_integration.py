#!/usr/bin/env python3
"""
AI MyTag DJ Assistant - Spotify API Integration
Phase 2: Audio Features Analysis via Spotify Web API
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import base64
from urllib.parse import quote

# Import the original tagger
from rekordbox_ai_tagger import RekordboxAITagger

class SpotifyIntegration:
    def __init__(self, client_id: str = None, client_secret: str = None):
        """Initialize Spotify API integration"""
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials required. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
        
        self.access_token = None
        self.token_expires_at = None
        self.base_url = 'https://api.spotify.com/v1'
        
        # Audio feature mappings to our tag system
        self.feature_mappings = {
            'energy': {
                'high': ['3-Peak Time', 'Energetic'],
                'medium': ['2-Build up'],
                'low': ['4-Cool Down', '5-After Hours']
            },
            'valence': {
                'high': ['Uplifting', 'Energetic'],
                'medium': ['Emotional'],
                'low': ['Dark', 'Emotional']
            },
            'danceability': {
                'high': ['3-Peak Time'],
                'medium': ['2-Build up'],
                'low': ['5-After Hours']
            },
            'acousticness': {
                'high': ['Piano', 'Strings'],
                'low': ['Synth Lead']
            },
            'instrumentalness': {
                'high': [],
                'low': ['Female Vocal', 'Male Vocal']
            }
        }
        
        # Initialize tagger for hierarchy sorting
        self.tagger = RekordboxAITagger()
        
    def authenticate(self) -> bool:
        """Authenticate with Spotify API using Client Credentials flow"""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return True
            
        auth_url = 'https://accounts.spotify.com/api/token'
        
        # Encode credentials
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = token_data['expires_in']
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # 60s buffer
            
            print(f"✅ Spotify authentication successful. Token expires at {self.token_expires_at}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Spotify authentication failed: {e}")
            return False
            
    def search_track(self, artist: str, title: str, album: str = None) -> Optional[Dict]:
        """Search for a track on Spotify"""
        if not self.authenticate():
            return None
            
        # Clean and format search query
        query_parts = [f'artist:"{artist}"', f'track:"{title}"']
        if album:
            query_parts.append(f'album:"{album}"')
            
        query = ' '.join(query_parts)
        
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        params = {
            'q': query,
            'type': 'track',
            'limit': 5  # Get multiple results for better matching
        }
        
        try:
            response = requests.get(f'{self.base_url}/search', headers=headers, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            tracks = search_results.get('tracks', {}).get('items', [])
            
            if not tracks:
                # Try simplified search
                simplified_query = f'{artist} {title}'
                params['q'] = simplified_query
                
                response = requests.get(f'{self.base_url}/search', headers=headers, params=params)
                response.raise_for_status()
                
                search_results = response.json()
                tracks = search_results.get('tracks', {}).get('items', [])
            
            # Find best match
            best_match = self._find_best_match(tracks, artist, title)
            return best_match
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Spotify search failed for {artist} - {title}: {e}")
            return None
            
    def _find_best_match(self, tracks: List[Dict], target_artist: str, target_title: str) -> Optional[Dict]:
        """Find the best matching track from search results"""
        if not tracks:
            return None
            
        def similarity_score(track):
            track_artist = track['artists'][0]['name'].lower()
            track_title = track['name'].lower()
            target_artist_lower = target_artist.lower()
            target_title_lower = target_title.lower()
            
            # Simple similarity scoring
            artist_score = 1.0 if target_artist_lower in track_artist or track_artist in target_artist_lower else 0.0
            title_score = 1.0 if target_title_lower in track_title or track_title in target_title_lower else 0.0
            
            return artist_score * 0.6 + title_score * 0.4
        
        # Sort by similarity and return best match
        tracks.sort(key=similarity_score, reverse=True)
        best_track = tracks[0]
        
        if similarity_score(best_track) > 0.3:  # Minimum similarity threshold
            return best_track
        
        return None
        
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """Get audio features for a Spotify track"""
        if not self.authenticate():
            return None
            
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(f'{self.base_url}/audio-features/{track_id}', headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get audio features for track {track_id}: {e}")
            return None
            
    def get_audio_analysis(self, track_id: str) -> Optional[Dict]:
        """Get detailed audio analysis for a Spotify track"""
        if not self.authenticate():
            return None
            
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(f'{self.base_url}/audio-analysis/{track_id}', headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get audio analysis for track {track_id}: {e}")
            return None
            
    def analyze_track_for_tags(self, artist: str, title: str, album: str = None) -> Tuple[List[str], float, Dict]:
        """Analyze a track and suggest tags based on Spotify audio features"""
        # Search for track
        track = self.search_track(artist, title, album)
        if not track:
            return [], 0.0, {}
            
        # Get audio features
        track_id = track['id']
        features = self.get_audio_features(track_id)
        if not features:
            return [], 0.0, {}
            
        # Generate tags based on audio features
        suggested_tags = []
        confidence_scores = []
        feature_analysis = {}
        
        # Energy-based tags
        energy = features.get('energy', 0)
        feature_analysis['energy'] = energy
        if energy > 0.7:
            suggested_tags.extend(self.feature_mappings['energy']['high'])
            confidence_scores.append(0.8)
        elif energy > 0.4:
            suggested_tags.extend(self.feature_mappings['energy']['medium'])
            confidence_scores.append(0.6)
        else:
            suggested_tags.extend(self.feature_mappings['energy']['low'])
            confidence_scores.append(0.7)
            
        # Valence-based tags (positivity)
        valence = features.get('valence', 0)
        feature_analysis['valence'] = valence
        if valence > 0.6:
            suggested_tags.extend(self.feature_mappings['valence']['high'])
            confidence_scores.append(0.7)
        elif valence > 0.3:
            suggested_tags.extend(self.feature_mappings['valence']['medium'])
            confidence_scores.append(0.6)
        else:
            suggested_tags.extend(self.feature_mappings['valence']['low'])
            confidence_scores.append(0.8)
            
        # Danceability
        danceability = features.get('danceability', 0)
        feature_analysis['danceability'] = danceability
        if danceability > 0.7:
            suggested_tags.extend(self.feature_mappings['danceability']['high'])
            confidence_scores.append(0.8)
        elif danceability > 0.5:
            suggested_tags.extend(self.feature_mappings['danceability']['medium'])
            confidence_scores.append(0.6)
        else:
            suggested_tags.extend(self.feature_mappings['danceability']['low'])
            confidence_scores.append(0.5)
            
        # Acousticness (acoustic vs electronic)
        acousticness = features.get('acousticness', 0)
        feature_analysis['acousticness'] = acousticness
        if acousticness > 0.5:
            suggested_tags.extend(self.feature_mappings['acousticness']['high'])
            confidence_scores.append(0.7)
        else:
            suggested_tags.extend(self.feature_mappings['acousticness']['low'])
            confidence_scores.append(0.6)
            
        # Instrumentalness (vocal vs instrumental)
        instrumentalness = features.get('instrumentalness', 0)
        feature_analysis['instrumentalness'] = instrumentalness
        if instrumentalness < 0.3:  # Likely has vocals
            suggested_tags.extend(self.feature_mappings['instrumentalness']['low'])
            confidence_scores.append(0.6)
            
        # Tempo-based situation tags
        tempo = features.get('tempo', 0)
        feature_analysis['tempo'] = tempo
        if tempo > 0:
            if tempo < 100:
                suggested_tags.extend(['5-After Hours'])
                confidence_scores.append(0.8)
            elif tempo < 115:
                suggested_tags.extend(['4-Cool Down'])
                confidence_scores.append(0.7)
            elif tempo < 125:
                suggested_tags.extend(['1-Opener', '2-Build up'])
                confidence_scores.append(0.6)
            elif tempo < 135:
                suggested_tags.extend(['2-Build up', '3-Peak Time'])
                confidence_scores.append(0.7)
            else:
                suggested_tags.extend(['3-Peak Time'])
                confidence_scores.append(0.8)
                
        # Genre analysis from Spotify track data
        if 'genres' in track.get('artists', [{}])[0]:
            genres = track['artists'][0]['genres']
            for genre in genres:
                if 'house' in genre.lower():
                    if 'deep' in genre.lower():
                        suggested_tags.append('Deep House')
                    elif 'progressive' in genre.lower():
                        suggested_tags.append('Progressive House')
                    else:
                        suggested_tags.append('Progressive House')
                    confidence_scores.append(0.9)
                elif 'techno' in genre.lower():
                    suggested_tags.append('Melodic Techno')
                    confidence_scores.append(0.9)
                    
        # Remove duplicates and sort by hierarchy
        unique_tags = list(set(suggested_tags))
        sorted_tags = self.tagger.sort_tags_by_hierarchy(unique_tags)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Add Spotify metadata
        feature_analysis.update({
            'spotify_track_id': track_id,
            'spotify_popularity': track.get('popularity', 0),
            'spotify_preview_url': track.get('preview_url'),
            'spotify_external_url': track.get('external_urls', {}).get('spotify'),
            'key': features.get('key', -1),
            'mode': features.get('mode', -1),
            'time_signature': features.get('time_signature', 4),
            'loudness': features.get('loudness', 0),
            'speechiness': features.get('speechiness', 0),
            'liveness': features.get('liveness', 0)
        })
        
        return sorted_tags, overall_confidence, feature_analysis
        
    def batch_analyze_tracks(self, tracks: Dict) -> Dict[str, Dict]:
        """Analyze multiple tracks in batch"""
        results = {}
        total_tracks = len(tracks)
        
        print(f"🎵 Starting Spotify analysis for {total_tracks} tracks...")
        
        for i, (track_id, track_info) in enumerate(tracks.items(), 1):
            artist = track_info.get('artist', '')
            title = track_info.get('title', '')
            
            print(f"[{i}/{total_tracks}] Analyzing: {artist} - {title}")
            
            try:
                tags, confidence, analysis = self.analyze_track_for_tags(artist, title)
                
                results[track_id] = {
                    'suggested_tags': tags,
                    'confidence': confidence,
                    'spotify_analysis': analysis,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                # Rate limiting - Spotify allows 100 requests per minute
                time.sleep(0.1)  # Small delay to be respectful
                
            except Exception as e:
                print(f"❌ Error analyzing {artist} - {title}: {e}")
                results[track_id] = {
                    'error': str(e),
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
        print(f"✅ Spotify analysis complete. Analyzed {len([r for r in results.values() if 'suggested_tags' in r])} tracks successfully.")
        return results
        
    def save_analysis_cache(self, analysis_results: Dict, cache_file: str = 'spotify_analysis_cache.json'):
        """Save analysis results to cache file"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"💾 Analysis cache saved to {cache_file}")
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
            
    def load_analysis_cache(self, cache_file: str = 'spotify_analysis_cache.json') -> Dict:
        """Load analysis results from cache file"""
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            print(f"📂 Loaded analysis cache from {cache_file}")
            return cache_data
        except FileNotFoundError:
            print(f"📂 No cache file found at {cache_file}")
            return {}
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return {}

def demo_spotify_integration():
    """Demo function to test Spotify integration"""
    print("🎵 AI MyTag DJ Assistant - Spotify Integration Demo")
    print("=" * 50)
    
    # Check for credentials
    if not os.getenv('SPOTIFY_CLIENT_ID') or not os.getenv('SPOTIFY_CLIENT_SECRET'):
        print("❌ Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
        print("   Get credentials from: https://developer.spotify.com/dashboard")
        return
        
    try:
        # Initialize Spotify integration
        spotify = SpotifyIntegration()
        
        # Test authentication
        if not spotify.authenticate():
            print("❌ Authentication failed")
            return
            
        # Test track analysis
        test_tracks = [
            ('Deadmau5', 'Strobe'),
            ('Eric Prydz', 'Opus'),
            ('Above & Beyond', 'Sun & Moon')
        ]
        
        for artist, title in test_tracks:
            print(f"\n🔍 Analyzing: {artist} - {title}")
            tags, confidence, analysis = spotify.analyze_track_for_tags(artist, title)
            
            print(f"   Suggested tags: {tags}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Energy: {analysis.get('energy', 'N/A'):.2f}")
            print(f"   Valence: {analysis.get('valence', 'N/A'):.2f}")
            print(f"   Tempo: {analysis.get('tempo', 'N/A'):.0f} BPM")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_spotify_integration()
