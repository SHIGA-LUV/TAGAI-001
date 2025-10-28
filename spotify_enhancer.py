#!/usr/bin/env python3
"""
Spotify API Integration for Rekordbox AI Tagger
Phase 2: Enhanced audio feature analysis
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import time
from typing import Dict, List, Optional, Tuple
import re

class SpotifyEnhancer:
    def __init__(self, client_id: str = None, client_secret: str = None):
        """
        Initialize Spotify API client
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.spotify = None
        
        # Audio feature mappings for AI suggestions
        self.feature_mappings = {
            'energy': {
                'high': ['3-Peak Time', 'Energetic'],
                'medium': ['2-Build up', 'Progressive House'],
                'low': ['4-Cool Down', 'Dreamy']
            },
            'danceability': {
                'high': ['3-Peak Time', 'Tribal'],
                'medium': ['2-Build up'],
                'low': ['5-After Hours', 'Emotional']
            },
            'valence': {
                'high': ['Uplifting', 'Energetic'],
                'medium': ['Progressive House'],
                'low': ['Dark', 'Emotional']
            },
            'acousticness': {
                'high': ['Piano', 'Strings'],
                'low': ['Synth Lead']
            }
        }
        
        if client_id and client_secret:
            self.initialize_spotify()
    
    def initialize_spotify(self):
        """Initialize Spotify client with credentials"""
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            print('✅ Spotify API initialized successfully')
            return True
        except Exception as e:
            print(f'❌ Spotify API initialization failed: {e}')
            return False
    
    def search_track(self, artist: str, title: str, album: str = None) -> Optional[Dict]:
        """
        Search for a track on Spotify
        
        Args:
            artist: Track artist
            title: Track title
            album: Track album (optional)
            
        Returns:
            Track information dict or None if not found
        """
        if not self.spotify:
            print('⚠️ Spotify not initialized')
            return None
            
        try:
            # Clean up search terms
            clean_artist = self.clean_search_term(artist)
            clean_title = self.clean_search_term(title)
            
            # Build search query
            query = f'artist:{clean_artist} track:{clean_title}'
            if album:
                clean_album = self.clean_search_term(album)
                query += f' album:{clean_album}'
            
            # Search
            results = self.spotify.search(q=query, type='track', limit=5)
            
            if results['tracks']['items']:
                # Return the best match
                best_match = results['tracks']['items'][0]
                return {
                    'spotify_id': best_match['id'],
                    'name': best_match['name'],
                    'artist': best_match['artists'][0]['name'],
                    'album': best_match['album']['name'],
                    'popularity': best_match['popularity'],
                    'preview_url': best_match['preview_url']
                }
            
            return None
            
        except Exception as e:
            print(f'❌ Spotify search error: {e}')
            return None
    
    def get_audio_features(self, spotify_id: str) -> Optional[Dict]:
        """
        Get audio features for a track
        
        Args:
            spotify_id: Spotify track ID
            
        Returns:
            Audio features dict or None if error
        """
        if not self.spotify:
            return None
            
        try:
            features = self.spotify.audio_features([spotify_id])[0]
            if features:
                return {
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'key': features['key'],
                    'loudness': features['loudness'],
                    'mode': features['mode'],
                    'speechiness': features['speechiness'],
                    'acousticness': features['acousticness'],
                    'instrumentalness': features['instrumentalness'],
                    'liveness': features['liveness'],
                    'valence': features['valence'],
                    'tempo': features['tempo']
                }
            return None
            
        except Exception as e:
            print(f'❌ Audio features error: {e}')
            return None
    
    def enhance_track_with_spotify(self, track_info: Dict) -> Dict:
        """
        Enhance track information with Spotify data
        
        Args:
            track_info: Original track info from Rekordbox
            
        Returns:
            Enhanced track info with Spotify data
        """
        enhanced_track = track_info.copy()
        
        # Search for track on Spotify
        spotify_track = self.search_track(
            artist=track_info.get('artist', ''),
            title=track_info.get('title', ''),
            album=track_info.get('album', '')
        )
        
        if spotify_track:
            enhanced_track['spotify_data'] = spotify_track
            
            # Get audio features
            audio_features = self.get_audio_features(spotify_track['spotify_id'])
            if audio_features:
                enhanced_track['audio_features'] = audio_features
                
                # Generate AI suggestions based on audio features
                ai_suggestions = self.generate_ai_suggestions_from_features(audio_features)
                enhanced_track['spotify_ai_suggestions'] = ai_suggestions
        
        return enhanced_track
    
    def generate_ai_suggestions_from_features(self, audio_features: Dict) -> List[str]:
        """
        Generate AI tag suggestions based on Spotify audio features
        
        Args:
            audio_features: Spotify audio features
            
        Returns:
            List of suggested tags
        """
        suggestions = []
        
        # Energy-based suggestions
        energy = audio_features.get('energy', 0)
        if energy > 0.7:
            suggestions.extend(self.feature_mappings['energy']['high'])
        elif energy > 0.4:
            suggestions.extend(self.feature_mappings['energy']['medium'])
        else:
            suggestions.extend(self.feature_mappings['energy']['low'])
        
        # Danceability-based suggestions
        danceability = audio_features.get('danceability', 0)
        if danceability > 0.7:
            suggestions.extend(self.feature_mappings['danceability']['high'])
        elif danceability > 0.4:
            suggestions.extend(self.feature_mappings['danceability']['medium'])
        else:
            suggestions.extend(self.feature_mappings['danceability']['low'])
        
        # Valence (mood) based suggestions
        valence = audio_features.get('valence', 0)
        if valence > 0.6:
            suggestions.extend(self.feature_mappings['valence']['high'])
        elif valence > 0.3:
            suggestions.extend(self.feature_mappings['valence']['medium'])
        else:
            suggestions.extend(self.feature_mappings['valence']['low'])
        
        # Acousticness-based suggestions
        acousticness = audio_features.get('acousticness', 0)
        if acousticness > 0.5:
            suggestions.extend(self.feature_mappings['acousticness']['high'])
        else:
            suggestions.extend(self.feature_mappings['acousticness']['low'])
        
        # Remove duplicates and return
        return list(set(suggestions))
    
    def clean_search_term(self, term: str) -> str:
        """Clean search terms for better Spotify matching"""
        if not term:
            return ''
        
        # Remove common problematic characters
        cleaned = re.sub(r'[\[\](){}]', '', term)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def batch_enhance_tracks(self, tracks: Dict, max_requests: int = 100) -> Dict:
        """
        Enhance multiple tracks with Spotify data (with rate limiting)
        
        Args:
            tracks: Dictionary of tracks to enhance
            max_requests: Maximum API requests to make
            
        Returns:
            Dictionary of enhanced tracks
        """
        enhanced_tracks = {}
        request_count = 0
        
        print(f'🎵 Enhancing {min(len(tracks), max_requests)} tracks with Spotify data...')
        
        for track_id, track_info in tracks.items():
            if request_count >= max_requests:
                print(f'⚠️ Reached API limit ({max_requests} requests)')
                break
                
            enhanced_track = self.enhance_track_with_spotify(track_info)
            enhanced_tracks[track_id] = enhanced_track
            
            request_count += 1
            
            # Rate limiting (Spotify allows ~100 requests per minute)
            if request_count % 10 == 0:
                print(f'Processed {request_count} tracks...')
                time.sleep(1)  # Brief pause
        
        print(f'✅ Enhanced {len(enhanced_tracks)} tracks with Spotify data')
        return enhanced_tracks

# Example usage and testing
if __name__ == '__main__':
    print('🎵 Spotify Enhancer - Test Mode')
    print('=' * 40)
    
    # Initialize without credentials for testing
    enhancer = SpotifyEnhancer()
    
    print('
📋 Feature Mappings:')
    for feature, mappings in enhancer.feature_mappings.items():
        print(f'   {feature}: {list(mappings.keys())}')
    
    print('
⚠️ To use Spotify features, you need:')
    print('1. Spotify Developer Account')
    print('2. Create an app at https://developer.spotify.com/')
    print('3. Get Client ID and Client Secret')
    print('4. Initialize with: SpotifyEnhancer(client_id, client_secret)')
