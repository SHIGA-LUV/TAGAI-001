#!/usr/bin/env python3
"""
Machine Learning Pattern Recognition for Rekordbox AI Tagger
Phase 2: Learning system that adapts to user preferences
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ML imports (will be available when scikit-learn is installed)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print('⚠️ scikit-learn not available - ML features will be limited')

class MLPatternLearner:
    def __init__(self, model_save_path: str = '/Users/shiraazoulay/ml_models'):
        """
        Initialize the ML pattern learning system
        
        Args:
            model_save_path: Directory to save trained models
        """
        self.model_save_path = model_save_path
        self.ensure_model_directory()
        
        # Learning data storage
        self.user_interactions = []
        self.tag_patterns = defaultdict(list)
        self.feature_importance = {}
        
        # Models
        self.tag_classifier = None
        self.feature_scaler = None
        self.clustering_model = None
        
        # Pattern recognition data
        self.learned_patterns = {
            'tag_sequences': defaultdict(int),
            'bpm_tag_correlations': defaultdict(list),
            'key_tag_correlations': defaultdict(list),
            'genre_tag_patterns': defaultdict(Counter),
            'time_based_patterns': defaultdict(list),
            'user_preferences': {}
        }
        
        # Tag hierarchy for learning
        self.tag_hierarchy = {
            'SITUATION': ['1-Opener', '2-Build up', '3-Peak Time', '4-Cool Down', '5-After Hours'],
            'GENRE': ['Melodic Techno', 'Ethnic House', 'Progressive House', 'Deep House', 'Afro House'],
            'COMPONENTS': ['Piano', 'Darbuka', 'Female Vocal', 'Male Vocal', 'Strings', 'Synth Lead', 'Percussion'],
            'MOOD': ['Tribal', 'Dreamy', 'Sexy', 'Energetic', 'Emotional', 'Dark', 'Uplifting']
        }
        
        self.load_existing_patterns()
    
    def ensure_model_directory(self):
        """Create model directory if it doesn't exist"""
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            print(f'✅ Created model directory: {self.model_save_path}')
    
    def record_user_interaction(self, track_info: Dict, suggested_tags: List[str], 
                              selected_tags: List[str], timestamp: str = None):
        """
        Record a user interaction for learning
        
        Args:
            track_info: Track metadata
            suggested_tags: Tags suggested by AI
            selected_tags: Tags actually selected by user
            timestamp: When the interaction occurred
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        interaction = {
            'timestamp': timestamp,
            'track_info': track_info,
            'suggested_tags': suggested_tags,
            'selected_tags': selected_tags,
            'acceptance_rate': len(set(suggested_tags) & set(selected_tags)) / len(suggested_tags) if suggested_tags else 0,
            'track_features': self.extract_track_features(track_info)
        }
        
        self.user_interactions.append(interaction)
        self.update_patterns_from_interaction(interaction)
        
        print(f'✅ Recorded interaction: {len(selected_tags)} tags selected')
    
    def extract_track_features(self, track_info: Dict) -> Dict:
        """
        Extract numerical features from track info for ML
        
        Args:
            track_info: Track metadata
            
        Returns:
            Dictionary of numerical features
        """
        features = {}
        
        # Basic features
        features['bpm'] = float(track_info.get('bpm', 0)) if track_info.get('bpm') else 0
        features['year'] = float(track_info.get('year', 0)) if track_info.get('year') else 0
        
        # Text features (will be vectorized later)
        features['artist'] = track_info.get('artist', '')
        features['title'] = track_info.get('title', '')
        features['genre'] = track_info.get('genre', '')
        features['key'] = track_info.get('key', '')
        
        # Spotify features (if available)
        if 'audio_features' in track_info:
            audio_features = track_info['audio_features']
            features.update({
                'danceability': audio_features.get('danceability', 0),
                'energy': audio_features.get('energy', 0),
                'valence': audio_features.get('valence', 0),
                'acousticness': audio_features.get('acousticness', 0),
                'instrumentalness': audio_features.get('instrumentalness', 0),
                'liveness': audio_features.get('liveness', 0),
                'speechiness': audio_features.get('speechiness', 0)
            })
        
        # Librosa features (if available)
        if 'spectral_features' in track_info:
            spectral = track_info['spectral_features']
            features.update({
                'spectral_centroid': spectral.get('spectral_centroid_mean', 0),
                'zero_crossing_rate': spectral.get('zero_crossing_rate_mean', 0),
                'rms_energy': spectral.get('rms_mean', 0)
            })
        
        return features
    
    def update_patterns_from_interaction(self, interaction: Dict):
        """
        Update learned patterns based on user interaction
        
        Args:
            interaction: User interaction data
        """
        track_features = interaction['track_features']
        selected_tags = interaction['selected_tags']
        
        # Update BPM-tag correlations
        bpm = track_features.get('bpm', 0)
        if bpm > 0:
            for tag in selected_tags:
                self.learned_patterns['bpm_tag_correlations'][tag].append(bpm)
        
        # Update key-tag correlations
        key = track_features.get('key', '')
        if key:
            for tag in selected_tags:
                self.learned_patterns['key_tag_correlations'][key].append(tag)
        
        # Update genre-tag patterns
        genre = track_features.get('genre', '').lower()
        if genre:
            for tag in selected_tags:
                self.learned_patterns['genre_tag_patterns'][genre][tag] += 1
        
        # Update tag sequences (co-occurrence patterns)
        for i, tag1 in enumerate(selected_tags):
            for tag2 in selected_tags[i+1:]:
                sequence = tuple(sorted([tag1, tag2]))
                self.learned_patterns['tag_sequences'][sequence] += 1
        
        # Update time-based patterns
        hour = datetime.fromisoformat(interaction['timestamp']).hour
        self.learned_patterns['time_based_patterns'][hour].extend(selected_tags)
    
    def train_tag_classifier(self) -> bool:
        """
        Train ML classifier for tag prediction
        
        Returns:
            True if training successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            print('⚠️ Cannot train classifier - scikit-learn not available')
            return False
        
        if len(self.user_interactions) < 10:
            print('⚠️ Need at least 10 interactions to train classifier')
            return False
        
        print('🤖 Training tag prediction classifier...')
        
        try:
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if len(X) == 0:
                print('❌ No valid training data')
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train classifier
            self.tag_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tag_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.tag_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f'✅ Classifier trained with {accuracy:.2f} accuracy')
            
            # Save model
            self.save_models()
            
            return True
            
        except Exception as e:
            print(f'❌ Classifier training error: {e}')
            return False
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from user interactions
        
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        for interaction in self.user_interactions:
            track_features = interaction['track_features']
            selected_tags = interaction['selected_tags']
            
            if not selected_tags:
                continue
            
            # Convert track features to numerical array
            feature_vector = []
            
            # Numerical features
            numerical_features = ['bpm', 'year', 'danceability', 'energy', 'valence', 
                                'acousticness', 'instrumentalness', 'liveness', 
                                'speechiness', 'spectral_centroid', 'zero_crossing_rate', 'rms_energy']
            
            for feature in numerical_features:
                feature_vector.append(track_features.get(feature, 0))
            
            # For each tag category, create a training example
            for category, category_tags in self.tag_hierarchy.items():
                selected_category_tags = [tag for tag in selected_tags if tag in category_tags]
                if selected_category_tags:
                    features.append(feature_vector)
                    labels.append(selected_category_tags[0])  # Use first tag as label
        
        return np.array(features), np.array(labels)
    
    def predict_tags_for_track(self, track_info: Dict) -> Dict[str, List[str]]:
        """
        Predict tags for a track using learned patterns
        
        Args:
            track_info: Track metadata
            
        Returns:
            Dictionary of predicted tags by category
        """
        predictions = {
            'pattern_based': self.predict_from_patterns(track_info),
            'ml_based': self.predict_from_ml_model(track_info) if SKLEARN_AVAILABLE else [],
            'confidence_scores': {}
        }
        
        return predictions
    
    def predict_from_patterns(self, track_info: Dict) -> List[str]:
        """
        Predict tags based on learned patterns
        
        Args:
            track_info: Track metadata
            
        Returns:
            List of predicted tags
        """
        suggestions = []
        track_features = self.extract_track_features(track_info)
        
        # BPM-based suggestions
        bpm = track_features.get('bpm', 0)
        if bpm > 0:
            for tag, bpm_list in self.learned_patterns['bpm_tag_correlations'].items():
                if bpm_list:
                    avg_bpm = np.mean(bpm_list)
                    if abs(bpm - avg_bpm) < 10:  # Within 10 BPM
                        suggestions.append(tag)
        
        # Genre-based suggestions
        genre = track_features.get('genre', '').lower()
        if genre and genre in self.learned_patterns['genre_tag_patterns']:
            genre_tags = self.learned_patterns['genre_tag_patterns'][genre]
            # Get most common tags for this genre
            suggestions.extend([tag for tag, count in genre_tags.most_common(3)])
        
        # Key-based suggestions
        key = track_features.get('key', '')
        if key and key in self.learned_patterns['key_tag_correlations']:
            key_tags = self.learned_patterns['key_tag_correlations'][key]
            # Get most common tags for this key
            tag_counts = Counter(key_tags)
            suggestions.extend([tag for tag, count in tag_counts.most_common(2)])
        
        return list(set(suggestions))
    
    def predict_from_ml_model(self, track_info: Dict) -> List[str]:
        """
        Predict tags using trained ML model
        
        Args:
            track_info: Track metadata
            
        Returns:
            List of predicted tags
        """
        if not self.tag_classifier or not self.feature_scaler:
            return []
        
        try:
            track_features = self.extract_track_features(track_info)
            
            # Prepare feature vector
            feature_vector = []
            numerical_features = ['bpm', 'year', 'danceability', 'energy', 'valence', 
                                'acousticness', 'instrumentalness', 'liveness', 
                                'speechiness', 'spectral_centroid', 'zero_crossing_rate', 'rms_energy']
            
            for feature in numerical_features:
                feature_vector.append(track_features.get(feature, 0))
            
            # Scale and predict
            feature_vector_scaled = self.feature_scaler.transform([feature_vector])
            prediction = self.tag_classifier.predict(feature_vector_scaled)
            
            return prediction.tolist()
            
        except Exception as e:
            print(f'❌ ML prediction error: {e}')
            return []
    
    def get_learning_statistics(self) -> Dict:
        """
        Get statistics about the learning system
        
        Returns:
            Dictionary with learning statistics
        """
        stats = {
            'total_interactions': len(self.user_interactions),
            'unique_tags_learned': len(set().union(*[i['selected_tags'] for i in self.user_interactions])),
            'most_common_tags': [],
            'bpm_ranges_learned': {},
            'genre_patterns_learned': len(self.learned_patterns['genre_tag_patterns']),
            'tag_sequences_learned': len(self.learned_patterns['tag_sequences']),
            'model_trained': self.tag_classifier is not None
        }
        
        # Most common tags
        all_tags = []
        for interaction in self.user_interactions:
            all_tags.extend(interaction['selected_tags'])
        
        if all_tags:
            tag_counts = Counter(all_tags)
            stats['most_common_tags'] = tag_counts.most_common(10)
        
        # BPM ranges for tags
        for tag, bpm_list in self.learned_patterns['bpm_tag_correlations'].items():
            if bpm_list:
                stats['bpm_ranges_learned'][tag] = {
                    'min': min(bpm_list),
                    'max': max(bpm_list),
                    'avg': np.mean(bpm_list)
                }
        
        return stats
    
    def save_models(self):
        """Save trained models and patterns to disk"""
        try:
            # Save patterns
            patterns_file = os.path.join(self.model_save_path, 'learned_patterns.json')
            with open(patterns_file, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                patterns_to_save = {}
                for key, value in self.learned_patterns.items():
                    if isinstance(value, defaultdict):
                        patterns_to_save[key] = dict(value)
                    else:
                        patterns_to_save[key] = value
                json.dump(patterns_to_save, f, indent=2)
            
            # Save ML models
            if self.tag_classifier:
                classifier_file = os.path.join(self.model_save_path, 'tag_classifier.pkl')
                with open(classifier_file, 'wb') as f:
                    pickle.dump(self.tag_classifier, f)
            
            if self.feature_scaler:
                scaler_file = os.path.join(self.model_save_path, 'feature_scaler.pkl')
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
            
            # Save interactions
            interactions_file = os.path.join(self.model_save_path, 'user_interactions.json')
            with open(interactions_file, 'w') as f:
                json.dump(self.user_interactions, f, indent=2)
            
            print('✅ Models and patterns saved successfully')
            
        except Exception as e:
            print(f'❌ Error saving models: {e}')
    
    def load_existing_patterns(self):
        """Load previously saved patterns and models"""
        try:
            # Load patterns
            patterns_file = os.path.join(self.model_save_path, 'learned_patterns.json')
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    loaded_patterns = json.load(f)
                    for key, value in loaded_patterns.items():
                        self.learned_patterns[key] = value
                print('✅ Loaded existing patterns')
            
            # Load interactions
            interactions_file = os.path.join(self.model_save_path, 'user_interactions.json')
            if os.path.exists(interactions_file):
                with open(interactions_file, 'r') as f:
                    self.user_interactions = json.load(f)
                print(f'✅ Loaded {len(self.user_interactions)} previous interactions')
            
            # Load ML models
            if SKLEARN_AVAILABLE:
                classifier_file = os.path.join(self.model_save_path, 'tag_classifier.pkl')
                if os.path.exists(classifier_file):
                    with open(classifier_file, 'rb') as f:
                        self.tag_classifier = pickle.load(f)
                    print('✅ Loaded trained classifier')
                
                scaler_file = os.path.join(self.model_save_path, 'feature_scaler.pkl')
                if os.path.exists(scaler_file):
                    with open(scaler_file, 'rb') as f:
                        self.feature_scaler = pickle.load(f)
                    print('✅ Loaded feature scaler')
            
        except Exception as e:
            print(f'⚠️ Could not load existing patterns: {e}')

# Example usage and testing
if __name__ == '__main__':
    print('🤖 ML Pattern Learner - Test Mode')
    print('=' * 45)
    
    learner = MLPatternLearner()
    
    print('
📋 Learning Capabilities:')
    print('- User interaction recording')
    print('- Pattern recognition from behavior')
    print('- ML-based tag prediction')
    print('- Continuous learning and adaptation')
    
    print('
🔧 Pattern Types:')
    print('- BPM-tag correlations')
    print('- Genre-tag patterns')
    print('- Key-tag relationships')
    print('- Tag co-occurrence sequences')
    print('- Time-based preferences')
    
    # Show current statistics
    stats = learner.get_learning_statistics()
    print(f'
📊 Current Statistics:')
    print(f'- Total interactions: {stats["total_interactions"]}')
    print(f'- Unique tags learned: {stats["unique_tags_learned"]}')
    print(f'- Model trained: {stats["model_trained"]}')
    
    print('
⚠️ To enable full ML features:')
    print('1. Install scikit-learn: pip install scikit-learn')
    print('2. Record user interactions with record_user_interaction()')
    print('3. Train models with train_tag_classifier()')
