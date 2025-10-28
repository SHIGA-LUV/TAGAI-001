#!/usr/bin/env python3
"""
AI MyTag DJ Assistant - Machine Learning Pattern Recognition
Phase 2: Learn from existing tags to improve suggestions
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ pandas not available. Install with: pip install pandas")
    PANDAS_AVAILABLE = False

# Import the original tagger
from rekordbox_ai_tagger import RekordboxAITagger

class MLPatternRecognizer:
    def __init__(self):
        """Initialize ML pattern recognizer"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ML features. Install with: pip install scikit-learn")
            
        self.tagger = RekordboxAITagger()
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.vectorizers = {}
        
        # Pattern analysis data
        self.tag_patterns = {}
        self.feature_importance = {}
        self.tag_clusters = {}
        
        # Training data
        self.training_data = []
        self.is_trained = False
        
    def extract_features_from_track(self, track_info: Dict) -> Dict:
        """Extract numerical features from track metadata"""
        features = {}
        
        # Basic numerical features
        features['bpm'] = float(track_info.get('bpm', 0)) if track_info.get('bpm') else 0
        features['year'] = self._extract_year(track_info.get('date_added', ''))
        
        # Genre encoding (one-hot style)
        genre = track_info.get('genre', '').lower()
        genre_keywords = ['house', 'techno', 'progressive', 'deep', 'melodic', 'ethnic', 'afro', 'trance']
        for keyword in genre_keywords:
            features[f'genre_{keyword}'] = 1 if keyword in genre else 0
            
        # Key analysis
        key = track_info.get('key', '')
        if key:
            features['has_key'] = 1
            # Major/minor detection
            features['is_minor'] = 1 if any(indicator in key.lower() for indicator in ['m', 'min', 'minor']) else 0
            # Key number (simplified)
            key_mapping = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
            key_letter = key[0].upper() if key else 'C'
            features['key_number'] = key_mapping.get(key_letter, 0)
        else:
            features['has_key'] = 0
            features['is_minor'] = 0
            features['key_number'] = 0
            
        # Text features (length, complexity)
        title = track_info.get('title', '')
        artist = track_info.get('artist', '')
        
        features['title_length'] = len(title)
        features['artist_length'] = len(artist)
        features['title_word_count'] = len(title.split())
        features['artist_word_count'] = len(artist.split())
        
        # Special characters (often indicate remixes, versions)
        features['has_parentheses'] = 1 if '(' in title or ')' in title else 0
        features['has_brackets'] = 1 if '[' in title or ']' in title else 0
        features['has_remix_indicator'] = 1 if any(word in title.lower() for word in ['remix', 'edit', 'mix', 'version']) else 0
        
        return features
        
    def _extract_year(self, date_string: str) -> int:
        """Extract year from date string"""
        try:
            if date_string:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y']:
                    try:
                        return datetime.strptime(date_string[:10], fmt).year
                    except:
                        continue
        except:
            pass
        return 2020  # Default year
        
    def prepare_training_data(self, tracks: Dict) -> Tuple[np.ndarray, Dict]:
        """Prepare training data from tracks with existing tags"""
        training_samples = []
        tag_labels = defaultdict(list)
        
        print("🔄 Preparing training data from existing tags...")
        
        for track_id, track_info in tracks.items():
            existing_tags = track_info.get('existing_tags', [])
            if not existing_tags:
                continue
                
            # Extract features
            features = self.extract_features_from_track(track_info)
            
            # Create training sample
            sample = {
                'track_id': track_id,
                'features': features,
                'tags': existing_tags,
                'artist': track_info.get('artist', ''),
                'title': track_info.get('title', '')
            }
            
            training_samples.append(sample)
            
            # Collect labels for each tag category
            for tag in existing_tags:
                tag_category = self._get_tag_category(tag)
                tag_labels[tag_category].append((features, tag))
        
        print(f"✅ Prepared {len(training_samples)} training samples")
        print(f"📊 Tag categories found: {list(tag_labels.keys())}")
        
        self.training_data = training_samples
        return training_samples, tag_labels
        
    def _get_tag_category(self, tag: str) -> str:
        """Determine which category a tag belongs to"""
        for category, info in self.tagger.tag_hierarchy.items():
            if tag in info['tags']:
                return category
        return 'OTHER'
        
    def train_models(self, tracks: Dict):
        """Train ML models on existing tag patterns"""
        print("🤖 Training ML models...")
        
        # Prepare training data
        training_samples, tag_labels = self.prepare_training_data(tracks)
        
        if not training_samples:
            print("❌ No training data available (no tracks with existing tags)")
            return
            
        # Train models for each tag category
        for category, category_data in tag_labels.items():
            if len(category_data) < 5:  # Need minimum samples
                print(f"⚠️ Skipping {category}: insufficient training data ({len(category_data)} samples)")
                continue
                
            print(f"🎯 Training model for {category} category...")
            
            # Prepare features and labels
            X = []
            y = []
            
            for features, tag in category_data:
                feature_vector = list(features.values())
                X.append(feature_vector)
                y.append(tag)
                
            X = np.array(X)
            y = np.array(y)
            
            # Handle case where all samples have the same label
            if len(set(y)) == 1:
                print(f"⚠️ Skipping {category}: all samples have the same label")
                continue
                
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[category] = scaler
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            
            # Cross-validation if enough samples
            if len(X) >= 10:
                cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(X)//2))
                print(f"   Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            clf.fit(X_scaled, y)
            self.models[category] = clf
            
            # Feature importance
            feature_names = list(training_samples[0]['features'].keys())
            importance_scores = dict(zip(feature_names, clf.feature_importances_))
            self.feature_importance[category] = importance_scores
            
            print(f"   ✅ Model trained for {category} with {len(X)} samples")
            print(f"   🔝 Top features: {sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
        # Train tag co-occurrence patterns
        self._analyze_tag_patterns(training_samples)
        
        # Cluster analysis
        self._perform_clustering_analysis(training_samples)
        
        self.is_trained = True
        print("🎉 ML model training complete!")
        
    def _analyze_tag_patterns(self, training_samples: List[Dict]):
        """Analyze patterns in tag combinations"""
        print("📈 Analyzing tag co-occurrence patterns...")
        
        # Tag co-occurrence matrix
        tag_cooccurrence = defaultdict(lambda: defaultdict(int))
        tag_frequency = Counter()
        
        for sample in training_samples:
            tags = sample['tags']
            tag_frequency.update(tags)
            
            # Count co-occurrences
            for i, tag1 in enumerate(tags):
                for j, tag2 in enumerate(tags):
                    if i != j:
                        tag_cooccurrence[tag1][tag2] += 1
                        
        # Calculate association strengths
        tag_associations = {}
        for tag1, cooccurrences in tag_cooccurrence.items():
            tag_associations[tag1] = {}
            for tag2, count in cooccurrences.items():
                # Calculate lift (association strength)
                lift = count / (tag_frequency[tag1] * tag_frequency[tag2] / len(training_samples))
                tag_associations[tag1][tag2] = lift
                
        self.tag_patterns = {
            'cooccurrence': dict(tag_cooccurrence),
            'frequency': dict(tag_frequency),
            'associations': tag_associations
        }
        
        print(f"   📊 Analyzed {len(tag_frequency)} unique tags")
        print(f"   🔗 Found {sum(len(assoc) for assoc in tag_associations.values())} tag associations")
        
    def _perform_clustering_analysis(self, training_samples: List[Dict]):
        """Perform clustering analysis to find track archetypes"""
        print("🎯 Performing clustering analysis...")
        
        if len(training_samples) < 10:
            print("   ⚠️ Insufficient data for clustering")
            return
            
        # Prepare feature matrix
        X = []
        track_info = []
        
        for sample in training_samples:
            feature_vector = list(sample['features'].values())
            X.append(feature_vector)
            track_info.append({
                'artist': sample['artist'],
                'title': sample['title'],
                'tags': sample['tags']
            })
            
        X = np.array(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        n_clusters = min(8, len(training_samples) // 3)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(track_info[i])
            
        # Find representative tags for each cluster
        cluster_profiles = {}
        for cluster_id, tracks in clusters.items():
            all_tags = []
            for track in tracks:
                all_tags.extend(track['tags'])
                
            tag_counts = Counter(all_tags)
            representative_tags = [tag for tag, count in tag_counts.most_common(5)]
            
            cluster_profiles[cluster_id] = {
                'size': len(tracks),
                'representative_tags': representative_tags,
                'sample_tracks': tracks[:3]  # First 3 tracks as examples
            }
            
        self.tag_clusters = {
            'model': kmeans,
            'scaler': scaler,
            'profiles': cluster_profiles
        }
        
        print(f"   🎯 Created {n_clusters} clusters")
        for cluster_id, profile in cluster_profiles.items():
            print(f"   Cluster {cluster_id}: {profile['size']} tracks, tags: {profile['representative_tags'][:3]}")
            
    def predict_tags(self, track_info: Dict) -> Tuple[List[str], float, Dict]:
        """Predict tags for a track using trained models"""
        if not self.is_trained:
            print("⚠️ Models not trained yet. Using basic suggestions.")
            return self.tagger.suggest_tags_for_track(track_info, {}), 0.5, {}
            
        # Extract features
        features = self.extract_features_from_track(track_info)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        predicted_tags = []
        confidence_scores = []
        prediction_details = {}
        
        # Predict for each category
        for category, model in self.models.items():
            if category in self.scalers:
                # Scale features
                X_scaled = self.scalers[category].transform(feature_vector)
                
                # Get prediction probabilities
                probabilities = model.predict_proba(X_scaled)[0]
                classes = model.classes_
                
                # Get top predictions
                top_indices = np.argsort(probabilities)[::-1][:2]  # Top 2 predictions
                
                for idx in top_indices:
                    if probabilities[idx] > 0.3:  # Confidence threshold
                        predicted_tags.append(classes[idx])
                        confidence_scores.append(probabilities[idx])
                        
                prediction_details[category] = {
                    'predictions': [(classes[i], probabilities[i]) for i in top_indices[:3]],
                    'top_prediction': classes[top_indices[0]],
                    'confidence': probabilities[top_indices[0]]
                }
                
        # Add pattern-based suggestions
        pattern_tags, pattern_confidence = self._get_pattern_based_suggestions(predicted_tags)
        predicted_tags.extend(pattern_tags)
        confidence_scores.extend([pattern_confidence] * len(pattern_tags))
        
        # Add cluster-based suggestions
        cluster_tags = self._get_cluster_based_suggestions(features)
        predicted_tags.extend(cluster_tags)
        confidence_scores.extend([0.6] * len(cluster_tags))
        
        # Remove duplicates and sort by hierarchy
        unique_tags = list(set(predicted_tags))
        sorted_tags = self.tagger.sort_tags_by_hierarchy(unique_tags)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return sorted_tags, overall_confidence, prediction_details
        
    def _get_pattern_based_suggestions(self, current_tags: List[str]) -> Tuple[List[str], float]:
        """Get additional tag suggestions based on learned patterns"""
        if not self.tag_patterns or not current_tags:
            return [], 0.0
            
        suggested_tags = []
        associations = self.tag_patterns.get('associations', {})
        
        for tag in current_tags:
            if tag in associations:
                # Get strongly associated tags
                for associated_tag, strength in associations[tag].items():
                    if strength > 1.5 and associated_tag not in current_tags:  # Strong association
                        suggested_tags.append(associated_tag)
                        
        return suggested_tags, 0.7
        
    def _get_cluster_based_suggestions(self, features: Dict) -> List[str]:
        """Get tag suggestions based on cluster similarity"""
        if not self.tag_clusters:
            return []
            
        try:
            # Transform features to cluster space
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            X_scaled = self.tag_clusters['scaler'].transform(feature_vector)
            
            # Find closest cluster
            cluster_id = self.tag_clusters['model'].predict(X_scaled)[0]
            
            # Get representative tags from that cluster
            cluster_profile = self.tag_clusters['profiles'].get(cluster_id, {})
            representative_tags = cluster_profile.get('representative_tags', [])
            
            return representative_tags[:3]  # Top 3 cluster tags
            
        except Exception as e:
            print(f"⚠️ Cluster prediction failed: {e}")
            return []
            
    def get_model_insights(self) -> Dict:
        """Get insights about the trained models"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
            
        insights = {
            'training_summary': {
                'total_samples': len(self.training_data),
                'categories_trained': list(self.models.keys()),
                'training_timestamp': datetime.now().isoformat()
            },
            'feature_importance': self.feature_importance,
            'tag_patterns': {
                'most_frequent_tags': sorted(
                    self.tag_patterns.get('frequency', {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:10],
                'strongest_associations': self._get_strongest_associations()
            },
            'cluster_analysis': {
                'num_clusters': len(self.tag_clusters.get('profiles', {})),
                'cluster_profiles': self.tag_clusters.get('profiles', {})
            }
        }
        
        return insights
        
    def _get_strongest_associations(self) -> List[Tuple[str, str, float]]:
        """Get the strongest tag associations"""
        associations = self.tag_patterns.get('associations', {})
        all_associations = []
        
        for tag1, assoc_dict in associations.items():
            for tag2, strength in assoc_dict.items():
                all_associations.append((tag1, tag2, strength))
                
        # Sort by strength and return top 10
        return sorted(all_associations, key=lambda x: x[2], reverse=True)[:10]
        
    def save_models(self, model_dir: str = 'ml_models'):
        """Save trained models to disk"""
        if not self.is_trained:
            print("⚠️ No trained models to save")
            return
            
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save models
        for category, model in self.models.items():
            with open(model_path / f'{category}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                
        # Save scalers
        for category, scaler in self.scalers.items():
            with open(model_path / f'{category}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                
        # Save patterns and clusters
        with open(model_path / 'tag_patterns.json', 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            patterns_json = {
                'cooccurrence': {k: dict(v) for k, v in self.tag_patterns['cooccurrence'].items()},
                'frequency': dict(self.tag_patterns['frequency']),
                'associations': self.tag_patterns['associations']
            }
            json.dump(patterns_json, f, indent=2)
            
        with open(model_path / 'clusters.pkl', 'wb') as f:
            pickle.dump(self.tag_clusters, f)
            
        print(f"💾 Models saved to {model_path}")
        
    def load_models(self, model_dir: str = 'ml_models'):
        """Load trained models from disk"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            print(f"⚠️ Model directory {model_path} not found")
            return False
            
        try:
            # Load models
            for model_file in model_path.glob('*_model.pkl'):
                category = model_file.stem.replace('_model', '')
                with open(model_file, 'rb') as f:
                    self.models[category] = pickle.load(f)
                    
            # Load scalers
            for scaler_file in model_path.glob('*_scaler.pkl'):
                category = scaler_file.stem.replace('_scaler', '')
                with open(scaler_file, 'rb') as f:
                    self.scalers[category] = pickle.load(f)
                    
            # Load patterns
            patterns_file = model_path / 'tag_patterns.json'
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.tag_patterns = {
                        'cooccurrence': defaultdict(lambda: defaultdict(int), 
                                                  {k: defaultdict(int, v) for k, v in patterns_data['cooccurrence'].items()}),
                        'frequency': Counter(patterns_data['frequency']),
                        'associations': patterns_data['associations']
                    }
                    
            # Load clusters
            clusters_file = model_path / 'clusters.pkl'
            if clusters_file.exists():
                with open(clusters_file, 'rb') as f:
                    self.tag_clusters = pickle.load(f)
                    
            self.is_trained = True
            print(f"📂 Models loaded from {model_path}")
            print(f"   Categories: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False

class EnhancedLearningAlgorithms:
    def __init__(self, ml_recognizer: MLPatternRecognizer):
        """Initialize enhanced learning algorithms"""
        self.ml_recognizer = ml_recognizer
        self.learning_history = []
        self.feedback_data = []
        self.adaptive_thresholds = {}
        self.user_preferences = {}
        
    def active_learning_suggest(self, uncertain_tracks: List[Dict], n_suggestions: int = 5) -> List[Dict]:
        """Suggest tracks for manual tagging using active learning"""
        if not self.ml_recognizer.is_trained:
            return uncertain_tracks[:n_suggestions]
            
        uncertainty_scores = []
        
        for track in uncertain_tracks:
            # Get prediction confidence
            _, confidence, details = self.ml_recognizer.predict_tags(track)
            
            # Calculate uncertainty (lower confidence = higher uncertainty)
            uncertainty = 1.0 - confidence
            
            # Add diversity bonus (prefer tracks different from training set)
            diversity_bonus = self._calculate_diversity_bonus(track)
            
            total_score = uncertainty + (diversity_bonus * 0.3)
            uncertainty_scores.append((track, total_score))
            
        # Sort by uncertainty score and return top suggestions
        uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
        return [track for track, _ in uncertainty_scores[:n_suggestions]]
        
    def _calculate_diversity_bonus(self, track: Dict) -> float:
        """Calculate diversity bonus for active learning"""
        if not self.ml_recognizer.training_data:
            return 0.0
            
        track_features = self.ml_recognizer.extract_features_from_track(track)
        
        # Compare with training data
        similarities = []
        for training_sample in self.ml_recognizer.training_data[:50]:  # Sample for efficiency
            training_features = training_sample['features']
            similarity = self._calculate_feature_similarity(track_features, training_features)
            similarities.append(similarity)
            
        # Higher diversity bonus for tracks less similar to training data
        avg_similarity = np.mean(similarities) if similarities else 0.5
        return 1.0 - avg_similarity
        
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature vectors"""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize and calculate similarity
                max_val = max(abs(val1), abs(val2), 1)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(similarity)
            else:
                # Binary features
                similarities.append(1.0 if val1 == val2 else 0.0)
                
        return np.mean(similarities) if similarities else 0.0
        
    def incremental_learning(self, new_tracks: Dict):
        """Incrementally update models with new tagged tracks"""
        print("🔄 Performing incremental learning...")
        
        # Extract new training samples
        new_samples = []
        for track_id, track_info in new_tracks.items():
            existing_tags = track_info.get('existing_tags', [])
            if existing_tags:
                features = self.ml_recognizer.extract_features_from_track(track_info)
                sample = {
                    'track_id': track_id,
                    'features': features,
                    'tags': existing_tags,
                    'artist': track_info.get('artist', ''),
                    'title': track_info.get('title', '')
                }
                new_samples.append(sample)
                
        if not new_samples:
            print("   No new tagged tracks to learn from")
            return
            
        # Add to training data
        self.ml_recognizer.training_data.extend(new_samples)
        
        # Update tag patterns
        self._update_tag_patterns(new_samples)
        
        # Retrain models with combined data (simplified incremental approach)
        print(f"   📚 Retraining with {len(new_samples)} new samples...")
        all_tracks = {}
        for sample in self.ml_recognizer.training_data:
            all_tracks[sample['track_id']] = {
                'artist': sample['artist'],
                'title': sample['title'],
                'existing_tags': sample['tags'],
                **sample['features']  # Include features as track info
            }
            
        self.ml_recognizer.train_models(all_tracks)
        
        print(f"   ✅ Incremental learning complete. Total training samples: {len(self.ml_recognizer.training_data)}")
        
    def _update_tag_patterns(self, new_samples: List[Dict]):
        """Update tag patterns with new samples"""
        if not hasattr(self.ml_recognizer, 'tag_patterns') or not self.ml_recognizer.tag_patterns:
            return
            
        # Update frequency counts
        for sample in new_samples:
            for tag in sample['tags']:
                self.ml_recognizer.tag_patterns['frequency'][tag] += 1
                
        # Update co-occurrence patterns
        for sample in new_samples:
            tags = sample['tags']
            for i, tag1 in enumerate(tags):
                for j, tag2 in enumerate(tags):
                    if i != j:
                        self.ml_recognizer.tag_patterns['cooccurrence'][tag1][tag2] += 1
                        
    def adaptive_threshold_learning(self, feedback_data: List[Dict]):
        """Learn adaptive confidence thresholds based on user feedback"""
        print("🎯 Learning adaptive confidence thresholds...")
        
        # Analyze feedback to adjust thresholds
        category_feedback = defaultdict(list)
        
        for feedback in feedback_data:
            category = feedback.get('category')
            confidence = feedback.get('confidence', 0.5)
            accepted = feedback.get('accepted', False)
            
            if category:
                category_feedback[category].append((confidence, accepted))
                
        # Calculate optimal thresholds for each category
        for category, feedback_list in category_feedback.items():
            if len(feedback_list) < 5:  # Need minimum feedback
                continue
                
            # Find threshold that maximizes accuracy
            confidences = [f[0] for f in feedback_list]
            acceptances = [f[1] for f in feedback_list]
            
            best_threshold = 0.5
            best_accuracy = 0.0
            
            for threshold in np.arange(0.3, 0.9, 0.05):
                predictions = [conf >= threshold for conf in confidences]
                accuracy = sum(p == a for p, a in zip(predictions, acceptances)) / len(acceptances)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    
            self.adaptive_thresholds[category] = best_threshold
            print(f"   {category}: threshold = {best_threshold:.2f}, accuracy = {best_accuracy:.1%}")
            
    def personalized_recommendations(self, user_id: str, track_info: Dict) -> Tuple[List[str], float]:
        """Generate personalized tag recommendations based on user preferences"""
        # Get base predictions
        base_tags, base_confidence, _ = self.ml_recognizer.predict_tags(track_info)
        
        # Apply user preferences if available
        if user_id in self.user_preferences:
            preferences = self.user_preferences[user_id]
            
            # Boost preferred tags
            preferred_tags = preferences.get('preferred_tags', [])
            boosted_tags = []
            
            for tag in base_tags:
                if tag in preferred_tags:
                    boosted_tags.insert(0, tag)  # Move to front
                else:
                    boosted_tags.append(tag)
                    
            # Add user-specific tags based on listening history
            genre_preference = preferences.get('genre_preference', {})
            track_genre = track_info.get('genre', '').lower()
            
            for genre, weight in genre_preference.items():
                if genre in track_genre and weight > 0.7:
                    # Add genre-specific tags
                    if 'house' in genre and 'Progressive House' not in boosted_tags:
                        boosted_tags.append('Progressive House')
                    elif 'techno' in genre and 'Melodic Techno' not in boosted_tags:
                        boosted_tags.append('Melodic Techno')
                        
            # Adjust confidence based on user agreement history
            user_accuracy = preferences.get('agreement_rate', 0.7)
            adjusted_confidence = base_confidence * user_accuracy
            
            return boosted_tags, adjusted_confidence
            
        return base_tags, base_confidence
        
    def learn_user_preferences(self, user_id: str, feedback_history: List[Dict]):
        """Learn user preferences from feedback history"""
        if not feedback_history:
            return
            
        preferences = {
            'preferred_tags': [],
            'disliked_tags': [],
            'genre_preference': {},
            'agreement_rate': 0.0
        }
        
        # Analyze tag preferences
        tag_feedback = defaultdict(list)
        total_feedback = len(feedback_history)
        agreements = 0
        
        for feedback in feedback_history:
            suggested_tags = feedback.get('suggested_tags', [])
            accepted_tags = feedback.get('accepted_tags', [])
            rejected_tags = feedback.get('rejected_tags', [])
            
            # Track agreements
            if set(accepted_tags).intersection(set(suggested_tags)):
                agreements += 1
                
            # Collect tag preferences
            for tag in accepted_tags:
                tag_feedback[tag].append(1)
            for tag in rejected_tags:
                tag_feedback[tag].append(0)
                
        # Calculate agreement rate
        preferences['agreement_rate'] = agreements / total_feedback if total_feedback > 0 else 0.7
        
        # Determine preferred and disliked tags
        for tag, feedback_list in tag_feedback.items():
            if len(feedback_list) >= 3:  # Minimum feedback for reliability
                acceptance_rate = sum(feedback_list) / len(feedback_list)
                if acceptance_rate >= 0.7:
                    preferences['preferred_tags'].append(tag)
                elif acceptance_rate <= 0.3:
                    preferences['disliked_tags'].append(tag)
                    
        # Analyze genre preferences
        genre_feedback = defaultdict(list)
        for feedback in feedback_history:
            track_genre = feedback.get('track_genre', '').lower()
            satisfaction = feedback.get('satisfaction_score', 0.5)
            if track_genre:
                genre_feedback[track_genre].append(satisfaction)
                
        for genre, scores in genre_feedback.items():
            if len(scores) >= 2:
                preferences['genre_preference'][genre] = np.mean(scores)
                
        self.user_preferences[user_id] = preferences
        
        print(f"📊 Learned preferences for user {user_id}:")
        print(f"   Agreement rate: {preferences['agreement_rate']:.1%}")
        print(f"   Preferred tags: {preferences['preferred_tags'][:5]}")
        print(f"   Genre preferences: {dict(list(preferences['genre_preference'].items())[:3])}")
        
    def ensemble_prediction(self, track_info: Dict, methods: List[str] = None) -> Tuple[List[str], float, Dict]:
        """Combine multiple prediction methods for better accuracy"""
        if methods is None:
            methods = ['ml_models', 'pattern_based', 'cluster_based', 'rule_based']
            
        all_predictions = {}
        all_confidences = {}
        
        # ML model predictions
        if 'ml_models' in methods and self.ml_recognizer.is_trained:
            ml_tags, ml_conf, ml_details = self.ml_recognizer.predict_tags(track_info)
            all_predictions['ml_models'] = ml_tags
            all_confidences['ml_models'] = ml_conf
            
        # Pattern-based predictions
        if 'pattern_based' in methods:
            pattern_tags, pattern_conf = self.ml_recognizer._get_pattern_based_suggestions([])
            all_predictions['pattern_based'] = pattern_tags
            all_confidences['pattern_based'] = pattern_conf
            
        # Rule-based predictions (original tagger)
        if 'rule_based' in methods:
            rule_tags = self.ml_recognizer.tagger.suggest_tags_for_track(track_info, {})
            all_predictions['rule_based'] = rule_tags
            all_confidences['rule_based'] = 0.6  # Default confidence
            
        # Combine predictions using weighted voting
        tag_votes = defaultdict(float)
        method_weights = {
            'ml_models': 0.4,
            'pattern_based': 0.3,
            'cluster_based': 0.2,
            'rule_based': 0.1
        }
        
        for method, tags in all_predictions.items():
            weight = method_weights.get(method, 0.1)
            confidence = all_confidences.get(method, 0.5)
            
            for tag in tags:
                tag_votes[tag] += weight * confidence
                
        # Select top tags based on votes
        sorted_tags = sorted(tag_votes.items(), key=lambda x: x[1], reverse=True)
        final_tags = [tag for tag, vote in sorted_tags if vote > 0.3][:8]  # Top 8 tags
        
        # Sort by hierarchy
        final_tags = self.ml_recognizer.tagger.sort_tags_by_hierarchy(final_tags)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(list(all_confidences.values())) if all_confidences else 0.5
        
        ensemble_details = {
            'method_predictions': all_predictions,
            'method_confidences': all_confidences,
            'tag_votes': dict(tag_votes),
            'methods_used': list(all_predictions.keys())
        }
        
        return final_tags, ensemble_confidence, ensemble_details

def demo_ml_pattern_recognition():
    """Demo function to test ML pattern recognition"""
    print("🤖 AI MyTag DJ Assistant - ML Pattern Recognition Demo")
    print("=" * 60)
    
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available. Install with: pip install scikit-learn")
        return
        
    try:
        # Initialize ML recognizer
        ml_recognizer = MLPatternRecognizer()
        
        # Create sample training data
        sample_tracks = {
            '1': {
                'artist': 'Deadmau5',
                'title': 'Strobe',
                'bpm': '128',
                'key': 'Db',
                'genre': 'Progressive House',
                'date_added': '2024-01-01',
                'existing_tags': ['Progressive House', '2-Build up', 'Emotional', 'Synth Lead']
            },
            '2': {
                'artist': 'Eric Prydz',
                'title': 'Opus',
                'bpm': '126',
                'key': 'F#m',
                'genre': 'Progressive House',
                'date_added': '2024-01-02',
                'existing_tags': ['Progressive House', '3-Peak Time', 'Energetic', 'Synth Lead']
            },
            '3': {
                'artist': 'Tale Of Us',
                'title': 'Monument',
                'bpm': '122',
                'key': 'Am',
                'genre': 'Melodic Techno',
                'date_added': '2024-01-03',
                'existing_tags': ['Melodic Techno', '2-Build up', 'Dark', 'Emotional']
            },
            '4': {
                'artist': 'Artbat',
                'title': 'Best Of Me',
                'bpm': '124',
                'key': 'Gm',
                'genre': 'Melodic Techno',
                'date_added': '2024-01-04',
                'existing_tags': ['Melodic Techno', '3-Peak Time', 'Dark', 'Synth Lead']
            },
            '5': {
                'artist': 'Nora En Pure',
                'title': 'Come With Me',
                'bpm': '118',
                'key': 'C',
                'genre': 'Deep House',
                'date_added': '2024-01-05',
                'existing_tags': ['Deep House', '1-Opener', 'Dreamy', 'Female Vocal']
            }
        }
        
        print(f"📊 Training on {len(sample_tracks)} sample tracks...")
        
        # Train models
        ml_recognizer.train_models(sample_tracks)
        
        # Test prediction on new track
        test_track = {
            'artist': 'Yotto',
            'title': 'The One You Left Behind',
            'bpm': '125',
            'key': 'Em',
            'genre': 'Progressive House',
            'date_added': '2024-01-06'
        }
        
        print(f"\n🔮 Predicting tags for: {test_track['artist']} - {test_track['title']}")
        predicted_tags, confidence, details = ml_recognizer.predict_tags(test_track)
        
        print(f"   Predicted tags: {predicted_tags}")
        print(f"   Overall confidence: {confidence:.1%}")
        print(f"   Prediction details: {details}")
        
        # Get model insights
        print(f"\n📈 Model Insights:")
        insights = ml_recognizer.get_model_insights()
        
        print(f"   Training samples: {insights['training_summary']['total_samples']}")
        print(f"   Categories trained: {insights['training_summary']['categories_trained']}")
        
        if 'most_frequent_tags' in insights['tag_patterns']:
            print(f"   Most frequent tags: {insights['tag_patterns']['most_frequent_tags'][:5]}")
            
        # Save models
        ml_recognizer.save_models()
        
        print("\n✅ ML Pattern Recognition demo complete!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_ml_pattern_recognition()