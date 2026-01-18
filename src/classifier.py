import os
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Tuple, Optional, List
from tqdm import tqdm

from config import (
    SVM_C, MODELS_PATH, CHARACTERS, CHAR_TO_LABEL, LABEL_TO_CHAR, RANDOM_SEED
)
from feature_extraction import extract_hog_features_batch, extract_combined_features_batch

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)


class FaceDetector:

    def __init__(self, C: float = SVM_C, use_color: bool = False, use_lbp: bool = True):
        self.C = C
        self.use_color = use_color
        self.use_lbp = use_lbp
        self.classifier = LinearSVC(C=C, max_iter=10000, dual=True)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, patches: List[np.ndarray], show_progress: bool = True) -> np.ndarray:
        if self.use_color or self.use_lbp:
            return extract_combined_features_batch(patches, use_color=self.use_color, use_lbp=self.use_lbp, show_progress=show_progress)
        else:
            return extract_hog_features_batch(patches, show_progress=show_progress)
    
    def train(self, X_patches: np.ndarray, y: np.ndarray, validate: bool = True):
        print("Extracting features for training...")
        X_features = self.extract_features(list(X_patches))
        
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_features)
        
        if validate:
            print("Performing cross-validation...")
            scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
            print(f"Cross-validation scores: {scores}")
            print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        print("Training SVM classifier...")
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        print("Training complete!")
    
    def predict(self, patches: List[np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet!")
        
        X_features = self.extract_features(patches, show_progress=False)
        X_scaled = self.scaler.transform(X_features)
        
        return self.classifier.predict(X_scaled)
    
    def predict_scores(self, patches: List[np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet!")
        
        X_features = self.extract_features(patches, show_progress=False)
        X_scaled = self.scaler.transform(X_features)
        
        return self.classifier.decision_function(X_scaled)
    
    def predict_single(self, patch: np.ndarray) -> Tuple[int, float]:
        X_features = self.extract_features([patch], show_progress=False)
        X_scaled = self.scaler.transform(X_features)
        
        prediction = self.classifier.predict(X_scaled)[0]
        score = self.classifier.decision_function(X_scaled)[0]
        
        return prediction, score
    
    def save(self, path: str = None):
        if path is None:
            os.makedirs(MODELS_PATH, exist_ok=True)
            path = os.path.join(MODELS_PATH, "face_detector.pkl")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler,
                'use_color': self.use_color,
                'use_lbp': self.use_lbp,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {path}")

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(MODELS_PATH, "face_detector.pkl")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.use_color = data['use_color']
        self.use_lbp = data.get('use_lbp', True)
        self.is_trained = data['is_trained']
        print(f"Model loaded from {path}")


class CharacterRecognizer:

    def __init__(self, C: float = SVM_C, use_color: bool = True, use_lbp: bool = True):
        self.C = C
        self.use_color = use_color
        self.use_lbp = use_lbp
        self.classifier = LinearSVC(C=C, max_iter=10000, multi_class='ovr', dual=True)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, patches: List[np.ndarray], show_progress: bool = True) -> np.ndarray:
        if self.use_color or self.use_lbp:
            return extract_combined_features_batch(patches, use_color=self.use_color, use_lbp=self.use_lbp, show_progress=show_progress)
        else:
            return extract_hog_features_batch(patches, show_progress=show_progress)
    
    def train(self, X_patches: np.ndarray, y: np.ndarray, validate: bool = True):
        valid_mask = (y >= 0) & (y < len(CHARACTERS))
        X_valid = X_patches[valid_mask]
        y_valid = y[valid_mask]
        
        print(f"Training on {len(y_valid)} samples")
        for i, char in enumerate(CHARACTERS):
            count = np.sum(y_valid == i)
            print(f"  {char}: {count} samples")
        
        print("Extracting features for training...")
        X_features = self.extract_features(list(X_valid))
        
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_features)
        
        if validate:
            print("Performing cross-validation...")
            scores = cross_val_score(self.classifier, X_scaled, y_valid, cv=5)
            print(f"Cross-validation scores: {scores}")
            print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        print("Training SVM classifier...")
        self.classifier.fit(X_scaled, y_valid)
        self.is_trained = True
        print("Training complete!")
    
    def predict(self, patches: List[np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet!")
        
        X_features = self.extract_features(patches, show_progress=False)
        X_scaled = self.scaler.transform(X_features)
        
        return self.classifier.predict(X_scaled)
    
    def predict_scores(self, patches: List[np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet!")
        
        X_features = self.extract_features(patches, show_progress=False)
        X_scaled = self.scaler.transform(X_features)
        
        return self.classifier.decision_function(X_scaled)
    
    def predict_single(self, patch: np.ndarray) -> Tuple[str, float]:
        X_features = self.extract_features([patch], show_progress=False)
        X_scaled = self.scaler.transform(X_features)
        
        prediction = self.classifier.predict(X_scaled)[0]
        scores = self.classifier.decision_function(X_scaled)[0]
        
        character = CHARACTERS[prediction]
        score = scores[prediction] if len(scores.shape) > 0 else scores
        
        return character, float(score)
    
    def save(self, path: str = None):
        if path is None:
            os.makedirs(MODELS_PATH, exist_ok=True)
            path = os.path.join(MODELS_PATH, "character_recognizer.pkl")

        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler,
                'use_color': self.use_color,
                'use_lbp': self.use_lbp,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {path}")

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(MODELS_PATH, "character_recognizer.pkl")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.use_color = data['use_color']
        self.use_lbp = data.get('use_lbp', True)
        self.is_trained = data['is_trained']
        print(f"Model loaded from {path}")


class CombinedDetector:
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.character_recognizer = CharacterRecognizer()
    
    def train(self, X_patches: np.ndarray, y_binary: np.ndarray, y_character: np.ndarray):
        print("=" * 50)
        print("Training Face Detector (Task 1)")
        print("=" * 50)
        self.face_detector.train(X_patches, y_binary)
        
        print("\n" + "=" * 50)
        print("Training Character Recognizer (Task 2)")
        print("=" * 50)
        positive_mask = y_binary == 1
        self.character_recognizer.train(X_patches[positive_mask], y_character[positive_mask])
    
    def predict(self, patches: List[np.ndarray]) -> List[Tuple[bool, str, float]]:
        results = []
        
        face_predictions = self.face_detector.predict(patches)
        face_scores = self.face_detector.predict_scores(patches)
        
        for i, patch in enumerate(patches):
            if face_predictions[i] == 1:
                character, char_score = self.character_recognizer.predict_single(patch)
                results.append((True, character, face_scores[i]))
            else:
                results.append((False, None, face_scores[i]))
        
        return results
    
    def save(self, path: str = None):
        if path is None:
            path = MODELS_PATH
        os.makedirs(path, exist_ok=True)
        
        self.face_detector.save(os.path.join(path, "face_detector.pkl"))
        self.character_recognizer.save(os.path.join(path, "character_recognizer.pkl"))
    
    def load(self, path: str = None):
        if path is None:
            path = MODELS_PATH
        
        self.face_detector.load(os.path.join(path, "face_detector.pkl"))
        self.character_recognizer.load(os.path.join(path, "character_recognizer.pkl"))


if __name__ == "__main__":
    from data_loader import prepare_training_data
    
    print("Preparing training data...")
    X, y_binary = prepare_training_data(include_character_labels=False)
    X2, y_character = prepare_training_data(include_character_labels=True)
    
    print("\nTraining face detector...")
    detector = FaceDetector()
    detector.train(X, y_binary)
    detector.save()
    
    print("\nTraining character recognizer...")
    recognizer = CharacterRecognizer()
    recognizer.train(X2, y_character)
    recognizer.save()
