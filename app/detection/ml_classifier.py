"""
ML-Based Scam Classifier
Trained on WhatsApp scam dataset for accurate scam detection
"""

import os
import re
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from app.api.schemas import ScamType

logger = logging.getLogger(__name__)

# Mapping from dataset scam types to our ScamType enum
SCAM_TYPE_MAPPING = {
    "Phishing Scam (Link Sharing)": ScamType.BANKING_FRAUD,
    "Fake Discount/Refund Scam": ScamType.UPI_FRAUD,
    "Fake Loan Approval Scam": ScamType.BANKING_FRAUD,
    "WhatsApp Account Hacking Scam": ScamType.IMPERSONATION,
    "Fake E-commerce Scam": ScamType.DELIVERY_SCAM,
    "Cryptocurrency Investment Scam": ScamType.CRYPTO_SCAM,
    "Fake Job Offer Scam": ScamType.JOB_SCAM,
    "Tax Refund Scam": ScamType.TAX_GST_SCAM,
    "WhatsApp Lottery/Prize Scam": ScamType.LOTTERY_SCAM,
    "Fake Technical Support Scam": ScamType.TECH_SUPPORT,
    "Friend in Distress Scam": ScamType.IMPERSONATION,
    "SIM Card Replacement Scam": ScamType.KYC_SCAM,
    "UPI Scam": ScamType.UPI_FRAUD,
    "Fake Charity/Donation Scam": ScamType.UPI_FRAUD,
}


@dataclass
class MLClassifierResult:
    """Result from ML classification"""
    score: float
    scam_type: ScamType
    original_type: str
    confidence: float
    top_features: List[str] = field(default_factory=list)


class ScamMLClassifier:
    """
    Machine Learning based scam classifier
    Uses TF-IDF features with ensemble voting
    Weight: 0.30 in ensemble
    """

    def __init__(self, weight: float = 0.30, dataset_path: str = None):
        self.weight = weight
        self.dataset_path = dataset_path or self._find_dataset()

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[VotingClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_trained = False

        # Model storage path
        self.model_dir = Path(__file__).parent.parent.parent / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Try to load pre-trained model or train new one
        if not self._load_model():
            self._train_model()

    def _find_dataset(self) -> str:
        """Find the dataset file"""
        possible_paths = [
            "/Users/apple/Desktop/guvi/whatsapp_scam_dataset.csv",
            "whatsapp_scam_dataset.csv",
            "../whatsapp_scam_dataset.csv",
            "../../whatsapp_scam_dataset.csv",
            "../../../whatsapp_scam_dataset.csv",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        logger.warning("Dataset not found, ML classifier will use fallback")
        return None

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for ML with enhanced feature extraction"""
        # Lowercase
        text = text.lower()

        # Extract urgency indicators before normalization
        urgency_words = ['urgent', 'immediately', 'now', 'today', 'expire', 'last chance', 'jaldi', 'abhi', 'turant']
        urgency_count = sum(1 for word in urgency_words if word in text)
        urgency_marker = ' URGENCY_HIGH ' if urgency_count >= 2 else (' URGENCY_MED ' if urgency_count == 1 else '')

        # Extract threat indicators
        threat_words = ['block', 'suspend', 'cancel', 'arrest', 'legal', 'police', 'fine', 'penalty']
        threat_count = sum(1 for word in threat_words if word in text)
        threat_marker = ' THREAT_HIGH ' if threat_count >= 2 else (' THREAT_MED ' if threat_count == 1 else '')

        # Extract authority claims
        authority_words = ['bank', 'government', 'rbi', 'official', 'department', 'ministry', 'court']
        authority_count = sum(1 for word in authority_words if word in text)
        authority_marker = ' AUTHORITY_CLAIM ' if authority_count >= 1 else ''

        # Remove URLs but keep indicator
        url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
        text = re.sub(r'https?://\S+|www\.\S+', ' URL_LINK ', text)
        if url_count > 1:
            text += ' MULTIPLE_URLS '

        # Remove UPI IDs but keep indicator
        upi_count = len(re.findall(r'[a-zA-Z0-9._-]+@[a-zA-Z]{2,10}(?!\.[a-z]{2,})', text))
        text = re.sub(r'[a-zA-Z0-9._-]+@[a-zA-Z]{2,10}(?!\.[a-z]{2,})', ' UPI_ID ', text)
        if upi_count > 0:
            text += ' HAS_UPI '

        # Remove phone numbers but keep indicator
        phone_count = len(re.findall(r'\+?91[\s-]?[6-9]\d{9}|\b[6-9]\d{9}\b', text))
        text = re.sub(r'\+?91[\s-]?[6-9]\d{9}|\b[6-9]\d{9}\b', ' PHONE_NUMBER ', text)
        if phone_count > 0:
            text += ' HAS_PHONE '

        # Remove amounts but keep indicator
        amount_matches = re.findall(r'₹[\d,]+|rs\.?\s*[\d,]+', text)
        text = re.sub(r'₹[\d,]+|rs\.?\s*[\d,]+', ' AMOUNT ', text)
        if amount_matches:
            text += ' HAS_AMOUNT '

        # Detect OTP/PIN requests
        if re.search(r'\b(otp|pin|password|cvv)\b', text):
            text += ' CREDENTIAL_REQUEST '

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Add extracted markers
        text = f"{urgency_marker} {threat_marker} {authority_marker} {text}"

        return text.strip()

    def _train_model(self):
        """Train the ML classifier on the dataset"""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            logger.warning("No dataset available for training")
            return

        try:
            logger.info(f"Loading dataset from {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)

            # Preprocess messages
            df['processed_message'] = df['message'].apply(self._preprocess_text)

            # Prepare features and labels
            X = df['processed_message'].values
            y = df['scam_type'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create TF-IDF vectorizer with optimized parameters
            self.vectorizer = TfidfVectorizer(
                max_features=8000,       # Increased for more features
                ngram_range=(1, 3),       # Capture phrases
                min_df=2,
                max_df=0.90,              # Stricter to remove very common words
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\b[A-Za-z_][A-Za-z_]+\b'  # Include our markers
            )

            # Fit and transform
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)

            # Create ensemble classifier with optimized parameters
            # Using 'balanced' for automatic class weight handling
            nb_clf = MultinomialNB(alpha=0.05)  # Lower alpha for less smoothing
            lr_clf = LogisticRegression(
                max_iter=2000,
                C=2.0,                    # Higher regularization
                class_weight='balanced',  # Auto-balance classes
                random_state=42,
                solver='lbfgs'
            )
            rf_clf = RandomForestClassifier(
                n_estimators=200,         # More trees
                max_depth=20,             # Limit depth to prevent overfitting
                min_samples_split=5,
                class_weight='balanced',  # Auto-balance classes
                random_state=42,
                n_jobs=-1
            )

            self.classifier = VotingClassifier(
                estimators=[
                    ('nb', nb_clf),
                    ('lr', lr_clf),
                    ('rf', rf_clf)
                ],
                voting='soft',
                weights=[0.2, 0.4, 0.4]   # Give more weight to LR and RF
            )

            # Train
            logger.info("Training ML classifier...")
            self.classifier.fit(X_train_tfidf, y_train)

            # Evaluate
            y_pred = self.classifier.predict(X_test_tfidf)
            accuracy = (y_pred == y_test).mean()
            logger.info(f"ML Classifier accuracy: {accuracy:.4f}")

            # Create label encoder for scam types
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df['scam_type'].unique())

            self.is_trained = True

            # Save model
            self._save_model()

            logger.info("ML Classifier training complete")

        except Exception as e:
            logger.error(f"Error training ML classifier: {e}")
            self.is_trained = False

    def _save_model(self):
        """Save trained model to disk"""
        try:
            model_path = self.model_dir / "scam_classifier.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'classifier': self.classifier,
                    'label_encoder': self.label_encoder
                }, f)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _load_model(self) -> bool:
        """Load pre-trained model from disk"""
        try:
            model_path = self.model_dir / "scam_classifier.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.classifier = data['classifier']
                    self.label_encoder = data['label_encoder']
                self.is_trained = True
                logger.info("Loaded pre-trained ML classifier")
                return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
        return False

    def predict(self, text: str) -> MLClassifierResult:
        """
        Predict scam type and confidence for given text

        Args:
            text: Message text to classify

        Returns:
            MLClassifierResult with prediction
        """
        if not self.is_trained:
            return self._fallback_predict(text)

        try:
            # Preprocess
            processed = self._preprocess_text(text)

            # Transform
            X = self.vectorizer.transform([processed])

            # Predict probabilities
            proba = self.classifier.predict_proba(X)[0]
            predicted_idx = proba.argmax()
            confidence = proba[predicted_idx]

            # Get scam type
            original_type = self.classifier.classes_[predicted_idx]
            mapped_type = SCAM_TYPE_MAPPING.get(original_type, ScamType.UNKNOWN)

            # Get top features
            top_features = self._get_top_features(processed)

            return MLClassifierResult(
                score=confidence,
                scam_type=mapped_type,
                original_type=original_type,
                confidence=confidence,
                top_features=top_features
            )

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._fallback_predict(text)

    def _get_top_features(self, processed_text: str, top_n: int = 5) -> List[str]:
        """Get top contributing features for the prediction"""
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_vector = self.vectorizer.transform([processed_text])

            # Get non-zero features
            non_zero_indices = tfidf_vector.nonzero()[1]
            feature_scores = [(feature_names[i], tfidf_vector[0, i])
                            for i in non_zero_indices]

            # Sort by score
            feature_scores.sort(key=lambda x: x[1], reverse=True)

            return [f[0] for f in feature_scores[:top_n]]
        except:
            return []

    def _fallback_predict(self, text: str) -> MLClassifierResult:
        """Fallback prediction when model is not available - enhanced heuristics"""
        text_lower = text.lower()

        # Enhanced keyword-based fallback with weights
        scam_indicators = {
            ScamType.UPI_FRAUD: {
                'keywords': ['upi', 'blocked', 'unblock', 'pay to', 'gpay', 'phonepe', 'paytm', 'bhim'],
                'high_signal': ['upi blocked', 'account blocked', 'unblock upi'],
                'weight': 1.0
            },
            ScamType.JOB_SCAM: {
                'keywords': ['job', 'offer', 'shortlisted', 'registration fee', 'hr', 'hiring', 'salary', 'work from home', 'part time', 'full time'],
                'high_signal': ['job offer', 'registration fee', 'selected for job', 'hiring immediately'],
                'weight': 1.0
            },
            ScamType.LOTTERY_SCAM: {
                'keywords': ['won', 'lottery', 'prize', 'congratulations', 'claim', 'winner', 'lucky', 'jackpot'],
                'high_signal': ['you won', 'claim prize', 'lottery winner', 'won lottery'],
                'weight': 1.0
            },
            ScamType.CRYPTO_SCAM: {
                'keywords': ['crypto', 'invest', 'bitcoin', 'returns', 'trading', 'forex', 'profit', 'guaranteed returns'],
                'high_signal': ['invest now', 'guaranteed returns', 'double your money'],
                'weight': 1.0
            },
            ScamType.TECH_SUPPORT: {
                'keywords': ['malware', 'teamviewer', 'support', 'device', 'virus', 'infected', 'remote access', 'anydesk'],
                'high_signal': ['virus detected', 'call support', 'install anydesk', 'remote access'],
                'weight': 1.0
            },
            ScamType.KYC_SCAM: {
                'keywords': ['kyc', 'sim', 'aadhaar', 'otp', 'upgrade', 'pan', 'verify identity', 'link aadhaar'],
                'high_signal': ['kyc update', 'sim block', 'verify kyc', 'aadhaar link'],
                'weight': 1.0
            },
            ScamType.BANKING_FRAUD: {
                'keywords': ['bank', 'account', 'verify', 'click here', 'credit card', 'debit card', 'net banking', 'ifsc'],
                'high_signal': ['bank account blocked', 'verify your account', 'suspicious activity'],
                'weight': 1.0
            },
            ScamType.TAX_GST_SCAM: {
                'keywords': ['tax', 'refund', 'gst', 'income tax', 'itr', 'pan verification'],
                'high_signal': ['tax refund', 'gst refund', 'claim refund'],
                'weight': 0.9
            },
            ScamType.IMPERSONATION: {
                'keywords': ['friend', 'stuck', 'urgently', 'otp', 'emergency', 'help me', 'borrow'],
                'high_signal': ['friend in trouble', 'urgently need money', 'stuck abroad', 'send otp'],
                'weight': 1.0
            },
            ScamType.DELIVERY_SCAM: {
                'keywords': ['order', 'delivery', 'prepay', 'package', 'customs', 'cod', 'shipping'],
                'high_signal': ['order stuck', 'prepay customs', 'delivery failed', 'pay customs'],
                'weight': 0.9
            }
        }

        best_match = ScamType.UNKNOWN
        best_score = 0
        matched_features = []

        for scam_type, config in scam_indicators.items():
            score = 0
            features = []

            # Check for high signal phrases (worth more)
            for phrase in config['high_signal']:
                if phrase in text_lower:
                    score += 2.0
                    features.append(f"high:{phrase}")

            # Check for individual keywords
            for kw in config['keywords']:
                if kw in text_lower:
                    score += 0.5
                    features.append(kw)

            # Apply type weight
            score *= config['weight']

            if score > best_score:
                best_score = score
                best_match = scam_type
                matched_features = features[:5]  # Top 5 features

        # Check for general urgency/threat signals
        urgency_signals = ['urgent', 'immediately', 'now', 'today only', 'last chance', 'expire']
        threat_signals = ['legal action', 'arrest', 'police', 'fine', 'penalty', 'suspend']

        urgency_count = sum(1 for s in urgency_signals if s in text_lower)
        threat_count = sum(1 for s in threat_signals if s in text_lower)

        # Boost score if urgency or threats detected
        if urgency_count > 0:
            best_score += urgency_count * 0.3
        if threat_count > 0:
            best_score += threat_count * 0.4

        # Calculate confidence (0 to 1)
        confidence = min(0.85, best_score * 0.12) if best_score > 0 else 0.1

        return MLClassifierResult(
            score=confidence,
            scam_type=best_match,
            original_type="fallback",
            confidence=confidence,
            top_features=matched_features
        )

    def batch_predict(self, texts: List[str]) -> List[MLClassifierResult]:
        """Predict for multiple texts"""
        return [self.predict(text) for text in texts]


class ScamSimilarityMatcher:
    """
    Finds similar scam messages from the dataset
    Useful for understanding scam patterns
    """

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or self._find_dataset()
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_dataset()

    def _find_dataset(self) -> str:
        possible_paths = [
            "/Users/apple/Desktop/guvi/whatsapp_scam_dataset.csv",
            "whatsapp_scam_dataset.csv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _load_dataset(self):
        """Load and prepare dataset for similarity matching"""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return

        try:
            self.df = pd.read_csv(self.dataset_path)

            # Create TF-IDF matrix
            self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['message'])

            logger.info(f"Loaded {len(self.df)} scam messages for similarity matching")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")

    def find_similar(self, text: str, top_n: int = 3) -> List[Dict]:
        """
        Find most similar scam messages from dataset

        Args:
            text: Query text
            top_n: Number of similar messages to return

        Returns:
            List of similar message dictionaries
        """
        if self.df is None or self.vectorizer is None:
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Transform query
            query_vec = self.vectorizer.transform([text])

            # Calculate similarity
            similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

            # Get top matches
            top_indices = similarities.argsort()[-top_n:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold
                    results.append({
                        'message': self.df.iloc[idx]['message'],
                        'scam_type': self.df.iloc[idx]['scam_type'],
                        'similarity': float(similarities[idx]),
                        'description': self.df.iloc[idx]['description']
                    })

            return results

        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []
