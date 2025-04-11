import numpy as np
import pandas as pd
from typing import List, Dict, Any
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class FeatureExtractor:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.count_vectorizer = CountVectorizer(max_features=5000)

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_word_length': np.mean([len(word) for word in text.split()]),
            'stopword_ratio': len([w for w in text.lower().split() if w in self.stop_words]) / len(text.split()),
            'uppercase_ratio': len(re.findall(r'[A-Z]', text)) / len(text),
            'digit_ratio': len(re.findall(r'\d', text)) / len(text),
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / len(text),
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        }

    def extract_structural_features(self, text: str) -> Dict[str, float]:
        sentences = text.split('.')
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]),
            'paragraph_count': len(text.split('\n')),
            'title_length': len(text.split('\n')[0]) if '\n' in text else len(text)
        }

    def extract_pos_features(self, text: str) -> Dict[str, float]:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        total_tags = sum(pos_counts.values())
        return {f'pos_{tag}': count/total_tags for tag, count in pos_counts.items()}

    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        return self.tfidf_vectorizer.fit_transform(texts).toarray()

    def extract_count_features(self, texts: List[str]) -> np.ndarray:
        return self.count_vectorizer.fit_transform(texts).toarray()

    def extract_all_features(self, texts: List[str]) -> pd.DataFrame:
        linguistic_features = []
        structural_features = []
        pos_features = []
        
        for text in texts:
            linguistic_features.append(self.extract_linguistic_features(text))
            structural_features.append(self.extract_structural_features(text))
            pos_features.append(self.extract_pos_features(text))
        
        df_linguistic = pd.DataFrame(linguistic_features)
        df_structural = pd.DataFrame(structural_features)
        df_pos = pd.DataFrame(pos_features)
        
        tfidf_features = self.extract_tfidf_features(texts)
        count_features = self.extract_count_features(texts)
        
        df_tfidf = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        df_count = pd.DataFrame(count_features, columns=[f'count_{i}' for i in range(count_features.shape[1])])
        
        return pd.concat([df_linguistic, df_structural, df_pos, df_tfidf, df_count], axis=1) 