import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer,
                 max_length: int = 512, use_features: bool = True) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_features = use_features
        self.stop_words = {}
        self._init_stopwords()
        DetectorFactory.seed = 0

    def _init_stopwords(self) -> None:
        languages = ['english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'russian']
        for lang in languages:
            try:
                self.stop_words[lang] = set(stopwords.words(lang))
            except LookupError:
                nltk.download('stopwords')
                self.stop_words[lang] = set(stopwords.words(lang))

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            lang_map = {
                'en': 'english',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'ru': 'russian'
            }
            return lang_map.get(lang, 'english')
        except (LangDetectException, Exception):
            return 'english'

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_features(self, text: str) -> Dict[str, float]:
        text = self.clean_text(text)
        lang = self.detect_language(text)
        blob = TextBlob(text)
        
        try:
            tokens = word_tokenize(text.lower())
            words = [word for word in tokens if word.isalnum()]
            stop_words = self.stop_words.get(lang, self.stop_words['english'])
            
            return {
                'sentiment': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'word_count': len(words),
                'unique_words': len(set(words)),
                'stopword_ratio': len([w for w in words if w in stop_words]) / len(words) if words else 0,
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'language': list(self.stop_words.keys()).index(lang) if lang in self.stop_words else 0
            }
        except Exception:
            return {
                'sentiment': 0.0,
                'subjectivity': 0.0,
                'word_count': 0,
                'unique_words': 0,
                'stopword_ratio': 0.0,
                'avg_word_length': 0.0,
                'language': 0
            }

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        if self.use_features:
            features = self.extract_features(text)
            item['features'] = torch.tensor(list(features.values()), dtype=torch.float)

        return item

class NewsDataModule:
    def __init__(self, data_path: str, tokenizer_name: str = 'bert-base-multilingual-cased',
                 batch_size: int = 32, max_length: int = 512, val_split: float = 0.1,
                 test_split: float = 0.1, random_state: int = 42, use_features: bool = True) -> None:
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.use_features = use_features

    def prepare_data(self) -> None:
        df = pd.read_csv(self.data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=self.val_split + self.test_split,
            random_state=self.random_state, stratify=labels
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=self.test_split / (self.val_split + self.test_split),
            random_state=self.random_state, stratify=temp_labels
        )

        self.train_dataset = NewsDataset(
            train_texts, train_labels, self.tokenizer, self.max_length, self.use_features
        )
        self.val_dataset = NewsDataset(
            val_texts, val_labels, self.tokenizer, self.max_length, self.use_features
        )
        self.test_dataset = NewsDataset(
            test_texts, test_labels, self.tokenizer, self.max_length, self.use_features
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        ) 