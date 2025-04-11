import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from typing import Tuple, List, Any

class DataProcessor:
    def __init__(self) -> None:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text

    def preprocess_text(self, text: str) -> str:
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def prepare_data(self, df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        X = df['processed_text'].values
        y = df[label_column].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def tokenize_text(self, texts: List[str], max_length: int = 512) -> Any:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt') 