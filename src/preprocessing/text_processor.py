# src/preprocessing/text_processor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from typing import List, Dict
import pandas as pd

class DocumentPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize document text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct]
        return tokens
    
    def preprocess_document(self, document_path: str) -> Dict:
        """Complete preprocessing pipeline for a document"""
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        processed_text = ' '.join(tokens)
        
        return {
            'original': text,
            'processed': processed_text,
            'tokens': tokens,
            'length': len(tokens)
        }