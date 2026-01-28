# src/features/feature_extractor.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Union

class FeatureExtractor:
    def __init__(self, method='tfidf', model_name='bert-base-uncased'):
        self.method = method
        self.model_name = model_name
        self.vectorizer = None
        self.tokenizer = None
        self.model = None
        
    def fit_transform(self, documents: List[str]):
        """Extract features based on selected method"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=5
            )
            features = self.vectorizer.fit_transform(documents)
            return features
        
        elif self.method == 'word2vec':
            tokenized_docs = [doc.split() for doc in documents]
            model = Word2Vec(
                sentences=tokenized_docs,
                vector_size=300,
                window=5,
                min_count=2,
                workers=4
            )
            # Document embedding as average of word vectors
            features = []
            for doc in tokenized_docs:
                vectors = [model.wv[word] for word in doc if word in model.wv]
                if vectors:
                    features.append(np.mean(vectors, axis=0))
                else:
                    features.append(np.zeros(300))
            return np.array(features)
        
        elif self.method == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            features = []
            for doc in documents:
                inputs = self.tokenizer(
                    doc,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Use [CLS] token representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(cls_embedding.flatten())
            
            return np.array(features)