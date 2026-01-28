# src/models/document_classifier.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import optuna

class HybridDocumentClassifier:
    def __init__(self, model_type='transformer'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        
    def build_model(self, input_dim=None, num_classes=None):
        """Build appropriate model based on type"""
        if self.model_type == 'transformer':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=num_classes
            )
            
        elif self.model_type == 'cnn':
            self.model = CNNTextClassifier(
                vocab_size=input_dim,
                embed_dim=300,
                num_classes=num_classes
            )
            
        elif self.model_type == 'lstm':
            self.model = LSTMTextClassifier(
                vocab_size=input_dim,
                embed_dim=300,
                hidden_dim=256,
                num_classes=num_classes
            )
            
        elif self.model_type == 'ensemble':
            self.model = EnsembleClassifier(
                classifiers=[
                    RandomForestClassifier(n_estimators=100),
                    XGBClassifier(n_estimators=100),
                    SVC(probability=True)
                ]
            )
            
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional hyperparameter optimization"""
        if self.model_type in ['transformer', 'cnn', 'lstm']:
            self._train_deep_learning(X_train, y_train, X_val, y_val)
        else:
            self.model.fit(X_train, y_train)
            
    def predict(self, X):
        """Make predictions on new documents"""
        if self.model_type in ['transformer', 'cnn', 'lstm']:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X)
            return predictions
        else:
            return self.model.predict(X)

class CNNTextClassifier(nn.Module):
    """CNN-based text classifier"""
    def __init__(self, vocab_size, embed_dim=300, num_classes=10):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        
        conv1 = torch.relu(self.conv1(embedded))
        conv2 = torch.relu(self.conv2(embedded))
        conv3 = torch.relu(self.conv3(embedded))
        
        pooled1 = nn.functional.max_pool1d(conv1, conv1.shape[2]).squeeze(2)
        pooled2 = nn.functional.max_pool1d(conv2, conv2.shape[2]).squeeze(2)
        pooled3 = nn.functional.max_pool1d(conv3, conv3.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled1, pooled2, pooled3), dim=1))
        output = self.fc(cat)
        return output