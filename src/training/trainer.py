# src/training/trainer.py
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import mlflow

class DocumentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return total_loss / len(dataloader), accuracy, report
    
    def train(self, train_loader, val_loader, epochs=10, lr=2e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        with mlflow.start_run():
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", epochs)
            
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader, optimizer, criterion)
                val_loss, val_accuracy, val_report = self.evaluate(val_loader, criterion)
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_accuracy:.4f}")
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                
                # Save best model
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    mlflow.log_artifact('best_model.pt')