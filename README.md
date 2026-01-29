# Intelligent Document Classification System

## 📋 Overview
An end-to-end intelligent document classification system that automatically categorizes documents using machine learning and deep learning techniques. The system supports multiple document types (PDF, Word, images, scanned documents) and uses state-of-the-art NLP and computer vision models for accurate classification.

## 🚀 Features

### **Multi-Modal Document Processing**
- **Text Documents**: PDF, DOCX, TXT, HTML
- **Handwritten Text**: OCR with post-processing
- **Mixed Documents**: Multi-page PDFs with text and images

### **Advanced Classification Capabilities**
- **Multi-label Classification**: Documents can belong to multiple categories
- **Confidence Scoring**: Probability scores for each classification
- **Zero-shot Classification**: Classify unseen document types without retraining
- **Transfer Learning**: Fine-tune pre-trained models on custom datasets


## 🛠️ Installation

### **Option 1: Local Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/intelligent-document-classification.git
cd intelligent-document-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Install Tesseract OCR (Linux)
sudo apt-get install tesseract-ocr