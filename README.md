# Fake News Detection System

This project implements a robust fake news detection system using various machine learning models including BERT, TF-IDF with SVM, and LSTM.

## Features

- Multiple model architectures (BERT, TF-IDF + SVM, LSTM)
- Comprehensive text preprocessing
- Training and evaluation pipeline
- Performance metrics visualization
- Easy-to-use command line interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

The system supports three different models:

1. BERT (default)
2. TF-IDF with SVM
3. LSTM

To train and evaluate a model:

```bash
python src/main.py --data_path path/to/your/dataset.csv --model_type bert --epochs 5 --batch_size 16
```

### Dataset Format

The input dataset should be a CSV file with at least two columns:
- `text`: The news article text
- `label`: Binary label (0 for real news, 1 for fake news)

## Model Comparison

- BERT: State-of-the-art transformer-based model, best for accuracy but requires more computational resources
- TF-IDF + SVM: Traditional machine learning approach, faster training but may have lower accuracy
- LSTM: Deep learning approach with good balance between performance and resource requirements

## Output

The system will:
1. Train the selected model
2. Save the best model weights
3. Generate performance metrics plots
4. Print evaluation metrics (accuracy, precision, recall, F1 score)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 