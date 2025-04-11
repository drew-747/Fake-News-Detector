import torch
from data_processor import DataProcessor
from models import BERTClassifier, TFIDFModel, LSTMModel
from train import Trainer
import argparse
import os
from typing import Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_type', type=str, default='bert', 
                       choices=['bert', 'tfidf', 'lstm'], help='Type of model to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()

    data_processor = DataProcessor()
    df = data_processor.load_data(args.data_path)
    X_train, X_test, y_train, y_test = data_processor.prepare_data(df)

    if args.model_type == 'bert':
        train_encodings = data_processor.tokenize_text(X_train.tolist())
        test_encodings = data_processor.tokenize_text(X_test.tolist())

        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            test_encodings['input_ids'],
            test_encodings['attention_mask'],
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size
        )

        model = BERTClassifier()
        trainer = Trainer(model)
        history = trainer.train(train_loader, test_loader, num_epochs=args.epochs)
        trainer.plot_metrics(history)

    elif args.model_type == 'tfidf':
        model = TFIDFModel(model_type='svm')
        model.fit(X_train.tolist(), y_train.tolist())
        predictions = model.predict(X_test.tolist())
        print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        print(f"Precision: {precision_score(y_test, predictions)}")
        print(f"Recall: {recall_score(y_test, predictions)}")
        print(f"F1 Score: {f1_score(y_test, predictions)}")

    elif args.model_type == 'lstm':
        vocab_size = 10000
        embedding_dim = 100
        hidden_dim = 256
        output_dim = 2
        n_layers = 2
        dropout = 0.5

        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
        trainer = Trainer(model)
        history = trainer.train(train_loader, test_loader, num_epochs=args.epochs)
        trainer.plot_metrics(history)

if __name__ == '__main__':
    main() 