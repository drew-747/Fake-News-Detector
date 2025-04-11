import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import logging
from typing import Dict, Any

from data_processor import FakeNewsDataset
from models.advanced_model import AdvancedModel
from hyperparameter_tuner import HyperparameterTuner

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization.log'),
            logging.StreamHandler()
        ]
    )

def load_data(data_dir: str, batch_size: int) -> Dict[str, DataLoader]:
    dataset = FakeNewsDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    data_loaders = load_data(args.data_dir, args.batch_size)
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        model_class=AdvancedModel,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        device=device,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage
    )
    
    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    best_params = tuner.optimize()
    
    # Visualize results
    logger.info("Visualizing results...")
    tuner.visualize_results()
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    model = AdvancedModel(
        hidden_dim=best_params['hidden_dim'],
        dropout=best_params['dropout'],
        num_heads=best_params['num_heads'],
        num_layers=best_params['num_layers'],
        num_filters=best_params['num_filters']
    ).to(device)
    
    # Save best model
    model_path = Path("models") / f"{args.study_name}_best_model.pt"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved best model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for fake news detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout for optimization in seconds")
    parser.add_argument("--study_name", type=str, default="fake_news_study", help="Name of the study")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", help="Storage URL for Optuna")
    
    args = parser.parse_args()
    main(args) 