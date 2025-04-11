import optuna
from optuna.trial import Trial
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
from pathlib import Path
import json

class HyperparameterTuner:
    def __init__(self, model_class: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 n_trials: int = 100, timeout: Optional[int] = None,
                 study_name: str = "fake_news_study",
                 storage: str = "sqlite:///optuna.db") -> None:
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize"
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{study_name}.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def objective(self, trial: Trial) -> float:
        # Define hyperparameters to optimize
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 128, 512, step=64),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'num_heads': trial.suggest_int('num_heads', 4, 16, step=2),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'num_filters': trial.suggest_int('num_filters', 50, 200, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'linear', 'none'])
        }
        
        # Initialize model with suggested parameters
        model = self.model_class(
            hidden_dim=params['hidden_dim'],
            dropout=params['dropout'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            num_filters=params['num_filters']
        ).to(self.device)
        
        # Setup optimizer
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        
        # Setup scheduler
        if params['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(self.train_loader)
            )
        elif params['scheduler'] == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=len(self.train_loader)
            )
        else:
            scheduler = None
        
        # Training loop
        best_val_f1 = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):  # Maximum 50 epochs
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch in self.train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                features = batch.get('features', None)
                if features is not None:
                    features = features.to(self.device)
                
                outputs = model(input_ids, attention_mask, features)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                train_loss += loss.item()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    features = batch.get('features', None)
                    if features is not None:
                        features = features.to(self.device)
                    
                    outputs = model(input_ids, attention_mask, features)
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Log progress
            self.logger.info(
                f"Trial {trial.number} - Epoch {epoch + 1}: "
                f"Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}"
            )
        
        return best_val_f1

    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Save results
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'all_trials': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                }
                for trial in self.study.trials
            ]
        }
        
        # Save to file
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / f"{self.study_name}_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best F1 score: {best_value:.4f}")
        
        return best_params

    def visualize_results(self) -> None:
        """Visualize optimization results."""
        try:
            import plotly.express as px
            import pandas as pd
            
            # Create DataFrame from trials
            trials_df = pd.DataFrame([
                {
                    'trial': trial.number,
                    'value': trial.value,
                    **trial.params
                }
                for trial in self.study.trials
            ])
            
            # Plot parallel coordinates
            fig = px.parallel_coordinates(
                trials_df,
                color='value',
                dimensions=list(self.study.best_params.keys()) + ['value'],
                title='Hyperparameter Optimization Results'
            )
            
            # Save plot
            output_dir = Path("optimization_results")
            fig.write_html(output_dir / f"{self.study_name}_parallel_coordinates.html")
            
            # Plot optimization history
            fig = px.line(
                trials_df,
                x='trial',
                y='value',
                title='Optimization History'
            )
            fig.write_html(output_dir / f"{self.study_name}_history.html")
            
        except ImportError:
            self.logger.warning(
                "Plotly not installed. Install with 'pip install plotly' to visualize results."
            ) 