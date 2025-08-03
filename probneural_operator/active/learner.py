"""Active learning implementation for neural operators."""

from typing import List, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .acquisition import AcquisitionFunction, AcquisitionFunctions


class ActiveLearner:
    """Active learning framework for neural operators.
    
    This class implements the active learning loop:
    1. Train model on current labeled data
    2. Score unlabeled data using acquisition function
    3. Select most informative points
    4. Query oracle for labels
    5. Update training set and repeat
    """
    
    def __init__(self,
                 model: nn.Module,
                 acquisition: Union[str, AcquisitionFunction] = "bald",
                 budget: int = 1000,
                 batch_size: int = 10,
                 initial_size: int = 50,
                 oracle_fn: Optional[Callable] = None):
        """Initialize active learner.
        
        Args:
            model: Probabilistic neural operator model
            acquisition: Acquisition function or name
            budget: Total annotation budget  
            batch_size: Number of points to query per iteration
            initial_size: Size of initial labeled set
            oracle_fn: Function to query for labels (x -> y)
        """
        self.model = model
        self.budget = budget
        self.batch_size = batch_size
        self.initial_size = initial_size
        self.oracle_fn = oracle_fn
        
        # Set up acquisition function
        if isinstance(acquisition, str):
            self.acquisition_fn = self._get_acquisition_function(acquisition)
        else:
            self.acquisition_fn = acquisition
        
        # State tracking
        self.labeled_data = {"inputs": [], "targets": []}
        self.unlabeled_pool = None
        self.iteration = 0
        self.annotation_count = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "test_error": [],
            "selected_indices": []
        }
    
    def _get_acquisition_function(self, name: str) -> AcquisitionFunction:
        """Get acquisition function by name."""
        if name == "bald":
            return AcquisitionFunctions.bald()
        elif name == "variance" or name == "max_variance":
            return AcquisitionFunctions.max_variance()
        elif name == "entropy" or name == "max_entropy":
            return AcquisitionFunctions.max_entropy()
        elif name == "random":
            return AcquisitionFunctions.random()
        else:
            raise ValueError(f"Unknown acquisition function: {name}")
    
    def initialize(self, 
                   pool_data: Tuple[torch.Tensor, torch.Tensor],
                   test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> None:
        """Initialize the active learning process.
        
        Args:
            pool_data: Tuple of (inputs, targets) for the unlabeled pool
            test_data: Optional test data for evaluation
        """
        pool_inputs, pool_targets = pool_data
        
        # Sample initial labeled set
        n_pool = pool_inputs.shape[0]
        initial_indices = torch.randperm(n_pool)[:self.initial_size]
        
        # Initialize labeled data
        self.labeled_data["inputs"] = [pool_inputs[initial_indices]]
        self.labeled_data["targets"] = [pool_targets[initial_indices]]
        
        # Remaining unlabeled pool
        remaining_indices = torch.randperm(n_pool)[self.initial_size:]
        self.unlabeled_pool = {
            "inputs": pool_inputs[remaining_indices],
            "targets": pool_targets[remaining_indices],
            "indices": remaining_indices
        }
        
        self.test_data = test_data
        self.annotation_count = self.initial_size
        
        print(f"Initialized with {self.initial_size} labeled examples, "
              f"{len(remaining_indices)} in unlabeled pool")
    
    def query(self, 
              retrain: bool = True,
              fit_posterior: bool = True) -> Tuple[torch.Tensor, List[int]]:
        """Query the most informative points.
        
        Args:
            retrain: Whether to retrain the model
            fit_posterior: Whether to fit posterior approximation
            
        Returns:
            Tuple of (selected_points, selected_indices)
        """
        if self.unlabeled_pool is None:
            raise RuntimeError("Must call initialize() first")
        
        if self.annotation_count >= self.budget:
            print(f"Budget exhausted ({self.budget})")
            return torch.empty(0), []
        
        # Train model on current labeled data
        if retrain:
            train_loader = self._create_dataloader(
                torch.cat(self.labeled_data["inputs"], dim=0),
                torch.cat(self.labeled_data["targets"], dim=0)
            )
            
            print(f"Training model on {train_loader.dataset.tensors[0].shape[0]} samples...")
            history = self.model.fit(train_loader, epochs=50, lr=1e-3)
            self.history["train_loss"].extend(history["train_loss"])
            
            # Fit posterior for uncertainty quantification
            if fit_posterior and hasattr(self.model, 'fit_posterior'):
                print("Fitting posterior approximation...")
                self.model.fit_posterior(train_loader)
        
        # Score unlabeled points
        pool_inputs = self.unlabeled_pool["inputs"]
        with torch.no_grad():
            acquisition_scores = self.acquisition_fn(self.model, pool_inputs)
        
        # Select most informative points
        batch_size = min(self.batch_size, len(pool_inputs), self.budget - self.annotation_count)
        _, top_indices = torch.topk(acquisition_scores, batch_size)
        
        selected_points = pool_inputs[top_indices]
        original_indices = self.unlabeled_pool["indices"][top_indices]
        
        print(f"Iteration {self.iteration}: Selected {batch_size} points "
              f"(scores: {acquisition_scores[top_indices].mean():.4f})")
        
        self.history["selected_indices"].append(original_indices.tolist())
        
        return selected_points, top_indices.tolist()
    
    def update(self, 
               query_points: torch.Tensor,
               query_labels: torch.Tensor,
               query_indices: List[int]) -> None:
        """Update the training set with new labeled data.
        
        Args:
            query_points: Points that were queried
            query_labels: Labels obtained from oracle
            query_indices: Indices of points in unlabeled pool
        """
        # Add to labeled data
        self.labeled_data["inputs"].append(query_points)
        self.labeled_data["targets"].append(query_labels)
        
        # Remove from unlabeled pool
        mask = torch.ones(len(self.unlabeled_pool["inputs"]), dtype=torch.bool)
        mask[query_indices] = False
        
        self.unlabeled_pool["inputs"] = self.unlabeled_pool["inputs"][mask]
        self.unlabeled_pool["targets"] = self.unlabeled_pool["targets"][mask]
        self.unlabeled_pool["indices"] = self.unlabeled_pool["indices"][mask]
        
        self.annotation_count += len(query_points)
        self.iteration += 1
        
        print(f"Updated: {self.annotation_count}/{self.budget} annotations used, "
              f"{len(self.unlabeled_pool['inputs'])} points remaining in pool")
    
    def active_learning_loop(self,
                           pool_data: Tuple[torch.Tensor, torch.Tensor],
                           test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                           max_iterations: int = 20) -> dict:
        """Run the complete active learning loop.
        
        Args:
            pool_data: Unlabeled pool data
            test_data: Test data for evaluation
            max_iterations: Maximum number of AL iterations
            
        Returns:
            Training history dictionary
        """
        # Initialize
        self.initialize(pool_data, test_data)
        
        for iteration in range(max_iterations):
            if self.annotation_count >= self.budget:
                break
            
            # Query most informative points
            query_points, query_indices = self.query()
            
            if len(query_points) == 0:
                break
            
            # Get labels from oracle
            if self.oracle_fn is not None:
                query_labels = self.oracle_fn(query_points)
            else:
                # Use ground truth from pool
                query_labels = self.unlabeled_pool["targets"][query_indices]
            
            # Update training set
            self.update(query_points, query_labels, query_indices)
            
            # Evaluate on test set
            if test_data is not None:
                test_error = self._evaluate_test_error(test_data)
                self.history["test_error"].append(test_error)
                print(f"Test error: {test_error:.6f}")
            
            print("-" * 50)
        
        print(f"Active learning completed after {self.iteration} iterations")
        return self.history
    
    def _create_dataloader(self, 
                          inputs: torch.Tensor,
                          targets: torch.Tensor,
                          batch_size: int = 32) -> DataLoader:
        """Create DataLoader from tensors."""
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _evaluate_test_error(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Evaluate test error."""
        test_inputs, test_targets = test_data
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(test_inputs)
            error = torch.nn.functional.mse_loss(predictions, test_targets).item()
        
        return error
    
    @property
    def test_error(self) -> float:
        """Get latest test error."""
        if self.history["test_error"]:
            return self.history["test_error"][-1]
        return float('inf')
    
    def get_labeled_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current labeled dataset.
        
        Returns:
            Tuple of (inputs, targets)
        """
        inputs = torch.cat(self.labeled_data["inputs"], dim=0)
        targets = torch.cat(self.labeled_data["targets"], dim=0)
        return inputs, targets
    
    def reset(self) -> None:
        """Reset the active learner state."""
        self.labeled_data = {"inputs": [], "targets": []}
        self.unlabeled_pool = None
        self.iteration = 0
        self.annotation_count = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "test_error": [],
            "selected_indices": []
        }