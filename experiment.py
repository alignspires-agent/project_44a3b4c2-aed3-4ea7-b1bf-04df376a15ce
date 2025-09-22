
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LPConformalPrediction:
    """
    Implementation of Conformal Prediction under Lévy–Prokhorov Distribution Shifts
    based on the paper methodology.
    """
    
    def __init__(self, epsilon: float = 0.1, rho: float = 0.05, alpha: float = 0.1):
        """
        Initialize the LP conformal prediction model.
        
        Parameters:
        epsilon (float): Local perturbation parameter (ε)
        rho (float): Global perturbation parameter (ρ)
        alpha (float): Significance level (1 - coverage)
        """
        self.epsilon = epsilon
        self.rho = rho
        self.alpha = alpha
        self.scores = None
        self.quantile = None
        
    def compute_scores(self, X_calib: np.ndarray, y_calib: np.ndarray) -> np.ndarray:
        """
        Compute conformity scores for calibration data.
        Using absolute error as a simple scoring function.
        
        Parameters:
        X_calib (np.ndarray): Calibration features
        y_calib (np.ndarray): Calibration targets
        
        Returns:
        np.ndarray: Conformity scores
        """
        try:
            # Simple mean prediction as placeholder - replace with actual model
            mean_pred = np.mean(y_calib)
            scores = np.abs(y_calib - mean_pred)
            return scores
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            sys.exit(1)
    
    def compute_worst_case_quantile(self, scores: np.ndarray) -> float:
        """
        Compute worst-case quantile under LP distribution shifts.
        
        Parameters:
        scores (np.ndarray): Conformity scores
        
        Returns:
        float: Worst-case quantile value
        """
        try:
            n = len(scores)
            sorted_scores = np.sort(scores)
            
            # Compute empirical quantile
            empirical_quantile_idx = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
            empirical_quantile = sorted_scores[empirical_quantile_idx]
            
            # Apply LP robustness adjustments
            # Local perturbation: ε affects the quantile position
            local_shift = int(np.floor(self.epsilon * n))
            local_quantile_idx = min(empirical_quantile_idx + local_shift, n - 1)
            local_quantile = sorted_scores[local_quantile_idx]
            
            # Global perturbation: ρ affects the quantile value
            global_adjustment = self.rho * (np.max(scores) - np.min(scores))
            worst_case_quantile = local_quantile + global_adjustment
            
            return worst_case_quantile
        except Exception as e:
            logger.error(f"Error computing worst-case quantile: {e}")
            sys.exit(1)
    
    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray):
        """
        Fit the conformal prediction model on calibration data.
        
        Parameters:
        X_calib (np.ndarray): Calibration features
        y_calib (np.ndarray): Calibration targets
        """
        try:
            logger.info("Fitting LP conformal prediction model...")
            self.scores = self.compute_scores(X_calib, y_calib)
            self.quantile = self.compute_worst_case_quantile(self.scores)
            logger.info(f"Model fitted. Worst-case quantile: {self.quantile:.4f}")
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            sys.exit(1)
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for test data.
        
        Parameters:
        X_test (np.ndarray): Test features
        
        Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of prediction intervals
        """
        try:
            # Simple mean prediction as placeholder
            mean_pred = np.mean([np.mean(row) for row in X_test]) if X_test.ndim > 1 else np.mean(X_test)
            
            lower_bounds = mean_pred - self.quantile
            upper_bounds = mean_pred + self.quantile
            
            return lower_bounds, upper_bounds
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            sys.exit(1)
    
    def evaluate_coverage(self, y_true: np.ndarray, lower_bounds: np.ndarray, 
                         upper_bounds: np.ndarray) -> float:
        """
        Evaluate coverage of prediction intervals.
        
        Parameters:
        y_true (np.ndarray): True target values
        lower_bounds (np.ndarray): Lower bounds of intervals
        upper_bounds (np.ndarray): Upper bounds of intervals
        
        Returns:
        float: Coverage percentage
        """
        try:
            covered = np.sum((y_true >= lower_bounds) & (y_true <= upper_bounds))
            coverage = covered / len(y_true)
            return coverage
        except Exception as e:
            logger.error(f"Error evaluating coverage: {e}")
            sys.exit(1)

def generate_synthetic_data(n_samples: int = 1000, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for experimentation.
    
    Parameters:
    n_samples (int): Number of samples to generate
    noise_level (float): Level of noise to add
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Features and targets
    """
    try:
        np.random.seed(42)
        X = np.random.randn(n_samples, 5)
        true_weights = np.array([1.0, -0.5, 2.0, -1.0, 0.3])
        y = X @ true_weights + noise_level * np.random.randn(n_samples)
        return X, y
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        sys.exit(1)

def simulate_distribution_shift(X: np.ndarray, y: np.ndarray, shift_strength: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a distribution shift for testing robustness.
    
    Parameters:
    X (np.ndarray): Original features
    y (np.ndarray): Original targets
    shift_strength (float): Strength of the shift
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: Shifted features and targets
    """
    try:
        X_shifted = X + shift_strength * np.random.randn(*X.shape)
        y_shifted = y + shift_strength * np.random.randn(len(y))
        return X_shifted, y_shifted
    except Exception as e:
        logger.error(f"Error simulating distribution shift: {e}")
        sys.exit(1)

def main():
    """Main experiment function."""
    logger.info("Starting LP Conformal Prediction Experiment")
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000)
    
    # Split data into calibration and test sets
    split_idx = int(0.7 * len(X))
    X_calib, y_calib = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Initialize and fit the model
    model = LPConformalPrediction(epsilon=0.1, rho=0.05, alpha=0.1)
    model.fit(X_calib, y_calib)
    
    # Make predictions on test data
    logger.info("Making predictions on test data...")
    lower_bounds, upper_bounds = model.predict(X_test)
    
    # Evaluate coverage on original test data
    coverage_original = model.evaluate_coverage(y_test, lower_bounds, upper_bounds)
    logger.info(f"Coverage on original test data: {coverage_original:.3f}")
    
    # Simulate distribution shift and test robustness
    logger.info("Simulating distribution shift...")
    X_test_shifted, y_test_shifted = simulate_distribution_shift(X_test, y_test, shift_strength=0.3)
    
    # Evaluate coverage on shifted data
    coverage_shifted = model.evaluate_coverage(y_test_shifted, lower_bounds, upper_bounds)
    logger.info(f"Coverage on shifted test data: {coverage_shifted:.3f}")
    
    # Compare with non-robust conformal prediction
    logger.info("Comparing with non-robust conformal prediction...")
    model_non_robust = LPConformalPrediction(epsilon=0.0, rho=0.0, alpha=0.1)
    model_non_robust.fit(X_calib, y_calib)
    lower_non_robust, upper_non_robust = model_non_robust.predict(X_test_shifted)
    coverage_non_robust = model_non_robust.evaluate_coverage(y_test_shifted, lower_non_robust, upper_non_robust)
    logger.info(f"Non-robust coverage on shifted data: {coverage_non_robust:.3f}")
    
    # Print final results
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"LP Robust Model Coverage (original): {coverage_original:.4f}")
    print(f"LP Robust Model Coverage (shifted): {coverage_shifted:.4f}")
    print(f"Non-Robust Model Coverage (shifted): {coverage_non_robust:.4f}")
    print(f"Coverage Drop (Robust): {abs(1 - model.alpha - coverage_shifted):.4f}")
    print(f"Coverage Drop (Non-Robust): {abs(1 - model.alpha - coverage_non_robust):.4f}")
    print(f"Robustness Improvement: {abs(coverage_shifted - coverage_non_robust):.4f}")
    print("="*50)
    
    # Validate theoretical properties
    logger.info("Validating theoretical properties...")
    
    # Check if coverage is maintained under shift
    coverage_maintained = coverage_shifted >= (1 - model.alpha - 0.05)  # Allow small tolerance
    robustness_demonstrated = coverage_shifted > coverage_non_robust
    
    print(f"Coverage maintained under shift: {coverage_maintained}")
    print(f"Robustness demonstrated: {robustness_demonstrated}")
    print(f"Worst-case quantile: {model.quantile:.4f}")
    
    if coverage_maintained and robustness_demonstrated:
        logger.info("Validation successful: Model demonstrates robustness to distribution shifts")
    else:
        logger.warning("Validation results show room for improvement")
    
    return coverage_shifted, coverage_non_robust

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        sys.exit(1)
