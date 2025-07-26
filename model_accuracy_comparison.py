#!/usr/bin/env python3
"""
Model Accuracy Comparison Script for AI Stock Screener

This script implements the 4th feature from the Future Work section of BENCHMARK_REPORT.md:
"Model Accuracy Comparison: Verify that GPU and CPU models produce equivalent prediction results"

The script trains both GPU and CPU versions of RandomForest and XGBoost models with identical
data and compares their prediction outputs to ensure they produce equivalent results.
"""

import time
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_stock_screener.ai_screener import run_screening, train_model, fetch_data, initialize_feature_columns
from ai_stock_screener.gpu_utils import get_gpu_manager, print_gpu_status
from ai_stock_screener.output_formatter import console


class ModelAccuracyComparison:
    """Compare GPU vs CPU model accuracy and prediction equivalence."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'comparisons': {},
            'summary': {}
        }
        self.gpu_manager = get_gpu_manager()
        
    def print_system_info(self):
        """Print system configuration information."""
        console.print("\n🔧 [bold blue]System Configuration[/bold blue]")
        print_gpu_status()
        
        # Store system info
        self.results['system_info'] = {
            'cuda_available': self.gpu_manager.cuda_available,
            'cuml_available': self.gpu_manager.cuml_available,
            'torch_available': self.gpu_manager.torch_available,
            'gpu_count': self.gpu_manager.gpu_count,
            'gpu_memory': self.gpu_manager.gpu_memory
        }
        
    def prepare_test_data(self, tickers: List[str], config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Prepare consistent test data for both GPU and CPU models."""
        console.print(f"\n📊 [bold blue]Preparing test data for {len(tickers)} tickers[/bold blue]")
        
        # Initialize feature columns
        initialize_feature_columns(tickers, config)
        
        # First, fetch SPY data for relative strength calculations
        console.print("📈 Fetching SPY market data...")
        spy_data = fetch_data("SPY", config, is_market=True)
        if spy_data is None or spy_data.empty:
            console.print("⚠️ Warning: Could not fetch SPY data, relative strength features will be missing")
            spy_close = None
        else:
            spy_close = spy_data["Close"]
            console.print(f"✅ SPY data fetched: {len(spy_close)} samples")
        
        # Fetch and combine data for all tickers
        all_data = []
        for ticker in tickers:
            console.print(f"📈 Fetching data for {ticker}...")
            data = fetch_data(ticker, config, is_market=False, spy_close=spy_close)
            if data is not None and not data.empty:
                all_data.append(data)
            else:
                console.print(f"⚠️ No data available for {ticker}")
        
        if not all_data:
            raise ValueError("No valid data available for any ticker")
            
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        console.print(f"✅ Combined dataset: {len(combined_data)} samples")
        
        # Check label distribution
        label_counts = combined_data['Label'].value_counts()
        console.print(f"📊 Label distribution: {dict(label_counts)}")
        
        return combined_data, {'label_distribution': dict(label_counts)}
        
    def train_model_pair(self, data: pd.DataFrame, model_type: str, config: Dict) -> Tuple[Optional[object], Optional[object], Dict]:
        """Train both GPU and CPU versions of the same model type."""
        console.print(f"\n🤖 [bold blue]Training {model_type.upper()} model pair[/bold blue]")
        
        results = {
            'model_type': model_type,
            'gpu_model': None,
            'cpu_model': None,
            'gpu_training_time': None,
            'cpu_training_time': None,
            'gpu_success': False,
            'cpu_success': False,
            'gpu_error': None,
            'cpu_error': None
        }
        
        # Train GPU model
        console.print("🚀 Training GPU model...")
        gpu_config = config.copy()
        gpu_config.update({
            'model': model_type,
            'use_gpu': True,
            'news_analysis': False,  # Disable to allow cuML
            'grid_search': 0,
            'ensemble_runs': 1
        })
        
        try:
            start_time = time.time()
            gpu_model = train_model(data, gpu_config)
            gpu_training_time = time.time() - start_time
            
            if gpu_model is not None:
                results['gpu_model'] = gpu_model
                results['gpu_training_time'] = gpu_training_time
                results['gpu_success'] = True
                console.print(f"✅ GPU model trained successfully in {gpu_training_time:.2f}s")
            else:
                results['gpu_error'] = "Model training returned None"
                console.print("❌ GPU model training failed (returned None)")
                
        except Exception as e:
            results['gpu_error'] = str(e)
            console.print(f"❌ GPU model training failed: {e}")
        
        # Train CPU model
        console.print("💻 Training CPU model...")
        cpu_config = config.copy()
        cpu_config.update({
            'model': model_type,
            'use_gpu': False,
            'news_analysis': False,
            'grid_search': 0,
            'ensemble_runs': 1
        })
        
        try:
            start_time = time.time()
            cpu_model = train_model(data, cpu_config)
            cpu_training_time = time.time() - start_time
            
            if cpu_model is not None:
                results['cpu_model'] = cpu_model
                results['cpu_training_time'] = cpu_training_time
                results['cpu_success'] = True
                console.print(f"✅ CPU model trained successfully in {cpu_training_time:.2f}s")
            else:
                results['cpu_error'] = "Model training returned None"
                console.print("❌ CPU model training failed (returned None)")
                
        except Exception as e:
            results['cpu_error'] = str(e)
            console.print(f"❌ CPU model training failed: {e}")
        
        return results['gpu_model'], results['cpu_model'], results
    
    def compare_predictions(self, gpu_model, cpu_model, test_data: pd.DataFrame, model_type: str) -> Dict:
        """Compare predictions between GPU and CPU models."""
        console.print(f"\n🔍 [bold blue]Comparing {model_type.upper()} predictions[/bold blue]")
        
        from ai_stock_screener.ai_screener import FEATURE_COLUMNS, safe_cuml_predict_proba
        
        X_test = test_data[FEATURE_COLUMNS]
        y_test = test_data['Label']
        
        comparison_results = {
            'model_type': model_type,
            'test_samples': len(X_test),
            'gpu_predictions': None,
            'cpu_predictions': None,
            'gpu_probabilities': None,
            'cpu_probabilities': None,
            'prediction_correlation': None,
            'probability_correlation': None,
            'prediction_agreement': None,
            'probability_mse': None,
            'probability_mae': None,
            'gpu_accuracy': None,
            'cpu_accuracy': None,
            'gpu_precision': None,
            'cpu_precision': None,
            'gpu_recall': None,
            'cpu_recall': None,
            'gpu_f1': None,
            'cpu_f1': None,
            'equivalent_predictions': False,
            'errors': []
        }
        
        try:
            # Get GPU predictions
            console.print("🚀 Getting GPU predictions...")
            if model_type == 'random_forest':
                gpu_probabilities = safe_cuml_predict_proba(gpu_model, X_test)
            else:  # xgboost
                gpu_probabilities = gpu_model.predict_proba(X_test)[:, 1]
            
            gpu_predictions = (gpu_probabilities > 0.5).astype(int)
            comparison_results['gpu_probabilities'] = gpu_probabilities.tolist()
            comparison_results['gpu_predictions'] = gpu_predictions.tolist()
            
            # Calculate GPU metrics
            comparison_results['gpu_accuracy'] = accuracy_score(y_test, gpu_predictions)
            comparison_results['gpu_precision'] = precision_score(y_test, gpu_predictions, zero_division=0)
            comparison_results['gpu_recall'] = recall_score(y_test, gpu_predictions, zero_division=0)
            comparison_results['gpu_f1'] = f1_score(y_test, gpu_predictions, zero_division=0)
            
            console.print(f"✅ GPU metrics - Accuracy: {comparison_results['gpu_accuracy']:.4f}, "
                         f"Precision: {comparison_results['gpu_precision']:.4f}, "
                         f"Recall: {comparison_results['gpu_recall']:.4f}, "
                         f"F1: {comparison_results['gpu_f1']:.4f}")
            
        except Exception as e:
            error_msg = f"GPU prediction error: {e}"
            comparison_results['errors'].append(error_msg)
            console.print(f"❌ {error_msg}")
            return comparison_results
        
        try:
            # Get CPU predictions
            console.print("💻 Getting CPU predictions...")
            cpu_probabilities = cpu_model.predict_proba(X_test)[:, 1]
            cpu_predictions = (cpu_probabilities > 0.5).astype(int)
            comparison_results['cpu_probabilities'] = cpu_probabilities.tolist()
            comparison_results['cpu_predictions'] = cpu_predictions.tolist()
            
            # Calculate CPU metrics
            comparison_results['cpu_accuracy'] = accuracy_score(y_test, cpu_predictions)
            comparison_results['cpu_precision'] = precision_score(y_test, cpu_predictions, zero_division=0)
            comparison_results['cpu_recall'] = recall_score(y_test, cpu_predictions, zero_division=0)
            comparison_results['cpu_f1'] = f1_score(y_test, cpu_predictions, zero_division=0)
            
            console.print(f"✅ CPU metrics - Accuracy: {comparison_results['cpu_accuracy']:.4f}, "
                         f"Precision: {comparison_results['cpu_precision']:.4f}, "
                         f"Recall: {comparison_results['cpu_recall']:.4f}, "
                         f"F1: {comparison_results['cpu_f1']:.4f}")
            
        except Exception as e:
            error_msg = f"CPU prediction error: {e}"
            comparison_results['errors'].append(error_msg)
            console.print(f"❌ {error_msg}")
            return comparison_results
        
        try:
            # Compare predictions
            console.print("🔍 Analyzing prediction agreement...")
            
            # Prediction agreement (exact match)
            prediction_agreement = np.mean(gpu_predictions == cpu_predictions)
            comparison_results['prediction_agreement'] = prediction_agreement
            
            # Probability correlations
            prob_pearson_r, prob_pearson_p = pearsonr(gpu_probabilities, cpu_probabilities)
            prob_spearman_r, prob_spearman_p = spearmanr(gpu_probabilities, cpu_probabilities)
            
            comparison_results['probability_correlation'] = {
                'pearson_r': prob_pearson_r,
                'pearson_p': prob_pearson_p,
                'spearman_r': prob_spearman_r,
                'spearman_p': prob_spearman_p
            }
            
            # Prediction correlations (for binary predictions)
            if len(np.unique(gpu_predictions)) > 1 and len(np.unique(cpu_predictions)) > 1:
                pred_pearson_r, pred_pearson_p = pearsonr(gpu_predictions, cpu_predictions)
                comparison_results['prediction_correlation'] = {
                    'pearson_r': pred_pearson_r,
                    'pearson_p': pred_pearson_p
                }
            
            # Probability differences
            prob_mse = np.mean((gpu_probabilities - cpu_probabilities) ** 2)
            prob_mae = np.mean(np.abs(gpu_probabilities - cpu_probabilities))
            comparison_results['probability_mse'] = prob_mse
            comparison_results['probability_mae'] = prob_mae
            
            # Determine if predictions are equivalent (high agreement and correlation)
            equivalent_threshold = 0.95  # 95% agreement threshold
            correlation_threshold = 0.95  # 95% correlation threshold
            
            comparison_results['equivalent_predictions'] = (
                prediction_agreement >= equivalent_threshold and
                prob_pearson_r >= correlation_threshold
            )
            
            console.print(f"📊 Prediction agreement: {prediction_agreement:.4f} ({prediction_agreement*100:.1f}%)")
            console.print(f"📊 Probability correlation (Pearson): {prob_pearson_r:.4f}")
            console.print(f"📊 Probability correlation (Spearman): {prob_spearman_r:.4f}")
            console.print(f"📊 Probability MSE: {prob_mse:.6f}")
            console.print(f"📊 Probability MAE: {prob_mae:.6f}")
            
            if comparison_results['equivalent_predictions']:
                console.print("✅ [bold green]Models produce equivalent predictions![/bold green]")
            else:
                console.print("⚠️ [bold yellow]Models show significant prediction differences[/bold yellow]")
            
        except Exception as e:
            error_msg = f"Comparison analysis error: {e}"
            comparison_results['errors'].append(error_msg)
            console.print(f"❌ {error_msg}")
        
        return comparison_results
    
    def run_accuracy_comparison(self, tickers: List[str] = None, config: Dict = None) -> Dict:
        """Run complete accuracy comparison for both model types."""
        if tickers is None:
            tickers = ['AAPL', 'GOOGL', 'TSLA', 'NVDA', 'META', 'MSFT', 'AMZN', 'NFLX']
        
        if config is None:
            config = {
                'period': '1y',  # Use 1 year for better data availability
                'future_days': 30,
                'threshold': 0.07,
                'seed': 42,
                'n_estimators': 100,  # Smaller for faster training
                'integrate_market': True,  # Enable market integration for SPY data
                'use_sharpe_labeling': 1.0  # Use Sharpe ratio labeling
            }
        
        console.print(f"\n🎯 [bold blue]Starting Model Accuracy Comparison[/bold blue]")
        console.print(f"📊 Tickers: {', '.join(tickers)}")
        console.print(f"⚙️ Config: {config}")
        
        self.print_system_info()
        
        try:
            # Prepare test data
            test_data, data_info = self.prepare_test_data(tickers, config)
            self.results['data_info'] = data_info
            
            # Test both model types
            model_types = ['random_forest', 'xgboost']
            
            for model_type in model_types:
                console.print(f"\n{'='*60}")
                console.print(f"🤖 [bold blue]Testing {model_type.upper()} Model[/bold blue]")
                console.print(f"{'='*60}")
                
                # Train model pair
                gpu_model, cpu_model, training_results = self.train_model_pair(test_data, model_type, config)
                
                if gpu_model is not None and cpu_model is not None:
                    # Compare predictions
                    comparison_results = self.compare_predictions(gpu_model, cpu_model, test_data, model_type)
                    
                    # Combine training and comparison results
                    full_results = {**training_results, **comparison_results}
                    self.results['comparisons'][model_type] = full_results
                    
                    # Cleanup GPU models to prevent memory leaks
                    if hasattr(gpu_model, '__module__') and 'cuml' in str(gpu_model.__module__):
                        self.gpu_manager.cleanup_cuml_model(gpu_model)
                        
                else:
                    console.print(f"❌ Skipping {model_type} comparison due to training failures")
                    self.results['comparisons'][model_type] = training_results
            
            # Generate summary
            self.generate_summary()
            
        except Exception as e:
            console.print(f"❌ [bold red]Accuracy comparison failed: {e}[/bold red]")
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    def generate_summary(self):
        """Generate summary of accuracy comparison results."""
        console.print(f"\n📋 [bold blue]Accuracy Comparison Summary[/bold blue]")
        
        summary = {
            'total_models_tested': 0,
            'successful_comparisons': 0,
            'equivalent_models': 0,
            'model_results': {}
        }
        
        for model_type, results in self.results['comparisons'].items():
            model_summary = {
                'gpu_training_success': results.get('gpu_success', False),
                'cpu_training_success': results.get('cpu_success', False),
                'comparison_completed': 'equivalent_predictions' in results,
                'predictions_equivalent': results.get('equivalent_predictions', False),
                'prediction_agreement': results.get('prediction_agreement', 0),
                'probability_correlation': results.get('probability_correlation', {}).get('pearson_r', 0) if results.get('probability_correlation') else 0
            }
            
            summary['total_models_tested'] += 1
            
            if model_summary['comparison_completed']:
                summary['successful_comparisons'] += 1
                if model_summary['predictions_equivalent']:
                    summary['equivalent_models'] += 1
            
            summary['model_results'][model_type] = model_summary
            
            # Print model-specific summary
            console.print(f"\n🤖 {model_type.upper()}:")
            console.print(f"  ✅ GPU Training: {'Success' if model_summary['gpu_training_success'] else 'Failed'}")
            console.print(f"  ✅ CPU Training: {'Success' if model_summary['cpu_training_success'] else 'Failed'}")
            
            if model_summary['comparison_completed']:
                console.print(f"  📊 Prediction Agreement: {model_summary['prediction_agreement']:.4f} ({model_summary['prediction_agreement']*100:.1f}%)")
                console.print(f"  📊 Probability Correlation: {model_summary['probability_correlation']:.4f}")
                console.print(f"  🎯 Equivalent Predictions: {'✅ Yes' if model_summary['predictions_equivalent'] else '❌ No'}")
            else:
                console.print(f"  ❌ Comparison not completed")
        
        # Overall summary
        console.print(f"\n📊 [bold blue]Overall Results[/bold blue]")
        console.print(f"  🤖 Models Tested: {summary['total_models_tested']}")
        console.print(f"  ✅ Successful Comparisons: {summary['successful_comparisons']}")
        console.print(f"  🎯 Equivalent Models: {summary['equivalent_models']}")
        
        if summary['successful_comparisons'] > 0:
            equivalence_rate = summary['equivalent_models'] / summary['successful_comparisons']
            console.print(f"  📈 Equivalence Rate: {equivalence_rate:.2f} ({equivalence_rate*100:.0f}%)")
        
        self.results['summary'] = summary
    
    def save_results(self, filename: str = None):
        """Save comparison results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_reports/accuracy_comparison_results_{timestamp}.json"
        
        # Ensure the generated_reports directory exists
        Path("generated_reports").mkdir(exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Remove non-serializable model objects before saving
        results_copy = self.results.copy()
        for model_type in results_copy.get('comparisons', {}):
            if 'gpu_model' in results_copy['comparisons'][model_type]:
                del results_copy['comparisons'][model_type]['gpu_model']
            if 'cpu_model' in results_copy['comparisons'][model_type]:
                del results_copy['comparisons'][model_type]['cpu_model']
        
        results_to_save = convert_numpy_types(results_copy)
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        console.print(f"💾 Results saved to: {filename}")
        return filename


def main():
    """Main function to run the accuracy comparison."""
    console.print("🎯 [bold blue]AI Stock Screener - Model Accuracy Comparison[/bold blue]")
    console.print("Implementing Feature #4 from BENCHMARK_REPORT.md Future Work")
    
    # Configuration for testing
    test_config = {
        'period': '1y',  # Use 1 year for better data availability
        'future_days': 30,
        'threshold': 0.07,
        'seed': 42,
        'n_estimators': 100,  # Smaller for faster training
        'integrate_market': True,  # Enable market integration for SPY data
        'use_sharpe_labeling': 1.0  # Use Sharpe ratio labeling
    }
    
    # Test with a smaller set of tickers for faster execution
    test_tickers = ['AAPL', 'GOOGL', 'TSLA', 'NVDA', 'META', 'MSFT']
    
    try:
        # Run accuracy comparison
        comparator = ModelAccuracyComparison()
        results = comparator.run_accuracy_comparison(test_tickers, test_config)
        
        # Save results
        results_file = comparator.save_results()
        
        console.print(f"\n🎉 [bold green]Accuracy comparison completed successfully![/bold green]")
        console.print(f"📄 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        console.print(f"❌ [bold red]Accuracy comparison failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()