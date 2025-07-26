#!/usr/bin/env python3
"""
Extended GPU vs CPU Benchmark Script for AI Stock Screener

This script implements the "Extended Benchmarking" feature mentioned in BENCHMARK_REPORT.md
to test with larger datasets and more tickers to better evaluate GPU scaling benefits.

Features:
- Tests with different dataset sizes (8, 20, 50, 100 tickers)
- Analyzes GPU scaling performance
- Compares RandomForest and XGBoost models across different scales
- Generates comprehensive scaling analysis reports

Usage:
    python extended_benchmark.py
"""

import time
import sys
import json
from datetime import datetime
from pathlib import Path
import statistics
from typing import Dict, List, Tuple

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_stock_screener.ai_screener import run_screening
from ai_stock_screener.gpu_utils import get_gpu_manager, print_gpu_status
from ai_stock_screener.output_formatter import console

class ExtendedBenchmarkRunner:
    """Handles extended benchmarking with different dataset sizes for GPU scaling analysis."""
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        
        # Comprehensive ticker lists for different dataset sizes
        self.ticker_sets = {
            8: ["AAPL", "GOOGL", "TSLA", "PLTR", "AMZN", "NVDA", "META", "MSFT"],
            20: ["AAPL", "GOOGL", "TSLA", "PLTR", "AMZN", "NVDA", "META", "MSFT", 
                 "NFLX", "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "ABNB",
                 "COIN", "RBLX", "SNOW", "ZM"],
            50: ["AAPL", "GOOGL", "TSLA", "PLTR", "AMZN", "NVDA", "META", "MSFT",
                 "NFLX", "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "ABNB",
                 "COIN", "RBLX", "SNOW", "ZM", "SHOP", "SQ", "ROKU", "TWLO",
                 "OKTA", "DDOG", "NET", "CRWD", "ZS", "ESTC", "MDB", "TEAM",
                 "WDAY", "NOW", "SPLK", "VEEV", "DOCU", "ZEN", "BILL", "SMAR",
                 "GTLB", "FROG", "AI", "SMCI", "ARM", "RIVN", "LCID", "NIO",
                 "XPEV", "LI"],
            100: None  # Will be populated with top 100 tech/growth stocks
        }
        
        # Populate 100-ticker set
        self.ticker_sets[100] = self.ticker_sets[50] + [
            "BABA", "JD", "PDD", "BIDU", "BILI", "IQ", "VIPS", "WB", "DIDI", "GRAB",
            "SE", "MELI", "GLOB", "CPNG", "COUPN", "GLBE", "VROOM", "CVNA", "KMX", "AN",
            "LAD", "ABG", "SAH", "GPI", "PAG", "SFM", "RUSHA", "RUSHB", "ORLY", "AZO",
            "AAP", "LKQX", "LKQ", "ADNT", "PRTS", "WRBY", "LENS", "VSH", "SITM", "FORM",
            "PTON", "NLS", "XPEL", "MODG", "GOOS", "CROX", "DECK", "BIRK", "ONON", "TPG"
        ]
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self.gpu_manager.get_gpu_info(),
            "extended_benchmark_config": {
                "dataset_sizes": [8, 20, 50, 100],
                "models": ["random_forest", "xgboost"],
                "runs_per_configuration": 2,  # Reduced for extended testing
                "future_days": 30,
                "threshold": 0.07,
                "news_analysis": False  # Disabled for faster benchmarking
            },
            "scaling_results": {},
            "analysis": {}
        }
    
    def print_system_info(self):
        """Print system and extended benchmark information."""
        console.print("\n" + "="*70)
        console.print("🔬 AI Stock Screener Extended GPU Scaling Benchmark")
        console.print("="*70)
        
        console.print("\n📊 Extended Benchmark Configuration:")
        config = self.results["extended_benchmark_config"]
        console.print(f"   • Dataset Sizes: {config['dataset_sizes']} tickers")
        console.print(f"   • Models: {', '.join(config['models'])}")
        console.print(f"   • Runs per Configuration: {config['runs_per_configuration']}")
        console.print(f"   • Future Days: {config['future_days']}")
        console.print(f"   • Threshold: {config['threshold']}")
        console.print(f"   • News Analysis: {config['news_analysis']}")
        
        console.print("\n🖥️ System Information:")
        print_gpu_status()
        
        if not self.gpu_manager.is_gpu_available():
            console.print("\n⚠️ WARNING: No GPU acceleration detected!")
            console.print("   Extended benchmark will compare CPU vs CPU (no meaningful scaling expected)")
            return False
        
        return True
    
    def run_single_scaling_benchmark(self, dataset_size: int, model: str, use_gpu: bool, run_number: int) -> Dict:
        """Run a single benchmark iteration for a specific dataset size and model."""
        mode_name = "GPU" if use_gpu else "CPU"
        console.print(f"\n🏃 Running {mode_name} {model} benchmark with {dataset_size} tickers (Run {run_number + 1})...")
        
        # Configuration for the screening
        config = {
            "period": "6mo",  # Reduced period for faster benchmarking
            "future_days": 30,
            "threshold": 0.07,
            "n_estimators": 200,  # Reduced for faster benchmarking
            "use_sharpe_labeling": 1.0,
            "model": model,
            "grid_search": 0,
            "ensemble_runs": 1,
            "integrate_market": True,
            "use_gpu": use_gpu
        }
        
        tickers = self.ticker_sets[dataset_size]
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Run the screening
            run_screening(tickers, config, mode="eval", news_analysis=False)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "dataset_size": dataset_size,
                "model": model,
                "run_number": run_number + 1,
                "execution_time": execution_time,
                "success": True,
                "error": None,
                "config": config.copy()
            }
            
            console.print(f"✅ {mode_name} {model} with {dataset_size} tickers completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "dataset_size": dataset_size,
                "model": model,
                "run_number": run_number + 1,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "config": config.copy()
            }
            
            console.print(f"❌ {mode_name} {model} with {dataset_size} tickers failed after {execution_time:.2f} seconds: {e}")
        
        return result
    
    def run_extended_benchmark_suite(self):
        """Run the complete extended benchmark suite across different dataset sizes."""
        if not self.print_system_info():
            console.print("\n⏭️ Proceeding with CPU-only scaling analysis...")
        
        dataset_sizes = self.results["extended_benchmark_config"]["dataset_sizes"]
        models = self.results["extended_benchmark_config"]["models"]
        runs_per_config = self.results["extended_benchmark_config"]["runs_per_configuration"]
        
        for dataset_size in dataset_sizes:
            console.print(f"\n{'='*50}")
            console.print(f"📈 Testing Dataset Size: {dataset_size} tickers")
            console.print(f"{'='*50}")
            
            for model in models:
                # Initialize results structure
                key = f"{dataset_size}_{model}"
                self.results["scaling_results"][key] = {
                    "dataset_size": dataset_size,
                    "model": model,
                    "gpu_results": [],
                    "cpu_results": []
                }
                
                # Run GPU benchmarks
                console.print(f"\n🚀 Starting GPU {model} benchmarks ({runs_per_config} runs)...")
                for i in range(runs_per_config):
                    result = self.run_single_scaling_benchmark(dataset_size, model, use_gpu=True, run_number=i)
                    self.results["scaling_results"][key]["gpu_results"].append(result)
                
                # Run CPU benchmarks
                console.print(f"\n💻 Starting CPU {model} benchmarks ({runs_per_config} runs)...")
                for i in range(runs_per_config):
                    result = self.run_single_scaling_benchmark(dataset_size, model, use_gpu=False, run_number=i)
                    self.results["scaling_results"][key]["cpu_results"].append(result)
    
    def analyze_scaling_results(self):
        """Analyze scaling performance across different dataset sizes."""
        console.print("\n" + "="*70)
        console.print("📈 Extended Benchmark Scaling Analysis")
        console.print("="*70)
        
        scaling_analysis = {}
        
        for key, data in self.results["scaling_results"].items():
            dataset_size = data["dataset_size"]
            model = data["model"]
            
            # Extract successful runs
            gpu_times = [r["execution_time"] for r in data["gpu_results"] if r["success"]]
            cpu_times = [r["execution_time"] for r in data["cpu_results"] if r["success"]]
            
            if not gpu_times or not cpu_times:
                console.print(f"❌ Insufficient successful runs for {model} with {dataset_size} tickers")
                continue
            
            # Calculate statistics
            gpu_mean = statistics.mean(gpu_times)
            cpu_mean = statistics.mean(cpu_times)
            speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0
            
            scaling_analysis[key] = {
                "dataset_size": dataset_size,
                "model": model,
                "gpu_mean_time": gpu_mean,
                "cpu_mean_time": cpu_mean,
                "speedup": speedup,
                "gpu_success_rate": len(gpu_times) / len(data["gpu_results"]) * 100,
                "cpu_success_rate": len(cpu_times) / len(data["cpu_results"]) * 100
            }
            
            # Print results
            console.print(f"\n📊 {model.upper()} with {dataset_size} tickers:")
            console.print(f"   • GPU Mean Time: {gpu_mean:.2f}s")
            console.print(f"   • CPU Mean Time: {cpu_mean:.2f}s")
            if speedup > 1:
                console.print(f"   • GPU Speedup: {speedup:.2f}x FASTER")
            else:
                console.print(f"   • CPU Performance: {1/speedup:.2f}x FASTER than GPU")
        
        self.results["analysis"]["scaling_analysis"] = scaling_analysis
        
        # Generate scaling trends
        self.generate_scaling_trends()
    
    def generate_scaling_trends(self):
        """Generate scaling trend analysis."""
        console.print(f"\n🔍 GPU Scaling Trends Analysis:")
        
        scaling_data = self.results["analysis"]["scaling_analysis"]
        
        for model in ["random_forest", "xgboost"]:
            console.print(f"\n📈 {model.upper()} Scaling Trends:")
            
            model_data = [(data["dataset_size"], data["speedup"]) 
                         for key, data in scaling_data.items() 
                         if data["model"] == model]
            
            if len(model_data) < 2:
                console.print("   • Insufficient data for trend analysis")
                continue
            
            model_data.sort(key=lambda x: x[0])  # Sort by dataset size
            
            for i, (size, speedup) in enumerate(model_data):
                trend = ""
                if i > 0:
                    prev_speedup = model_data[i-1][1]
                    if speedup > prev_speedup * 1.05:  # 5% improvement threshold
                        trend = " 📈 (Improving)"
                    elif speedup < prev_speedup * 0.95:  # 5% degradation threshold
                        trend = " 📉 (Degrading)"
                    else:
                        trend = " ➡️ (Stable)"
                
                console.print(f"   • {size} tickers: {speedup:.2f}x speedup{trend}")
            
            # Calculate overall trend
            if len(model_data) >= 2:
                first_speedup = model_data[0][1]
                last_speedup = model_data[-1][1]
                overall_trend = (last_speedup - first_speedup) / first_speedup * 100
                
                console.print(f"   • Overall Trend: {overall_trend:+.1f}% from {model_data[0][0]} to {model_data[-1][0]} tickers")
    
    def save_extended_results(self):
        """Save extended benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_reports/extended_benchmark_results_{timestamp}.json"
        
        # Ensure the generated_reports directory exists
        Path("generated_reports").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n💾 Extended benchmark results saved to: {filename}")
        return filename
    
    def generate_summary_report(self):
        """Generate a summary report for documentation updates."""
        console.print(f"\n📋 Extended Benchmark Summary Report:")
        console.print(f"{'='*50}")
        
        scaling_data = self.results["analysis"]["scaling_analysis"]
        
        # Best performing configurations
        best_gpu_speedup = max(scaling_data.values(), key=lambda x: x["speedup"])
        console.print(f"\n🏆 Best GPU Performance:")
        console.print(f"   • Model: {best_gpu_speedup['model'].upper()}")
        console.print(f"   • Dataset Size: {best_gpu_speedup['dataset_size']} tickers")
        console.print(f"   • Speedup: {best_gpu_speedup['speedup']:.2f}x")
        
        # Scaling insights
        console.print(f"\n🔍 Key Scaling Insights:")
        
        for model in ["random_forest", "xgboost"]:
            model_results = [(data["dataset_size"], data["speedup"]) 
                           for data in scaling_data.values() 
                           if data["model"] == model]
            
            if len(model_results) >= 2:
                model_results.sort()
                min_size, min_speedup = model_results[0]
                max_size, max_speedup = model_results[-1]
                
                console.print(f"   • {model.upper()}: {min_speedup:.2f}x → {max_speedup:.2f}x ({min_size} → {max_size} tickers)")
        
        return {
            "best_configuration": {
                "model": best_gpu_speedup['model'],
                "dataset_size": best_gpu_speedup['dataset_size'],
                "speedup": best_gpu_speedup['speedup']
            },
            "scaling_summary": scaling_data
        }

def main():
    """Main extended benchmark execution."""
    try:
        benchmark = ExtendedBenchmarkRunner()
        
        # Run the extended benchmark suite
        benchmark.run_extended_benchmark_suite()
        
        # Analyze scaling results
        benchmark.analyze_scaling_results()
        
        # Generate summary report
        summary = benchmark.generate_summary_report()
        
        # Save results
        results_file = benchmark.save_extended_results()
        
        console.print(f"\n🎉 Extended benchmark completed successfully!")
        console.print(f"📄 Detailed results saved in: {results_file}")
        console.print(f"\n💡 Use these results to update BENCHMARK_REPORT.md with Extended Benchmarking section")
        
    except KeyboardInterrupt:
        console.print("\n⏹️ Extended benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Extended benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()