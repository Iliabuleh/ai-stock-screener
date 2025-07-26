#!/usr/bin/env python3
"""
GPU vs CPU Benchmark Script for AI Stock Screener

This script benchmarks the performance difference between GPU-accelerated
and CPU-only modes using the specified screening command:

poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA,PLTR,AMZN,NVDA,META,MSFT --news --threshold 0.07 --future_days 30

Usage:
    python benchmark_gpu_vs_cpu.py
"""

import time
import sys
import subprocess
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

class BenchmarkRunner:
    """Handles benchmarking of GPU vs CPU performance."""
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self.gpu_manager.get_gpu_info(),
            "benchmark_config": {
                "tickers": ["AAPL", "GOOGL", "TSLA", "PLTR", "AMZN", "NVDA", "META", "MSFT"],
                "mode": "eval",
                "news": True,
                "threshold": 0.07,
                "future_days": 30,
                "runs_per_mode": 3  # Multiple runs for statistical significance
            },
            "gpu_results": [],
            "cpu_results": [],
            "comparison": {}
        }
    
    def print_system_info(self):
        """Print system and GPU information."""
        console.print("\n" + "="*60)
        console.print("🔬 AI Stock Screener GPU vs CPU Benchmark")
        console.print("="*60)
        
        console.print("\n📊 Benchmark Configuration:")
        config = self.results["benchmark_config"]
        console.print(f"   • Tickers: {', '.join(config['tickers'])}")
        console.print(f"   • Mode: {config['mode']}")
        console.print(f"   • News Analysis: {config['news']}")
        console.print(f"   • Threshold: {config['threshold']}")
        console.print(f"   • Future Days: {config['future_days']}")
        console.print(f"   • Runs per Mode: {config['runs_per_mode']}")
        
        console.print("\n🖥️ System Information:")
        print_gpu_status()
        
        if not self.gpu_manager.is_gpu_available():
            console.print("\n⚠️ WARNING: No GPU acceleration detected!")
            console.print("   This benchmark will compare CPU vs CPU (no meaningful difference expected)")
            return False
        
        return True
    
    def run_single_benchmark(self, use_gpu: bool, run_number: int) -> Dict:
        """Run a single benchmark iteration."""
        mode_name = "GPU" if use_gpu else "CPU"
        console.print(f"\n🏃 Running {mode_name} benchmark (Run {run_number + 1})...")
        
        # Configuration for the screening
        config = {
            "period": "1y",
            "future_days": 30,
            "threshold": 0.07,
            "n_estimators": 300,
            "use_sharpe_labeling": 1.0,
            "model": "random_forest",  # Test with RandomForest first
            "grid_search": 0,  # Disable grid search for consistent timing
            "ensemble_runs": 1,
            "integrate_market": True,
            "use_gpu": use_gpu
        }
        
        tickers = ["AAPL", "GOOGL", "TSLA", "PLTR", "AMZN", "NVDA", "META", "MSFT"]
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Capture stdout to avoid cluttering benchmark output
            import io
            import contextlib
            
            # Run the screening
            run_screening(tickers, config, mode="eval", news_analysis=True)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "run_number": run_number + 1,
                "execution_time": execution_time,
                "success": True,
                "error": None,
                "config": config.copy()
            }
            
            console.print(f"✅ {mode_name} Run {run_number + 1} completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "run_number": run_number + 1,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "config": config.copy()
            }
            
            console.print(f"❌ {mode_name} Run {run_number + 1} failed after {execution_time:.2f} seconds: {e}")
        
        return result
    
    def run_benchmark_suite(self):
        """Run the complete benchmark suite."""
        if not self.print_system_info():
            console.print("\n⏭️ Proceeding with CPU-only comparison...")
        
        runs_per_mode = self.results["benchmark_config"]["runs_per_mode"]
        
        # Run GPU benchmarks
        console.print(f"\n🚀 Starting GPU Benchmarks ({runs_per_mode} runs)...")
        for i in range(runs_per_mode):
            result = self.run_single_benchmark(use_gpu=True, run_number=i)
            self.results["gpu_results"].append(result)
        
        # Run CPU benchmarks
        console.print(f"\n💻 Starting CPU Benchmarks ({runs_per_mode} runs)...")
        for i in range(runs_per_mode):
            result = self.run_single_benchmark(use_gpu=False, run_number=i)
            self.results["cpu_results"].append(result)
    
    def analyze_results(self):
        """Analyze and compare the benchmark results."""
        console.print("\n" + "="*60)
        console.print("📈 Benchmark Results Analysis")
        console.print("="*60)
        
        # Extract successful runs
        gpu_times = [r["execution_time"] for r in self.results["gpu_results"] if r["success"]]
        cpu_times = [r["execution_time"] for r in self.results["cpu_results"] if r["success"]]
        
        if not gpu_times or not cpu_times:
            console.print("❌ Insufficient successful runs for comparison")
            return
        
        # Calculate statistics
        gpu_stats = {
            "mean": statistics.mean(gpu_times),
            "median": statistics.median(gpu_times),
            "min": min(gpu_times),
            "max": max(gpu_times),
            "stdev": statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0
        }
        
        cpu_stats = {
            "mean": statistics.mean(cpu_times),
            "median": statistics.median(cpu_times),
            "min": min(cpu_times),
            "max": max(cpu_times),
            "stdev": statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0
        }
        
        # Calculate performance improvement
        speedup = cpu_stats["mean"] / gpu_stats["mean"] if gpu_stats["mean"] > 0 else 0
        time_saved = cpu_stats["mean"] - gpu_stats["mean"]
        percent_improvement = ((cpu_stats["mean"] - gpu_stats["mean"]) / cpu_stats["mean"]) * 100
        
        # Store comparison results
        self.results["comparison"] = {
            "gpu_stats": gpu_stats,
            "cpu_stats": cpu_stats,
            "speedup": speedup,
            "time_saved_seconds": time_saved,
            "percent_improvement": percent_improvement
        }
        
        # Print results
        console.print(f"\n🚀 GPU Performance (RandomForest with cuML):")
        console.print(f"   • Mean Time: {gpu_stats['mean']:.2f}s")
        console.print(f"   • Median Time: {gpu_stats['median']:.2f}s")
        console.print(f"   • Min Time: {gpu_stats['min']:.2f}s")
        console.print(f"   • Max Time: {gpu_stats['max']:.2f}s")
        console.print(f"   • Std Dev: {gpu_stats['stdev']:.2f}s")
        
        console.print(f"\n💻 CPU Performance (scikit-learn RandomForest):")
        console.print(f"   • Mean Time: {cpu_stats['mean']:.2f}s")
        console.print(f"   • Median Time: {cpu_stats['median']:.2f}s")
        console.print(f"   • Min Time: {cpu_stats['min']:.2f}s")
        console.print(f"   • Max Time: {cpu_stats['max']:.2f}s")
        console.print(f"   • Std Dev: {cpu_stats['stdev']:.2f}s")
        
        console.print(f"\n📊 Performance Comparison:")
        if speedup > 1:
            console.print(f"   • GPU is {speedup:.2f}x FASTER than CPU")
            console.print(f"   • Time Saved: {time_saved:.2f} seconds ({percent_improvement:.1f}% improvement)")
        elif speedup < 1:
            console.print(f"   • CPU is {1/speedup:.2f}x FASTER than GPU")
            console.print(f"   • GPU Overhead: {abs(time_saved):.2f} seconds ({abs(percent_improvement):.1f}% slower)")
        else:
            console.print(f"   • Performance is roughly equivalent")
        
        # Success rates
        gpu_success_rate = len([r for r in self.results["gpu_results"] if r["success"]]) / len(self.results["gpu_results"]) * 100
        cpu_success_rate = len([r for r in self.results["cpu_results"] if r["success"]]) / len(self.results["cpu_results"]) * 100
        
        console.print(f"\n✅ Success Rates:")
        console.print(f"   • GPU: {gpu_success_rate:.1f}% ({len(gpu_times)}/{len(self.results['gpu_results'])} runs)")
        console.print(f"   • CPU: {cpu_success_rate:.1f}% ({len(cpu_times)}/{len(self.results['cpu_results'])} runs)")
    
    def test_xgboost_comparison(self):
        """Run additional benchmark with XGBoost model."""
        console.print(f"\n🔄 Running XGBoost GPU vs CPU Comparison...")
        
        # XGBoost GPU test
        config_gpu = {
            "period": "1y", "future_days": 30, "threshold": 0.07, "n_estimators": 300,
            "use_sharpe_labeling": 1.0, "model": "xgboost", "grid_search": 0,
            "ensemble_runs": 1, "integrate_market": True, "use_gpu": True
        }
        
        # XGBoost CPU test
        config_cpu = {
            "period": "1y", "future_days": 30, "threshold": 0.07, "n_estimators": 300,
            "use_sharpe_labeling": 1.0, "model": "xgboost", "grid_search": 0,
            "ensemble_runs": 1, "integrate_market": True, "use_gpu": False
        }
        
        tickers = ["AAPL", "GOOGL", "TSLA", "PLTR"]  # Smaller set for quick test
        
        try:
            # GPU XGBoost
            start_time = time.time()
            run_screening(tickers, config_gpu, mode="eval", news_analysis=False)
            gpu_time = time.time() - start_time
            
            # CPU XGBoost
            start_time = time.time()
            run_screening(tickers, config_cpu, mode="eval", news_analysis=False)
            cpu_time = time.time() - start_time
            
            xgb_speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            console.print(f"\n🌲 XGBoost Performance Comparison:")
            console.print(f"   • GPU XGBoost: {gpu_time:.2f}s")
            console.print(f"   • CPU XGBoost: {cpu_time:.2f}s")
            if xgb_speedup > 1:
                console.print(f"   • GPU is {xgb_speedup:.2f}x FASTER for XGBoost")
            else:
                console.print(f"   • CPU is {1/xgb_speedup:.2f}x FASTER for XGBoost")
                
        except Exception as e:
            console.print(f"❌ XGBoost comparison failed: {e}")
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_reports/benchmark_results_{timestamp}.json"
        
        # Ensure the generated_reports directory exists
        Path("generated_reports").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """Main benchmark execution."""
    try:
        benchmark = BenchmarkRunner()
        
        # Run the main benchmark suite
        benchmark.run_benchmark_suite()
        
        # Analyze results
        benchmark.analyze_results()
        
        # Test XGBoost as well
        benchmark.test_xgboost_comparison()
        
        # Save results
        results_file = benchmark.save_results()
        
        console.print(f"\n🎉 Benchmark completed successfully!")
        console.print(f"📄 Detailed results saved in: {results_file}")
        
    except KeyboardInterrupt:
        console.print("\n⏹️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()