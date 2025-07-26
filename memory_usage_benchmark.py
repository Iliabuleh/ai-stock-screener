#!/usr/bin/env python3
"""
Memory Usage Benchmark for AI Stock Screener
Analyzes GPU vs CPU memory consumption patterns across different dataset sizes.

This script implements the 3rd feature from BENCHMARK_REPORT.md Future Work section:
"Memory Usage Analysis: Compare GPU vs CPU memory consumption patterns"
"""

import time
import sys
import json
import psutil
import threading
from datetime import datetime
from pathlib import Path
import statistics
from typing import Dict, List, Tuple, Optional
import gc

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_stock_screener.ai_screener import run_screening
from ai_stock_screener.gpu_utils import get_gpu_manager, print_gpu_status
from ai_stock_screener.output_formatter import console

try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    console.print("⚠️ GPUtil not available - GPU memory monitoring will be limited")


class MemoryMonitor:
    """Monitors system and GPU memory usage during benchmark execution."""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = None
        self.sample_interval = 0.1  # Sample every 100ms
        
    def start_monitoring(self):
        """Start memory monitoring in a separate thread."""
        self.monitoring = True
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return {
            "system_memory": self.memory_samples,
            "gpu_memory": self.gpu_memory_samples,
            "peak_system_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "peak_gpu_memory_mb": max(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            "avg_system_memory_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "avg_gpu_memory_mb": statistics.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
            "sample_count": len(self.memory_samples)
        }
    
    def _monitor_memory(self):
        """Internal method to continuously monitor memory usage."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # System memory usage (RSS - Resident Set Size)
                memory_info = process.memory_info()
                system_memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                self.memory_samples.append(system_memory_mb)
                
                # GPU memory usage (if available)
                if GPU_MONITORING_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            # Use the first GPU's memory usage
                            gpu_memory_mb = gpus[0].memoryUsed
                            self.gpu_memory_samples.append(gpu_memory_mb)
                        else:
                            self.gpu_memory_samples.append(0)
                    except Exception:
                        self.gpu_memory_samples.append(0)
                else:
                    self.gpu_memory_samples.append(0)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                # Continue monitoring even if individual samples fail
                continue


class MemoryBenchmarkRunner:
    """Handles memory usage benchmarking with different dataset sizes for GPU vs CPU analysis."""
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.memory_monitor = MemoryMonitor()
        
        # Same ticker sets as extended benchmark for consistency
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
            "memory_benchmark_config": {
                "dataset_sizes": [8, 20, 50, 100],
                "models": ["random_forest", "xgboost"],
                "runs_per_configuration": 2,
                "future_days": 30,
                "threshold": 0.07,
                "news_analysis": False,  # Disabled for consistent memory measurement
                "memory_sample_interval_ms": 100
            },
            "memory_results": {},
            "analysis": {}
        }
    
    def print_system_info(self):
        """Print system and memory benchmark information."""
        console.print("\n" + "="*70)
        console.print("🧠 AI Stock Screener Memory Usage Benchmark")
        console.print("="*70)
        
        console.print("\n📊 Memory Benchmark Configuration:")
        config = self.results["memory_benchmark_config"]
        console.print(f"   • Dataset Sizes: {config['dataset_sizes']} tickers")
        console.print(f"   • Models: {', '.join(config['models'])}")
        console.print(f"   • Runs per Configuration: {config['runs_per_configuration']}")
        console.print(f"   • Memory Sampling: Every {config['memory_sample_interval_ms']}ms")
        console.print(f"   • GPU Memory Monitoring: {'✅ Available' if GPU_MONITORING_AVAILABLE else '❌ Limited'}")
        
        console.print("\n🖥️ System Information:")
        print_gpu_status()
        
        # System memory info
        memory = psutil.virtual_memory()
        console.print(f"   • System RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        
        if not self.gpu_manager.is_gpu_available():
            console.print("\n⚠️ WARNING: No GPU acceleration detected!")
            console.print("   Memory benchmark will compare CPU vs CPU memory usage")
            return False
        
        return True
    
    def run_single_memory_benchmark(self, dataset_size: int, model: str, use_gpu: bool, run_number: int) -> Dict:
        """Run a single memory benchmark iteration for a specific dataset size and model."""
        mode_name = "GPU" if use_gpu else "CPU"
        console.print(f"\n🧠 Running {mode_name} {model} memory benchmark with {dataset_size} tickers (Run {run_number + 1})...")
        
        # Force garbage collection before benchmark
        gc.collect()
        
        # Get baseline memory usage
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
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
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Run the screening
            run_screening(tickers, config, mode="eval", news_analysis=False)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Stop memory monitoring and collect data
            memory_data = self.memory_monitor.stop_monitoring()
            
            # Calculate memory metrics
            peak_memory_usage = memory_data["peak_system_memory_mb"]
            memory_overhead = peak_memory_usage - baseline_memory
            
            result = {
                "dataset_size": dataset_size,
                "model": model,
                "run_number": run_number + 1,
                "execution_time": execution_time,
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory_usage,
                "memory_overhead_mb": memory_overhead,
                "avg_memory_mb": memory_data["avg_system_memory_mb"],
                "peak_gpu_memory_mb": memory_data["peak_gpu_memory_mb"],
                "avg_gpu_memory_mb": memory_data["avg_gpu_memory_mb"],
                "memory_samples": len(memory_data["system_memory"]),
                "success": True,
                "error": None,
                "config": config.copy()
            }
            
            console.print(f"✅ {mode_name} {model} with {dataset_size} tickers:")
            console.print(f"   • Execution Time: {execution_time:.2f}s")
            console.print(f"   • Peak Memory: {peak_memory_usage:.1f} MB")
            console.print(f"   • Memory Overhead: {memory_overhead:.1f} MB")
            if use_gpu and memory_data["peak_gpu_memory_mb"] > 0:
                console.print(f"   • Peak GPU Memory: {memory_data['peak_gpu_memory_mb']:.1f} MB")
            
        except Exception as e:
            # Stop monitoring even if benchmark fails
            memory_data = self.memory_monitor.stop_monitoring()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            result = {
                "dataset_size": dataset_size,
                "model": model,
                "run_number": run_number + 1,
                "execution_time": execution_time,
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": 0,
                "memory_overhead_mb": 0,
                "avg_memory_mb": 0,
                "peak_gpu_memory_mb": 0,
                "avg_gpu_memory_mb": 0,
                "memory_samples": 0,
                "success": False,
                "error": str(e),
                "config": config.copy()
            }
            
            console.print(f"❌ {mode_name} {model} with {dataset_size} tickers failed: {str(e)}")
        
        # Force garbage collection after benchmark
        gc.collect()
        
        return result
    
    def run_memory_benchmark_suite(self):
        """Run the complete memory benchmark suite across all dataset sizes and models."""
        console.print("\n🚀 Starting Memory Usage Benchmark Suite...")
        
        dataset_sizes = self.results["memory_benchmark_config"]["dataset_sizes"]
        models = self.results["memory_benchmark_config"]["models"]
        runs_per_config = self.results["memory_benchmark_config"]["runs_per_configuration"]
        
        for dataset_size in dataset_sizes:
            console.print(f"\n📊 Testing dataset size: {dataset_size} tickers")
            
            for model in models:
                console.print(f"\n🤖 Testing model: {model}")
                
                # Initialize results for this configuration
                config_key = f"{dataset_size}_{model}"
                self.results["memory_results"][config_key] = {
                    "gpu_runs": [],
                    "cpu_runs": []
                }
                
                # Run GPU benchmarks
                for run in range(runs_per_config):
                    result = self.run_single_memory_benchmark(dataset_size, model, use_gpu=True, run_number=run)
                    self.results["memory_results"][config_key]["gpu_runs"].append(result)
                    time.sleep(2)  # Brief pause between runs
                
                # Run CPU benchmarks
                for run in range(runs_per_config):
                    result = self.run_single_memory_benchmark(dataset_size, model, use_gpu=False, run_number=run)
                    self.results["memory_results"][config_key]["cpu_runs"].append(result)
                    time.sleep(2)  # Brief pause between runs
        
        console.print("\n✅ Memory benchmark suite completed!")
    
    def analyze_memory_results(self):
        """Analyze memory benchmark results and generate insights."""
        console.print("\n📈 Analyzing Memory Usage Results...")
        
        analysis = {
            "memory_efficiency_by_dataset": {},
            "memory_scaling_trends": {},
            "gpu_vs_cpu_memory_comparison": {},
            "key_insights": []
        }
        
        for config_key, results in self.results["memory_results"].items():
            dataset_size, model = config_key.split("_", 1)
            dataset_size = int(dataset_size)
            
            # Analyze successful runs only
            gpu_successful = [r for r in results["gpu_runs"] if r["success"]]
            cpu_successful = [r for r in results["cpu_runs"] if r["success"]]
            
            if not gpu_successful or not cpu_successful:
                continue
            
            # Calculate memory statistics
            gpu_peak_memory = [r["peak_memory_mb"] for r in gpu_successful]
            cpu_peak_memory = [r["peak_memory_mb"] for r in cpu_successful]
            gpu_overhead = [r["memory_overhead_mb"] for r in gpu_successful]
            cpu_overhead = [r["memory_overhead_mb"] for r in cpu_successful]
            
            gpu_avg_peak = statistics.mean(gpu_peak_memory)
            cpu_avg_peak = statistics.mean(cpu_peak_memory)
            gpu_avg_overhead = statistics.mean(gpu_overhead)
            cpu_avg_overhead = statistics.mean(cpu_overhead)
            
            # Memory efficiency ratio (lower is better)
            memory_efficiency_ratio = gpu_avg_peak / cpu_avg_peak if cpu_avg_peak > 0 else 1.0
            
            config_analysis = {
                "dataset_size": dataset_size,
                "model": model,
                "gpu_avg_peak_memory_mb": gpu_avg_peak,
                "cpu_avg_peak_memory_mb": cpu_avg_peak,
                "gpu_avg_overhead_mb": gpu_avg_overhead,
                "cpu_avg_overhead_mb": cpu_avg_overhead,
                "memory_efficiency_ratio": memory_efficiency_ratio,
                "gpu_memory_advantage": "Lower" if memory_efficiency_ratio < 1.0 else "Higher",
                "memory_difference_mb": gpu_avg_peak - cpu_avg_peak,
                "memory_difference_percent": ((gpu_avg_peak - cpu_avg_peak) / cpu_avg_peak * 100) if cpu_avg_peak > 0 else 0
            }
            
            analysis["memory_efficiency_by_dataset"][config_key] = config_analysis
        
        # Generate scaling trends
        for model in ["random_forest", "xgboost"]:
            model_configs = [k for k in analysis["memory_efficiency_by_dataset"].keys() if k.endswith(model)]
            model_configs.sort(key=lambda x: int(x.split("_")[0]))
            
            scaling_data = []
            for config_key in model_configs:
                config_data = analysis["memory_efficiency_by_dataset"][config_key]
                scaling_data.append({
                    "dataset_size": config_data["dataset_size"],
                    "gpu_memory_mb": config_data["gpu_avg_peak_memory_mb"],
                    "cpu_memory_mb": config_data["cpu_avg_peak_memory_mb"],
                    "efficiency_ratio": config_data["memory_efficiency_ratio"]
                })
            
            analysis["memory_scaling_trends"][model] = scaling_data
        
        # Generate key insights
        insights = []
        
        # Overall memory efficiency
        all_ratios = [config["memory_efficiency_ratio"] for config in analysis["memory_efficiency_by_dataset"].values()]
        if all_ratios:
            avg_ratio = statistics.mean(all_ratios)
            if avg_ratio < 1.0:
                insights.append(f"GPU implementations use {(1-avg_ratio)*100:.1f}% less memory on average")
            else:
                insights.append(f"GPU implementations use {(avg_ratio-1)*100:.1f}% more memory on average")
        
        # Model-specific insights
        for model in ["random_forest", "xgboost"]:
            model_ratios = [config["memory_efficiency_ratio"] for config in analysis["memory_efficiency_by_dataset"].values() if config["model"] == model]
            if model_ratios:
                avg_model_ratio = statistics.mean(model_ratios)
                model_name = "RandomForest" if model == "random_forest" else "XGBoost"
                if avg_model_ratio < 1.0:
                    insights.append(f"{model_name} GPU uses {(1-avg_model_ratio)*100:.1f}% less memory than CPU")
                else:
                    insights.append(f"{model_name} GPU uses {(avg_model_ratio-1)*100:.1f}% more memory than CPU")
        
        # Scaling insights
        for model in ["random_forest", "xgboost"]:
            if model in analysis["memory_scaling_trends"]:
                scaling_data = analysis["memory_scaling_trends"][model]
                if len(scaling_data) >= 2:
                    first_ratio = scaling_data[0]["efficiency_ratio"]
                    last_ratio = scaling_data[-1]["efficiency_ratio"]
                    model_name = "RandomForest" if model == "random_forest" else "XGBoost"
                    
                    if last_ratio < first_ratio:
                        insights.append(f"{model_name} GPU memory efficiency improves with larger datasets")
                    elif last_ratio > first_ratio:
                        insights.append(f"{model_name} GPU memory efficiency decreases with larger datasets")
                    else:
                        insights.append(f"{model_name} GPU memory efficiency remains consistent across dataset sizes")
        
        analysis["key_insights"] = insights
        self.results["analysis"] = analysis
        
        # Print summary
        console.print("\n📊 Memory Analysis Summary:")
        for insight in insights:
            console.print(f"   • {insight}")
    
    def generate_memory_report(self):
        """Generate a comprehensive memory usage report."""
        console.print("\n📋 Generating Memory Usage Report...")
        
        report_lines = []
        report_lines.append("## Memory Usage Analysis")
        report_lines.append("")
        report_lines.append(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
        report_lines.append(f"**Feature Status:** ✅ **COMPLETED**")
        report_lines.append("")
        report_lines.append("This section presents the results of memory usage analysis comparing GPU vs CPU memory consumption patterns across different dataset sizes.")
        report_lines.append("")
        
        # Configuration
        config = self.results["memory_benchmark_config"]
        report_lines.append("### Memory Benchmark Configuration")
        report_lines.append("")
        report_lines.append(f"- **Dataset Sizes**: {', '.join(map(str, config['dataset_sizes']))} tickers")
        report_lines.append(f"- **Models Tested**: {', '.join(config['models'])}")
        report_lines.append(f"- **Runs per Configuration**: {config['runs_per_configuration']} (for statistical significance)")
        report_lines.append(f"- **Historical Period**: 6 months (optimized for consistent memory measurement)")
        report_lines.append(f"- **Future Days**: {config['future_days']}")
        report_lines.append(f"- **Threshold**: {config['threshold']} ({config['threshold']*100}% growth threshold)")
        report_lines.append(f"- **News Analysis**: {'Enabled' if config['news_analysis'] else 'Disabled'} (for consistent memory timing)")
        report_lines.append(f"- **Memory Sampling**: Every {config['memory_sample_interval_ms']}ms")
        report_lines.append("")
        
        # Memory efficiency results
        if "analysis" in self.results and "memory_efficiency_by_dataset" in self.results["analysis"]:
            analysis = self.results["analysis"]
            
            # RandomForest memory analysis
            report_lines.append("#### RandomForest Memory Usage")
            report_lines.append("")
            report_lines.append("| Dataset Size | GPU Memory (MB) | CPU Memory (MB) | Memory Ratio | Trend |")
            report_lines.append("|--------------|-----------------|-----------------|--------------|-------|")
            
            rf_configs = [(k, v) for k, v in analysis["memory_efficiency_by_dataset"].items() if v["model"] == "random_forest"]
            rf_configs.sort(key=lambda x: x[1]["dataset_size"])
            
            for config_key, config_data in rf_configs:
                dataset_size = config_data["dataset_size"]
                gpu_memory = config_data["gpu_avg_peak_memory_mb"]
                cpu_memory = config_data["cpu_avg_peak_memory_mb"]
                ratio = config_data["memory_efficiency_ratio"]
                
                if ratio < 1.0:
                    trend = "📉 GPU Lower"
                    ratio_text = f"{ratio:.2f}x"
                else:
                    trend = "📈 GPU Higher" 
                    ratio_text = f"{ratio:.2f}x"
                
                report_lines.append(f"| {dataset_size} tickers | {gpu_memory:.1f} | {cpu_memory:.1f} | {ratio_text} | {trend} |")
            
            report_lines.append("")
            
            # XGBoost memory analysis
            report_lines.append("#### XGBoost Memory Usage")
            report_lines.append("")
            report_lines.append("| Dataset Size | GPU Memory (MB) | CPU Memory (MB) | Memory Ratio | Trend |")
            report_lines.append("|--------------|-----------------|-----------------|--------------|-------|")
            
            xgb_configs = [(k, v) for k, v in analysis["memory_efficiency_by_dataset"].items() if v["model"] == "xgboost"]
            xgb_configs.sort(key=lambda x: x[1]["dataset_size"])
            
            for config_key, config_data in xgb_configs:
                dataset_size = config_data["dataset_size"]
                gpu_memory = config_data["gpu_avg_peak_memory_mb"]
                cpu_memory = config_data["cpu_avg_peak_memory_mb"]
                ratio = config_data["memory_efficiency_ratio"]
                
                if ratio < 1.0:
                    trend = "📉 GPU Lower"
                    ratio_text = f"{ratio:.2f}x"
                else:
                    trend = "📈 GPU Higher"
                    ratio_text = f"{ratio:.2f}x"
                
                report_lines.append(f"| {dataset_size} tickers | {gpu_memory:.1f} | {cpu_memory:.1f} | {ratio_text} | {trend} |")
            
            report_lines.append("")
            
            # Key insights
            if "key_insights" in analysis and analysis["key_insights"]:
                report_lines.append("### Key Memory Usage Findings")
                report_lines.append("")
                for i, insight in enumerate(analysis["key_insights"], 1):
                    report_lines.append(f"{i}. **{insight}**")
                report_lines.append("")
            
            # Memory efficiency recommendations
            report_lines.append("### Memory Usage Recommendations")
            report_lines.append("")
            
            # Calculate overall trends
            all_configs = list(analysis["memory_efficiency_by_dataset"].values())
            rf_ratios = [c["memory_efficiency_ratio"] for c in all_configs if c["model"] == "random_forest"]
            xgb_ratios = [c["memory_efficiency_ratio"] for c in all_configs if c["model"] == "xgboost"]
            
            if rf_ratios:
                avg_rf_ratio = statistics.mean(rf_ratios)
                if avg_rf_ratio < 1.0:
                    report_lines.append("- **RandomForest**: GPU implementation is more memory-efficient than CPU")
                else:
                    report_lines.append("- **RandomForest**: CPU implementation is more memory-efficient than GPU")
            
            if xgb_ratios:
                avg_xgb_ratio = statistics.mean(xgb_ratios)
                if avg_xgb_ratio < 1.0:
                    report_lines.append("- **XGBoost**: GPU implementation is more memory-efficient than CPU")
                else:
                    report_lines.append("- **XGBoost**: CPU implementation is more memory-efficient than GPU")
            
            report_lines.append("- **Production Guidance**: Consider memory constraints when choosing between GPU and CPU modes")
            report_lines.append("- **Large Datasets**: Monitor memory usage closely with 50+ tickers to avoid out-of-memory errors")
            report_lines.append("")
        
        # Methodology
        report_lines.append("### Memory Benchmark Methodology")
        report_lines.append("")
        report_lines.append("**Memory Metrics Collected:**")
        report_lines.append("- Peak memory usage (RSS - Resident Set Size)")
        report_lines.append("- Average memory usage during execution")
        report_lines.append("- Memory overhead (peak - baseline)")
        report_lines.append("- GPU memory usage (when available)")
        report_lines.append("- Memory sampling every 100ms during execution")
        report_lines.append("")
        report_lines.append("**Memory Monitoring Tools:**")
        report_lines.append("- System Memory: psutil library")
        report_lines.append("- GPU Memory: GPUtil library (when available)")
        report_lines.append("- Continuous monitoring during benchmark execution")
        report_lines.append("- Garbage collection before and after each benchmark")
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_memory_results(self):
        """Save memory benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_reports/memory_benchmark_results_{timestamp}.json"
        
        # Ensure the generated_reports directory exists
        Path("generated_reports").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        console.print(f"\n💾 Memory benchmark results saved to: {filename}")
        return filename


def main():
    """Main function to run memory usage benchmark."""
    runner = MemoryBenchmarkRunner()
    
    # Print system information
    if not runner.print_system_info():
        console.print("\n❌ Cannot run meaningful memory benchmark without GPU acceleration")
        return
    
    try:
        # Run the memory benchmark suite
        runner.run_memory_benchmark_suite()
        
        # Analyze results
        runner.analyze_memory_results()
        
        # Generate and print report
        report = runner.generate_memory_report()
        console.print("\n" + "="*70)
        console.print("📋 MEMORY USAGE ANALYSIS REPORT")
        console.print("="*70)
        console.print(report)
        
        # Save results
        results_file = runner.save_memory_results()
        
        console.print(f"\n✅ Memory usage benchmark completed successfully!")
        console.print(f"📊 Report content ready for BENCHMARK_REPORT.md integration")
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Memory benchmark interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Memory benchmark failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()