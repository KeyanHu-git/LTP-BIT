from pathlib import Path
from typing import Dict, List

from .base import BaseWriter


class TxtWriter(BaseWriter):
    def write_single(self, result: Dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Task: {result['task_id']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write("-" * 50 + "\n")
            
            if result["status"] == "success":
                f.write(f"Samples: {result['num_samples']}\n")
                f.write(f"Time: {result.get('elapsed_time', 0):.2f}s\n")
                f.write("\nMetrics:\n")
                for name, value in result["metrics"].items():
                    f.write(f"  {name}: {value:.4f}\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown')}\n")
    
    def write_summary(self, results: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "failed"]
            
            f.write(f"Total Tasks: {len(results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            if successful:
                f.write("-" * 60 + "\n")
                f.write("RESULTS\n")
                f.write("-" * 60 + "\n\n")
                
                for r in successful:
                    f.write(f"Method: {r['task_id']}\n")
                    for name, value in r["metrics"].items():
                        f.write(f"  {name}: {value:.4f}\n")
                    f.write(f"  Samples: {r['num_samples']}\n")
                    f.write(f"  Time: {r.get('elapsed_time', 0):.2f}s\n")
                    f.write("\n")
            
            if failed:
                f.write("-" * 60 + "\n")
                f.write("FAILED TASKS\n")
                f.write("-" * 60 + "\n\n")
                for r in failed:
                    f.write(f"Method: {r['task_id']}\n")
                    f.write(f"  Error: {r.get('error', 'Unknown')}\n\n")



