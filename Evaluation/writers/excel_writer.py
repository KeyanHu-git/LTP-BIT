import pandas as pd
from pathlib import Path
from typing import Dict, List

from .base import BaseWriter


class ExcelWriter(BaseWriter):
    def write_single(self, result: Dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if result["status"] == "failed":
            df = pd.DataFrame([{"error": result.get("error", "Unknown error")}])
        else:
            metrics = result["metrics"]
            row = {
                "num_samples": f"{result['num_samples']}/{result['num_samples']}",
                "elapsed_time(s)": f"{result.get('elapsed_time', 0):.2f}"
            }
            row.update(metrics)
            df = pd.DataFrame([row])
        
        df.to_excel(output_path, index=False)
    
    def write_summary(self, results: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for r in results:
            if r["status"] == "success":
                row = {"method": r["task_id"]}
                row.update(r["metrics"])
                row["samples"] = f"{r['num_samples']}/{r['num_samples']}"
                row["time(s)"] = f"{r.get('elapsed_time', 0):.2f}"
                rows.append(row)
            else:
                rows.append({
                    "method": r["task_id"],
                    "error": r.get("error", "Failed")
                })
        
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False)



