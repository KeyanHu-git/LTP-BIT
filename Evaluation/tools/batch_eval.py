import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import ConfigLoader

from .. import BatchEvaluator, register_all


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Evaluation Runner")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--gpu-ids", default=None, help="GPU IDs, e.g., '0,1,2'")
    parser.add_argument("--max-workers", type=int, default=None, help="Max workers")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    register_all()

    config = ConfigLoader.load_recursive(args.config)

    if args.gpu_ids:
        config["execution"]["gpu_ids"] = args.gpu_ids
    if args.max_workers:
        config["execution"]["max_workers"] = args.max_workers
    if args.output_dir:
        config.setdefault("outputs", {})["root_dir"] = args.output_dir
    
    print(f"Starting batch evaluation...")
    print(f"Config: {args.config}")
    print(f"Mode: {config.get('mode', 'multi_folder')}")
    print(f"GPU IDs: {config['execution']['gpu_ids']}")
    print(f"Max Workers: {config['execution']['max_workers']}")
    print(f"Distributed per task: {config['execution'].get('distributed_per_task', False)}")
    print("-" * 60)
    
    evaluator = BatchEvaluator(config)
    results = evaluator.execute()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Total Tasks: {len(results)}")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nResults:")
        for r in successful:
            print(f"  ✓ {r['task_id']}: {r['metrics']}")
    
    if failed:
        print("\nFailed Tasks:")
        for r in failed:
            print(f"  ✗ {r['task_id']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {config['outputs']['root_dir']}")


if __name__ == "__main__":
    main()



