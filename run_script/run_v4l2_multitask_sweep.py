"""Compare LoRA-r4, LoRA-r8 and direct last-four tuning with depth, mask and normal."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_script.run_v4l2_lora_sweep import main

if __name__ == '__main__':
    raise SystemExit(main(ROOT / 'configs/experiments/v4l2_multitask_sweep.json'))
