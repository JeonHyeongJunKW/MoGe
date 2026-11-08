"""Compare LoRA r4 alone with LoRA r4 plus the metric scale head."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_script.run_v4l2_lora_sweep import main
if __name__ == '__main__':
    raise SystemExit(main(ROOT / 'configs/experiments/v4l2_lora_freeze_sweep.json'))
