"""Compare rank-8 LoRA with 10% and 20% V4L2 sampling on two devices."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_script.run_v4l2_lora_sweep import main


if __name__ == '__main__':
    raise SystemExit(main(ROOT / 'configs/experiments/v4l2_r8_real_fraction_sweep.json'))
