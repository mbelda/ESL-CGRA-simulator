"""Run the original-versus-Simple-Address GEM ablation."""
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--sizes', nargs='+', type=int, default=[16, 32, 64, 128])
parser.add_argument('--output', default=str(root / 'benchmarks/DISCOvsOE/gem/simple_address_results.json'))
args = parser.parse_args()
subprocess.run([sys.executable, str(root / 'benchmarks/DISCOvsOE/simple_address.py'),
                '--benchmark', 'gem', '--sizes', *map(str, args.sizes),
                '--output', args.output], check=True)
