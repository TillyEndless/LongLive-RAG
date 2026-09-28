import argparse, json
from pathlib import Path
from unified_evaluation_schema import as_csv_row, write_csv

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('summaries', nargs='+'); ap.add_argument('--output', required=True); a = ap.parse_args()
    summaries = [json.loads(Path(p).read_text()) for p in a.summaries]
    write_csv(a.output, [as_csv_row(x) for x in summaries])
if __name__ == '__main__': main()
