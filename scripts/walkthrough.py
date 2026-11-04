"""Print the M2 computation walkthrough for both mixer types as plain text.

Usage (inside the nano-interpretability environment):
    python scripts/walkthrough.py [--config configs/trace_short.yaml] [--output runs/trace]

Runs the same trace as ``nanoinspect trace`` (offline, local snapshot), then
prints each worked-example step, the residual-stream effects and the parity
verdicts. Exits non-zero if any parity or calibration check fails.
"""
import argparse
import json
import sys
from nanoinspect.trace import run_trace_command


def show(value):
    return json.dumps(value, default=str) if not isinstance(value, str) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trace_short.yaml")
    parser.add_argument("--output", default="runs/trace")
    args = parser.parse_args()
    result = run_trace_command(args.config, args.output)
    meta = result["metadata"]
    print(f"Prompt tokens: {list(enumerate(meta['tokens']))}")
    print(f"Selected position {meta['selected_token_position']} ({meta['selected_token']!r}); "
          f"next-token candidates {[s['token'] for s in meta['selected_next_tokens']]}\n")
    for example in result["worked_examples"]:
        head = example.get("query_head", example.get("value_head"))
        print(f"=== Layer {example['layer']} {example['mixer']}, head {head} ===")
        for number, step in enumerate(example["steps"], 1):
            details = {k: v for k, v in step.items() if k != "step"}
            print(f"{number:2d}. {step['step']}\n    {show(details)[:600]}")
        print()
    for effect in result["residual_effects"]:
        print(f"=== Layer {effect['layer']} residual stream at position {effect['token_position']} ===")
        for site, entry in effect["components"].items():
            print(f"  {site:20s} norm {entry['norm']:.3f}  direct logits {show(entry['direct_logit'])}")
        for name, entry in effect["ablations_at_selected_token"].items():
            print(f"  {name}: logit change {show(entry['logit_change'])}")
        print()
    failed = [p for p in result["parity"] + result.get("calibration", {}).get("checks", []) if not p["passed"]]
    print(f"Parity: {result['parity_summary']['passed']}/{result['parity_summary']['checks']} passed; "
          f"calibration all passed: {result.get('calibration', {}).get('all_passed')}")
    if failed:
        print("FAILED:", show(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
