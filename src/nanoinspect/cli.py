"""Command-line entry points. All model loads are local-only."""
import argparse
import json
from .runtime import baseline, doctor, download, write_json, load_model, read_config, compare_baselines


def main():
    parser = argparse.ArgumentParser(prog="nanoinspect")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    for name in ("download", "baseline", "inventory", "trace", "experiment"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/model.yaml")
        command.add_argument("--output", default=f"runs/{name}")
        if name == "baseline":
            command.add_argument("--offline-check", action="store_true")
            command.add_argument("--compare", help="Directory containing the reference baseline")
        if name == "trace":
            command.add_argument("--prompt")
            command.add_argument("--layer", type=int, action="append")
            command.add_argument("--token", type=int, action="append")
            command.add_argument("--head", type=int)
            command.add_argument("--site", action="append")
    report = commands.add_parser("report")
    report.add_argument("--run", required=True)
    report.add_argument("--compare")
    args = parser.parse_args()
    if args.command == "doctor":
        data = doctor(); write_json("runs/doctor.json", data)
    elif args.command == "download": data = download(args.config)
    elif args.command == "baseline":
        data = baseline(args.config, args.output, args.offline_check)
        if args.compare: compare_baselines(args.compare, args.output)
    elif args.command == "inventory":
        from .inventory import build_inventory
        data = build_inventory(args.config, args.output)
    elif args.command == "report":
        from .reporting import generate_report
        data = generate_report(args.run, compare=args.compare)
    elif args.command == "trace":
        from .trace import run_trace
        cfg = read_config(args.config)
        for key in ("prompt", "layer", "token", "head", "site"):
            if getattr(args, key) is not None: cfg[key] = getattr(args, key)
        data = run_trace(load_model(cfg), cfg, args.output)
    else:
        raise SystemExit("Trace/experiment implementation is being integrated.")
    if isinstance(data, dict):
        print(json.dumps({k: v for k, v in data.items() if k in ["kind", "generation", "elapsed_seconds", "accounting", "cuda_available", "gpu", "model_dir"]}, indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    else: print(data)


if __name__ == "__main__": main()
