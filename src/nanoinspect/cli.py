"""Command-line entry points. All model loads are local-only."""
import argparse
import json
from .runtime import baseline, doctor, download, write_json, compare_baselines


def main():
    parser = argparse.ArgumentParser(prog="nanoinspect")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_command = commands.add_parser("doctor")
    doctor_command.add_argument("--output", default="runs/doctor.json")
    for name in ("download", "baseline", "inventory", "trace", "experiment"):
        command = commands.add_parser(name)
        command.add_argument("--config", default={"trace": "configs/trace_short.yaml",
                                                  "experiment": "configs/retrieval_patch.yaml"}.get(name, "configs/model.yaml"))
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
    serve = commands.add_parser("serve", help="Interactive workbench on localhost")
    serve.add_argument("--config", default="configs/model.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-preload", action="store_true", help="Load the model on the first analysis request")
    args = parser.parse_args()
    if args.command == "serve":
        from .server import serve as run_server
        return run_server(args.host, args.port, args.config, preload=not args.no_preload)
    if args.command == "doctor":
        data = doctor(); write_json(args.output, data)
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
        from .trace import run_trace_command
        overrides = {key: getattr(args, key) for key in ("prompt", "layer", "token", "head", "site")}
        data = run_trace_command(args.config, args.output, overrides)
        print(json.dumps(data["parity_summary"]))
        if "calibration" in data:
            print(json.dumps({"calibration_all_passed": data["calibration"]["all_passed"]}))
        if not data["parity_summary"]["all_passed"] or not data.get("calibration", {}).get("all_passed", True):
            raise SystemExit("Trace parity failed; see trace.json")
    elif args.command == "experiment":
        from .experiments import run_experiment_command
        data = run_experiment_command(args.config, args.output)
    if isinstance(data, dict):
        print(json.dumps({k: v for k, v in data.items() if k in ["kind", "generation", "elapsed_seconds", "accounting", "cuda_available", "gpu", "model_dir", "summary"]}, indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    else: print(data)


if __name__ == "__main__": main()
