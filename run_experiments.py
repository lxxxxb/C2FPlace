import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


DEFAULT_BENCHMARKS = [
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue2",
    "bigblue3",
    "bigblue4",
]


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def experiment_name(benchmark, ripup_strategy, bbox_order, ripup_ratio, coarse_grid, seed, random_init_order=False, ripup_ratio_strategy="fixed", stage3_iters=0, stage3_order="hpwl", stage3_mode="fine", replace_order="auto"):
    if ripup_strategy == "bbox":
        strategy_tag = f"bbox_{bbox_order}"
    else:
        strategy_tag = "random"
    ratio_tag = f"r{ripup_ratio:g}".replace(".", "p")
    mode_tags = []
    if random_init_order:
        mode_tags.append("randinit")
    if ripup_ratio_strategy == "random_stage":
        mode_tags.append("randratio")
    elif ripup_ratio_strategy == "decile_random":
        mode_tags.append("decilerand")
    if replace_order != "auto":
        mode_tags.append(f"repl{replace_order}")
    if stage3_iters > 0:
        stage3_order_tag = {"random": "rand", "node_id": "nodeid"}.get(stage3_order, "hpwl")
        mode_tags.append(f"s3{stage3_mode}_{stage3_order_tag}{stage3_iters}")
    mode_tag = "_" + "_".join(mode_tags) if mode_tags else ""
    return f"{benchmark}_{strategy_tag}_{ratio_tag}_cg{coarse_grid}_seed{seed}{mode_tag}"


def build_command(args, benchmark, ripup_strategy, bbox_order, replace_order, ripup_ratio, coarse_grid, seed, ripup_ratio_strategy, output_path):
    command = [
        sys.executable,
        "run_placement.py",
        "--benchmark",
        benchmark,
        "--coarse_grid",
        str(coarse_grid),
        "--pop_size",
        str(args.pop_size),
        "--stage1_iters",
        str(args.stage1_iters),
        "--stage2_iters",
        str(args.stage2_iters),
        "--stage3_iters",
        str(args.stage3_iters),
        "--stage3_mode",
        args.stage3_mode,
        "--stage3_order",
        args.stage3_order,
        "--ripup_ratio",
        str(ripup_ratio),
        "--ripup_strategy",
        ripup_strategy,
        "--seed",
        str(seed),
        "--ripup_ratio_strategy",
        ripup_ratio_strategy,
        "--output",
        str(output_path),
        "--bbox_order",
        str(bbox_order),
        "--replace_order",
        replace_order,
    ]
    if args.random_init_order:
        command.append("--random_init_order")
    return command


def run_experiment(command, command_text, log_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"Command: {command_text}\n\n")
        log_file.flush()
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    return result.returncode, log_path


def main():
    parser = argparse.ArgumentParser(description="Run C2FPlace-EA experiments over benchmark/parameter grids.")
    parser.add_argument("--benchmarks", type=parse_list, default=DEFAULT_BENCHMARKS,
                        help="Comma-separated benchmark directory names under benchmark/ (default: adaptec1; data must be downloaded separately).")
    parser.add_argument("--ripup_strategies", type=parse_list, default=["random"],
                        help="Comma-separated ripup strategies. Example: random,bbox")
    # "bbox"]
    parser.add_argument("--bbox_orders", type=parse_list, default=["random"],
                        help="Comma-separated bbox candidate rankings used when ripup_strategy=bbox.")
    parser.add_argument("--replace_orders", "--replace_order", dest="replace_orders", type=parse_list, default=["node_id"],
                        help="Comma-separated replacement orders: auto,preserve,node_id,random,area_group.")
    # auto,preserve,node_id,random,area_group
    # ["count_desc", "count_asc", "random", "area_asc", "node_id"]
    parser.add_argument("--seeds", type=parse_list, default=[str(seed) for seed in range(0, 5)],
                        help="Comma-separated random seeds.")
    # ["1", "2", "3", "4", "5", ..., "20"]
    parser.add_argument("--ripup_ratios", type=parse_list,
                        default=["0.2"],
                        help="Comma-separated rip-up ratios.")
    # ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
    parser.add_argument("--coarse_grids", type=parse_list, default=["512"],
                        help="Comma-separated coarse grid sizes.")
    # ["128", "224", "256", "512", "1024"]
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--stage1_iters", type=int, default=5000)
    parser.add_argument("--stage2_iters", type=int, default=5000)
    parser.add_argument("--stage3_iters", type=int, default=50)
    parser.add_argument("--stage3_mode", type=str, default="both_2", choices=["coarse", "fine", "both_1", "both_2"])
    parser.add_argument("--stage3_order", type=str, default="node_id", choices=["hpwl", "random", "node_id"])
    parser.add_argument("--results_dir", type=Path, default=Path("results_experiments"))
    parser.add_argument("--jobs", type=int, default=1, help="Maximum number of experiments to run in parallel.")
    parser.add_argument("--random_init_order", action="store_true",
                        help="Shuffle macro placement order for each greedy initialization.")
    parser.add_argument("--ripup_ratio_strategies", type=parse_list, default=["random_stage"],
                        help="Comma-separated rip-up ratio schedules: fixed, random_stage, decile_random")
    # ["fixed", "random_stage", "decile_random"]
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running later experiments if one fails.")
    args = parser.parse_args()

    valid_replace_orders = {"auto", "preserve", "node_id", "random", "area_group"}
    invalid_replace_orders = sorted(set(args.replace_orders) - valid_replace_orders)
    if invalid_replace_orders:
        parser.error(f"invalid replace order(s): {', '.join(invalid_replace_orders)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / timestamp
    logs_dir = run_dir / "logs"
    placements_dir = run_dir / "placements"

    experiments = []
    seeds = [int(seed) for seed in args.seeds]
    coarse_grids = [int(coarse_grid) for coarse_grid in args.coarse_grids]
    ripup_ratios = [float(ripup_ratio) for ripup_ratio in args.ripup_ratios]
    for benchmark, ripup_strategy, ripup_ratio, coarse_grid, seed, ripup_ratio_strategy, replace_order in itertools.product(
        args.benchmarks, args.ripup_strategies, ripup_ratios, coarse_grids, seeds, args.ripup_ratio_strategies, args.replace_orders
    ):
        if ripup_strategy == "bbox":
            for bbox_order in args.bbox_orders:
                experiments.append((benchmark, ripup_strategy, bbox_order, replace_order, ripup_ratio, coarse_grid, seed, ripup_ratio_strategy))
        else:
            experiments.append((benchmark, ripup_strategy, args.bbox_orders[0], replace_order, ripup_ratio, coarse_grid, seed, ripup_ratio_strategy))

    print(f"Total experiments: {len(experiments)}")
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        placements_dir.mkdir(parents=True, exist_ok=True)

    jobs = max(1, args.jobs)
    failures = []
    prepared = []
    for exp_idx, (benchmark, ripup_strategy, bbox_order, replace_order, ripup_ratio, coarse_grid, seed, ripup_ratio_strategy) in enumerate(experiments, start=1):
        name = experiment_name(
            benchmark,
            ripup_strategy,
            bbox_order,
            ripup_ratio,
            coarse_grid,
            seed,
            args.random_init_order,
            ripup_ratio_strategy,
            args.stage3_iters,
            args.stage3_order,
            args.stage3_mode,
            replace_order,
        )
        output_path = placements_dir / benchmark / name / f"{benchmark}_c2fplace.pl"
        log_path = logs_dir / f"{name}.log"
        command = build_command(args, benchmark, ripup_strategy, bbox_order, replace_order, ripup_ratio, coarse_grid, seed, ripup_ratio_strategy, output_path)
        command_text = " ".join(command)
        prepared.append((exp_idx, name, command, command_text, log_path, output_path))

    for exp_idx, name, _, command_text, _, _ in prepared:
        print(f"[{exp_idx}/{len(experiments)}] {name}")
        print(command_text)

    if args.dry_run:
        return

    if jobs == 1:
        for exp_idx, name, command, command_text, log_path, output_path in prepared:
            print(f"START [{exp_idx}/{len(experiments)}] {name}")
            returncode, log_path = run_experiment(command, command_text, log_path, output_path)
            if returncode != 0:
                failures.append((name, returncode, log_path))
                print(f"FAILED {name}, returncode={returncode}, log={log_path}")
                if not args.continue_on_error:
                    break
            else:
                print(f"DONE {name}, log={log_path}")
    else:
        print(f"Running up to {jobs} experiments in parallel")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {}
            for exp_idx, name, command, command_text, log_path, output_path in prepared:
                print(f"START [{exp_idx}/{len(experiments)}] {name}")
                future = executor.submit(run_experiment, command, command_text, log_path, output_path)
                futures[future] = (name, log_path)

            for future in as_completed(futures):
                name, log_path = futures[future]
                returncode, log_path = future.result()
                if returncode != 0:
                    failures.append((name, returncode, log_path))
                    print(f"FAILED {name}, returncode={returncode}, log={log_path}")
                else:
                    print(f"DONE {name}, log={log_path}")

    if failures:
        print("\nFailures:")
        for name, returncode, log_path in failures:
            print(f"  {name}: returncode={returncode}, log={log_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
