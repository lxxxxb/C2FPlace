import os
import time
import argparse
from place_db import PlaceDB
from c2fplace_ea import C2FPlaceEA


def main():
    parser = argparse.ArgumentParser(description="C2FPlace-EA: Coarse-to-Fine Hierarchical Macro Placement via Evolutionary Search")
    parser.add_argument("--benchmark", type=str, default="adaptec1",
                        help="Benchmark directory name under benchmark/ (default: adaptec1; data must be downloaded separately)")
    parser.add_argument("--coarse_grid", type=int, default=256, help="Coarse grid size (default: 256)")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size (default: 20)")
    parser.add_argument("--stage1_iters", type=int, default=5000, help="Stage 1 iterations (default: 5000)")
    parser.add_argument("--stage2_iters", type=int, default=5000, help="Stage 2 iterations (default: 5000)")
    parser.add_argument("--stage3_iters", type=int, default=50, help="Stage 3 iterations (default: 50)")
    parser.add_argument("--stage3_mode", type=str, default="both_2", choices=["coarse", "fine", "both_1", "both_2"],
                        help="Stage 3: both_1 runs all coarse iterations then all fine iterations; both_2 runs coarse then fine each iteration (default: both_2)")
    parser.add_argument("--stage3_order", type=str, default="node_id", choices=["hpwl", "random", "node_id"],
                        help="Stage 3 macro order: hpwl, random, or node_id (default: node_id)")
    parser.add_argument("--ripup_ratio", type=float, default=0.2, help="Ripup ratio for destruction-reconstruction (default: 0.2)")
    parser.add_argument("--ripup_strategy", type=str, default="random", choices=["random", "bbox"],
                        help="Rip-up selection strategy: random or bbox (default: random)")
    # ["random", "bbox"]
    parser.add_argument("--bbox_order", type=str, default="random", choices=["random", "count_desc", "count_asc", "area_asc", "node_id"],
                        help="Ranking used to select bbox rip-up macros; node_id follows topology order (default: random)")
    parser.add_argument("--replace_order", type=str, default="auto", choices=["auto", "preserve", "node_id", "random", "area_group"],
                        help="Order for re-placing selected macros; area_group uses descending area groups with random order inside each group")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--random_init_order", action="store_true",
                        help="Shuffle macro placement order for each greedy initialization")
    parser.add_argument("--ripup_ratio_strategy", type=str, default="fixed",
                        choices=["fixed", "random_stage", "decile_random"],
                        help="Rip-up ratio schedule; decile_random samples [0.9,1.0] down to [0.0,0.1] across Stage 1")
    parser.add_argument("--output", type=str, default=None, help="Output .pl file path")
    args = parser.parse_args()

    print("=" * 70)
    print("C2FPlace-EA: Coarse-to-Fine Macro Placement")
    print("=" * 70)

    print(f"\n[1/4] Loading benchmark: {args.benchmark}")
    placedb = PlaceDB(args.benchmark)
    placedb.debug_str()

    print(f"\n[2/4] Initializing C2FPlace-EA")
    print(f"  - Coarse grid size: {args.coarse_grid}x{args.coarse_grid}")
    print(f"  - Population size: {args.pop_size}")
    print(f"  - Stage 1 iterations: {args.stage1_iters}")
    print(f"  - Stage 2 iterations: {args.stage2_iters}")
    print(f"  - Stage 3 iterations: {args.stage3_iters}")
    print(f"  - Stage 3 mode: {args.stage3_mode}")
    print(f"  - Stage 3 order: {args.stage3_order}")
    print(f"  - Ripup ratio: {args.ripup_ratio}")
    print(f"  - Ripup strategy: {args.ripup_strategy}")
    print(f"  - BBox order: {args.bbox_order}")
    print(f"  - Replace order: {args.replace_order}")
    print(f"  - Random initialization order: {args.random_init_order}")
    print(f"  - Ripup ratio strategy: {args.ripup_ratio_strategy}")
    print(f"  - Random seed: {args.seed}")

    ea = C2FPlaceEA(
        placedb,
        coarse_grid_size=args.coarse_grid,
        pop_size=args.pop_size,
        stage1_iters=args.stage1_iters,
        stage2_iters=args.stage2_iters,
        stage3_iters=args.stage3_iters,
        stage3_mode=args.stage3_mode,
        stage3_order=args.stage3_order,
        ripup_ratio=args.ripup_ratio,
        ripup_strategy=args.ripup_strategy,
        bbox_order=args.bbox_order,
        replace_order=args.replace_order,
        random_init_order=args.random_init_order,
        ripup_ratio_strategy=args.ripup_ratio_strategy,
        seed=args.seed
    )

    if args.output:
        output_path = args.output
    else:
        result_tag = args.ripup_strategy
        if args.ripup_strategy == "bbox":
            result_tag = f"bbox_{args.bbox_order}"
        if args.ripup_ratio_strategy == "random_stage":
            result_tag = f"{result_tag}_randratio"
        elif args.ripup_ratio_strategy == "decile_random":
            result_tag = f"{result_tag}_decilerand"
        else:
            ratio_tag = f"r{args.ripup_ratio:g}".replace(".", "p")
            result_tag = f"{result_tag}_{ratio_tag}"
        if args.replace_order != "auto":
            result_tag = f"{result_tag}_repl{args.replace_order}"
        output_dir = f"results/{args.benchmark}/{result_tag}/seed{args.seed}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{args.benchmark}_c2fplace.pl"

    print(f"\n[3/4] Running evolutionary search...")
    start_time = time.time()
    best = ea.run(verbose=True, save_path=output_path)
    elapsed = time.time() - start_time

    validation = ea.validate_individual(best)

    print(f"\n[4/4] Final Results")
    print("-" * 50)
    print(f"HPWL:       {best.hpwl:.4e}")
    print(f"Overlap:    {best.overlap:.4e}")
    print(f"Congestion: {best.congestion:.4e}")
    print(f"Fitness:    {best.fitness:.4e}")
    print(f"Runtime:    {elapsed:.2f}s")
    print(f"Valid:      {validation.is_valid}")
    if validation.errors:
        print("Validation errors:")
        for error in validation.errors[:10]:
            print(f"  - {error}")
        if len(validation.errors) > 10:
            print(f"  - ... {len(validation.errors) - 10} more")
    print("-" * 50)

    print(f"\nSaving placement to: {output_path}")
    saved_path = ea._save_best_individual(best, output_path, placedb.benchmark, include_hpwl=True)
    print(f"Done! Saved to: {saved_path}")


if __name__ == "__main__":
    main()
