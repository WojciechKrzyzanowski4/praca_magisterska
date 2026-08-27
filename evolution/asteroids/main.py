import argparse

from .asteroid_runner import play_game, train_model, train_single_phase, visualize_phase
from .asteroid_trainer import DEAPConfig


def main():
    parser = argparse.ArgumentParser(description="Curriculum phases runner & visualizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for p in (1, 2, 3):
        rp = sub.add_parser(f"run_phase{p}", help=f"Train only phase {p} and save its genome in the Asteroids output directory")
        rp.add_argument("--seed", type=str, default=None, help="Optional path to seed genome .npy (defaults: previous phase output)")
        rp.add_argument("--out", type=str, default=None, help="Optional output path for the trained genome .npy")
        rp.add_argument("--hidden-size", type=int, default=48)
        rp.add_argument("--population", type=int, default=None, help="Override population size")
        rp.add_argument("--generations", type=int, default=None, help="Override number of generations")
        rp.add_argument("--cx-prob", type=float, default=None)
        rp.add_argument("--mut-prob", type=float, default=None)
        rp.add_argument("--mutation-sigma", type=float, default=None)
        rp.add_argument("--eval-max-steps", type=int, default=None)
        rp.add_argument("--random-seed", type=int, default=None)

    v = sub.add_parser("visualize", help="Visualize a single phase genome")
    v.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    v.add_argument("--genome", type=str, default=None, help="Path to genome .npy (defaults to the saved genome for the selected phase)")
    v.add_argument("--hidden-size", type=int, default=48)
    v.add_argument("--fps", type=int, default=60)
    v.add_argument(
        "--scenario",
        type=int,
        nargs="+",
        default=[2],
        help="One or more scenario offsets to show (default: 2)",
    )

    allp = sub.add_parser("run_all", help="Run P1 -> P2 -> P3 sequentially")
    allp.add_argument("--hidden-size", type=int, default=48)

    play = sub.add_parser("play", help="Play the game yourself")

    args = parser.parse_args()

    def cfg_from_args() -> DEAPConfig:
        base_cfg = DEAPConfig()
        if getattr(args, "population", None) is not None:
            base_cfg.population_size = args.population
        if getattr(args, "generations", None) is not None:
            base_cfg.generations = args.generations
        if getattr(args, "cx_prob", None) is not None:
            base_cfg.cx_prob = args.cx_prob
        if getattr(args, "mut_prob", None) is not None:
            base_cfg.mut_prob = args.mut_prob
        if getattr(args, "mutation_sigma", None) is not None:
            base_cfg.mutation_sigma = args.mutation_sigma
        if getattr(args, "eval_max_steps", None) is not None:
            base_cfg.eval_max_steps = args.eval_max_steps
        if getattr(args, "random_seed", None) is not None:
            base_cfg.random_seed = args.random_seed
        return base_cfg

    if args.cmd.startswith("run_phase"):
        phase = int(args.cmd[-1])
        cfg = cfg_from_args()
        train_single_phase(
            phase=phase,
            cfg=cfg,
            hidden_size=args.hidden_size,
            seed_path=args.seed,
            out_path=args.out,
            override_generations=args.generations,
        )
    elif args.cmd == "visualize":
        visualize_phase(
            phase=args.phase,
            genome_path=args.genome,
            hidden_size=args.hidden_size,
            fps=args.fps,
            scenario_cycle=args.scenario,
        )
    elif args.cmd == "run_all":
        cfg = cfg_from_args()
        train_model(cfg=cfg, hidden_size=args.hidden_size)

    elif args.cmd == "play":
        play_game()

if __name__ == "__main__":
    main()
