import argparse

from .snake_runner import play_game, train_model, visualize_result
from .snake_trainer import DEAPConfig


def main():
    parser = argparse.ArgumentParser(description="Snake trainer and visualizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train a snake agent and save the best genome")
    train.add_argument("--seed", type=str, default=None, help="Optional path to a seed genome .npy")
    train.add_argument("--out", type=str, default=None, help="Optional output path for the trained genome .npy")
    train.add_argument("--hidden-size", type=int, default=48)
    train.add_argument("--population", type=int, default=None, help="Override population size")
    train.add_argument("--generations", type=int, default=None, help="Override number of generations")
    train.add_argument("--cx-prob", type=float, default=None)
    train.add_argument("--mut-prob", type=float, default=None)
    train.add_argument("--mutation-sigma", type=float, default=None)
    train.add_argument("--eval-max-steps", type=int, default=None)
    train.add_argument("--eval-episodes", type=int, default=None)
    train.add_argument("--random-seed", type=int, default=None)

    vis = sub.add_parser("visualize", help="Visualize a trained snake genome")
    vis.add_argument("--genome", type=str, default=None, help="Path to genome .npy (defaults to snake output)")
    vis.add_argument("--hidden-size", type=int, default=48)
    vis.add_argument("--fps", type=int, default=20)
    vis.add_argument("--max-steps", type=int, default=None)
    vis.add_argument("--seed", type=int, default=None, help="Optional fixed food-seed for deterministic playback")

    sub.add_parser("play", help="Play the game yourself")

    args = parser.parse_args()

    def cfg_from_args() -> DEAPConfig:
        cfg = DEAPConfig()
        if getattr(args, "population", None) is not None:
            cfg.population_size = args.population
        if getattr(args, "generations", None) is not None:
            cfg.generations = args.generations
        if getattr(args, "cx_prob", None) is not None:
            cfg.cx_prob = args.cx_prob
        if getattr(args, "mut_prob", None) is not None:
            cfg.mut_prob = args.mut_prob
        if getattr(args, "mutation_sigma", None) is not None:
            cfg.mutation_sigma = args.mutation_sigma
        if getattr(args, "eval_max_steps", None) is not None:
            cfg.eval_max_steps = args.eval_max_steps
        if getattr(args, "eval_episodes", None) is not None:
            cfg.eval_episodes = args.eval_episodes
        if getattr(args, "random_seed", None) is not None:
            cfg.random_seed = args.random_seed
        return cfg

    if args.cmd == "train":
        train_model(
            cfg=cfg_from_args(),
            hidden_size=args.hidden_size,
            seed_path=args.seed,
            out_path=args.out,
        )
    elif args.cmd == "visualize":
        visualize_result(
            genome_path=args.genome,
            hidden_size=args.hidden_size,
            fps=args.fps,
            max_steps=args.max_steps if args.max_steps is not None else DEAPConfig().eval_max_steps,
            seed=args.seed,
        )
    elif args.cmd == "play":
        play_game()

if __name__ == "__main__":
    main()
