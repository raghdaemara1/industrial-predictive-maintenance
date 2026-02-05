from __future__ import annotations

import argparse

from pm.common.logging import setup_logging
from pm.pipelines.build_dataset import run as build_dataset
from pm.pipelines.evaluate import run as evaluate_run
from pm.pipelines.score_batch import run as score_batch
from pm.pipelines.train_model import run as train_model


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(prog="pm.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--config", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--model", required=True)

    eval_cmd = sub.add_parser("evaluate")
    eval_cmd.add_argument("--run-id", required=True)

    score = sub.add_parser("score-batch")
    score.add_argument("--run-id", required=True)
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "build-dataset":
        build_dataset(args.config)
    elif args.command == "train":
        run_id = train_model(args.config, args.model)
        print(run_id)
    elif args.command == "evaluate":
        evaluate_run(args.run_id)
    elif args.command == "score-batch":
        score_batch(args.run_id, args.input, args.output)


if __name__ == "__main__":
    main()
