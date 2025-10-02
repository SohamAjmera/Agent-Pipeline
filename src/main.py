import argparse
from pathlib import Path

from .agentic_pipeline.config import Config
from .agentic_pipeline.controller.agent import AgentController
from .agentic_pipeline.logging_utils import console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini Agentic Pipeline")
    parser.add_argument("--query", type=str, required=True, help="User query to answer")
    parser.add_argument("--save-trace", action="store_true", help="Save trace JSON under results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_env()
    controller = AgentController(config=config)

    final_answer, trace, outfile = controller.run(query=args.query, save_trace=args.save_trace)

    console.rule("Final Answer")
    console.print(final_answer)
    console.rule()
    if outfile:
        console.print(f"Trace saved: {outfile}")


if __name__ == "__main__":
    main()


