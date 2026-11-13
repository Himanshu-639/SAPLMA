"""
Master Runner for Replicating "The Internal State of an LLM Knows When It's Lying"
(Amos Azaria and Tom Mitchell, EMNLP Findings 2023 / arXiv:2304.13734v2)

Provides CLI commands to run every experiment and baseline in the paper.
"""

import os
import sys
import argparse

# Ensure implementation root is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from saplma.extract_activations import extract_and_save_embeddings
from saplma.evaluate_cross_topic import run_cross_topic_evaluation
from saplma.evaluate_generated import evaluate_on_generated
from saplma.permutation_test import run_permutation_test
from saplma.table5_analysis import run_table5_analysis
from baselines.bert_baseline import run_bert_baseline
from baselines.few_shot_baseline import run_few_shot_baseline
from baselines.it_is_true_baseline import run_it_is_true_baseline

def main():
    parser = argparse.ArgumentParser(
        description="Master reproduction runner for Azaria & Mitchell (EMNLP Findings 2023)."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="cross_topic",
        choices=[
            "extract",
            "cross_topic",
            "generated",
            "permutation",
            "table5",
            "bert",
            "few_shot",
            "it_is_true",
            "all"
        ],
        help="Which paper experiment to run."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="opt",
        choices=["opt", "llama3", "llama3-instruct", "llama2"],
        help="Which LLM backbone to evaluate: 'opt' (OPT-6.7b), 'llama3' (Llama-3.2-3B), 'llama3-instruct' (Llama-3.2-3B-Instruct), or 'llama2'."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "8bit", "4bit"],
        help="Quantization precision: 'fp16' (full), '8bit' (BitsAndBytes 8-bit), or '4bit' (BitsAndBytes NF4)."
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=3,
        help="Number of training repetitions (paper specifies 3 for cross-topic, 14 for generated)."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    if args.experiment == "extract":
        extract_and_save_embeddings(
            args.model,
            precision=args.precision,
            data_dir=args.data_dir,
            output_dir=args.embeddings_dir
        )
    elif args.experiment == "cross_topic":
        run_cross_topic_evaluation(
            model_name=args.model,
            precision=args.precision,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            repeat_each=args.rep
        )
    elif args.experiment == "generated":
        evaluate_on_generated(
            model_name=args.model,
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir,
            repeat_each=14 if args.rep == 3 else args.rep
        )
    elif args.experiment == "permutation":
        run_permutation_test(
            model_name=args.model,
            embeddings_dir=args.embeddings_dir
        )
    elif args.experiment == "table5":
        run_table5_analysis(
            embeddings_dir=args.embeddings_dir,
            output_dir=args.output_dir
        )
    elif args.experiment == "bert":
        run_bert_baseline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            repeat_each=args.rep
        )
    elif args.experiment == "few_shot":
        run_few_shot_baseline(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    elif args.experiment == "it_is_true":
        run_it_is_true_baseline(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    elif args.experiment == "all":
        print("\n=== RUNNING FULL PAPER REPRODUCTION SUITE ===\n")
        run_cross_topic_evaluation(args.model, args.embeddings_dir, args.output_dir, args.rep)
        evaluate_on_generated(args.model, args.embeddings_dir, args.output_dir, 14)
        run_permutation_test(args.model, -12, args.embeddings_dir)
        run_bert_baseline(args.data_dir, args.output_dir, args.rep)

if __name__ == "__main__":
    main()
