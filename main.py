#!/bin/python

from time import sleep
import argparse
from typing import Any, List
from dataclasses import dataclass
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser, Namespace

from rich import inspect, print as p
from rich.console import Console
from rich.syntax import Syntax, SyntaxTheme

console = Console()

NP_TYPE = cp.byte

PROMPT = """PROMPT
In an interview, a human resource (HR) specialist asks candidates 20 questions with 5 possible multiple choice answers to each question.
What is the probability of the following candidates for success given the following assumptions:

 1. HR has the last answer always correct, candidates select answers at random.
 2. HR’s solution is in a random location and student selects an answer at random.

Note:
Assume the interview has only 3 questions, trace some (approx. 10) random candidates for each
  scenario but run a larger sample on the computer and do not respond philosophically without support.
"""


def title(str: Any):
    console.rule(str)


def log(str: Any):
    console.log(str)


def toPercent(num) -> str:
    return f'{np.round(num * 100, 2)}%'


def setupArgParser() -> Namespace:
    parser = ArgumentParser(description=PROMPT)
    parser.add_argument('-answer-type', metavar='A', type=str, default='random', choices=[
                        'last', 'random'], help='Which choice is correct for each question')
    # parser.add_argument('-trials', metavar='T', type=int, default=130000000, # Max size for last
    parser.add_argument('-trials', metavar='T', type=int, default=120000000,  # Max size for random
                        help='Number of trials to run (job candidates taking the exam)')
    parser.add_argument('-questions', metavar='Q', type=int, default=3, help='Number of question per exam')
    parser.add_argument('-choices', metavar='C', type=int, default=5, help='Number of choices per question')
    return parser.parse_args()


def displayResults(answers: np.ndarray, attempt: np.ndarray, correct: np.ndarray):
    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(answers)   # type: ignore
    axes[0].set_title('Answers')   # type: ignore
    axes[0].set_ylabel('Questions')   # type: ignore
    axes[0].set_xticks([])   # type: ignore
    axes[0].set_yticks([])   # type: ignore

    axes[1].imshow(attempt)   # type: ignore
    axes[1].set_title('Attempt')   # type: ignore
    axes[1].set_xlabel('Choices')   # type: ignore
    axes[1].set_xticks([])   # type: ignore
    axes[1].set_yticks([])   # type: ignore

    axes[2].imshow(correct)   # type: ignore
    axes[2].set_title('Correct')   # type: ignore
    axes[2].axis('off')   # type: ignore

    sns.set()
    fig.suptitle(f'Exam Results [Score={(correct.sum()/len(correct)) * 100}%]')
    fig.tight_layout()  # type: ignore
    plt.show()
    pass


def main(args: Namespace):
    trials = args.trials
    q_cnt = args.questions
    c_cnt = args.choices
    exams_shape = (trials, q_cnt)

    with console.status('Computing on GPU'):
        #! Create exams
        if args.answer_type == 'last':
            exams = cp.full(exams_shape, c_cnt - 1, dtype=NP_TYPE)
        elif args.answer_type == 'random':
            exams = cp.random.choice(c_cnt, exams_shape).astype(NP_TYPE)
        else:
            log("[bold red] Incorrect answer strategy!")
            exit(-1)

        #! Create attempts
        attempts = cp.random.choice(c_cnt, exams_shape).astype(NP_TYPE)

        grades = (exams == attempts)
        correct = grades.sum(axis=1).astype(NP_TYPE)
        scores = (correct / q_cnt) * 100

    # displayResults(exam, trial, graded)
    # log(f'Score = {score:0.2f}%')
    # log(locals())

    # Move to CPU and sync
    cp.cuda.Stream.null.synchronize()
    scores = scores.get()

    passing = (scores > 70).sum()
    trace = np.random.choice(scores, 10).round(2)
    simulation_result = (passing / len(scores))

    title('[bold purple]Results')
    p(f'[green]Scores (trace 10): [blue][{"] [".join(trace.astype(str))}]')
    p(f'[green]Percent of Successes (> 70%): [blue]{toPercent(simulation_result)}')  # type: ignore
    p(f'[green]Average: [blue] {scores.mean()}')

    formula = '    (1 / c_cnt) ** np.ceil(q_cnt * 0.7)'
    formula_result = eval(formula)
    syntax = Syntax(formula, "python")
    error = (simulation_result - formula_result) / formula_result

    title('[bold yellow]Conclusion')
    p('Each question is independent, therefor:')
    p('    P(q1,q2,q3)\n    = P(q1)P(q2)P(q3)')
    p('Each question has the same probability of being answered correctly')
    p('    = P(q1)^N\n    where N is the number of questions')
    p('and in this case where success is > 70%...')
    p(syntax, f'    = {toPercent(formula_result)}')
    p('The simulation results support this conclusion:')
    p(f'Experimental Error = {toPercent(error)}')


if __name__ == '__main__':
    args = setupArgParser()

    title('[bold cyan]Assumptions')
    p('- Interview has 3 questions instead of 20')
    p('- Success means a 70% or higher')
    p('  - That means 3/3 given the above assumption')

    title('[bold blue]Parameters')
    inspect(args, title='Simulation Params', docs=False)

    main(args)
