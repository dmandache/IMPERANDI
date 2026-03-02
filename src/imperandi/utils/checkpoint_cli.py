from __future__ import annotations

import argparse


def add_checkpoint_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_rows: int,
    default_sec: int,
    include_resume: bool = True,
    include_strict: bool = True,
) -> None:
    parser.add_argument(
        "--checkpoint_every_rows",
        type=int,
        default=default_rows,
        help="Flush checkpoint files every N processed rows.",
    )
    parser.add_argument(
        "--checkpoint_every_sec",
        type=int,
        default=default_sec,
        help="Flush checkpoint files every T seconds.",
    )
    if include_resume:
        parser.add_argument(
            "--no_resume",
            action="store_false",
            dest="resume",
            default=True,
            help="Disable resume from matching checkpoint state.",
        )
    if include_strict:
        parser.add_argument(
            "--strict_resume",
            action="store_true",
            default=False,
            help="Use content hashing for input fingerprint when resuming.",
        )
