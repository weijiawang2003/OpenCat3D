"""Minimal cat skeleton metadata for the placeholder sandbox."""

from __future__ import annotations

JOINT_NAMES: tuple[str, ...] = (
    "root",
    "chest",
    "head",
    "ear_l",
    "ear_r",
    "fore_l_paw",
    "fore_r_paw",
    "hind_l_paw",
    "hind_r_paw",
    "tail_base",
    "tail_tip",
)

JOINT_PARENTS: tuple[int, ...] = (-1, 0, 1, 2, 2, 1, 1, 0, 0, 0, 9)

