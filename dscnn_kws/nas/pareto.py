from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParetoItem:
    uid: str
    acc: float
    mults: int
    params: int
    payload: dict


def _dominates(a: ParetoItem, b: ParetoItem) -> bool:
    no_worse = (a.acc >= b.acc) and (a.mults <= b.mults) and (a.params <= b.params)
    strictly = (a.acc > b.acc) or (a.mults < b.mults) or (a.params < b.params)
    return no_worse and strictly


def pareto_front(items: list[ParetoItem]) -> list[ParetoItem]:
    front: list[ParetoItem] = []
    for i, x in enumerate(items):
        dominated = False
        for j, y in enumerate(items):
            if i == j:
                continue
            if _dominates(y, x):
                dominated = True
                break
        if not dominated:
            front.append(x)
    return front
