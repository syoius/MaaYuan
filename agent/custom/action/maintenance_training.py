"""Turn the existing training options into stock deficits and scoped subtask overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from custom.action.fixed_reward_reporting import parse_fixed_reward_plan


SET_COST = (180, 570, 1440, 2550, 3420, 2100)
BOOK_XP = {"bingshucanjuan": 100, "bingshuquanjuan": 1000, "liutaobingshu": 10000}
ELEMENT_GROUPS = (
    ("wind_fire", "风火历练"),
    ("earth_water", "地水历练"),
    ("yin_yang", "阴阳历练"),
)
CONFIG = "历练奖励配置"


def book_experience(counts: dict[str, int]) -> int:
    return sum(counts.get(key, 0) * amount for key, amount in BOOK_XP.items())


def merge_fields(dest: dict, source: dict):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            merge_fields(dest[key], value)
        else:
            dest[key] = deepcopy(value)


@dataclass
class Stage:
    name: str
    rewards: dict[str, int]
    override: dict


@dataclass
class Goal:
    kind: str
    label: str
    targets: dict[str, int]
    experience: int
    stages: list[Stage]

    def gaps(self, stock: dict[str, int]) -> dict[str, int] | None:
        if self.experience:
            xp = book_experience(stock)
            if xp >= self.experience:
                return {}
            if not BOOK_XP.keys() <= stock.keys():
                return None
            return {"经验": self.experience - xp}
        if not self.targets.keys() <= stock.keys():
            return None
        return {
            key: amount - stock[key]
            for key, amount in self.targets.items()
            if amount > stock[key]
        }

    def plan_batches(
        self, gaps: dict[str, int], unavailable: set[str]
    ) -> list[tuple[Stage, int]]:
        stages = [s for s in self.stages if s.name not in unavailable]
        if self.experience:
            yields = [{"经验": book_experience(s.rewards)} for s in stages]
        else:
            yields = [s.rewards for s in stages]

        def allocate(order):
            remaining = dict(gaps)
            batches = []
            for key in order:
                if remaining[key] <= 0:
                    continue
                candidates = [
                    (y.get(key, 0), i) for i, y in enumerate(yields) if y.get(key, 0)
                ]
                if not candidates:
                    continue
                rate, index = max(candidates)
                runs = (remaining[key] + rate - 1) // rate
                batches.append((stages[index], runs))
                for item, amount in yields[index].items():
                    if item in remaining:
                        remaining[item] -= amount * runs
            return batches

        plans = [allocate(gaps), allocate(reversed(gaps))]
        # With equal total runs, a single stage saves a second trip through the menus.
        for stage, rates in zip(stages, yields):
            if gaps and gaps.keys() <= rates.keys():
                runs = max(
                    (n + rates[key] - 1) // rates[key] for key, n in gaps.items()
                )
                plans.append([(stage, runs)])
        return min(
            plans, key=lambda batches: (sum(n for _, n in batches), len(batches))
        )


def load_goals(root: Path, config: dict) -> list[Goal]:
    path = root / "interface.json"
    if not path.exists():
        path = root / "assets/interface.json"
    options = json.loads(path.read_text("utf-8-sig"))["option"]
    goals = []
    for group_key, label in (*ELEMENT_GROUPS, ("experience", "经验历练")):
        xp = group_key == "experience"
        target = config.get("experience_target" if xp else f"{group_key}_sets", 0)
        # Earlier task configurations used a nonzero quantity as their opt-in.
        enabled = config.get(f"{group_key}_enabled", target != 0)
        if type(enabled) is not bool:
            raise ValueError(f"{label}开关无效")
        allowed = (0, 2386300, 4772600, 7158900) if xp else range(4)
        if type(target) is not int or target not in allowed:
            raise ValueError(f"{label}预设数量无效")
        if not enabled or not target:
            continue
        kind = "经验" if xp else "修为"
        if xp:
            group = {
                CONFIG: {
                    "attach": {
                        "acquisition_channel": "历练-经验",
                        "reward_item_ids": ["bingshucanjuan", "bingshuquanjuan"],
                    }
                }
            }
            targets = {}
        else:
            case = next(
                c for c in options["修为历练类别"]["cases"] if c["name"] == label
            )
            group = case["pipeline_override"]
            ids = group[CONFIG]["attach"]["reward_item_ids"]
            targets = {
                item: cost * target for item, cost in zip(ids, SET_COST, strict=True)
            }
        stages = []
        for level in options[f"{kind}历练级别"]["cases"]:
            override = deepcopy(group)
            merge_fields(override, level["pipeline_override"])
            plan = parse_fixed_reward_plan(
                {**override[CONFIG]["attach"], "sweep_count": 1}
            )
            stages.append(Stage(level["name"], dict(plan.rewards), override))
        goals.append(Goal(kind, label, targets, target if xp else 0, stages))
    return goals
