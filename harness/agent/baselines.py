"""Reference agents: the cheapest task-quality test a benchmark can run.

Two agents, both implementing the same ``Agent`` interface as the LLM
controller, so they are drop-in and can run in CI:

- **OracleAgent** proves a task is *solvable*. If the oracle fails, the task is
  broken and every model's zero on it says nothing about the models. It also
  supplies the efficiency denominator: ``agent_steps / oracle_steps`` is the
  manipulation analogue of SPL's shortest-path ratio, which is otherwise
  undefined because there is no geodesic optimum for a pick-and-place.
- **NullAgent** does nothing. If it *passes*, the success check is vacuous --
  the task is satisfied by the initial state.

The oracle is written against the environment's object-aware query API
(``list_objects`` / ``get_object_pos`` / ``get_goal_pos`` / ``get_ee_pos``)
rather than against any one simulator, so it transfers to any env that exposes
them.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from harness.agent.base import Agent
from harness.envs.base import Env
from harness.types import Action, Episode

_EPS = 1e-9


class NullAgent(Agent):
    """Takes no action. A task this passes has a vacuous success check."""

    name = "null_agent"

    def __init__(self, *, steps: int = 1, **kwargs) -> None:
        self._steps = max(0, int(steps))

    def run(self, env: Env, *, seed: Optional[int] = None) -> Episode:
        ep = Episode(metadata={"mode": "null", "agent": self.name, "env": env.name})
        obs = env.reset(seed=seed)
        ep.observations.append(obs)
        for _ in range(self._steps):
            result = env.step(Action(kind="noop", comment="null agent"))
            ep.actions.append(Action(kind="noop", comment="null agent"))
            ep.rewards.append(result.reward)
            ep.infos.append(result.info)
            ep.observations.append(result.obs)
            if result.success or result.terminated or result.truncated:
                break
        ep.success = bool(ep.infos and ep.infos[-1].get("success", False))
        ep.total_reward = sum(ep.rewards)
        ep.metadata["steps"] = ep.steps
        return ep


class OracleAgent(Agent):
    """Scripted solver driven by the env's own object/goal queries.

    Handles the task families the 2D suite defines: reach-style goals (drive the
    end-effector to a target, routing around obstacles) and place-style goals
    (pick each object, carry it to its goal, release). Both reduce to one
    primitive -- move to a point without entering an obstacle -- plus gripper
    toggles.
    """

    name = "oracle"

    def __init__(
        self,
        *,
        max_steps: int = 400,
        step_size: float = 0.06,
        reach_tol: float = 0.02,
        obstacle_margin: float = 0.035,
        **kwargs,
    ) -> None:
        self._max_steps = max_steps
        self._step = step_size
        self._tol = reach_tol
        self._margin = obstacle_margin

    # -- geometry -------------------------------------------------------- #
    def _obstacles(self, env: Env) -> list[tuple[np.ndarray, float]]:
        spec = getattr(env, "task_spec", None)
        out = []
        for o in list(getattr(spec, "obstacles", []) or []):
            out.append((np.asarray(o["pos"], dtype=float)[:2], float(o.get("radius", 0.1))))
        return out

    def _avoid(self, pos: np.ndarray, delta: np.ndarray, obstacles) -> np.ndarray:
        """Steer a step tangentially when it would enter an obstacle.

        Local avoidance is enough for the convex single-obstacle layouts the
        suite generates; it deliberately does not attempt general planning.
        """
        nxt = pos + delta
        for centre, radius in obstacles:
            keep_out = radius + self._margin
            if float(np.linalg.norm(nxt - centre)) >= keep_out:
                continue
            # rotate the step to run around the obstacle rather than into it
            away = pos - centre
            n = float(np.linalg.norm(away))
            if n < _EPS:
                away, n = np.array([1.0, 0.0]), 1.0
            radial = away / n
            tangent = np.array([-radial[1], radial[0]])
            if float(np.dot(tangent, delta)) < 0:
                tangent = -tangent
            # push outward if already inside the keep-out ring
            outward = radial * max(0.0, keep_out - n)
            steered = tangent * float(np.linalg.norm(delta)) + outward
            nrm = float(np.linalg.norm(steered))
            return steered / nrm * float(np.linalg.norm(delta)) if nrm > _EPS else delta
        return delta

    # -- primitives ------------------------------------------------------ #
    @staticmethod
    def _segment_clear(p0, p1, obstacles, margin: float) -> bool:
        """True when the straight segment p0->p1 misses every obstacle."""
        p0, p1 = np.asarray(p0, dtype=float)[:2], np.asarray(p1, dtype=float)[:2]
        seg = p1 - p0
        seg_len2 = float(np.dot(seg, seg))
        for centre, radius in obstacles:
            if seg_len2 < _EPS:
                dist = float(np.linalg.norm(p0 - centre))
            else:
                t = max(0.0, min(1.0, float(np.dot(centre - p0, seg)) / seg_len2))
                dist = float(np.linalg.norm(p0 + t * seg - centre))
            if dist < radius + margin:
                return False
        return True

    def _move_to(self, env, ep, target, obstacles, *, gripper=None) -> bool:
        """Drive the end-effector to ``target``; True once within tolerance.

        Takes a single absolute move when the path is clear. An efficiency
        denominator has to use the cheapest primitive the action space offers,
        or every agent that uses a coarser one scores better than "optimal".
        Incremental steering is reserved for routing around obstacles.
        """
        target = np.asarray(target, dtype=float)[:2]
        here = np.asarray(env.get_ee_pos(), dtype=float)[:2]
        if self._segment_clear(here, target, obstacles, self._margin):
            return self._apply(
                env, ep,
                Action(kind="ee_pose", value=target.astype(np.float32),
                       gripper=gripper, comment="oracle move_to"),
            ) or bool(
                float(np.linalg.norm(np.asarray(env.get_ee_pos(), dtype=float)[:2] - target))
                <= self._tol
            )

        for _ in range(self._max_steps):
            if ep.steps >= self._max_steps:
                return False
            pos = np.asarray(env.get_ee_pos(), dtype=float)[:2]
            gap = target - pos
            dist = float(np.linalg.norm(gap))
            if dist <= self._tol:
                return True
            delta = gap / dist * min(self._step, dist)
            delta = self._avoid(pos, delta, obstacles)
            if not self._apply(env, ep, Action(kind="ee_delta", value=delta.astype(np.float32),
                                               gripper=gripper, comment="oracle move")):
                return False
        return False

    def _set_gripper(self, env, ep, value: float) -> bool:
        return self._apply(
            env, ep,
            Action(kind="noop", gripper=float(value),
                   comment="oracle grasp" if value > 0.5 else "oracle release"),
        )

    def _apply(self, env, ep, action: Action) -> bool:
        """Step the env and record it; False when the episode is over."""
        result = env.step(action)
        ep.actions.append(action)
        ep.rewards.append(result.reward)
        ep.infos.append(result.info)
        ep.observations.append(result.obs)
        if result.info.get("collided"):
            return False
        return not (result.success or result.terminated or result.truncated)

    # -- plan ------------------------------------------------------------ #
    def _place_pairs(self, env: Env) -> list[tuple[str, np.ndarray]]:
        """(object, destination) pairs the task requires, in a sane order."""
        spec = getattr(env, "task_spec", None)
        pairs: list[tuple[str, np.ndarray]] = []

        # explicit object -> goal assignments
        for o in list(getattr(spec, "objects", []) or []):
            target = o.get("target")
            if target:
                dest = env.get_goal_pos(target)
                if dest is not None:
                    pairs.append((o["name"], np.asarray(dest, dtype=float)))

        # stacking has no goals dict: the destination is another object
        if not pairs and getattr(spec, "kind", "") == "stack":
            names = env.list_objects()
            if "top" in names and "base" in names:
                base = env.get_object_pos("base")
                if base is not None:
                    pairs.append(("top", np.asarray(base, dtype=float)))
        return pairs

    def run(self, env: Env, *, seed: Optional[int] = None) -> Episode:
        ep = Episode(metadata={"mode": "oracle", "agent": self.name, "env": env.name})
        obs = env.reset(seed=seed)
        ep.observations.append(obs)

        obstacles = self._obstacles(env)
        spec = getattr(env, "task_spec", None)
        kind = getattr(spec, "kind", "") or ""
        ep.metadata["oracle_plan"] = []

        try:
            if kind in ("reach", "reach_avoid") or getattr(spec, "ee_target", None) is not None:
                ep.metadata["oracle_plan"] = ["move_to_ee_target"]
                self._move_to(env, ep, np.asarray(spec.ee_target, dtype=float), obstacles)
            else:
                for name, dest in self._place_pairs(env):
                    ep.metadata["oracle_plan"] += [f"pick({name})", f"place({name})"]
                    here = env.get_object_pos(name)
                    if here is None:
                        continue
                    if not self._move_to(env, ep, here, obstacles):
                        break
                    if not self._set_gripper(env, ep, 1.0):
                        break
                    # carry: the grasped object tracks the end-effector, so the
                    # obstacle test applies to the pair, not just the gripper
                    if not self._move_to(env, ep, dest, obstacles, gripper=1.0):
                        break
                    if not self._set_gripper(env, ep, 0.0):
                        break
        except Exception as e:  # noqa: BLE001 - a broken oracle is a finding, not a crash
            ep.metadata["oracle_error"] = f"{type(e).__name__}: {e}"

        ep.success = bool(
            (ep.infos and ep.infos[-1].get("success", False)) or env.is_success()
        )
        ep.total_reward = sum(ep.rewards)
        ep.metadata["steps"] = ep.steps
        return ep


class ScriptedPickPlaceAgent(Agent):
    """Told which object to move where, then executes a fixed 3-D pick-and-place.

    This is the solvability probe for envs whose goals are stated in language
    rather than as coordinates -- RoboLab says "put the cube in the bowl", so the
    generic OracleAgent (which reads ``list_goals``) has nothing to aim at. Being
    *given* the source and target is the point: it answers "can this task be
    solved through this action interface at all?", which is the question a 0%
    model score cannot answer on its own. A task this fails is not evidence about
    any model.

    Written against the env's query API and absolute end-effector moves, so it
    works anywhere both exist.
    """

    name = "scripted_pick_place"

    def __init__(self, *, source: str = "", target: str = "", max_steps: int = 260,
                 hover: float = 0.12, tol: float = 0.02, settle: int = 6,
                 phase_budget: int = 32, top_down: bool = True, **kwargs) -> None:
        self._source = source
        self._target = target
        self._max_steps = int(max_steps)
        self._hover = float(hover)       # approach height above an object
        self._tol = float(tol)           # position tolerance before advancing
        self._settle = int(settle)       # steps to hold after a gripper change
        #: Steps allowed per waypoint. This must be generous enough for the arm to
        #: actually ARRIVE: a controller that clips each step to a few centimetres
        #: needs tens of steps to cross a 12cm approach, and a phase that times out
        #: early closes the gripper while the arm is still descending. Diagnostic
        #: for that failure: total steps equal to waypoints x (budget + settle)
        #: exactly, meaning no phase ever reached its tolerance.
        self._phase_budget = int(phase_budget)
        #: Command a downward approach where the env offers one. A wrist left in
        #: its start orientation cannot close on an object lying on a table, so
        #: without this the probe reaches the right place and still never grasps.
        self._top_down = bool(top_down)

    def _pos(self, env: Env, name: str):
        """Locate a named thing, whether the env calls it an object or a goal.

        RoboLab's containers are scene objects ("the bowl"); tabletop's targets
        are goals with coordinates. Accepting both keeps one probe for both.
        """
        for attr in ("get_object_pos", "get_goal_pos"):
            getter = getattr(env, attr, None)
            if getter is None:
                continue
            try:
                p = getter(name)
            except Exception:  # noqa: BLE001 - unknown name for this getter
                continue
            if p is not None:
                return np.asarray(p, dtype=np.float32).ravel()[:3]
        return None

    @staticmethod
    def _as3(p):
        """Pad to three dimensions so the waypoint arithmetic is uniform.

        Planar envs report (x, y); the hover offset then has nothing to act on,
        which is correct rather than special-cased.
        """
        a = np.zeros(3, dtype=np.float32)
        a[:min(3, len(p))] = p[:3]
        return a

    def run(self, env: Env, *, seed: Optional[int] = None) -> Episode:  # noqa: C901
        ep = Episode(metadata={"mode": "scripted", "agent": self.name, "env": env.name,
                               "source": self._source, "target": self._target})
        obs = env.reset(seed=seed)
        ep.observations.append(obs)

        src_native = self._pos(env, self._source)
        dst_native = self._pos(env, self._target)
        src = None if src_native is None else self._as3(src_native)
        dst = None if dst_native is None else self._as3(dst_native)
        if src is None or dst is None:
            ep.metadata["error"] = (
                f"cannot locate source {self._source!r} or target {self._target!r}; "
                f"available: {getattr(env, 'list_objects', lambda: [])()}")
            ep.success = False
            ep.metadata["steps"] = 0
            return ep

        # emit actions in the env's own dimensionality: a planar env expects (x, y)
        dim = int(len(src_native))
        grasp_quat = None
        if self._top_down:
            getter = getattr(env, "grasp_orientation", None)
            if getter is not None:
                grasp_quat = np.asarray(getter(), dtype=np.float32).ravel()[:4]
        up = np.array([0.0, 0.0, self._hover], dtype=np.float32)
        # (target position, gripper) -- the gripper is commanded on every step,
        # because a binary gripper action that is not repeated is released by the
        # next command that omits it.
        waypoints = [
            (src + up, 0.0),   # hover over the object, open
            (src, 0.0),        # descend onto it
            (src, 1.0),        # close
            (src + up, 1.0),   # lift it clear
            (dst + up, 1.0),   # carry it over the target
            (dst + up * 0.4, 1.0),  # lower toward the target
            (dst + up * 0.4, 0.0),  # release
        ]

        done = False
        for point, grip in waypoints:
            if done:
                break
            for _ in range(self._phase_budget):
                if ep.steps >= self._max_steps:
                    break
                value = np.asarray(point, dtype=np.float32)[:dim]
                if grasp_quat is not None and dim >= 3:
                    value = np.concatenate([value, grasp_quat])
                action = Action(kind="ee_pose", value=value, gripper=grip,
                                comment=f"scripted -> {point.round(3)}")
                result = env.step(action)
                ep.actions.append(action)
                ep.rewards.append(result.reward)
                ep.infos.append(result.info)
                ep.observations.append(result.obs)
                if result.success or result.terminated or result.truncated:
                    done = True
                    break
                ee = getattr(env, "get_ee_pos", lambda: None)()
                if ee is not None:
                    ee = self._as3(np.asarray(ee, dtype=np.float32).ravel())
                    if float(np.linalg.norm(ee[:dim] - point[:dim])) <= self._tol:
                        break
            # hold after a gripper transition so the physics can settle
            for _ in range(self._settle if grip in (0.0, 1.0) else 0):
                if done or ep.steps >= self._max_steps:
                    break
                value = np.asarray(point, dtype=np.float32)[:dim]
                if grasp_quat is not None and dim >= 3:
                    value = np.concatenate([value, grasp_quat])
                action = Action(kind="ee_pose", value=value, gripper=grip,
                                comment="settle")
                result = env.step(action)
                ep.actions.append(action)
                ep.rewards.append(result.reward)
                ep.infos.append(result.info)
                ep.observations.append(result.obs)
                if result.success or result.terminated or result.truncated:
                    done = True

        ep.success = bool(ep.infos and ep.infos[-1].get("success", False)) or bool(
            getattr(env, "is_success", lambda: False)())
        ep.total_reward = sum(ep.rewards)
        ep.metadata["steps"] = ep.steps
        return ep


def get_baseline_agent(name: str, **kwargs) -> Agent:
    """Factory for the reference agents."""
    key = (name or "").strip().lower()
    if key in ("oracle", "oracle_agent"):
        return OracleAgent(**kwargs)
    if key in ("null", "null_agent", "nop", "noop"):
        return NullAgent(**kwargs)
    if key in ("scripted_pick_place", "pick_place", "scripted"):
        return ScriptedPickPlaceAgent(**kwargs)
    raise KeyError(
        f"Unknown baseline agent '{name}'. Use 'oracle', 'null' or 'scripted_pick_place'.")
