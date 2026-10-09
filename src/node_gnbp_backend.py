"""Owner-computes GN/BP with synchronous packets and node-local math kernels.

Only transport/scheduling lives here; all residuals, Jacobians, prior, Schur
messages, damping and tangent updates are the existing backend's kernels.
Threads own disjoint UAVNode instances. Every cross-node input is a read-only
snapshot, and each map call is a barrier (also in serial mode).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import time

import numpy as np

from src import multi_agent_bearing_gabp as kernel

FactorId = tuple[str, str, int]
MessageValue = tuple[np.ndarray, np.ndarray]


def readonly(array: np.ndarray) -> np.ndarray:
    """Return an isolated immutable array for a stage snapshot."""
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PoseSnapshot:
    """Read-only pose packet; neither attributes nor array entries can be changed."""
    rotation: np.ndarray
    translation: np.ndarray


def freeze_states(states: Mapping[str, kernel.PoseState]) -> Mapping[str, PoseSnapshot]:
    """Copy pose arrays; workers may neither replace entries nor edit arrays."""
    return MappingProxyType({node: PoseSnapshot(readonly(s.rotation), readonly(s.translation))
                             for node, s in states.items()})


@dataclass(frozen=True)
class MessagePacket:
    """One canonical Gaussian message, routed between variable and factor owners."""
    window_id: str
    gn_round: int
    bp_round: int
    source: str
    destination: str
    factor_id: FactorId
    kind: str
    Lambda: np.ndarray
    eta: np.ndarray

    def __post_init__(self) -> None:
        if self.kind not in ("v2f", "f2v"):
            raise ValueError("Unknown message direction.")
        if np.shape(self.Lambda) != (6, 6) or np.shape(self.eta) != (6,):
            raise ValueError("Messages require (6,6) information and (6,) vector.")
        if not np.all(np.isfinite(self.Lambda)) or not np.all(np.isfinite(self.eta)):
            raise ValueError("Nonfinite message.")
        object.__setattr__(self, "Lambda", readonly(self.Lambda))
        object.__setattr__(self, "eta", readonly(self.eta))


@dataclass(frozen=True)
class FactorDescriptor:
    """Remote incidence metadata; contains no remote mutable factor object."""
    factor_id: FactorId
    owner: str
    endpoints: tuple[str, str]


@dataclass
class ObservationFactor:
    """One original directed measurement, maintained exclusively by its observer."""
    factor_id: FactorId
    measurement: kernel.AnchorBearingFactor | kernel.PairwiseBearingFactor
    linearized: kernel.UnaryInfoFactor | kernel.PairwiseInfoFactor | None = None

    @property
    def owner(self) -> str:
        """Return measurement observer (also the sole compute owner)."""
        return self.factor_id[0]

    def relinearize(self, states: Mapping[str, kernel.PoseState]) -> None:
        """Rebuild this factor in the current GN tangent space using legacy math."""
        if isinstance(self.measurement, kernel.AnchorBearingFactor):
            self.linearized = kernel._linearize_anchor_factor(self.measurement, states)
        else:
            self.linearized = kernel._linearize_pairwise_factor(self.measurement, states)


@dataclass
class UAVNode:
    """Local trajectories, owned factors, remote incidence and double-buffered BP."""
    node_id: str
    local_trajectory: np.ndarray | None = None
    state: kernel.PoseState | None = None
    initial_state: kernel.PoseState | None = None
    owned_factors: dict[FactorId, ObservationFactor] = field(default_factory=dict)
    incident_factor_ids: dict[FactorId, FactorDescriptor] = field(default_factory=dict)
    segments: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, list[Any]] = field(default_factory=dict)
    selected: dict[str, list[Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    phi: MessageValue = field(default_factory=kernel._zero_message)
    belief: MessageValue = field(default_factory=kernel._zero_message)
    factor_messages: dict[FactorId, MessageValue] = field(default_factory=dict)
    delta: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def prepare_gn(self, payload: tuple) -> float:
        """Rebuild phi from anchors and the original relinearized pose prior once."""
        states, priors, covariance = payload
        unary, cost = [], 0.0
        for factor in self.owned_factors.values():
            factor.relinearize(states)
            cost += factor.linearized.residual_norm ** 2
            if isinstance(factor.linearized, kernel.UnaryInfoFactor):
                unary.append(factor.linearized)
        if covariance is not None:
            prior = kernel._build_prior_info_factors(
                {self.node_id: priors[self.node_id]}, states, covariance)
            unary.extend(prior)
            cost += sum(f.residual_norm ** 2 for f in prior)
        self.phi = kernel._sum_unary_information([self.node_id], unary)[self.node_id]
        return cost

    def reset_bp(self, payload: Any = None) -> None:
        """Legacy backend resets factor messages at every GN linearization."""
        self.factor_messages = {fid: kernel._zero_message() for fid in self.incident_factor_ids}

    def make_variable_messages(self, payload: tuple) -> list[MessagePacket]:
        """Exclude the destination factor and read only the previous BP buffer."""
        window, gn, bp = payload
        packets = []
        for fid, description in sorted(self.incident_factor_ids.items()):
            info, eta = kernel._copy_message(self.phi)
            for other_id, message in sorted(self.factor_messages.items()):
                if other_id != fid:
                    info += message[0]
                    eta += message[1]
            packets.append(MessagePacket(window, gn, bp, self.node_id,
                description.owner, fid, "v2f", info, eta))
        return packets

    def make_owned_factor_messages(self, payload: tuple) -> list[MessagePacket]:
        """The sole owner computes both outputs using this round's variable inbox."""
        window, gn, bp, inbox, cfg = payload
        packets = []
        local = inbox[self.node_id]
        for fid, factor in sorted(self.owned_factors.items()):
            info = factor.linearized
            if not isinstance(info, kernel.PairwiseInfoFactor):
                continue
            first, second = info.first_id, info.second_id
            incoming_first = local[(fid, first)]
            incoming_second = local[(fid, second)]
            outputs = (
                (first, kernel._factor_to_first_message(info,
                    (incoming_second.Lambda, incoming_second.eta), cfg.diagonal_regularization)),
                (second, kernel._factor_to_second_message(info,
                    (incoming_first.Lambda, incoming_first.eta), cfg.diagonal_regularization)),
            )
            for destination, (matrix, vector) in outputs:
                packets.append(MessagePacket(window, gn, bp, self.node_id,
                    destination, fid, "f2v", matrix, vector))
        return packets

    def update_belief(self, payload: tuple) -> float:
        """Damp into a fresh buffer, form phi+messages, then atomically commit."""
        inbox, cfg = payload
        next_messages, change = {}, 0.0
        info, eta = kernel._copy_message(self.phi)
        for fid in sorted(self.incident_factor_ids):
            packet = inbox[self.node_id][(fid, self.incident_factor_ids[fid].owner)]
            previous = self.factor_messages[fid]
            updated = kernel._damp_message(previous, (packet.Lambda, packet.eta), cfg.message_damping)
            change = max(change, float(np.linalg.norm(updated[0] - previous[0])
                                       + np.linalg.norm(updated[1] - previous[1])))
            next_messages[fid] = updated
            info += updated[0]
            eta += updated[1]
        self.belief = (info, eta)
        self.delta = np.linalg.solve(info + cfg.diagonal_regularization * np.eye(6), eta)
        self.factor_messages = next_messages
        return change

    def commit_increment(self, cfg: kernel.GaBPSettings) -> float:
        """Apply the legacy left rotation/additive translation update locally."""
        step = cfg.step_damping * self.delta
        kernel._apply_tangent_update(self.state, step)
        if not np.all(np.isfinite(self.state.rotation)) or not np.all(np.isfinite(self.state.translation)):
            raise ValueError(f"Nonfinite updated pose: {self.node_id}")
        return float(np.linalg.norm(step))


class NodeExecutor:
    """Identical deterministic stage interface for serial and local thread workers."""
    def __init__(self, mode: str = "serial", workers: int = 2) -> None:
        if mode not in ("serial", "thread") or workers < 1:
            raise ValueError("Executor must be serial/thread with positive workers.")
        self.pool = ThreadPoolExecutor(max_workers=workers) if mode == "thread" else None

    def map_nodes(self, nodes: Mapping[str, UAVNode], operation: Any, snapshot: Any) -> list:
        """Return in ID order only after every worker has completed this stage."""
        def invoke(node: UAVNode) -> Any:
            return getattr(node, operation)(snapshot) if isinstance(operation, str) else operation(node, snapshot)
        ordered = [nodes[key] for key in sorted(nodes)]
        return list(self.pool.map(invoke, ordered)) if self.pool else [invoke(n) for n in ordered]

    def __enter__(self) -> "NodeExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        if self.pool:
            self.pool.shutdown(wait=True)


class MessageRouter:
    """Reliable synchronous transport; validates rounds, owners and completeness."""
    def __init__(self, nodes: Mapping[str, UAVNode]) -> None:
        self.ids = tuple(sorted(nodes))
        self.registry = {}
        for node in nodes.values():
            self.registry.update(node.incident_factor_ids)
        self.cross_owner_packets = 0
        self.local_packets = 0

    def route(self, batches: list[list[MessagePacket]], window: str, gn: int, bp: int,
              kind: str) -> Mapping:
        """Reject stale/duplicate/missing or misdirected messages before any commit."""
        boxes = {node: {} for node in self.ids}
        expected = set()
        for fid, description in self.registry.items():
            for endpoint in description.endpoints:
                source, destination = ((endpoint, description.owner) if kind == "v2f"
                                       else (description.owner, endpoint))
                expected.add((destination, fid, source))
        seen = set()
        for batch in batches:
            for packet in batch:
                if (packet.window_id, packet.gn_round, packet.bp_round, packet.kind) != (window, gn, bp, kind):
                    raise ValueError("Stale/wrong window, GN round, BP round or packet direction.")
                key = (packet.destination, packet.factor_id, packet.source)
                if key not in expected or key in seen:
                    raise ValueError("Unknown, misdirected or duplicate factor message.")
                seen.add(key)
                boxes[packet.destination][(packet.factor_id, packet.source)] = packet
        if seen != expected:
            raise ValueError("Incomplete synchronous message round.")
        self.cross_owner_packets += sum(source != destination for destination, _, source in seen)
        self.local_packets += sum(source == destination for destination, _, source in seen)
        return MappingProxyType({node: MappingProxyType(box) for node, box in boxes.items()})


def register_incidence(nodes: Mapping[str, UAVNode]) -> None:
    """Register both endpoints without duplicating remote owned observations."""
    for node in nodes.values():
        node.incident_factor_ids.clear()
    seen = set()
    for owner, node in sorted(nodes.items()):
        for fid, factor in sorted(node.owned_factors.items()):
            if fid in seen or fid != factor.factor_id or factor.owner != owner:
                raise ValueError("Duplicate or incorrectly owned observation factor.")
            seen.add(fid)
            raw = factor.measurement
            if raw.observer_id != owner:
                raise ValueError("Measurement observer differs from owner.")
            if isinstance(raw, kernel.PairwiseBearingFactor):
                if raw.target_id != fid[1] or raw.target_id == owner or raw.target_id not in nodes:
                    raise ValueError("Invalid pairwise factor endpoint.")
                description = FactorDescriptor(fid, owner, (owner, raw.target_id))
                for endpoint in description.endpoints:
                    nodes[endpoint].incident_factor_ids[fid] = description


def solve_nodes_linearized(nodes: Mapping[str, UAVNode], executor: NodeExecutor,
                           router: MessageRouter, settings: kernel.GaBPSettings,
                           window_id: str, gn_round: int) -> dict:
    """Run synchronous BP with three map/route barriers and global stopping."""
    executor.map_nodes(nodes, "reset_bp", None)
    for bp in range(settings.max_bp_iterations):
        v2f = executor.map_nodes(nodes, "make_variable_messages", (window_id, gn_round, bp))
        variable_inbox = router.route(v2f, window_id, gn_round, bp, "v2f")
        f2v = executor.map_nodes(nodes, "make_owned_factor_messages",
            (window_id, gn_round, bp, variable_inbox, settings))
        factor_inbox = router.route(f2v, window_id, gn_round, bp, "f2v")
        changes = executor.map_nodes(nodes, "update_belief", (factor_inbox, settings))
        change = max(changes, default=0.0)
        if change < settings.bp_tolerance:
            break
    converged = change < settings.bp_tolerance
    return {"bp_iterations": bp + 1, "bp_converged": converged,
            "bp_stop_reason": "message_tolerance" if converged else "iteration_limit",
            "max_message_change": change}


def refine_nodes(nodes: Mapping[str, UAVNode], *, executor: NodeExecutor,
                 settings: kernel.GaBPSettings, prior_covariance: np.ndarray | None,
                 window_id: str) -> dict:
    """Refine a fixed owned observation set without forming global normal equations."""
    if settings.max_gn_iterations < 1 or settings.max_bp_iterations < 1:
        raise ValueError("GN and BP iteration limits must be positive.")
    started = time.perf_counter()
    register_incidence(nodes)
    router = MessageRouter(nodes)
    priors = freeze_states({key: node.initial_state for key, node in nodes.items()})
    history = []
    for gn in range(settings.max_gn_iterations):
        states = freeze_states({key: node.state for key, node in nodes.items()})
        costs = executor.map_nodes(nodes, "prepare_gn", (states, priors, prior_covariance))
        stats = solve_nodes_linearized(nodes, executor, router, settings, window_id, gn)
        increments = executor.map_nodes(nodes, "commit_increment", settings)
        max_increment = max(increments, default=0.0)
        history.append({"iteration": gn, "residual_norm": float(np.sqrt(sum(costs))),
                        "max_increment_norm": max_increment, **stats})
        if max_increment < settings.gn_tolerance:
            break
    converged = max_increment < settings.gn_tolerance
    return {"history": history, "gn_iterations": len(history), "gn_converged": converged,
            "gn_stop_reason": "increment_tolerance" if converged else "iteration_limit",
            "bp_converged": all(h["bp_converged"] for h in history),
            "cross_owner_packets": router.cross_owner_packets,
            "local_packets": router.local_packets, "gnbp_wall_seconds": time.perf_counter() - started}
