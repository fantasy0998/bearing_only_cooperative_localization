"""Geometry, ownership, probability-model and deterministic-execution regressions."""
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest

from src import multi_agent_bearing_gabp as old
from src.bearing_segmentation import segment_bearings_buchin, choose_segment_representatives, supplement_segment_candidates
from src.boepnp_gnbp_nodes import (
    NodeGNBPConfig, DirectedObservations, WindowInput, solve_window, sample_candidates,
    stage_rng, segment_node, anchor_matrix, select_pairwise, build_measurement,
)
from src.node_gnbp_backend import (
    UAVNode, ObservationFactor, NodeExecutor, MessagePacket, MessageRouter,
    register_incidence, solve_nodes_linearized, freeze_states,
)
from src.gnbp_keypoint_selection import EdgeTimeKey, _Atom, _add_atom, _schur
from src.bgPnp_GABP import _build_prior_covariance
from src.bo_epnp_solver import BO_EPnP
from src.exp_node_gnbp import validate_records, save_records


def synthetic_window(n_estimated=2, count=32, *, use_underscores=False, extra_anchor=False):
    """Small nondegenerate synthetic fixture, never a replacement for real data."""
    rng = np.random.default_rng(53)
    ids = tuple(f"ifo_{i:03}" if use_underscores else f"ifo{i:03}" for i in range(n_estimated+1+int(extra_anchor)))
    anchors = ids[:1+int(extra_anchor)]
    estimated = ids[len(anchors):]
    world = {key: rng.normal(size=(3,count)) + np.array([3*i, i, .3])[:,None] for i,key in enumerate(ids)}
    truth = {key: old.PoseState(old.so3_exp(rng.normal(size=3)*.1), rng.normal(size=3)*.2) for key in estimated}
    local = {key: truth[key].rotation @ world[key] + truth[key].translation[:,None] for key in estimated}
    edges, adjacency = [], np.zeros((len(ids), len(ids)), dtype=int)
    for i in estimated:
        q = np.array([old.so3_exp(rng.normal(size=3)*.1) for _ in range(count)])
        for j in ids:
            if i == j:
                continue
            b3 = truth[i].rotation @ (world[j]-world[i])
            raw = np.einsum("nji,jn->in", q, b3)
            raw /= np.linalg.norm(raw, axis=0)
            edges.append(DirectedObservations(i,j,np.arange(count),raw,.01,.02,q))
            adjacency[ids.index(i),ids.index(j)] = 1
    return WindowInput("synthetic",ids,anchors,estimated,adjacency,np.arange(count),
        np.cumsum(rng.uniform(.1,1.,count)), {k:world[k] for k in anchors},local,tuple(edges)), truth


def fast_cfg(**kwargs):
    return NodeGNBPConfig(max_gn_iterations=2,max_bp_iterations=4,**kwargs)


@pytest.mark.parametrize("representatives,expected", [(1,[2]),(2,[0,3]),(3,[0,2,3])])
def test_real_time_midpoint_and_endpoints(representatives, expected):
    segments = (np.arange(4),)
    # Array midpoint would be 1, actual time midpoint is nearest sample 2.
    result = choose_segment_representatives(segments, per_segment=representatives, timestamps=np.array([0.,1.,5.,10.]))
    assert result[0].tolist() == expected
    short = choose_segment_representatives((np.array([0]),), per_segment=representatives, timestamps=np.array([0.]))
    assert short[0].tolist() == [0]


def test_midpoint_ties_and_absent_timestamps():
    segments = (np.array([4,7,9,12]),)
    ts = np.arange(13,dtype=float)
    assert choose_segment_representatives(segments,per_segment=1,timestamps=ts)[0].tolist() == [7]
    assert choose_segment_representatives(segments,per_segment=1)[0].tolist() == [7]
    cfg = NodeGNBPConfig()
    assert cfg.representatives_per_segment == 1
    assert cfg.initialization_method == "bo_epnp"
    assert (cfg.max_gn_iterations,cfg.max_bp_iterations) == (8,30)


def test_segment_diameter_oracle_and_rotation_invariance():
    rng = np.random.default_rng(7)
    for _ in range(12):
        u = rng.normal(size=(3,12)); u /= np.linalg.norm(u,axis=0)
        theta = rng.uniform(.3,2.5)
        def split(data):
            return segment_bearings_buchin(data,np.arange(12),np.arange(12.),max_angle_rad=theta)
        a, b = split(u), split(old.so3_exp(rng.normal(size=3)) @ u)
        assert [s.tolist() for s in a.segments] == [s.tolist() for s in b.segments]
        dp = [0] + [100]*12
        for end in range(1,13):
            for start in range(end):
                gram = u[:,start:end].T @ u[:,start:end]
                if np.max(np.arccos(np.clip(gram,-1,1))) <= theta + 1e-9:
                    dp[end] = min(dp[end],dp[start]+1)
        assert len(a.segments) == dp[-1]
    angle = np.deg2rad([0.,1.,2.,3.,4.])
    u = np.array([np.cos(angle),np.sin(angle),np.zeros(5)])
    result = segment_bearings_buchin(u,np.arange(5),np.arange(5.),max_angle_rad=np.deg2rad(3))
    assert [s.tolist() for s in result.segments] == [[0,1,2,3],[4]]


def test_midpoint_supplement_covers_both_boundary_gaps():
    keep,reps,added = supplement_segment_candidates({"B_A":(np.arange(20),)}, per_segment=1,minimum_count=12,timestamps=np.arange(20.))
    assert reps["B_A"][0].tolist() == [9]
    assert len(keep["B_A"]) == 12 and len(added["B_A"]) == 11
    assert set(keep["B_A"]) <= set(range(20))


def test_graph_non_symmetric_roles_and_absolute_masks():
    window,_ = synthetic_window(use_underscores=True)
    retained = tuple(e for e in window.edges if (e.observer_id,e.target_id) != (window.estimated_ids[1],window.estimated_ids[0]))
    matrix = window.observation_adjacency.copy()
    matrix[2,1] = 0
    window = replace(window, observation_adjacency=matrix,edges=retained,absolute_indices=np.arange(5,25))
    result = solve_window(window,fast_cfg())
    assert result.record["status"] == "completed"
    factors = [key for node in result.nodes.values() for key in node.owned_factors]
    assert all(5 <= k[2] < 25 for k in factors)
    assert not any(k[:2] == (window.estimated_ids[1],window.estimated_ids[0]) for k in factors)
    with pytest.raises(ValueError,match="roles"):
        replace(window,estimated_ids=(window.anchor_ids[0],))
    matrix[0,1] = 1
    with pytest.raises(ValueError,match="anchor-as-observer"):
        replace(window,observation_adjacency=matrix)
    edge = retained[0]
    with pytest.raises(ValueError,match="indices"):
        replace(edge,absolute_indices=np.zeros(len(edge.absolute_indices),dtype=int))


def test_segmentation_all_edges_b3_raw_and_invalid_gaps():
    window,_ = synthetic_window()
    edge = window.edges[0]
    mask = np.ones(len(edge.absolute_indices),dtype=bool); mask[5] = False
    edge = replace(edge,valid_mask=mask)
    window = replace(window,edges=(edge,*window.edges[1:]))
    node = UAVNode(edge.observer_id,local_trajectory=window.estimated_position_local[edge.observer_id])
    segment_node(node,(window,fast_cfg(max_angle_deg=179.,candidate_multiplier=1.)))
    assert len(node.segments) == 2
    assert 5 not in node.segments[edge.edge_key]["candidate_indices"]
    for segment in node.segments[edge.edge_key]["segments"]:
        assert not (min(segment["indices"]) < 5 < max(segment["indices"]))
    key = node.candidates["anchor"][0]
    raw_factor = build_measurement(window,key)
    pos = np.searchsorted(edge.absolute_indices,key.time_index)
    np.testing.assert_allclose(raw_factor.observer_extrinsic_rotation,edge.rotation_b3_sensor[pos].T)
    np.testing.assert_allclose(raw_factor.measured_angles,old.vector_to_az_el(edge.bearings[:,pos]))
    assert raw_factor.sigma_azimuth == edge.sigma_azimuth
    assert not edge.bearings.flags.writeable
    for name in node.segments:
        assert node.segments[name]["candidate_count"] > 0


def test_uniform_time_and_random_streams():
    keys = [EdgeTimeKey("B","A",i) for i in range(5)]
    ts = np.array([0.,.01,.02,4.9,10.])
    cfg = NodeGNBPConfig()
    chosen = sample_candidates(keys,3,"uniform_time",ts,stage_rng(cfg,"w","B","anchor"))
    assert [k.time_index for k in chosen] == [0,3,4]
    a = sample_candidates(keys,3,"random",ts,stage_rng(cfg,"w","B","anchor"))
    b = sample_candidates(keys,3,"random",ts,stage_rng(cfg,"w","B","anchor"))
    assert a == b and len(set(a)) == 3


@pytest.mark.parametrize("anchor_strategy", ["random","uniform_time","rank_logdet"])
@pytest.mark.parametrize("pairwise_strategy", ["random","uniform_time","schur_greedy"])
def test_strategies_are_independent(anchor_strategy,pairwise_strategy):
    window,_ = synthetic_window()
    cfg = fast_cfg(anchor_selection_strategy=anchor_strategy,pairwise_selection_strategy=pairwise_strategy)
    result = solve_window(window,cfg)
    assert result.record["status"] == "completed"
    for node in result.nodes.values():
        assert len(node.selected["anchor"]) == 12 and len(node.selected["pairwise"]) == 6
        diag = node.diagnostics["pairwise_selection"]
        if pairwise_strategy != "schur_greedy":
            assert diag["candidate_linearizations"] == 0 and diag["trial_score_count"] == 0
    validate_records([result.record])


@pytest.mark.parametrize("budgets", [(6,4),(12,6),(18,8)])
def test_budgets_fixed_sets_and_retained_anchors(budgets):
    window,_ = synthetic_window()
    result = solve_window(window,fast_cfg(anchor_budget_per_node=budgets[0],pairwise_budget_per_node=budgets[1]))
    assert result.record["status"] == "completed"
    for node in result.nodes.values():
        assert len(node.owned_factors) == sum(budgets)
        assert len(node.selected["anchor"]) == budgets[0]
        assert len(node.selected["pairwise"]) == budgets[1]
    validate_records([result.record])


@pytest.mark.parametrize("estimated", [2,3])
@pytest.mark.parametrize("strategies", [("rank_logdet","schur_greedy"),("random","random")])
def test_serial_thread_and_node_order(estimated,strategies):
    window,_ = synthetic_window(n_estimated=estimated)
    cfg = fast_cfg(anchor_selection_strategy=strategies[0],pairwise_selection_strategy=strategies[1])
    serial = solve_window(window,cfg)
    parallel = solve_window(replace(window,estimated_ids=tuple(reversed(window.estimated_ids))),replace(cfg,executor="thread",workers=3))
    assert serial.record["status"] == parallel.record["status"] == "completed"
    for key in serial.nodes:
        a,b = serial.nodes[key],parallel.nodes[key]
        assert a.selected == b.selected
        np.testing.assert_allclose(a.state.rotation,b.state.rotation,atol=1e-12)
        np.testing.assert_allclose(a.state.translation,b.state.translation,atol=1e-12)
        for fid in a.factor_messages:
            np.testing.assert_allclose(a.factor_messages[fid][0],b.factor_messages[fid][0],atol=1e-10)
            np.testing.assert_allclose(a.factor_messages[fid][1],b.factor_messages[fid][1],atol=1e-10)


def test_initialization_failure_barrier_no_fallback_or_resampling():
    window,_ = synthetic_window()
    victim = window.estimated_ids[0]
    edges = tuple(e for e in window.edges if not (e.observer_id == victim and e.target_id in window.anchor_ids))
    result = solve_window(replace(window,edges=edges),fast_cfg())
    assert result.record["status"] == "initialization_failed"
    assert result.record["failed_node_ids"] == [victim]
    assert result.record["backend"] is None
    assert result.record["graph"]["missing_declared_edges"]
    with patch("src.boepnp_gnbp_nodes.solve_frontend",side_effect=RuntimeError("front end failed")) as call:
        result = solve_window(window,fast_cfg(anchor_selection_strategy="random"))
    assert call.call_count == 2
    assert len(result.record["failed_node_ids"]) == 2
    assert all("front end failed" in n.diagnostics["initialization"]["reason"] for n in result.nodes.values())


def test_rank_deficiency_and_frontend_configuration():
    window,truth = synthetic_window()
    world = dict(window.anchor_position_world)
    world[window.anchor_ids[0]] = np.zeros_like(next(iter(world.values())))
    result = solve_window(replace(window,anchor_position_world=world),fast_cfg())
    assert result.record["status"] == "initialization_failed"
    assert all("rank insufficient" in n.diagnostics["initialization"]["reason"] for n in result.nodes.values())
    with patch("src.boepnp_gnbp_nodes.solve_frontend",side_effect=lambda w,keys,method: truth[keys[0].observer_id].copy()) as call:
        result = solve_window(window,fast_cfg(initialization_method="sdp"))
    assert result.record["status"] == "completed"
    assert all(c.args[2] == "sdp" for c in call.call_args_list)


def test_multiple_anchors_share_one_budget_and_parameterization():
    window,_ = synthetic_window(extra_anchor=True)
    result = solve_window(window,fast_cfg())
    assert result.record["status"] == "completed"
    for node in result.nodes.values():
        assert len(node.selected["anchor"]) == 12
        blocks, diag = anchor_matrix(window,node.candidates["anchor"])
        assert blocks.shape[-2:] == (2,12) and diag["effective_target_rank"] == 12
        assert len(node.selected["pairwise"]) == 6


def test_accumulated_schur_equals_direct_joint_elimination():
    rng = np.random.default_rng(11)
    cfg = NodeGNBPConfig().schur
    base = np.eye(6)*2
    self_block,cross,neighbor = base.copy(),{"C":np.zeros((6,6))},{"C":np.eye(6)*3}
    joint = np.zeros((12,12)); joint[:6,:6] = base; joint[6:,6:] = neighbor["C"]
    for k in range(4):
        jac = rng.normal(size=(2,12)); h = jac.T @ jac
        atom = _Atom(EdgeTimeKey("B","C",k),None,h[:6,:6],h[:6,6:],h[6:,6:])
        self_block,cross,neighbor = _add_atom(self_block,cross,neighbor,atom)
        joint += h
    actual,_ = _schur(self_block,cross,neighbor,cfg)
    expected = joint[:6,:6] - joint[:6,6:] @ np.linalg.solve(joint[6:,6:],joint[6:,:6])
    np.testing.assert_allclose(actual,expected,atol=1e-12)


@pytest.mark.parametrize("prior_enabled", [True,False])
def test_backend_probability_model_matches_existing_refinement(prior_enabled):
    window,_ = synthetic_window(n_estimated=3)
    cfg = fast_cfg(prior_enabled=prior_enabled)
    result = solve_window(window,cfg)
    assert result.record["status"] == "completed"
    initial = {key:n.initial_state for key,n in result.nodes.items()}
    factors = [f.measurement for n in result.nodes.values() for f in n.owned_factors.values()]
    anchors = [f for f in factors if isinstance(f,old.AnchorBearingFactor)]
    pairwise = [f for f in factors if isinstance(f,old.PairwiseBearingFactor)]
    covariance = _build_prior_covariance(5.,.5) if prior_enabled else None
    reference = old.refine_with_gabp(initial,anchors,pairwise,prior_states=initial,
                                    prior_covariance=covariance,settings=cfg.backend_settings())
    for key,node in result.nodes.items():
        np.testing.assert_allclose(node.state.rotation,reference.states[key].rotation,atol=1e-10)
        np.testing.assert_allclose(node.state.translation,reference.states[key].translation,atol=1e-10)
    for a,b in zip(result.record["backend"]["history"],reference.history):
        np.testing.assert_allclose(a["residual_norm"],b.residual_norm,atol=1e-9)


def linear_fixture(cycle=False):
    nodes = {key:UAVNode(key) for key in ("B","C","D")}
    for i,node in enumerate(nodes.values()):
        node.phi = (np.eye(6)*2,np.arange(6.)*.1+i*.2)
    for idx,(i,j) in enumerate([("B","C"),("C","D")]+([("D","B")] if cycle else [])):
        fid = (i,j,idx)
        dummy = old.PairwiseBearingFactor(i,j,np.ones(3),np.zeros(2))
        # Unit Gaussian on xi-xj=.1: positive, strictly anchored linear graph.
        info = old.PairwiseInfoFactor(i,j,np.eye(6),-np.eye(6),-np.eye(6),np.eye(6),np.ones(6)*.1,-np.ones(6)*.1)
        nodes[i].owned_factors[fid] = ObservationFactor(fid,dummy,info)
    register_incidence(nodes)
    return nodes


@pytest.mark.parametrize("cycle", [False,True])
def test_linear_tree_and_convergent_cycle_match_centralized(cycle):
    nodes = linear_fixture(cycle)
    settings = old.GaBPSettings(max_bp_iterations=100,message_damping=1.,diagonal_regularization=1e-12)
    router = MessageRouter(nodes)
    with NodeExecutor() as executor:
        result = solve_nodes_linearized(nodes,executor,router,settings,"linear",0)
    assert result["bp_converged"]
    matrix,vector = np.zeros((18,18)),np.zeros(18)
    ids = list(nodes)
    for key,node in nodes.items():
        i = ids.index(key)*6
        matrix[i:i+6,i:i+6] += node.phi[0]; vector[i:i+6] += node.phi[1]
        for f in node.owned_factors.values():
            info = f.linearized; j = ids.index(info.second_id)*6
            matrix[i:i+6,i:i+6] += info.omega_ii; matrix[i:i+6,j:j+6] += info.omega_ij
            matrix[j:j+6,i:i+6] += info.omega_ji; matrix[j:j+6,j:j+6] += info.omega_jj
            vector[i:i+6] += info.eta_i; vector[j:j+6] += info.eta_j
    expected = np.linalg.solve(matrix,vector)
    np.testing.assert_allclose(np.concatenate([n.delta for n in nodes.values()]),expected,atol=1e-8)


def test_message_exclusion_first_round_eta_phi_and_routing_guards():
    nodes = linear_fixture()
    settings = old.GaBPSettings(max_bp_iterations=1,message_damping=1.)
    router = MessageRouter(nodes)
    with NodeExecutor() as executor:
        stats = solve_nodes_linearized(nodes,executor,router,settings,"w",0)
    assert not stats["bp_converged"]
    node = nodes["C"]
    assert len(node.incident_factor_ids) == 2 and len(node.owned_factors) == 1
    packets = node.make_variable_messages(("w",0,1))
    for packet in packets:
        incoming = [message for fid,message in node.factor_messages.items() if fid != packet.factor_id]
        np.testing.assert_allclose(packet.Lambda,node.phi[0]+sum(m[0] for m in incoming))
        np.testing.assert_allclose(packet.eta,node.phi[1]+sum(m[1] for m in incoming))
    assert any(np.linalg.norm(m[1]) > 0 for m in node.factor_messages.values())
    np.testing.assert_allclose(node.belief[0],node.phi[0]+sum(m[0] for m in node.factor_messages.values()))
    assert router.cross_owner_packets == 4 and router.local_packets == 4
    with pytest.raises(ValueError,match="Stale"):
        router.route([packets],"w",1,1,"v2f")
    with pytest.raises(ValueError,match="Incomplete"):
        router.route([packets],"w",0,1,"v2f")
    with pytest.raises(ValueError,match="duplicate"):
        router.route([packets,packets],"w",0,1,"v2f")
    with pytest.raises(ValueError):
        packets[0].eta[0] = 3


@pytest.mark.parametrize("gn,bp", [(1,1),(3,2),(2,5)])
def test_independent_gn_bp_limits_and_window_isolation(gn,bp):
    window,_ = synthetic_window()
    cfg = NodeGNBPConfig(max_gn_iterations=gn,max_bp_iterations=bp)
    a = solve_window(window,cfg); b = solve_window(replace(window,window_id="second"),cfg)
    assert a.record["status"] == b.record["status"] == "completed"
    assert 1 <= a.record["backend"]["gn_iterations"] <= gn
    assert all(1 <= h["bp_iterations"] <= bp for h in a.record["backend"]["history"])
    for key in a.nodes:
        assert a.nodes[key].owned_factors is not b.nodes[key].owned_factors
        np.testing.assert_allclose(a.nodes[key].state.translation,b.nodes[key].state.translation)


def test_relinearization_fixed_factor_set_and_early_stop():
    window,_ = synthetic_window()
    original = ObservationFactor.relinearize
    calls = []
    def traced(self,states):
        calls.append((self.factor_id,states[self.owner].translation.copy()))
        return original(self,states)
    with patch.object(ObservationFactor,"relinearize",traced):
        result = solve_window(window,NodeGNBPConfig(max_gn_iterations=3,gn_tolerance=1e-30,max_bp_iterations=2))
    factor_count = sum(len(n.owned_factors) for n in result.nodes.values())
    assert len(calls) == factor_count*3
    assert {x[0] for x in calls[:factor_count]} == {x[0] for x in calls[-factor_count:]}
    assert any(not np.array_equal(calls[i][1],calls[i+factor_count][1]) for i in range(factor_count))
    early = solve_window(window,NodeGNBPConfig(gn_tolerance=1e6,bp_tolerance=1e6,max_gn_iterations=7,max_bp_iterations=9))
    assert early.record["backend"]["gn_iterations"] == 1
    assert early.record["backend"]["history"][0]["bp_iterations"] == 1


def test_save_roundtrip_without_method_filter(tmp_path):
    window,_ = synthetic_window()
    record = solve_window(window,fast_cfg()).record
    record["dataset"] = {"observation_path":"synthetic_test_only"}
    save_records(tmp_path/"run",[record],{})
    loaded = json.loads((tmp_path/"run"/"records.json").read_text())
    assert validate_records(loaded) == {"completed":1}
    with pytest.raises(FileExistsError):
        save_records(tmp_path/"run",[record],{})


@pytest.mark.parametrize("rows", [12,24])
def test_wtls_includes_right_null_vector_at_six_correspondences(rows):
    rng = np.random.default_rng(137)
    matrix = rng.normal(size=(rows,12))
    truth = rng.normal(size=12)
    rhs = matrix @ truth
    solution = BO_EPnP._solve_wtls(matrix,rhs)
    np.testing.assert_allclose(solution,truth,atol=1e-10)
    np.testing.assert_allclose(matrix@solution,rhs,atol=1e-10)


def test_six_point_noiseless_bo_pose_accuracy():
    window,truth = synthetic_window()
    result = solve_window(window,fast_cfg(anchor_budget_per_node=6))
    assert result.record["status"] == "completed"
    for key,node in result.nodes.items():
        np.testing.assert_allclose(node.initial_state.rotation,truth[key].rotation,atol=1e-7)
        np.testing.assert_allclose(node.initial_state.translation,truth[key].translation,atol=1e-7)


def test_shortfall_is_reported_and_segmentation_off_keeps_full_pool():
    window,_ = synthetic_window(count=8)
    result = solve_window(window,fast_cfg(segmentation_enabled=False,pairwise_budget_per_node=12))
    assert result.record["status"] == "completed"
    for node in result.nodes.values():
        assert len(node.candidates["anchor"]) == 8
        assert len(node.selected["anchor"]) == len(node.selected["pairwise"]) == 8
        assert node.diagnostics["anchor_selection"]["budget_shortfall_reason"]
        assert node.diagnostics["pairwise_selection"]["budget_shortfall_reason"]
    validate_records([result.record])


def test_missing_time_fallback_and_zero_pairwise_budget():
    window,_ = synthetic_window()
    result = solve_window(replace(window,timestamps=None),fast_cfg(pairwise_budget_per_node=0))
    assert result.record["status"] == "completed"
    assert result.record["backend"]["cross_owner_packets"] == 0
    for node in result.nodes.values():
        assert not node.incident_factor_ids and not node.selected["pairwise"]
        assert all(s["time_rule"] == "lower_middle_index_fallback" for s in node.segments.values())


def test_snapshot_attributes_and_arrays_are_immutable():
    from dataclasses import FrozenInstanceError
    snapshots = freeze_states({"B":old.PoseState(np.eye(3),np.zeros(3))})
    with pytest.raises(FrozenInstanceError):
        snapshots["B"].translation = np.ones(3)
    with pytest.raises(ValueError):
        snapshots["B"].rotation[0,0] = 2.
