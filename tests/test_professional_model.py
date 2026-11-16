"""Contracts must govern real validation, Search/CMA, extension and resume paths."""

import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from autooptlib import (
    AlgorithmGraph,
    ComponentContract,
    ComponentRequest,
    ComponentResult,
    DesignSpace,
    ExecutionSpec,
    GraphValidationError,
    NodeSpace,
    Operator,
    ParameterSpec,
    Pathway,
    Solution,
    SolutionSet,
    Stage,
    algorithm_to_dict,
    autoopt,
    get_component_contract,
    list_components,
    load_algorithm,
    make_problem,
    register_operator,
    save_algorithm,
)
from autooptlib.components import _custom_component_specs, _custom_components
from autooptlib.utils.design import Design
from autooptlib.utils.design._parameter_bank import ParameterBank
from autooptlib.utils.design._search_actions import (
    _decode_mixed_parameters,
    _encode_mixed_parameters,
    _mixed_cma_spec,
)
from autooptlib.utils.design._stream_graph import named_initial_genotype
from autooptlib.utils.space import space


def record():
    return SimpleNamespace(
        type=["continuous", "static", "certain"],
        bound=np.array([[-5.0] * 4, [5.0] * 4]),
        N=4,
        Gmax=5,
    )


def graph():
    return AlgorithmGraph(
        (
            Pathway((Stage(Operator("search_mu_uniform", (0.1,))),)),
            Pathway((Stage(Operator("search_mu_uniform", (0.2,))),)),
        ),
        population_size=4,
    )


def options(domain, seed=7):
    return space(
        record(),
        SimpleNamespace(
            DesignSpace=domain,
            ParameterScope=domain.parameter_scope,
            GraphSemantics="stream_graph_v2",
            AlgP=domain.max_pathways,
            AlgQ=domain.max_stages,
            PopulationSizeSpace=[4],
            OffspringSizeSpace=[4],
            ProbN=4,
            ProbFE=16,
            AlgRuns=1,
            rng=np.random.default_rng(seed),
        ),
    )


@pytest.mark.parametrize("name", list_components())
def test_every_builtin_has_a_complete_machine_readable_contract(name):
    c = get_component_contract(name, record())
    assert c.name == name and c.output_kind and c.input_kind
    assert c.backends == ("python", "native")
    assert c.version == 1 and c.parent_arity in (1, 2)
    json.dumps(c.to_dict(), allow_nan=False)
    assert ComponentContract.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_validation_happens_before_objective_and_reports_the_node(tmp_path):
    with pytest.raises(ValueError, match="expects 1 parameters"):
        Operator("search_mu_uniform", (0.1, 0.2, 0.3))
    g = AlgorithmGraph((Pathway((Stage("cross_point_two"),)),), population_size=4)
    small = record()
    small.bound = np.array([[-5.0], [5.0]])
    with pytest.raises(GraphValidationError, match=r"pathways\[0\].stages\[0\].search"):
        g.validate(small)
    calls = []
    p = make_problem(
        lambda x, _: calls.append(x) or float((x * x).sum()), bounds=(-5, 5)
    )
    bad = AlgorithmGraph((Pathway((Stage("search_bit_one"),)),), population_size=4)
    with pytest.raises(GraphValidationError, match="does not support continuous"):
        autoopt(
            Mode="solve",
            Algorithm=bad,
            Problem=p,
            InstanceSolve=[4],
            ProbFE=12,
            AlgRuns=1,
            OutputDir=tmp_path,
        )
    assert not calls


def test_execution_and_search_parameter_domains_are_distinct():
    g = AlgorithmGraph(
        (Pathway((Stage(Operator("search_mu_uniform", (0.9,))),)),), population_size=4
    )
    g.validate(record())
    DesignSpace.from_graph(g).resolve(record())
    with pytest.raises(ValueError):
        Operator("search_mu_uniform", (1.2,))


def test_integer_node_intervals_validate_reachable_levels_and_singleton_domains():
    d = DesignSpace(
        search=("cross_point_n",),
        max_pathways=1,
        max_stages=1,
        population_sizes=(4,),
        offspring_sizes=(4,),
        nodes=(
            NodeSpace(
                0, 0, "search", ("cross_point_n",), {"cross_point_n": ((1.2, 2.8),)}
            ),
        ),
    )
    d.resolve(record())
    p = record()
    p.bound = np.zeros((2, 1))
    for name in ("cross_point_n", "cross_point_two"):
        with pytest.raises(GraphValidationError, match="dimension"):
            AlgorithmGraph((Pathway((Stage(name),)),), population_size=4).validate(p)
    p.type = ["permutation", "static", "certain"]
    with pytest.raises(GraphValidationError, match="dimension"):
        AlgorithmGraph((Pathway((Stage("search_swap"),)),), population_size=4).validate(
            p
        )


def test_node_bank_roundtrip_cma_coordinates_and_independent_values():
    g = graph()
    d = DesignSpace(
        choose=("choose_random",),
        search=("search_mu_uniform",),
        update=("update_greedy",),
        max_stages=1,
        population_sizes=(4,),
        offspring_sizes=(4,),
        nodes=(
            NodeSpace(
                0,
                0,
                "search",
                ("search_mu_uniform",),
                {"search_mu_uniform": ((0.05, 0.15),)},
            ),
            NodeSpace(
                1,
                0,
                "search",
                ("search_mu_uniform",),
                {"search_mu_uniform": ((0.16, 0.28),)},
            ),
        ),
    )
    assert DesignSpace.from_dict(json.loads(json.dumps(d.to_dict()))) == d
    s = options(d)
    ops, ps, config = named_initial_genotype(g.to_initial_design(), s)
    a = Design.from_genotype(ops, ps, record(), s, configuration=config)
    assert isinstance(a.parameter, ParameterBank)
    spec = _mixed_cma_spec(a, s)
    assert sum(source == "node" for source, _ in spec) == 2
    v = _encode_mixed_parameters(a.parameter, a.configuration, spec, a)
    changed = v.copy()
    changed[0] += 1
    bank, _ = _decode_mixed_parameters(changed, a.parameter, a.configuration, spec, a)
    keys = list(a.parameter.nodes)
    searchkeys = [k for k in keys if k[2] == "search"]
    assert bank.nodes[searchkeys[0]][0] != a.parameter.nodes[searchkeys[0]][0]
    assert bank.nodes[searchkeys[1]][0] == a.parameter.nodes[searchkeys[1]][0]
    restored = pickle.loads(pickle.dumps(bank))
    assert restored.nodes.keys() == bank.nodes.keys()


def test_fixed_graph_space_is_evaluated_without_silent_parameter_collapse(tmp_path):
    g = graph()
    p = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
    args = dict(
        Mode="design",
        Problem=p,
        InstanceTrain=[4],
        InstanceTest=[4],
        DesignSpace=DesignSpace.from_graph(g),
        SearchInitialDesigns=[g],
        ProbN=4,
        ProbFE=12,
        AlgN=1,
        AlgFE=2,
        AlgRuns=1,
        Seed=2,
    )
    selected, _ = autoopt(
        **args, OutputDir=tmp_path, CheckpointDir=tmp_path / "checkpoints"
    )
    out = AlgorithmGraph.from_design(selected[0])
    assert [p.stages[0].search.parameters for p in out.pathways] == [(0.1,), (0.2,)]
    restored, _ = autoopt(
        **args,
        OutputDir=tmp_path / "resumed",
        CheckpointDir=tmp_path / "checkpoints",
        Resume=True,
    )
    assert (
        algorithm_to_dict(restored[0])["pathways"]
        == algorithm_to_dict(selected[0])["pathways"]
    )


def test_execution_contract_serialization_rejects_changed_rules(tmp_path):
    g = replace(graph(), execution_semantics="population_batch_v1")
    document = algorithm_to_dict(g)
    assert (
        document["programming_model"] == ExecutionSpec("population_batch_v1").to_dict()
    )
    document["programming_model"]["random"] = "another_rng"
    from autooptlib import algorithm_from_dict

    with pytest.raises(ValueError, match="rules differ"):
        algorithm_from_dict(document)
    with pytest.raises(GraphValidationError, match="requires population_batch"):
        g.validate(record(), backend="python")


def test_saved_batch_contract_selects_default_backend_and_rejects_override(tmp_path):
    from autooptlib.runtime.component_library import load_component_library

    try:
        load_component_library()
    except ModuleNotFoundError as error:
        if error.name != "autooptlib.runtime._native._native_compatibility":
            raise
        pytest.skip("optional native core is not installed")
    g = replace(graph(), execution_semantics="population_batch_v1")
    file = save_algorithm(g, tmp_path / "pinned.json")
    p = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
    common = dict(
        Mode="solve",
        Problem=p,
        InstanceSolve=[4],
        AlgFile=file,
        ProbFE=12,
        AlgRuns=1,
        Seed=3,
    )
    best, _ = autoopt(**common, OutputDir=tmp_path / "default")
    assert np.isfinite(best[0][0].fit)
    with pytest.raises(GraphValidationError, match="requires population_batch"):
        autoopt(**common, ComponentBackend="python", OutputDir=tmp_path / "invalid")


def test_node_constraints_govern_real_cma_search_and_structure_mutation(tmp_path):
    d = replace(
        DesignSpace.from_graph(graph()),
        nodes=(
            NodeSpace(
                0,
                0,
                "search",
                ("search_mu_uniform",),
                {"search_mu_uniform": ((0.05, 0.15),)},
            ),
            NodeSpace(
                1,
                0,
                "search",
                ("search_mu_uniform",),
                {"search_mu_uniform": ((0.16, 0.28),)},
            ),
        ),
    )
    s = options(d)
    ops, ps, config = named_initial_genotype(graph().to_initial_design(), s)
    a = Design.from_genotype(ops, ps, record(), s, configuration=config)
    from autooptlib.utils.design._parameter_bank import matches_nodes
    from autooptlib.utils.design._stream_graph import mutate_structure

    for _ in range(25):
        paths, bank, _ = mutate_structure(a, s)
        assert matches_nodes(paths, s)
        b = Design.from_genotype(paths, bank, record(), s, configuration=config)
        assert (
            0.05 <= b.parameter.nodes[(0, 0, "search", "search_mu_uniform")][0] <= 0.15
        )
        assert (
            0.16 <= b.parameter.nodes[(1, 0, "search", "search_mu_uniform")][0] <= 0.28
        )

    p = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
    selected, _ = autoopt(
        Mode="design",
        Problem=p,
        InstanceTrain=[4],
        InstanceTest=[5],
        DesignSpace=d,
        SearchInitialDesigns=[graph()],
        ProbN=4,
        ProbFE=12,
        AlgN=1,
        AlgFE=4,
        AlgRuns=1,
        Seed=4,
        SearchParameterActionProbability=1.0,
        SearchActionProbabilityGain=0.0,
        ParameterCMAOffspring=2,
        ParameterCMABlockGenerations=1,
        OutputDir=tmp_path,
    )
    stats = json.loads((tmp_path / "experiment.json").read_text())["options"][
        "SearchControllerStats"
    ]
    assert stats["semantic_candidate_evaluations"]["parameter"] == 4
    g = AlgorithmGraph.from_design(selected[0])
    assert 0.05 <= g.pathways[0].stages[0].search.parameters[0] <= 0.15
    assert 0.16 <= g.pathways[1].stages[0].search.parameters[0] <= 0.28


def test_checkpoint_identity_includes_node_constraints_and_contract_versions():
    from autooptlib.utils.general.process import _design_checkpoint_signature

    d = DesignSpace.from_graph(graph())
    first = _design_checkpoint_signature("sphere", options(d))
    changed = replace(
        d, nodes=tuple(n for n in d.nodes if n.location != (0, 0, "choose"))
    )
    assert _design_checkpoint_signature("sphere", options(changed)) != first


def test_warm_start_preserves_pinned_execution_and_legacy_pickle_is_not_preparsed():
    from autooptlib.programming import _configure_model_options

    opts = dict(
        Mode="design",
        SearchInitialDesigns=[
            replace(graph(), execution_semantics="population_batch_v1")
        ],
    )
    _configure_model_options(opts)
    assert opts["ComponentBackend"] == "native" and opts["StreamBackend"] == "batch"
    with pytest.raises(GraphValidationError, match="requires population_batch"):
        _configure_model_options(
            dict(
                Mode="design",
                ComponentBackend="python",
                SearchInitialDesigns=[
                    replace(graph(), execution_semantics="population_batch_v1")
                ],
            )
        )
    # Legacy file loading remains the responsibility of input_algorithm; do not
    # try to parse pickle bytes as JSON while expanding model defaults.
    _configure_model_options(dict(Mode="solve", AlgFile="legacy.pkl"))


def test_typed_components_have_independent_state_and_reject_inplace_rng_change():
    source = SolutionSet([Solution(np.zeros(4), 0.0, 0.0, 0.0)])
    rng = np.random.default_rng(1)
    state = {"rng": rng}
    names = ("search_mu_typed_a", "search_mu_typed_b", "search_mu_typed_bad")
    try:
        for name in names[:2]:
            adapter = register_operator(
                name,
                _typed_mutation,
                contract=ComponentContract(
                    name,
                    "search",
                    parameters=(ParameterSpec("scale", 0, 1, default=0.1),),
                    state_fields=("calls",),
                ),
                search_bounds=((0.01, 0.2),),
            )
            _, state = adapter(source, record(), None, state, 1, 1, None, "execute")
        assert state["_typed_components"][names[0]]["calls"] == 1
        assert state["_typed_components"][names[1]]["calls"] == 1
        assert state["rng"] is rng

        def bad(request):
            request.state["rng"] = np.random.default_rng(3)
            return ComponentResult(request.decisions, request.state)

        adapter = register_operator(
            names[2], bad, contract=ComponentContract(names[2], "search")
        )
        with pytest.raises(ValueError, match="authoritative rng"):
            adapter(source, record(), None, state, 1, 1, None, "execute")
    finally:
        for name in names:
            _custom_components.pop(name, None)
            _custom_component_specs.pop(name, None)


def test_typed_selector_and_update_execute_with_context_and_isolated_state(tmp_path):
    names = ("choose_typed_model", "update_typed_model")

    def choose(request):
        assert request.selection_count in (1, 2)
        state = dict(request.state)
        state["calls"] = state.get("calls", 0) + 1
        indices = request.rng.integers(
            len(request.source), size=request.selection_count
        )
        return ComponentResult(indices, state)

    def update(request):
        assert request.active_search in ("cross_point_one", "search_mu_uniform")
        state = dict(request.state)
        state["calls"] = state.get("calls", 0) + 1
        survivors = sorted(request.source, key=lambda row: row.fit)[: request.problem.N]
        return ComponentResult(SolutionSet(survivors), state)

    try:
        for name, function, role in zip(names, (choose, update), ("choose", "update")):
            register_operator(
                name,
                function,
                contract=ComponentContract(
                    name, role, needs_fitness=True, state_fields=("calls",)
                ),
            )
        g = AlgorithmGraph(
            (
                Pathway(
                    (
                        Stage("cross_point_one", choose=names[0]),
                        Stage(Operator("search_mu_uniform", (0.1,)), choose=names[0]),
                    ),
                    update=names[1],
                ),
            ),
            population_size=4,
        )
        p = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
        best, _ = autoopt(
            Mode="solve",
            Algorithm=g,
            Problem=p,
            InstanceSolve=[4, 5],
            ProbFE=20,
            AlgRuns=1,
            Seed=4,
            EvalWorkers=2,
            OutputDir=tmp_path,
        )
        assert all(np.isfinite(rows[0].fit) for rows in best)
    finally:
        for name in names:
            _custom_components.pop(name, None)
            _custom_component_specs.pop(name, None)


def _typed_mutation(request: ComponentRequest):
    state = dict(request.state)
    state["calls"] = state.get("calls", 0) + 1
    x = request.decisions.copy()
    x += request.rng.normal(size=x.shape) * request.parameters[0]
    return ComponentResult(x, state)


def test_typed_extension_runs_search_workers_and_checkpoint_protocol(tmp_path):
    name = "search_mu_typed_model"
    contract = ComponentContract(
        name,
        "search",
        parameters=(ParameterSpec("scale", 0.0, 1.0, default=0.1),),
        problem_types=("continuous",),
        state_fields=("calls",),
    )
    try:
        adapter = register_operator(
            name, _typed_mutation, contract=contract, search_bounds=((0.01, 0.2),)
        )
        c = get_component_contract(name)
        assert c == adapter.contract
        g = AlgorithmGraph(
            (Pathway((Stage(Operator(name, (0.1,))),)),), population_size=4
        )
        p = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
        common = dict(Problem=p, ProbN=4, ProbFE=12, AlgRuns=1, Seed=2)
        result, _ = autoopt(
            Mode="solve",
            Algorithm=g,
            InstanceSolve=[4, 5],
            EvalWorkers=2,
            OutputDir=tmp_path / "solve",
            **common,
        )
        assert all(np.isfinite(r[0].fit) for r in result)
        with pytest.raises(GraphValidationError, match="no native implementation"):
            g.validate(record(), backend="native")
        d = DesignSpace(
            choose=("choose_random", "choose_tournament"),
            search=(name,),
            update=("update_greedy",),
            max_stages=1,
            population_sizes=(4,),
            offspring_sizes=(4,),
        )
        selected, _ = autoopt(
            Mode="design",
            DesignSpace=d,
            InstanceTrain=[4],
            InstanceTest=[5],
            AlgN=1,
            AlgFE=2,
            OutputDir=tmp_path / "design",
            CheckpointDir=tmp_path / "typed-checkpoints",
            **common,
        )
        resumed, _ = autoopt(
            Mode="design",
            DesignSpace=d,
            InstanceTrain=[4],
            InstanceTest=[5],
            AlgN=1,
            AlgFE=2,
            OutputDir=tmp_path / "design-resumed",
            CheckpointDir=tmp_path / "typed-checkpoints",
            Resume=True,
            **common,
        )
        assert (
            algorithm_to_dict(resumed[0])["pathways"]
            == algorithm_to_dict(selected[0])["pathways"]
        )
        file = save_algorithm(selected[0], tmp_path / "typed.json")
        assert load_algorithm(file).operator_pheno[0]
    finally:
        _custom_components.pop(name, None)
        _custom_component_specs.pop(name, None)


def test_typed_result_protocol_catches_rng_replacement():
    name = "search_mu_bad_rng"
    try:
        c = ComponentContract(name, "search")
        adapter = register_operator(
            name,
            lambda r: ComponentResult(r.decisions, {"rng": np.random.default_rng(2)}),
            contract=c,
        )
        from autooptlib.utils.solve import Solution, SolutionSet

        source = SolutionSet([Solution(np.zeros(4), 0.0, 0.0, 0.0)])
        with pytest.raises(ValueError, match="authoritative rng"):
            adapter(
                source,
                record(),
                None,
                {"rng": np.random.default_rng(1)},
                1,
                1,
                None,
                "execute",
            )
    finally:
        _custom_components.pop(name, None)
        _custom_component_specs.pop(name, None)


def test_interrupted_typed_graph_restores_rng_identity_and_per_path_state(
    tmp_path, monkeypatch
):
    import importlib

    solve = importlib.import_module("autooptlib.utils.solve")
    name = "search_mu_typed_resume"
    try:
        register_operator(
            name,
            _typed_mutation,
            contract=ComponentContract(
                name,
                "search",
                parameters=(ParameterSpec("scale", 0, 1, default=0.1),),
                state_fields=("calls",),
            ),
            search_bounds=((0.01, 0.2),),
        )
        g = AlgorithmGraph(
            tuple(Pathway((Stage(Operator(name, (v,))),)) for v in (0.1, 0.2)),
            population_size=4,
        )
        p = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
        settings = dict(
            Mode="solve",
            Algorithm=g,
            Problem=p,
            InstanceSolve=[4],
            ProbFE=40,
            AlgRuns=1,
            Seed=7,
            CheckpointEvery=1,
        )
        best, history = autoopt(
            **settings,
            OutputDir=tmp_path / "baseline",
            CheckpointDir=tmp_path / "baseline-checkpoints",
        )
        original = solve._write_checkpoint

        def interrupt(path, payload):
            original(path, payload)
            if not payload["complete"]:
                raise RuntimeError("injected typed graph interruption")

        monkeypatch.setattr(solve, "_write_checkpoint", interrupt)
        with pytest.raises(RuntimeError, match="injected typed graph interruption"):
            autoopt(
                **settings,
                OutputDir=tmp_path / "interrupted",
                CheckpointDir=tmp_path / "resume-checkpoints",
            )
        monkeypatch.setattr(solve, "_write_checkpoint", original)
        resumed, resumed_history = autoopt(
            **settings,
            Resume=True,
            OutputDir=tmp_path / "resumed",
            CheckpointDir=tmp_path / "resume-checkpoints",
        )
        np.testing.assert_array_equal(best[0][0].dec, resumed[0][0].dec)
        assert [row.fit for row in history[0]] == [
            row.fit for row in resumed_history[0]
        ]

        def state(folder):
            with next(folder.glob("*.pkl")).open("rb") as f:
                return pickle.load(f)["aux_cache"]

        first = state(tmp_path / "baseline-checkpoints")
        second = state(tmp_path / "resume-checkpoints")
        for left, right in zip(first, second):
            a = left["_stream_stage_aux"][0]["_typed_components"][name]
            b = right["_stream_stage_aux"][0]["_typed_components"][name]
            assert a["calls"] == b["calls"] and b["calls"] > 1
            assert b["rng"] is right["_stream_stage_aux"][0]["rng"]
    finally:
        _custom_components.pop(name, None)
        _custom_component_specs.pop(name, None)
