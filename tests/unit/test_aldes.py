from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import autoopt, get_component, make_problem
from autooptlib.aldes import (
    AutoOptEvaluator,
    EvaluationConfig,
    SequenceValidationError,
    allowed_next_tokens,
    decode_sequence,
    validate_sequence,
)

SIMPLE_SEQUENCE = [17, 0, 29, 8, 29, 12, 29, 18]


def _binary_problem():
    return make_problem(
        lambda decision, _data: -float(np.sum(decision)),
        bounds=(0, 1),
        problem_type="discrete",
        name="negative_onemax",
    )


def test_aldes_sequence_validation_and_masks():
    assert validate_sequence(SIMPLE_SEQUENCE) == SIMPLE_SEQUENCE
    assert np.flatnonzero(allowed_next_tokens([17])).tolist() == [0, 1, 2, 3]
    assert np.flatnonzero(allowed_next_tokens([17, 0, 29])).tolist() == [
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert np.flatnonzero(allowed_next_tokens([17, 0, 29, 8, 29, 12, 29])).tolist() == [
        18
    ]

    with pytest.raises(SequenceValidationError, match="choose"):
        validate_sequence([17, 8, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="parameter"):
        validate_sequence([17, 0, 29, 9, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="final update"):
        validate_sequence([17, 0, 29, 8, 29, 12, 30, 19, 18])
    with pytest.raises(SequenceValidationError, match="more than once"):
        validate_sequence([17, 0, 29, 8, 29, 8, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="global search"):
        validate_sequence([17, 0, 29, 11, 29, 4, 29, 8, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="crossover"):
        validate_sequence([17, 0, 29, 4, 29, 12, 29, 18])

    four_component_prefix = [17, 0, 29, 8, 29, 9, 19, 29, 10, 20, 29]
    continuation = allowed_next_tokens(four_component_prefix)
    assert not continuation[4:8].any()
    assert continuation[11]
    assert continuation[12]
    after_four_searches = allowed_next_tokens(four_component_prefix + [11, 29])
    assert np.flatnonzero(after_four_searches).tolist() == [12, 13, 14, 15, 16]


def test_aldes_mask_enforces_global_search_and_distinct_fork_modes():
    assert np.flatnonzero(allowed_next_tokens([17, 0, 31])).tolist() == [
        21,
        22,
    ]

    # cross_point_one is always global. A following parameterized search
    # therefore receives only the legacy local parameter tokens 0.1--0.3.
    prefix = [17, 0, 29, 4, 29, 9]
    assert np.flatnonzero(allowed_next_tokens(prefix)).tolist() == [19, 20, 21]

    # A local parameter remains available before a global search is present.
    assert np.flatnonzero(allowed_next_tokens([17, 0, 29, 6])).tolist() == list(
        range(19, 29)
    )

    # Parameter 0.4 makes cross_point_n global, so always-global components
    # cannot be selected later in the same algorithm.
    after_global = allowed_next_tokens([17, 0, 29, 6, 22, 29, 8, 29])
    assert not after_global[4]
    assert not after_global[5]
    assert not after_global[11]


def test_aldes_fork_has_exactly_one_executable_search_step():
    assert np.flatnonzero(allowed_next_tokens([17, 0, 31, 21, 11, 29])).tolist() == [
        12,
        13,
        14,
        15,
        16,
    ]
    assert np.flatnonzero(allowed_next_tokens([17, 0, 31, 21, 4, 29])).tolist() == [
        8,
        9,
        10,
    ]
    assert np.flatnonzero(
        allowed_next_tokens([17, 0, 31, 21, 4, 29, 8, 29])
    ).tolist() == [12, 13, 14, 15, 16]

    validate_sequence([17, 0, 31, 21, 11, 29, 12, 29, 18])
    validate_sequence([17, 0, 31, 22, 4, 29, 8, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="branch-mode"):
        validate_sequence([17, 0, 31, 24, 11, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="mutation-branch"):
        validate_sequence([17, 0, 31, 22, 11, 29, 12, 29, 18])
    with pytest.raises(SequenceValidationError, match="one search step"):
        validate_sequence([17, 0, 31, 21, 11, 29, 8, 29, 12, 29, 18])


def test_aldes_codec_builds_executable_autooptlib_design():
    evaluator = AutoOptEvaluator(
        _binary_problem(),
        [6],
        config=EvaluationConfig(population_size=4, evaluations=20, runs=2, seed=7),
    )
    design = decode_sequence(SIMPLE_SEQUENCE, evaluator.problems, evaluator.setting)
    pathway = design.operator_pheno[0][0]
    assert pathway.choose == "choose_traverse"
    assert pathway.search[0].primary == "search_reset_one"
    assert pathway.update == "update_greedy"
    assert design.aldes_sequence == SIMPLE_SEQUENCE

    performance = evaluator.evaluate(SIMPLE_SEQUENCE)
    assert performance.shape == (1, 2)
    assert np.all(np.isfinite(performance))
    assert np.all(performance <= 0)


def test_aldes_codec_maps_parameters_and_crossover_pair():
    evaluator = AutoOptEvaluator(
        _binary_problem(),
        [10],
        config=EvaluationConfig(population_size=4, evaluations=20, runs=1),
    )
    sequence = [17, 1, 29, 7, 22, 29, 10, 20, 29, 13, 29, 18]
    design = decode_sequence(sequence, evaluator.problems, evaluator.setting)
    pathway = design.operator_pheno[0][0]
    parameters = design.parameter_pheno[0][0]
    assert pathway.search[0].primary == "cross_point_uniform"
    assert pathway.search[0].secondary == "search_reset_rand"
    assert parameters.search[0].primary == pytest.approx([0.2])
    assert parameters.search[0].secondary == pytest.approx([0.1])


def test_aldes_fork_executes_first_search_row_per_branch(monkeypatch):
    import autooptlib.utils.solve as solve_module

    calls = {"cross": 0, "reset": 0}
    original_lookup = solve_module.get_component
    original_cross = original_lookup("cross_point_one")
    original_reset = original_lookup("search_reset_one")

    def cross(*args):
        if args[-1] == "execute":
            calls["cross"] += 1
        return original_cross(*args)

    def reset(*args):
        if args[-1] == "execute":
            calls["reset"] += 1
        return original_reset(*args)

    def lookup(name):
        if name == "cross_point_one":
            return cross
        if name == "search_reset_one":
            return reset
        return original_lookup(name)

    monkeypatch.setattr(solve_module, "get_component", lookup)
    evaluator = AutoOptEvaluator(
        _binary_problem(),
        [6],
        config=EvaluationConfig(population_size=4, evaluations=8, runs=1, seed=5),
    )
    sequence = [17, 3, 31, 22, 4, 29, 8, 29, 12, 29, 18]
    performance = evaluator.evaluate(sequence)

    assert performance.shape == (1, 1)
    assert calls == {"cross": 1, "reset": 2}


def test_search_reset_n_component_uses_distinct_positions():
    component = get_component("search_reset_n")
    solution = np.zeros((3, 8), dtype=int)
    problem = SimpleNamespace(
        bound=np.vstack((np.zeros(8, dtype=int), np.ones(8, dtype=int)))
    )
    changed, _ = component(
        solution,
        problem,
        np.array([3.0]),
        {"rng": np.random.default_rng(4)},
        "execute",
    )
    np.testing.assert_array_equal(changed.sum(axis=1), np.full(3, 3))


def test_dimension_dependent_aldes_parameters_use_problem_dimension():
    problem_8d = SimpleNamespace(bound=np.zeros((2, 8)))
    problem_5d = SimpleNamespace(bound=np.zeros((2, 5)))

    cross_bounds, _ = get_component("cross_point_n")(
        [problem_8d, problem_5d], "parameter"
    )
    reset_bounds, _ = get_component("search_reset_n")(
        [problem_8d, problem_5d], "parameter"
    )

    assert cross_bounds == [1, 5]
    np.testing.assert_array_equal(reset_bounds, [1.0, 5.0])


def test_aldes_torch_api_is_optional(tmp_path):
    import autooptlib.aldes as aldes

    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="PyTorch"):
            _ = aldes.ALDesGenerator
    else:  # pragma: no cover - exercised in the ALDes optional-dependency job
        torch.manual_seed(4)
        model = aldes.ALDesGenerator(
            aldes.GeneratorConfig(layers=1, feedforward_dim=64, max_length=50)
        )
        result = model.generate(candidates=2)
        assert result.sequences.shape[0] == 2
        for sequence in result.sequences:
            validate_sequence(sequence.numpy())
        scores = model.score(None, result.sequences)
        assert torch.isfinite(scores).all()

        checkpoint = tmp_path / "generator.pt"
        model.save_checkpoint(checkpoint, vocabulary="aldes-discrete-v1")
        restored, metadata = aldes.ALDesGenerator.load_checkpoint(checkpoint)
        assert metadata == {"vocabulary": "aldes-discrete-v1"}
        assert restored.config == model.config


def test_aldes_feature_conditioning_is_opt_in():
    torch = pytest.importorskip("torch")
    import autooptlib.aldes as aldes

    single = aldes.ALDesGenerator(
        aldes.GeneratorConfig(layers=1, feedforward_dim=64, dropout=0.0, max_length=50)
    )
    with pytest.raises(ValueError, match="does not accept"):
        single.generate(torch.zeros(32))

    continual = aldes.ALDesGenerator(
        aldes.GeneratorConfig(
            layers=1,
            feedforward_dim=64,
            dropout=0.0,
            max_length=50,
            condition_on_features=True,
        )
    )
    continual.eval()
    tokens = torch.tensor([[17]])
    zero_logits = continual.logits(torch.zeros(32), tokens)
    one_logits = continual.logits(torch.ones(32), tokens)
    assert not torch.equal(zero_logits, one_logits)
    with pytest.raises(ValueError, match="requires"):
        continual.generate()


def test_aldes_schema_one_checkpoint_loads_as_legacy_conditioned_model(tmp_path):
    torch = pytest.importorskip("torch")
    import autooptlib.aldes as aldes

    legacy = aldes.ALDesGenerator(
        aldes.GeneratorConfig(
            layers=1,
            feedforward_dim=64,
            max_length=50,
            condition_on_features=True,
            position_encoding="learned",
        )
    )
    config = dict(vars(legacy.config))
    config.pop("condition_on_features")
    config.pop("position_encoding")
    checkpoint = tmp_path / "legacy-generator.pt"
    torch.save(
        {
            "schema": "autooptlib.aldes.generator",
            "schema_version": 1,
            "config": config,
            "state_dict": legacy.state_dict(),
            "metadata": {},
        },
        checkpoint,
    )
    restored, _ = aldes.ALDesGenerator.load_checkpoint(checkpoint)
    assert restored.config.condition_on_features is True
    assert restored.config.position_encoding == "learned"


def test_aldes_evaluator_reuses_initial_populations_and_candidate_streams():
    seen = []

    def objective(decision, _dimension):
        seen.append(np.asarray(decision, dtype=int).copy())
        return -float(np.sum(decision))

    problem = make_problem(
        objective,
        bounds=(0, 1),
        problem_type="discrete",
        name="recording_onemax",
    )
    initial = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    evaluator = AutoOptEvaluator(
        problem,
        [6],
        config=EvaluationConfig(
            population_size=4,
            evaluations=4,
            runs=1,
            seed=7,
            initial_populations=initial,
        ),
    )
    evaluator.evaluate(SIMPLE_SEQUENCE)
    np.testing.assert_array_equal(np.vstack(seen[:4]), initial)

    sequence_b = [17, 0, 29, 11, 29, 12, 29, 18]
    first = AutoOptEvaluator(
        _binary_problem(),
        [6],
        config=EvaluationConfig(population_size=4, evaluations=12, runs=1, seed=19),
    )
    second = AutoOptEvaluator(
        _binary_problem(),
        [6],
        config=EvaluationConfig(population_size=4, evaluations=12, runs=1, seed=19),
    )
    _, forward = first.evaluate_many([SIMPLE_SEQUENCE, sequence_b])
    _, reverse = second.evaluate_many([sequence_b, SIMPLE_SEQUENCE])
    np.testing.assert_array_equal(forward[0], reverse[1])
    np.testing.assert_array_equal(forward[1], reverse[0])


def test_aldes_pbo_parallel_evaluation_deduplicates_and_preserves_order(
    monkeypatch,
):
    import autooptlib.aldes.evaluator as evaluator_module

    calls = []

    def evaluate(_problem_id, sequence, _instances, _config):
        calls.append(sequence)
        value = float(sum(sequence))
        return value, np.asarray([[value]])

    monkeypatch.setattr(evaluator_module, "_evaluate_pbo_sequence", evaluate)
    first = np.asarray([17, 0, 29, 8, 29, 12, 29, 18])
    second = np.asarray([17, 0, 29, 11, 29, 12, 29, 18])
    means, performances = evaluator_module._evaluate_pbo_sequences(
        np.vstack((first, second, first)),
        1,
        [1],
        EvaluationConfig(population_size=4, evaluations=4, runs=1),
        workers=1,
    )

    assert len(calls) == 2
    assert means == [sum(first), sum(second), sum(first)]
    np.testing.assert_array_equal(performances[0], performances[2])


def test_aldes_high_level_designer_and_ppo(tmp_path):
    torch = pytest.importorskip("torch")
    import autooptlib.aldes as aldes

    torch.manual_seed(3)
    model = aldes.ALDesGenerator(
        aldes.GeneratorConfig(layers=1, feedforward_dim=64, max_length=50)
    )
    problem = _binary_problem()
    algorithms, trace = autoopt(
        Mode="design",
        Designer="aldes",
        Problem=problem,
        InstanceTrain=[4],
        InstanceTest=[5],
        ALDesModel=model,
        ALDesCandidates=2,
        AlgN=1,
        AlgFE=1,
        AlgRuns=1,
        ProbN=4,
        ProbFE=12,
        InnerFE=4,
        Seed=3,
        OutputDir=tmp_path,
    )
    assert len(algorithms) == 1
    assert algorithms[0].performance.shape == (2, 1)
    assert trace
    assert (tmp_path / "Algorithm_1.json").exists()
    assert not (tmp_path / "Algorithm_2.json").exists()

    evaluator = AutoOptEvaluator(
        problem,
        [4],
        config=EvaluationConfig(population_size=4, evaluations=12, runs=1, seed=4),
    )
    trainer = aldes.PPOTrainer(model, aldes.PPOConfig(candidates=2, update_epochs=1))
    model.eval()
    unchanged = model.generate(candidates=2)
    np.testing.assert_allclose(
        unchanged.log_probabilities.sum(dim=1).detach().numpy(),
        model.score(None, unchanged.sequences).detach().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    metrics = trainer.step(
        None, lambda sequences: evaluator.evaluate_many(sequences)[0]
    )
    assert set(metrics) == {
        "loss",
        "mean_cost",
        "baseline",
        "best_cost",
        "learning_rate",
    }
    assert np.all(np.isfinite(list(metrics.values())))

    continual_model = aldes.ALDesGenerator(
        aldes.GeneratorConfig(
            layers=1,
            feedforward_dim=64,
            max_length=50,
            condition_on_features=True,
        )
    )
    initial = np.zeros((1, 4, 4), dtype=int)
    feature_bundle = SimpleNamespace(
        features=np.zeros(32, dtype=np.float32),
        initial_populations={0: initial, 1: initial},
    )
    continual_algorithms, _ = autoopt(
        Mode="design",
        Designer="aldes",
        ALDesMode="continual",
        Problem=problem,
        InstanceTrain=[4],
        InstanceTest=[4],
        ALDesModel=continual_model,
        ALDesFeatures=feature_bundle,
        ALDesCandidates=2,
        AlgN=1,
        AlgFE=1,
        AlgRuns=1,
        ProbN=4,
        ProbFE=8,
        InnerFE=4,
        Seed=3,
        OutputDir=tmp_path / "continual",
    )
    assert len(continual_algorithms) == 1


def test_aldes_ioh_pbo_adapter_smoke():
    pytest.importorskip("ioh")
    from autooptlib.aldes import make_pbo_problem

    evaluator = AutoOptEvaluator(
        make_pbo_problem(1),
        [5],
        config=EvaluationConfig(population_size=4, evaluations=8, runs=1, seed=1),
    )
    performance = evaluator.evaluate(SIMPLE_SEQUENCE)
    assert performance.shape == (1, 1)
    assert np.isfinite(performance).all()


def test_aldes_pbo_feature_extraction_is_reproducible():
    pytest.importorskip("ioh")
    pytest.importorskip("pflacco")
    from autooptlib.aldes import extract_pbo_features

    options = dict(
        dimension=10,
        trials=2,
        sample_factor=10,
        population_size=4,
        seed=17,
    )
    first = extract_pbo_features(1, **options)
    second = extract_pbo_features(1, **options)

    assert first.features.shape == (32,)
    assert first.samples.shape == (2, 100, 10)
    assert first.initial_populations.shape == (2, 4, 10)
    assert np.isfinite(first.features).all()
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.samples, second.samples)
