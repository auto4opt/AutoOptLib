#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <eo>
#include <es.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <ga.h>
#include <ioh/problem/bbob.hpp>
#include <ioh/problem/pbo.hpp>
#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;
using Real = eoReal<eoMinimizingFitness>;
using Bits = eoBit<eoMaximizingFitness, int>;

constexpr std::size_t kMaxConsecutiveZeroEvaluationGenerations = 1000;

struct Arguments {
    std::string config;
    std::string suite;
    int function_id = 0;
    int dimension = 0;
    int instance = 0;
    std::uint32_t seed = 0;
    std::size_t budget = 0;
    bool json = false;
    bool cost_only = false;
    bool describe_space = false;
};

std::string require_value(int& index, const int argc, char* argv[]) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index - 1]);
    }
    return argv[index];
}

Arguments parse_arguments(const int argc, char* argv[]) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--config") args.config = require_value(i, argc, argv);
        else if (option == "--suite") args.suite = require_value(i, argc, argv);
        else if (option == "--function") args.function_id = std::stoi(require_value(i, argc, argv));
        else if (option == "--dimension") args.dimension = std::stoi(require_value(i, argc, argv));
        else if (option == "--instance") args.instance = std::stoi(require_value(i, argc, argv));
        else if (option == "--seed") args.seed = static_cast<std::uint32_t>(std::stoul(require_value(i, argc, argv)));
        else if (option == "--budget") args.budget = std::stoull(require_value(i, argc, argv));
        else if (option == "--json") args.json = true;
        else if (option == "--cost-only") args.cost_only = true;
        else if (option == "--describe-space") args.describe_space = true;
        else throw std::invalid_argument("unknown argument: " + option);
    }
    if (args.describe_space) return args;
    if (args.config.empty() || (args.suite != "bbob" && args.suite != "pbo") ||
        args.function_id <= 0 || args.dimension <= 0 || args.instance <= 0 || args.budget == 0) {
        throw std::invalid_argument(
            "required: --config PATH --suite {bbob,pbo} --function N --dimension N "
            "--instance N --seed N --budget N [--json|--cost-only]");
    }
    return args;
}

Json parameter_space() {
    return {
        {"schema", "autooptlib.eofastga-space"},
        {"schema_version", 2},
        {"shared", {
            {"crossover_rate", {{"type", "float"}, {"lower", 0.0}, {"upper", 1.0}}},
            {"mutation_rate", {{"type", "float"}, {"lower", 0.0}, {"upper", 1.0}}},
            {"crossover_selector", {{"type", "categorical"}, {"choices", {"random", "stochastic_tournament", "traverse", "proportional", "deterministic_tournament", "elite_fraction"}}}},
            {"aftercross_selector", {{"type", "categorical"}, {"choices", {"random", "stochastic_tournament", "traverse", "proportional", "deterministic_tournament", "elite_fraction"}}}},
            {"mutation_selector", {{"type", "categorical"}, {"choices", {"random", "stochastic_tournament", "traverse", "proportional", "deterministic_tournament", "elite_fraction"}}}},
            {"replacement", {{"type", "categorical"}, {"choices", {"plus", "ssga_worst", "ssga_stochastic_tournament", "ssga_deterministic_tournament"}}}},
            {"population_size", {{"type", "integer"}, {"lower", 4}, {"upper", 200}}},
            {"offspring_size", {{"type", "integer"}, {"lower", 1}, {"upper", 200}}},
            {"boundary_handling", {{"type", "categorical"}, {"choices", {"clip", "reflect", "resample"}}}},
            {"elite_fraction", {{"type", "float"}, {"lower", 0.0}, {"upper", 1.0}}},
            {"de_f", {{"type", "float"}, {"lower", 0.0}, {"upper", 1.0}}},
            {"de_cr", {{"type", "float"}, {"lower", 0.0}, {"upper", 1.0}}},
            {"de_p", {{"type", "float"}, {"lower", 0.0}, {"upper", 1.0}}}
        }},
        {"bbob", {
            {"crossover", {{"type", "categorical"}, {"choices", {"segment", "hypercube", "uniform", "sbx"}}}},
            {"mutation", {{"type", "categorical"}, {"choices", {"uniform", "deterministic_uniform", "normal", "current_to_pbest"}}}}
        }},
        {"pbo", {
            {"crossover", {{"type", "categorical"}, {"choices", {"uniform", "one_point", "three_point", "five_point"}}}},
            {"mutation", {{"type", "categorical"}, {"choices", {"uniform", "standard", "conditional", "shifted", "normal", "fast", "one_bit", "three_bit", "five_bit"}}}}
        }},
        {"constraints", Json::array({
            "BBOB selectors exclude proportional selection (index 3) for minimizing fitness",
            "SSGA replacements (indices 1-3) require offspring_size <= population_size",
            "BBOB mutation 3 is current-to-pbest/1/bin and disables external crossover"
        })}
    };
}

Json read_configuration(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open configuration: " + path);
    Json artifact;
    stream >> artifact;
    if (artifact.contains("configuration")) artifact = artifact.at("configuration");
    if (!artifact.is_object()) throw std::invalid_argument("configuration must be a JSON object");
    return artifact;
}

template <typename T>
T config_value(const Json& config, const std::string& name, const T fallback) {
    if (!config.contains(name)) return fallback;
    return config.at(name).get<T>();
}

std::size_t checked_index(const Json& config, const std::string& name,
                          const std::size_t fallback, const std::size_t size) {
    const auto value = config_value<std::size_t>(config, name, fallback);
    if (value >= size) {
        throw std::out_of_range(name + "=" + std::to_string(value) +
                                " is outside [0," + std::to_string(size - 1) + "]");
    }
    return value;
}

std::string target_key(const double target) {
    std::ostringstream stream;
    stream << std::setprecision(12) << std::defaultfloat << target;
    return stream.str();
}

std::vector<double> bbob_targets() {
    std::vector<double> targets;
    targets.reserve(51);
    for (int i = 0; i <= 50; ++i) targets.push_back(std::pow(10.0, 2.0 - 0.2 * i));
    return targets;
}

template <typename EOT, typename Problem, typename Scalar, bool Minimize>
class TrackedEval final : public eoEvalFunc<EOT> {
public:
    TrackedEval(std::shared_ptr<Problem> problem, std::vector<double> targets,
                const double optimum, const int boundary_handling = 0)
        : problem_(std::move(problem)), targets_(std::move(targets)),
          best_(Minimize ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity()),
          optimum_(optimum), boundary_handling_(boundary_handling) {
        if (boundary_handling_ < 0 || boundary_handling_ > 2) {
            throw std::out_of_range("boundary_handling must be 0, 1, or 2");
        }
    }

    void operator()(EOT& individual) override {
        if (!individual.invalid()) return;
        repair_boundary(individual);
        std::vector<Scalar> decision(individual.begin(), individual.end());
        const double raw = (*problem_)(decision);
        individual.fitness(raw);
        const bool improved = Minimize ? raw < best_ : raw > best_;
        if (improved) best_ = raw;
        const auto evaluations = static_cast<std::size_t>(problem_->state().evaluations);
        for (const double target : targets_) {
            const bool hit = Minimize ? best_ - optimum_ <= target : optimum_ - best_ <= target;
            const auto key = target_key(target);
            if (hit && target_hits_.find(key) == target_hits_.end()) target_hits_[key] = evaluations;
        }
        if (improved) trajectory_.push_back({{"evaluations", evaluations}, {"best_raw", best_}});
    }

    [[nodiscard]] std::size_t evaluations() const {
        return static_cast<std::size_t>(problem_->state().evaluations);
    }
    [[nodiscard]] double best() const { return best_; }
    [[nodiscard]] double optimum() const { return optimum_; }
    [[nodiscard]] const std::map<std::string, std::size_t>& target_hits() const { return target_hits_; }
    [[nodiscard]] const Json& trajectory() const { return trajectory_; }

private:
    void repair_boundary(EOT& individual) {
        if constexpr (std::is_floating_point_v<Scalar>) {
            const auto& bounds = problem_->bounds();
            for (std::size_t index = 0; index < individual.size(); ++index) {
                const double lower = bounds.lb.at(index);
                const double upper = bounds.ub.at(index);
                double value = static_cast<double>(individual[index]);
                if (!std::isfinite(value)) {
                    value = boundary_handling_ == 2
                        ? rng.uniform(lower, upper)
                        : lower + 0.5 * (upper - lower);
                }
                if (value >= lower && value <= upper) {
                    individual[index] = static_cast<Scalar>(value);
                    continue;
                }
                if (boundary_handling_ == 0) {
                    value = std::max(lower, std::min(upper, value));
                } else if (boundary_handling_ == 1) {
                    const double width = upper - lower;
                    if (width <= 0.0) {
                        value = lower;
                    } else {
                        double phase = std::fmod(value - lower, 2.0 * width);
                        if (phase < 0.0) phase += 2.0 * width;
                        value = lower + (phase <= width ? phase : 2.0 * width - phase);
                    }
                } else {
                    value = rng.uniform(lower, upper);
                }
                individual[index] = static_cast<Scalar>(value);
            }
        }
    }

    std::shared_ptr<Problem> problem_;
    std::vector<double> targets_;
    double best_;
    double optimum_;
    int boundary_handling_;
    std::map<std::string, std::size_t> target_hits_;
    Json trajectory_ = Json::array();
};

template <typename EOT>
class EliteFractionSelect final : public eoSelectOne<EOT> {
public:
    explicit EliteFractionSelect(const double fraction) : fraction_(fraction) {
        if (!std::isfinite(fraction_) || fraction_ < 0.0 || fraction_ > 1.0) {
            throw std::out_of_range("elite_fraction must be in [0,1]");
        }
    }

    const EOT& operator()(const eoPop<EOT>& population) override {
        if (population.empty()) throw std::invalid_argument("cannot select from an empty population");
        std::vector<std::size_t> order(population.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](const auto left, const auto right) {
            return static_cast<double>(population[left].fitness()) <
                   static_cast<double>(population[right].fitness());
        });
        const auto count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(fraction_ * population.size())));
        return population[order[rng.random(static_cast<std::uint32_t>(count))]];
    }

    std::string className() const override { return "EliteFractionSelect"; }

private:
    double fraction_;
};

template <typename EOT>
class TraverseSelect final : public eoSelectOne<EOT> {
public:
    void setup(const eoPop<EOT>& population) override {
        // eoFastGA calls setup() before *every* offspring.  eoSequentialSelect
        // resets its cursor in setup(), so using it here repeatedly selects
        // only the first sorted individual.  Keep the cursor across those
        // calls to implement the same round-robin semantics as Search's
        // choose_traverse component.
        if (population.empty()) cursor_ = 0;
        else cursor_ %= population.size();
    }

    const EOT& operator()(const eoPop<EOT>& population) override {
        if (population.empty()) {
            throw std::invalid_argument("cannot traverse an empty population");
        }
        const auto index = cursor_ % population.size();
        cursor_ = (index + 1) % population.size();
        return population[index];
    }

    std::string className() const override { return "TraverseSelect"; }

private:
    std::size_t cursor_ = 0;
};

class CurrentToPBestMutation final : public eoMonOp<Real> {
public:
    CurrentToPBestMutation(eoPop<Real>& population, const double scale,
                           const double crossover_rate, const double elite_fraction)
        : population_(population), scale_(scale), crossover_rate_(crossover_rate),
          elite_fraction_(elite_fraction) {
        for (const auto value : {scale_, crossover_rate_, elite_fraction_}) {
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                throw std::out_of_range("DE F, CR, and p must be in [0,1]");
            }
        }
    }

    bool operator()(Real& target) override {
        const std::size_t size = population_.size();
        if (size == 0 || target.empty()) return false;
        const std::size_t target_index = find_target_index(target);
        std::vector<std::size_t> order(size);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](const auto left, const auto right) {
            return static_cast<double>(population_[left].fitness()) <
                   static_cast<double>(population_[right].fitness());
        });
        const auto elite_count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(elite_fraction_ * size)));
        const auto pbest = order[rng.random(static_cast<std::uint32_t>(elite_count))];
        const auto first = sample_excluding({target_index});
        const auto second = sample_excluding({target_index, first});
        const auto forced = rng.random(static_cast<std::uint32_t>(target.size()));
        for (std::size_t coordinate = 0; coordinate < target.size(); ++coordinate) {
            const double current = static_cast<double>(target[coordinate]);
            const double donor = current
                + scale_ * (static_cast<double>(population_[pbest][coordinate]) - current)
                + scale_ * (static_cast<double>(population_[first][coordinate])
                            - static_cast<double>(population_[second][coordinate]));
            if (coordinate == forced || rng.flip(crossover_rate_)) {
                target[coordinate] = donor;
            }
        }
        // Search evaluates every generated trial, including the rare case in
        // which its coordinates equal the target. Returning true preserves
        // identical FE accounting in eoFastGA.
        return true;
    }

    std::string className() const override { return "CurrentToPBestMutation"; }

private:
    std::size_t find_target_index(const Real& target) const {
        for (std::size_t index = 0; index < population_.size(); ++index) {
            const auto& candidate = population_[index];
            if (candidate.size() == target.size()
                && candidate.fitness() == target.fitness()
                && std::equal(candidate.begin(), candidate.end(), target.begin())) {
                return index;
            }
        }
        throw std::logic_error(
            "current-to-pbest target is not a clone of the current population");
    }

    std::size_t sample_excluding(std::initializer_list<std::size_t> excluded) const {
        std::vector<std::size_t> available;
        for (std::size_t index = 0; index < population_.size(); ++index) {
            if (std::find(excluded.begin(), excluded.end(), index) == excluded.end()) {
                available.push_back(index);
            }
        }
        if (available.empty()) return rng.random(static_cast<std::uint32_t>(population_.size()));
        return available[rng.random(static_cast<std::uint32_t>(available.size()))];
    }

    eoPop<Real>& population_;
    double scale_;
    double crossover_rate_;
    double elite_fraction_;
};

template <typename EOT>
void add_common_selectors(eoAlgoFoundryFastGA<EOT>& foundry, const double elite_fraction) {
    for (auto& operators : {std::ref(foundry.crossover_selectors),
                            std::ref(foundry.aftercross_selectors),
                            std::ref(foundry.mutation_selectors)}) {
        operators.get().template add<eoRandomSelect<EOT>>();
        operators.get().template add<eoStochTournamentSelect<EOT>>(0.75);
        operators.get().template add<TraverseSelect<EOT>>();
        operators.get().template add<eoProportionalSelect<EOT>>();
        operators.get().template add<eoDetTournamentSelect<EOT>>(2);
        operators.get().template add<EliteFractionSelect<EOT>>(elite_fraction);
    }
}

template <typename EOT>
void add_common_replacements(eoAlgoFoundryFastGA<EOT>& foundry) {
    foundry.replacements.template add<eoPlusReplacement<EOT>>();
    foundry.replacements.template add<eoSSGAWorseReplacement<EOT>>();
    foundry.replacements.template add<eoSSGAStochTournamentReplacement<EOT>>(0.75);
    foundry.replacements.template add<eoSSGADetTournamentReplacement<EOT>>(2);
}

template <typename EOT, typename Eval>
class EvaluationProgressContinue final : public eoContinue<EOT> {
public:
    EvaluationProgressContinue(Eval& eval, const std::size_t maximum_stalled_generations)
        : eval_(eval), last_evaluations_(eval.evaluations()),
          maximum_stalled_generations_(maximum_stalled_generations) {}

    bool operator()(const eoPop<EOT>&) override {
        const auto current = eval_.evaluations();
        if (current > last_evaluations_) {
            last_evaluations_ = current;
            stalled_generations_ = 0;
            return true;
        }
        ++stalled_generations_;
        return stalled_generations_ < maximum_stalled_generations_;
    }

private:
    Eval& eval_;
    std::size_t last_evaluations_;
    std::size_t stalled_generations_ = 0;
    std::size_t maximum_stalled_generations_;
};

class ScopedStreamRedirect final {
public:
    ScopedStreamRedirect(std::ostream& stream, std::streambuf* replacement)
        : stream_(stream), previous_(stream.rdbuf(replacement)) {}
    ~ScopedStreamRedirect() { stream_.rdbuf(previous_); }
    ScopedStreamRedirect(const ScopedStreamRedirect&) = delete;
    ScopedStreamRedirect& operator=(const ScopedStreamRedirect&) = delete;

private:
    std::ostream& stream_;
    std::streambuf* previous_;
};

template <typename EOT, typename Eval>
void execute_foundry(eoAlgoFoundryFastGA<EOT>& foundry, eoInit<EOT>& init, Eval& eval,
                     const Json& config, const std::size_t budget) {
    const auto population_size = std::min(
        budget, std::max<std::size_t>(1, config_value<std::size_t>(config, "population_size", 20)));
    const auto offspring_size = std::max<std::size_t>(
        1, config_value<std::size_t>(config, "offspring_size", population_size));
    eoPop<EOT> population;
    population.append(population_size, init);
    eoPopLoopEval<EOT> initial_evaluation(eval);
    initial_evaluation(population, population);
    if (eval.evaluations() >= budget) return;

    if constexpr (std::is_same_v<EOT, Real>) {
        foundry.mutations.template add<CurrentToPBestMutation>(
            std::ref(population),
            config_value<double>(config, "de_f", 0.5),
            config_value<double>(config, "de_cr", 0.5),
            config_value<double>(config, "de_p", 0.2));
    }

    // eoFastGA skips objective calls for still-valid cloned offspring. Some
    // legal rate/operator combinations can therefore produce zero new FE for
    // ever. Stop such a candidate deterministically instead of hanging a
    // configurator worker for hours.
    foundry.continuators.template add<EvaluationProgressContinue<EOT, Eval>>(
        std::ref(eval), kMaxConsecutiveZeroEvaluationGenerations);

    const double crossover_rate = config_value<double>(config, "crossover_rate", 0.8);
    const double mutation_rate = config_value<double>(config, "mutation_rate", 0.8);
    if (crossover_rate < 0.0 || crossover_rate > 1.0 ||
        mutation_rate < 0.0 || mutation_rate > 1.0) {
        throw std::out_of_range("crossover_rate and mutation_rate must be in [0,1]");
    }
    const auto replacement = checked_index(
        config, "replacement", 0, foundry.replacements.size());
    if (replacement != 0 && offspring_size > population_size) {
        throw std::invalid_argument(
            "SSGA replacement requires offspring_size <= population_size");
    }
    foundry.select({
        crossover_rate,
        checked_index(config, "crossover_selector", 0, foundry.crossover_selectors.size()),
        checked_index(config, "crossover", 0, foundry.crossovers.size()),
        checked_index(config, "aftercross_selector", 0, foundry.aftercross_selectors.size()),
        mutation_rate,
        checked_index(config, "mutation_selector", 0, foundry.mutation_selectors.size()),
        checked_index(config, "mutation", 0, foundry.mutations.size()),
        replacement,
        std::size_t{0},
        offspring_size,
    });
    // Some ParadisEO replacements print internal diagnostics to stdout. Keep
    // the runner's stdout contract machine-readable (one cost or one JSON).
    std::ostringstream component_diagnostics;
    {
        ScopedStreamRedirect redirect(std::cout, component_diagnostics.rdbuf());
        foundry(population);
    }
    if (eval.evaluations() != budget) {
        throw std::runtime_error("eoFastGA stopped after " + std::to_string(eval.evaluations()) +
                                 " evaluations; expected " + std::to_string(budget));
    }
}

Json run_bbob(const Arguments& args, const Json& config) {
    for (const std::string selector : {
             "crossover_selector", "aftercross_selector", "mutation_selector"}) {
        if (config_value<std::size_t>(config, selector, 0) == 3) {
            throw std::invalid_argument(
                "BBOB minimizing fitness does not support proportional selector index 3");
        }
    }
    auto problem = ioh::problem::ProblemRegistry<ioh::problem::BBOB>::instance().create(
        args.function_id, args.instance, args.dimension);
    TrackedEval<Real, ioh::problem::BBOB, double, true> eval(
        problem, bbob_targets(), problem->optimum().y,
        config_value<int>(config, "boundary_handling", 0));
    const auto& ioh_bounds = problem->bounds();
    eoRealVectorBounds bounds(ioh_bounds.lb, ioh_bounds.ub);
    eoRealInitBounded<Real> init(bounds);
    const auto initial_size = std::min(
        args.budget, std::max<std::size_t>(1, config_value<std::size_t>(config, "population_size", 20)));
    eoAlgoFoundryFastGA<Real> foundry(init, eval, args.budget - initial_size);
    add_common_selectors(foundry, config_value<double>(config, "elite_fraction", 0.2));
    add_common_replacements(foundry);
    foundry.crossovers.add<eoSegmentCrossover<Real>>(std::ref(bounds));
    foundry.crossovers.add<eoHypercubeCrossover<Real>>(std::ref(bounds));
    foundry.crossovers.add<eoRealUXover<Real>>(0.5f);
    foundry.crossovers.add<eoSBXCrossover<Real>>(std::ref(bounds), 15.0);
    const double coordinate_probability = 1.0 / static_cast<double>(args.dimension);
    foundry.mutations.add<eoUniformMutation<Real>>(std::ref(bounds), 0.1, coordinate_probability);
    foundry.mutations.add<eoDetUniformMutation<Real>>(std::ref(bounds), 0.1, 1u);
    static double sigma = 0.2;
    foundry.mutations.add<eoNormalMutation<Real>>(std::ref(bounds), std::ref(sigma), coordinate_probability);
    execute_foundry(foundry, init, eval, config, args.budget);
    return {
        {"evaluations", eval.evaluations()}, {"best_raw", eval.best()},
        {"optimum_raw", eval.optimum()}, {"target_hits", eval.target_hits()},
        {"trajectory", eval.trajectory()}
    };
}

double checked_pbo_optimum(const Arguments& args, const double reported) {
    if (args.function_id != 22) return reported;
    // IOH 0.3.22 transforms MIS (F22) optimum metadata twice but evaluates
    // objective values once. Recover the instance affine map from OneMax at
    // adjacent dimensions, independently of the Python adapter.
    const int even_dimension = args.dimension - args.dimension % 2;
    const int native_optimum = even_dimension % 4 == 0
        ? even_dimension / 2 : even_dimension / 2 + 1;
    double corrected = static_cast<double>(native_optimum);
    double repeated = corrected;
    if (args.instance != 1) {
        auto& registry = ioh::problem::ProblemRegistry<ioh::problem::PBO>::instance();
        const double transformed_d = registry.create(1, args.instance, args.dimension)->optimum().y;
        const double transformed_d1 = registry.create(1, args.instance, args.dimension + 1)->optimum().y;
        const double scale = transformed_d1 - transformed_d;
        const double shift = transformed_d - scale * args.dimension;
        if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(shift)) {
            throw std::runtime_error("invalid PBO F22 instance affine transformation");
        }
        corrected = scale * native_optimum + shift;
        repeated = scale * corrected + shift;
    }
    const double tolerance = 1e-8 * std::max({1.0, std::abs(reported),
                                               std::abs(corrected), std::abs(repeated)});
    const auto matches = [=](const double expected) {
        return std::abs(reported - expected) <=
            std::max(tolerance, 1e-10 * std::max(std::abs(reported), std::abs(expected)));
    };
    if (!std::isfinite(reported) || (!matches(corrected) && !matches(repeated))) {
        throw std::runtime_error("PBO F22 optimum metadata is neither corrected nor known double-transformed");
    }
    return corrected;
}

Json run_pbo(const Arguments& args, const Json& config) {
    auto problem = ioh::problem::ProblemRegistry<ioh::problem::PBO>::instance().create(
        args.function_id, args.instance, args.dimension);
    const double optimum = checked_pbo_optimum(args, problem->optimum().y);
    std::vector<double> targets;
    if (std::isfinite(optimum)) targets.push_back(0.0);
    TrackedEval<Bits, ioh::problem::PBO, int, false> eval(problem, targets, optimum);
    eoBooleanGenerator<int> generator(0.5);
    eoInitFixedLength<Bits> init(args.dimension, generator);
    const auto initial_size = std::min(
        args.budget, std::max<std::size_t>(1, config_value<std::size_t>(config, "population_size", 20)));
    eoAlgoFoundryFastGA<Bits> foundry(init, eval, args.budget - initial_size);
    add_common_selectors(foundry, 0.2);
    add_common_replacements(foundry);
    foundry.crossovers.add<eoUBitXover<Bits>>(0.5);
    foundry.crossovers.add<eo1PtBitXover<Bits>>();
    foundry.crossovers.add<eoNPtsBitXover<Bits>>(3);
    foundry.crossovers.add<eoNPtsBitXover<Bits>>(5);
    foundry.mutations.add<eoUniformBitMutation<Bits>>();
    foundry.mutations.add<eoStandardBitMutation<Bits>>();
    foundry.mutations.add<eoConditionalBitMutation<Bits>>();
    foundry.mutations.add<eoShiftedBitMutation<Bits>>();
    foundry.mutations.add<eoNormalBitMutation<Bits>>();
    foundry.mutations.add<eoFastBitMutation<Bits>>();
    foundry.mutations.add<eoDetSingleBitFlip<Bits>>(1);
    foundry.mutations.add<eoDetSingleBitFlip<Bits>>(3);
    foundry.mutations.add<eoDetSingleBitFlip<Bits>>(5);
    execute_foundry(foundry, init, eval, config, args.budget);
    return {
        {"evaluations", eval.evaluations()}, {"best_raw", eval.best()},
        {"optimum_raw", eval.optimum()}, {"target_hits", eval.target_hits()},
        {"trajectory", eval.trajectory()}
    };
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const auto args = parse_arguments(argc, argv);
        if (args.describe_space) {
            std::cout << parameter_space().dump(2) << '\n';
            return 0;
        }
        eo::log << eo::setlevel(eo::warnings);
        rng.reseed(args.seed);
        const auto config = read_configuration(args.config);
        Json result = args.suite == "bbob" ? run_bbob(args, config) : run_pbo(args, config);
        result.update({
            {"schema", "autooptlib.external-run-result"}, {"schema_version", 1},
            {"suite", args.suite}, {"function", args.function_id},
            {"dimension", args.dimension}, {"instance", args.instance},
            {"seed", args.seed}, {"budget", args.budget}
        });
        if (args.cost_only) {
            const double optimum = result.at("optimum_raw").get<double>();
            const double best = result.at("best_raw").get<double>();
            const double cost = args.suite == "bbob" ? std::max(0.0, best - optimum)
                                                        : (std::isfinite(optimum) ? std::max(0.0, optimum - best) : -best);
            std::cout << std::setprecision(17) << cost << '\n';
        } else {
            std::cout << result.dump(args.json ? -1 : 2) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "autooptlib-fastga-runner: " << error.what() << '\n';
        return 2;
    }
}
