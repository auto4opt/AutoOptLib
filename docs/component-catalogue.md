# Component catalogue

The registry is the authority for component names. This catalogue lists the
87 built-in components. `list_components()` also includes registered extensions;
`describe_component(name)` exposes category, parent arity and native/batch support.
`get_component_contract(name, problem_record)` additionally supplies parameter
validity, data-flow kinds, fitness dependencies, state fields and backend rules.
All catalogue entries have a version-1 machine-readable contract.

## Roles

| Family | Count | Role |
| --- | ---: | --- |
| choose | 11 | Selection indices |
| cross | 14 | Paired-parent crossover |
| search | 39 | Variation or search |
| reinit | 3 | Reinitialization |
| update | 13 | Replacement |
| archive | 4 | Archive or statistics |
| para | 3 | Algorithm state update |

All 87 built-ins have native batch entries. The current problem-specific Search
catalogue is a subset; native execution requires the optional extension.
State/archives and selectors do not all return decision matrices. See the
[programming model](graph-programming-model.md) and
[CPU execution guide](cpu-execution.md).

## Registered built-ins

| Component | Execution category | Parent arity |
| --- | --- | ---: |
| `archive_best` | archive | 1 |
| `archive_diversity` | archive | 1 |
| `archive_statistic` | archive | 1 |
| `archive_tabu` | archive | 1 |
| `choose_brainstorm` | choose | 1 |
| `choose_elite_fraction` | choose | 1 |
| `choose_fastga_proportional` | choose | 1 |
| `choose_fastga_stochastic_tournament` | choose | 1 |
| `choose_ica` | choose | 1 |
| `choose_nich` | choose | 1 |
| `choose_random` | choose | 1 |
| `choose_roulette_wheel` | choose | 1 |
| `choose_stochastic_tournament` | choose | 1 |
| `choose_tournament` | choose | 1 |
| `choose_traverse` | choose | 1 |
| `cross_arithmetic` | search | 2 |
| `cross_fastga_five` | search | 2 |
| `cross_fastga_one` | search | 2 |
| `cross_fastga_three` | search | 2 |
| `cross_fastga_uniform` | search | 2 |
| `cross_order_n` | search | 2 |
| `cross_order_two` | search | 2 |
| `cross_point_five` | search | 2 |
| `cross_point_n` | search | 2 |
| `cross_point_one` | search | 2 |
| `cross_point_three` | search | 2 |
| `cross_point_two` | search | 2 |
| `cross_point_uniform` | search | 2 |
| `cross_sim_binary` | search | 2 |
| `para_cma` | parameter | 1 |
| `para_cmaes` | parameter | 1 |
| `para_pso` | parameter | 1 |
| `reinit_continuous` | search | 1 |
| `reinit_discrete` | search | 1 |
| `reinit_permutation` | search | 1 |
| `search_bit_conditional` | search | 1 |
| `search_bit_fast` | search | 1 |
| `search_bit_five` | search | 1 |
| `search_bit_normal` | search | 1 |
| `search_bit_one` | search | 1 |
| `search_bit_shifted` | search | 1 |
| `search_bit_standard` | search | 1 |
| `search_bit_three` | search | 1 |
| `search_bit_uniform` | search | 1 |
| `search_cma` | search | 1 |
| `search_de_current` | search | 1 |
| `search_de_current_best` | search | 1 |
| `search_de_current_to_pbest` | search | 1 |
| `search_de_random` | search | 1 |
| `search_eda` | search | 1 |
| `search_fastga_bit_conditional` | search | 1 |
| `search_fastga_bit_fast` | search | 1 |
| `search_fastga_bit_five` | search | 1 |
| `search_fastga_bit_normal` | search | 1 |
| `search_fastga_bit_one` | search | 1 |
| `search_fastga_bit_shifted` | search | 1 |
| `search_fastga_bit_standard` | search | 1 |
| `search_fastga_bit_three` | search | 1 |
| `search_fastga_bit_uniform` | search | 1 |
| `search_fastga_de_current_to_pbest` | search | 1 |
| `search_ica` | search | 1 |
| `search_insert` | search | 1 |
| `search_mu_cauchy` | search | 1 |
| `search_mu_gaussian` | search | 1 |
| `search_mu_polynomial` | search | 1 |
| `search_mu_uniform` | search | 1 |
| `search_pso` | search | 1 |
| `search_reset_creep` | search | 1 |
| `search_reset_n` | search | 1 |
| `search_reset_one` | search | 1 |
| `search_reset_rand` | search | 1 |
| `search_scramble` | search | 1 |
| `search_swap` | search | 1 |
| `search_swap_multi` | search | 1 |
| `update_always` | update | 1 |
| `update_fastga_plus` | update | 1 |
| `update_greedy` | update | 1 |
| `update_iterated_local_search` | update | 1 |
| `update_pairwise` | update | 1 |
| `update_round_robin` | update | 1 |
| `update_simulated_annealing` | update | 1 |
| `update_ssga_deterministic_tournament` | update | 1 |
| `update_ssga_fastga_deterministic_tournament` | update | 1 |
| `update_ssga_fastga_stochastic_tournament` | update | 1 |
| `update_ssga_fastga_worst` | update | 1 |
| `update_ssga_stochastic_tournament` | update | 1 |
| `update_ssga_worst` | update | 1 |
