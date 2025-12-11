"""Search for scenarios where tweaking the first approval ballot increases its cost satisfaction."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import itertools
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from pabutools.election import (
    ApprovalBallot,
    ApprovalProfile,
    Cost_Sat,
    Cardinality_Sat,
    Instance,
    Project,
)
from pabutools.rules import BudgetAllocation
from pabutools.rules import (
    greedy_utilitarian_welfare, 
    method_of_equal_shares, 
    sequential_phragmen, 
    maximin_support, 
    completion_by_rule_combination
)

@dataclass(slots=True)
class SearchConfig:
    """User-configurable bounds for the search."""

    sat_func: str = "Cost_Sat"
    algo: str = "greedy_utilitarian_welfare"
    max_project_count: int = 5
    max_value_per_project: int = 10
    max_voters: int = 10
    min_project_count: int = 2
    min_value_per_project: int = 1
    min_voters: int = 2
    thread_count: int = 16


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate project cost assignments and look for cases where changing "
            "the first voter's approval ballot increases their cost_sat score."
        )
    )
    parser.add_argument("--sat-func", type=str, default="cost_sat")
    parser.add_argument("--algorithm", type=str, default="greedy_utilitarian_welfare")
    parser.add_argument("--max-project-count", type=int, default=4)
    parser.add_argument("--min-project-count", type=int, default=2)
    parser.add_argument("--max-value-per-project", type=int, default=5)
    parser.add_argument("--min-value-per-project", type=int, default=1)
    parser.add_argument("--max-voters", type=int, default=4)
    parser.add_argument("--min-voters", type=int, default=2)
    parser.add_argument("--thread-count", type=int, default=16)

    args = parser.parse_args()
    return SearchConfig(
        sat_func=args.sat_func,
        algo=args.algorithm,
        max_project_count=args.max_project_count,
        max_value_per_project=args.max_value_per_project,
        max_voters=args.max_voters,
        min_project_count=args.min_project_count,
        min_value_per_project=args.min_value_per_project,
        min_voters=args.min_voters,
        thread_count=args.thread_count,
    )


pb_rule = None
sat_func = None
testing_func = None

def calculate_cost_sat(
    instance: Instance,
    truthful_profile: ApprovalProfile,
    truthful_first_ballot: ApprovalBallot,
    alternate_ballot: ApprovalBallot,
    stop_event: threading.Event,
) -> bool:
    if stop_event.is_set():
        return False
    try: 
        original_results = pb_rule(instance,truthful_profile, sat_class=sat_func, resoluteness=False)
    except TypeError:
        original_results = pb_rule(instance, truthful_profile, resoluteness=False)

    if isinstance(original_results, BudgetAllocation):
        original_results = [original_results]
    truthful_satisfaction = sat_func(instance, truthful_profile, truthful_first_ballot)
    initial_sat = max(truthful_satisfaction.sat(result) for result in original_results)

    tweaked_profile = ApprovalProfile(truthful_profile, instance=instance)
    tweaked_profile[0] = alternate_ballot
    
    try:
        tweaked_results = pb_rule(instance, tweaked_profile, sat_class=sat_func, resoluteness=False)
    except TypeError:
        tweaked_results = pb_rule(instance, tweaked_profile, resoluteness=False)

    if isinstance(tweaked_results, BudgetAllocation):
        tweaked_results = [tweaked_results]
    tweaked_sat = min(truthful_satisfaction.sat(result) for result in tweaked_results)

    if tweaked_sat > initial_sat:
        print("Found improved Cost_Sat after tweaking the first ballot.")
        print(f"n={truthful_profile.num_ballots()}")
        print(f"{instance}")
        print(f"L={instance.budget_limit}")
        print(f"truthful_profile={truthful_profile}")
        print(f"a'_1={tweaked_profile[0]}")
        print(f"W={original_results}")
        print(f"W'={tweaked_results}")
        print(f"original_cost_sat={initial_sat}")
        print(f"new_cost_sat={tweaked_sat}")
        stop_event.set()
        return True
    return False


def calculate_sat_with_ES_U(instance: Instance, truthful_profile: ApprovalProfile, truthful_first_ballot: ApprovalBallot, alternate_ballot: ApprovalBallot):
    original_results = completion_by_rule_combination(
        instance,
        truthful_profile,
        [method_of_equal_shares, greedy_utilitarian_welfare],
        [{"sat_class": sat_func}, {"sat_class": sat_func}],
        resoluteness=False,
    )

    if isinstance(original_results, BudgetAllocation):
        original_results = [original_results]
    truthful_satisfaction = sat_func(
        instance, truthful_profile, truthful_first_ballot
    )
    initial_sat = max([truthful_satisfaction.sat(result) for result in original_results])

    tweaked_profile = ApprovalProfile(truthful_profile, instance=instance)
    tweaked_profile[0] = alternate_ballot
    tweaked_results = completion_by_rule_combination(
        instance, tweaked_profile, 
        [method_of_equal_shares, greedy_utilitarian_welfare],
        [{"sat_class": sat_func}, {"sat_class": sat_func}],
        resoluteness=False,
    )
    if isinstance(tweaked_results, BudgetAllocation):
        tweaked_results = [tweaked_results]
    tweaked_sat = min([truthful_satisfaction.sat(result) for result in tweaked_results])

    if tweaked_sat > initial_sat:
        eq_res: list[BudgetAllocation] = method_of_equal_shares(
            instance, truthful_profile, sat_class=sat_func, resoluteness=False
        )

        if any(instance.is_exhaustive(res) or len(res) == 0 for res in eq_res):
            print(F"found potential candidate, but thrown as ES did not lead to normal output: lens: {[len(res) for res in eq_res]} exhaustive: {[instance.is_exhaustive(res) for res in eq_res]}")
            return
        eq_res_tweaked: list[BudgetAllocation] = method_of_equal_shares(
            instance, tweaked_profile, sat_class=sat_func, resoluteness=False
        )
        if any(instance.is_exhaustive(res) or len(res) == 0 for res in eq_res_tweaked):
            return

        print("Found improved Cost_Sat after tweaking the first ballot.")
        print(f"n={truthful_profile.num_ballots()}")
        print(f"{instance}")
        print(f"L={instance.budget_limit}")
        print(
            f"truthful_profile={truthful_profile}"
        )
        print(f"a'_1={tweaked_profile[0]}")
        print(
            f"W={original_results}"
        )
        print(f"W'={tweaked_results}")
        print(f"original_cost_sat={initial_sat}")
        print(f"new_cost_sat={tweaked_sat}")
        print(f"ES: W={eq_res}")
        print(f"tweaked ES: W={eq_res_tweaked}")

        sys.exit(0)

def calculate_sat_with_ES_add1U(
    instance: Instance, truthful_profile: ApprovalProfile, truthful_first_ballot: ApprovalBallot, alternate_ballot: ApprovalBallot,
    stop_event: threading.Event,
):
    if stop_event.is_set():
        return False
    original_results = completion_by_rule_combination(
        instance,
        truthful_profile,
        [method_of_equal_shares, greedy_utilitarian_welfare],
        [
            {"sat_class": sat_func, "voter_budget_increment": 1}, 
            {"sat_class": sat_func}
        ],
        resoluteness=False,
    )

    if isinstance(original_results, BudgetAllocation):
        original_results = [original_results]
    truthful_satisfaction = sat_func(
        instance, truthful_profile, truthful_first_ballot
    )
    initial_sat = max([truthful_satisfaction.sat(result) for result in original_results])

    tweaked_profile = ApprovalProfile(truthful_profile, instance=instance)
    tweaked_profile[0] = alternate_ballot
    tweaked_results = completion_by_rule_combination(
        instance, tweaked_profile, 
        [method_of_equal_shares, greedy_utilitarian_welfare],
        [{"sat_class": sat_func, "voter_budget_increment": 1}, {"sat_class": sat_func}],
        resoluteness=False,
    )
    if isinstance(tweaked_results, BudgetAllocation):
        tweaked_results = [tweaked_results]
    tweaked_sat = min([truthful_satisfaction.sat(result) for result in tweaked_results])

    if tweaked_sat > initial_sat:
        eq_results = method_of_equal_shares(
            instance, truthful_profile, sat_class=sat_func, resoluteness=False
        )
        if any(instance.is_exhaustive(res) or len(res) == 0 for res in eq_results):
            return False
        
        eq_res_add1 = method_of_equal_shares(
            instance, truthful_profile, sat_class=sat_func, resoluteness=False, voter_budget_increment=1
        )
        for res in eq_res_add1:
            res.sort()
        for eq_result in eq_results:
            eq_result.sort()
            if eq_result in eq_res_add1:
                return False
        if any(instance.is_exhaustive(res) or len(res) == 0 for res in eq_res_add1):
            return False
        
        eq_res_tweaked = method_of_equal_shares(
            instance, tweaked_profile, sat_class=sat_func, resoluteness=False
        )
        if any(instance.is_exhaustive(res) or len(res) == 0 for res in eq_res_tweaked):
            return False
        eq_res_tweaked_add1 = method_of_equal_shares(
            instance, tweaked_profile, sat_class=sat_func, resoluteness=False, voter_budget_increment=1
        )
        for res in eq_res_tweaked_add1:
            res.sort()
        for eq_result in eq_res_tweaked:
            eq_result.sort()
            if eq_result in eq_res_tweaked_add1:
                return False
        if any(instance.is_exhaustive(res) or len(res) == 0 for res in eq_res_tweaked_add1):
            return False
    

        print("Found improved Add1U with sat after tweaking the first ballot.")
        print(f"n={truthful_profile.num_ballots()}")
        print(f"{instance}")
        print(f"L={instance.budget_limit}")
        print(
            f"truthful_profile={truthful_profile}"
        )
        print(f"a'_1={tweaked_profile[0]}")
        print(
            f"W={original_results}"
        )
        print(f"W'={tweaked_results}")
        print(f"original_cost_sat={initial_sat}")
        print(f"new_cost_sat={tweaked_sat}")
        print("ES: W=", eq_results)
        print("ES add1U: W=", eq_res_add1)
        
        print("tweaked ES: W=", eq_res_tweaked)
        print("tweaked ES add1U: W=", eq_res_tweaked_add1)
        stop_event.set()
        return True
    return False

def _search_num_voters(
    num_voters: int,
    instance: Instance,
    projects: list[Project],
    stop_event: threading.Event,
) -> bool:
    if stop_event.is_set():
        return False
    possible_ballots = itertools.chain.from_iterable(
        itertools.combinations(projects, r) for r in range(len(projects) + 1)
    )
    print(f"\t\t\tSearching with n={num_voters}...")
    for profile in itertools.product(possible_ballots, repeat=num_voters):
        if stop_event.is_set():
            return False
        if any(len(ballot) == 0 for ballot in profile):
            continue
        truthful_profile = ApprovalProfile(instance=instance)
        for ballot in profile:
            truthful_profile.append(ApprovalBallot(ballot))
        truthful_first_ballot = ApprovalBallot(truthful_profile[0])

        allocation = method_of_equal_shares(
            instance, truthful_profile, sat_class=sat_func, resoluteness=False
        )
		
        for alternate_ballot in itertools.chain.from_iterable(
            itertools.combinations(projects, r) for r in range(len(projects) + 1)
        ):
            if stop_event.is_set():
                return False
            alternate_ballot = ApprovalBallot(alternate_ballot)
            if alternate_ballot == truthful_first_ballot:
                continue
            found = testing_func(
                instance,
                truthful_profile,
                truthful_first_ballot,
                alternate_ballot,
                stop_event,
            )
            if found:
                return True
    
    return False


def _search_cost_vector(project_costs, config, stop_event) -> bool:
    if stop_event.is_set():
        return False
    projects = [Project(f"p{i+1}", cost) for i, cost in enumerate(project_costs)]
    min_budget = min(project_costs) + 1
    max_budget = sum(project_costs)

    for budget in range(min_budget, max_budget + 1):
        print(f"\tSearching with costs={project_costs} and L={budget}...")
        if stop_event.is_set():
            return False
        instance = Instance(projects, budget)
        for num_voters in range(config.min_voters, config.max_voters + 1):
            if stop_event.is_set():
                return False
            if _search_num_voters(num_voters, instance, projects, stop_event):
                stop_event.set()
                return True
    return False


def main() -> None:
    config: SearchConfig = parse_args()
    manager = mp.Manager()
    stop_event = manager.Event()
    match config.sat_func.lower():
        case "cost_sat":
            global sat_func
            sat_func = Cost_Sat
        case "cardinality_sat":
            sat_func = Cardinality_Sat
        case _:
            print(f"Unknown satisfaction function: {config.sat_func}")
            sys.exit(1)
    global testing_func
    testing_func = calculate_cost_sat
    match config.algo.lower():
        case "greedy_utilitarian_welfare":
            global pb_rule
            pb_rule = greedy_utilitarian_welfare
        case "method_of_equal_shares":
            pb_rule = method_of_equal_shares
        case "sequential_phragmen":
            pb_rule = sequential_phragmen
        case "maximin_support":
            pb_rule = maximin_support
        case "method_of_equal_shares_u":
            pb_rule = method_of_equal_shares
            testing_func = calculate_sat_with_ES_U
        case "method_of_equal_shares_add1u":
            pb_rule = method_of_equal_shares
            testing_func = calculate_sat_with_ES_add1U
        case _:
            print(f"Unknown algorithm: {config.algo}")
            sys.exit(1)

    for project_count in range(config.min_project_count, config.max_project_count + 1):
        if stop_event.is_set():
            break
        print(f"Searching with |P|={project_count}...")
        cost_vectors = list(
            itertools.combinations_with_replacement(
                range(config.min_value_per_project, config.max_value_per_project + 1),
                project_count,
            )
        )
        # Multi-thereading to speed up the search over different cost vectors
        with ThreadPoolExecutor(max_workers=min(len(cost_vectors), config.thread_count)) as executor:
            futures = {
                executor.submit(_search_cost_vector, costs, config, stop_event): costs
                for costs in cost_vectors
            }
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                if future.result():
                    stop_event.set()
                    break

    if stop_event.is_set():
        sys.exit(0)

    print("No scenario increased the first voter's Cost_Sat within the configured search bounds.")


if __name__ == "__main__":
    main()
