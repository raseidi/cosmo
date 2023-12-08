import pandas as pd
from cosmo.simulation.utils import load_stuff, get_constraints
from cosmo.simulation import simulate
    
""" ------------------ SETTINGS AND HYPERPARAMS ------------------"""
SETTINGS = {
    "sepsis": {
        "existence": {
            "activity_to_simulate": "IV Antibiotics",
        },
        "choice": {
            "activity_to_simulate": "ER Sepsis Triage",
            "activity_to_simulate_2": "LacticAcid",
        },
        "positive relations": {
            "activity_to_simulate": "ER Sepsis Triage",
            "activity_to_simulate_2": "CRP",
        },
    },
    "bpi13_problems": {
        "existence": {
            "activity_to_simulate": "Queued-Awaiting Assignment_complete",
        },
        "choice": {
            "activity_to_simulate": "Accepted-In Progress_complete",
            "activity_to_simulate_2": "Accepted-Wait_complete",
        },
        "positive relations": {
            "activity_to_simulate": "Accepted-Assigned_complete",
            "activity_to_simulate_2": "Accepted-In Progress_complete",
        },
    },
    "bpi20_permit": {
        "existence": {
            "activity_to_simulate": "Permit APPROVED by BUDGET OWNER",
        },
        "choice": {
            "activity_to_simulate": "Permit SUBMITTED by EMPLOYEE",
            "activity_to_simulate_2": "Send Reminder",
        },
        "positive relations": {
            "activity_to_simulate": "Permit APPROVED by ADMINISTRATION",
            "activity_to_simulate_2": "Permit APPROVED by BUDGET OWNER",
        },
    },    
}

def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sepsis")
    parser.add_argument("--n_simulations", type=int, default=20)
    parser.add_argument("--n_rule_sets", type=int, default=15)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # sepsis: relu
    # others: gelu
    
    args = read_args()
    dataset = args["dataset"]
    n_simulations = args["n_simulations"]
    n_rule_sets = args["n_rule_sets"]
    conditioned_log = pd.DataFrame()

    scores = []
    # """ ------------------ EXISTENCE SIMULATION ------------------"""
    template = "existence"
    activity_to_simulate = SETTINGS[dataset][template]["activity_to_simulate"]
    model, event_dataset, log, declare = load_stuff(
        dataset=dataset, 
        template=template
    )
    rules = get_constraints(template=template, activity_to_simulate=activity_to_simulate)

    cases = (
        log[(log.activity != "<EOS>") & (log.split == "test")]
        .groupby(["case_id"])
        .filter(lambda x: activity_to_simulate not in x.activity.values)
        # .case_id.unique()
    )
    cases = cases.groupby("case_id").size().sort_values().tail(n_rule_sets).index.values
    # cases = log[(log.case_id.isin(cases)) & (log.split == "test")].case_id.unique()

    data_config = dict(
        rules=rules,
        cases=cases, 
        event_dataset=event_dataset, 
        activity_to_simulate=activity_to_simulate, 
        activity_to_simulate_2=None,
        template=template,
    )
    simulated_log = simulate(
        model=model, 
        data_config=data_config,
        n_simulations=n_simulations,
        temperature=0.1,
    )

    conditioned_log = pd.concat([conditioned_log, simulated_log], ignore_index=True)
    # evaluation
    a = simulated_log.groupby(["case_id"]).apply(lambda x: activity_to_simulate in x.activity.values).sum(), simulated_log.case_id.nunique()
    print("Rate Existence (higher the better ", a)

    scores.append({"dataset": dataset, "template": template, "rate": a[0] / a[1]})

    """ ------------------ CHOICE SIMULATION ------------------

    for exclusive choice:

    A or B have to occur at least
    once but not both.

    if SATISFY RULE = 1, the query below should return 0
    if SATISFY RULE = 0, the query below should return 1

    """
    template = "choice"
    activity_to_simulate = SETTINGS[dataset][template]["activity_to_simulate"]
    activity_to_simulate_2 = SETTINGS[dataset][template]["activity_to_simulate_2"]
    CHOICE = "Exclusive Choice"
    SATISFY_RULE = 0

    model, event_dataset, log, declare = load_stuff(
        dataset=dataset, 
        template=template
    )
    rules = get_constraints(template=template, activity_to_simulate=activity_to_simulate, activity_to_simulate_2=activity_to_simulate_2)

    sorted(declare.columns)
    d_rule = f"{CHOICE}[{activity_to_simulate}, {activity_to_simulate_2}] | |"
    cases = declare[declare[d_rule] == SATISFY_RULE].case_id.unique()
    cases = log[(log.case_id.isin(cases)) & (log.split == "test")].groupby("case_id").size().sort_values().tail(n_rule_sets).index.values
    # cases = log[(log.case_id.isin(cases)) & (log.split == "test")].case_id.unique()

    data_config = dict(
        rules=rules,
        cases=cases, 
        event_dataset=event_dataset, 
        activity_to_simulate=activity_to_simulate, 
        activity_to_simulate_2=activity_to_simulate_2,
        template=template,
    )
    simulated_log = simulate(
        model=model, 
        data_config=data_config,
        n_simulations=n_simulations,
        temperature=0.1,
    )
    conditioned_log = pd.concat([conditioned_log, simulated_log], ignore_index=True)

    # (A and not B) OR (not A and B)
    cases = (
        simulated_log.groupby(["case_id"])
        .apply(lambda x: (activity_to_simulate in x.activity.values and activity_to_simulate_2 not in x.activity.values) or (activity_to_simulate not in x.activity.values and activity_to_simulate_2 in x.activity.values))
        .sum()
    )
    print("Rate Choice (higher the better)", cases / len(simulated_log.case_id.unique()))
    print()
    rate = cases / len(simulated_log.case_id.unique())
    scores.append({"dataset": dataset, "template": template, "rate": rate})

    # """ ------------------ POSITIVE SIMULATION ------------------

    # for chain response (if a, b is next)
    # if SATISFY RULE = 1, the rate should be 0
    # if SATISFY RULE = 0, the rate should be 1

    # """
    template = "positive relations"
    activity_to_simulate = SETTINGS[dataset][template]["activity_to_simulate"]
    activity_to_simulate_2 = SETTINGS[dataset][template]["activity_to_simulate_2"]
    CHOICE = "Chain Response"
    SATISFY_RULE = 1

    model, event_dataset, log, declare = load_stuff(
        dataset=dataset, 
        template=template
    )
    rules = get_constraints(
        template=template, 
        activity_to_simulate=activity_to_simulate, 
        activity_to_simulate_2=activity_to_simulate_2
    )
    d_rule = f"{CHOICE}[{activity_to_simulate}, {activity_to_simulate_2}] | |"
    cases = declare[declare[d_rule] == SATISFY_RULE].case_id.unique()
    cases = log[(log.case_id.isin(cases)) & (log.split == "test")].groupby("case_id").size().sort_values().tail(n_rule_sets).index.values
    # cases = log[(log.case_id.isin(cases)) & (log.split == "test")].case_id.unique()

    data_config = dict(
        rules=rules,
        cases=cases, 
        event_dataset=event_dataset, 
        activity_to_simulate=activity_to_simulate, 
        activity_to_simulate_2=activity_to_simulate_2,
        template=template,
    )
    simulated_log = simulate(
        model=model, 
        data_config=data_config,
        n_simulations=n_simulations,
        temperature=0.1,
    )
    conditioned_log = pd.concat([conditioned_log, simulated_log], ignore_index=True)


    def check_rule(grp):
        activities = grp.activity.values
        try:
            a = [ix for ix, e in enumerate(activities) if e == activity_to_simulate]
        except ValueError:
            return False
        try:
            b = [ix for ix, e in enumerate(activities) if e == activity_to_simulate_2]
        except ValueError:
            return False
        
        for i in a:
            if i+1 in b:
                return True
        return False

    cases = (
        simulated_log.groupby(["case_id"])
        .apply(check_rule)
        .sum()
    )
    print("Rate Positive relations (lower the better)", cases / len(simulated_log.case_id.unique()))
    rate = 1 - (cases / len(simulated_log.case_id.unique())) # here we subtract because check_rule returns True if the rule is satisfied; so we want to invert it
    scores.append({"dataset": dataset, "template": template, "rate": rate})


    """ SAVE """
    conditioned_log.to_csv(f"data/simulation/dataset={dataset}-n_simulations={n_simulations}-n_rule_sets={n_rule_sets}.csv", index=False)

    try:
        all_scores = pd.read_csv("data/simulation/scores.csv")
    except:
        all_scores = pd.DataFrame()

    all_scores = pd.concat([all_scores, pd.DataFrame(scores)], ignore_index=True)
    all_scores.to_csv("data/simulation/scores.csv", index=False)