import torch
from cosmo.event_logs import LOG_READERS
from cosmo.event_logs.reader import get_declare
from cosmo.models import NeuralNet
from cosmo.event_logs import ConstrainedContinuousTraces
from cosmo.utils import get_existing_experiments

def load_stuff(dataset: str = "sepsis", template: str = "existence"):
    """ Loading and setting up dataset and model"""

    runs = get_existing_experiments(project="cosmo-ltl", force_fetch=False)
    runs = runs[(runs.dataset == dataset) & (runs.template == template)]
    config = runs.to_dict("records")[0]
    config["grad_clip"] = None # from csv it comes as nan as it ruins the model loading

    log = LOG_READERS[config["dataset"]]()
    declare_constraints = get_declare(config["dataset"], templates=config["template"])
    log = log.sort_values(by=["case_id", "timestamp", "split"])
    not_found_constraints = set(log.case_id.unique()) - set(
        declare_constraints.case_id.unique()
    )
    log = log[~log.case_id.isin(not_found_constraints)]

    t, test_set = log[log["split"] == "train"], log[log["split"] == "test"]
    t = ConstrainedContinuousTraces(
        log=t,
        constraints=declare_constraints.copy(),
        continuous_features=["remaining_time_norm"],
        categorical_features=["activity"],
        dataset_name=config["dataset"] + "_" + config["template"],
        train=True,
        device=config["device"],
    )
    event_dataset = ConstrainedContinuousTraces(
        log=test_set,
        constraints=declare_constraints.copy(),
        continuous_features=["remaining_time_norm"],
        categorical_features=["activity"],
        dataset_name=config["dataset"] + "_" + config["template"],
        train=False,
        vocab=t.get_vocabs(),
        device=config["device"],
    )

    run_name = f"{config['dataset']}-templates={config['template']}-n_features={event_dataset.num_features}-lr={config['lr']}-bs={config['batch_size']}-wd={config['weight_decay']}-epochs{config['epochs']}-hidden={config['hidden_size']}-input={config['input_size']}-gradclip={config['grad_clip']}-nlayers={config['n_layers']}"
    state = torch.load(f"models/{config['dataset']}/{run_name}.pth")

    # model
    model = NeuralNet(
        vocabs=event_dataset.feature2idx,
        continuous_size=event_dataset.num_cont_features,
        constraint_size=event_dataset.num_constraints,
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        n_layers=int(config["n_layers"]),
    )
    model.load_state_dict(state["net"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, event_dataset, log, declare_constraints

def get_constraints(template, activity_to_simulate, activity_to_simulate_2=None):
    rules = {
        "existence": [
            f"Existence1[{activity_to_simulate}] | |", 
            f"Exactly1[{activity_to_simulate}] | |", 
            f"Absence1[{activity_to_simulate}] | |", 
        ],
        "choice": [
            f"Choice[{activity_to_simulate}, {activity_to_simulate_2}] | |", 
            f"Exclusive Choice[{activity_to_simulate}, {activity_to_simulate_2}] | |", 
        ],
        "positive relations": [
            f"Chain Response[{activity_to_simulate}, {activity_to_simulate_2}] | |", 
            f"Alternate Response[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Chain Precedence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Precedence[{activity_to_simulate}, {activity_to_simulate_2}] | |",
            f"Response[{activity_to_simulate}, {activity_to_simulate_2}] | |",
        ],
    }
    
    return rules[template]