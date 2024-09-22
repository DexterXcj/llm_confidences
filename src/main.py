
from models import models
from params import experiment_types
import demographic_class
import data_preprocess
import inquirer
import yaml

def prompt_questions():
    experiments = [
        inquirer.List("Experiment",
                    message="What experiment do you want to run?",
                    choices=experiment_types,
                ),
        inquirer.List("Model",
                    message="What model do you want to use?",
                    choices=models
                )
    ]
    return inquirer.prompt(experiments)

def load_config():
    with open("config.yaml") as config:
        return yaml.safe_load(config)


if __name__ == "__main__":
    config = load_config()
    user_answers = prompt_questions()
    data_preprocess.login_huggingface(config["huggingface"]["token"])
    data_preprocess.preprocess()

    demographic_class.run_experiment(user_answers["Experiment"], user_answers["Model"], config["groq"]["tokens"])