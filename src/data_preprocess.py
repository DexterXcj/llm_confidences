from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset
import random
import pathlib

def convert_answers_to_columns(mmlu_dataset):
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    choices_list = mmlu_dataset['test']['choices']

    # Extract choices into separate columns
    A = [choices[0] for choices in choices_list]
    B = [choices[1] for choices in choices_list]
    C = [choices[2] for choices in choices_list]
    D = [choices[3] for choices in choices_list]

    return DatasetDict({
        'test': Dataset.from_dict({
            'question': mmlu_dataset['test']['question'],
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'answer': [index_to_letter[i] for i in mmlu_dataset['test']['answer']]
        })
    })


def shuffle_answers(example):
    answers = [
        example['Correct Answer'],
        example['Incorrect Answer 1'],
        example['Incorrect Answer 2'],
        example['Incorrect Answer 3']
    ]

    random.shuffle(answers)
    correct_position = answers.index(example['Correct Answer'])

    return {
        'A': answers[0],
        'B': answers[1],
        'C': answers[2],
        'D': answers[3],
        'Correct Answer Position': ['A', 'B', 'C', 'D'][correct_position]
    }


def convert_gpqa(gpqa_dataset):
    # Extract the relevant fields from gpqa_physics
    questions = gpqa_dataset['Question']
    choices = [[row['A'], row['B'], row['C'], row['D']] for row in gpqa_dataset]
    correct_answers = gpqa_dataset['Correct Answer Position']

    # Create a new Dataset with the desired structure
    gpqa_dataset = DatasetDict({
        'test': Dataset.from_dict({
            'question': questions,
            'A': [choice[0] for choice in choices],
            'B': [choice[1] for choice in choices],
            'C': [choice[2] for choice in choices],
            'D': [choice[3] for choice in choices],
            'answer': correct_answers
        })
    })

    return gpqa_dataset

def login_huggingface(token):
    login(token)

def preprocess():
    # create a new folder for all csv files
    current_dir = pathlib.Path.cwd()
    raw_data_dir = current_dir.parent / "raw"
    print(raw_data_dir)
    if not raw_data_dir.exists():
        raw_data_dir.mkdir(parents=True)

        ds_cp = convert_answers_to_columns(load_dataset("tasksource/mmlu", "college_physics")).shuffle(seed=40)
        ds_hp = convert_answers_to_columns(load_dataset("tasksource/mmlu", "high_school_physics")).shuffle(seed=40)
        ds_cb = convert_answers_to_columns(load_dataset("tasksource/mmlu", "college_biology")).shuffle(seed=40)
        ds_hb = convert_answers_to_columns(load_dataset("tasksource/mmlu", "high_school_biology")).shuffle(seed=40)
        ds_cc = convert_answers_to_columns(load_dataset("tasksource/mmlu", "college_chemistry")).shuffle(seed=40)
        ds_hc = convert_answers_to_columns(load_dataset("tasksource/mmlu", "high_school_chemistry")).shuffle(seed=40)
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")
        selected_columns = ds.select_columns(
            ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3',
            'High-level domain'])
        Physics_data = selected_columns['train'].filter(lambda x: x['High-level domain'] == 'Physics')
        Chem_data = selected_columns['train'].filter(lambda x: x['High-level domain'] == 'Chemistry')
        Bio_data = selected_columns['train'].filter(lambda x: x['High-level domain'] == 'Biology')

        Bio_data_processed = Bio_data.map(shuffle_answers)
        Chem_data_processed = Chem_data.map(shuffle_answers)
        Physics_data_processed = Physics_data.map(shuffle_answers)

        gpqa_physics = convert_gpqa(Physics_data_processed).shuffle(seed=40)
        gpqa_chem = convert_gpqa(Chem_data_processed).shuffle(seed=40)
        gpqa_bio = convert_gpqa(Bio_data_processed).shuffle(seed=40)

        # Save datasets as CSV files
        ds_cp['test'].to_csv(f'{raw_data_dir}/college_physics.csv')
        ds_hp['test'].to_csv(f'{raw_data_dir}/high_school_physics.csv')
        ds_cb['test'].to_csv(f'{raw_data_dir}/college_biology.csv')
        ds_hb['test'].to_csv(f'{raw_data_dir}/high_school_biology.csv')
        ds_cc['test'].to_csv(f'{raw_data_dir}/college_chemistry.csv')
        ds_hc['test'].to_csv(f'{raw_data_dir}/high_school_chemistry.csv')

        gpqa_physics['test'].to_csv(f'{raw_data_dir}/gpqa_physics.csv')
        gpqa_chem['test'].to_csv(f'{raw_data_dir}/gpqa_chemistry.csv')
        gpqa_bio['test'].to_csv(f'{raw_data_dir}/gpqa_biology.csv')
    else:
        print("Data already exist")
