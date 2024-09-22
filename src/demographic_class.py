from api_utils import completion, initialize_client
import pandas as pd
import re
import warnings
import pathlib
from params import (
    ages,
    genders,
    races
)

# Suppress only FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def combine_dataframes(answer_only, estimate_only):
    # Create a copy of the answer_only DataFrame
    combined_df = answer_only.copy()
    
    # Fill missing values in answer_only from estimate_only where applicable
    combined_df['estimates'].fillna(estimate_only['estimates'], inplace=True)
    combined_df['llm_confidences'].fillna(estimate_only['llm_confidences'], inplace=True)
    
    # Fill missing values in estimate_only from answer_only
    combined_df['llm_answers'].fillna(answer_only['llm_answers'], inplace=True)
    combined_df['quiz_accuracy'].fillna(answer_only['quiz_accuracy'], inplace=True)
    
    # Add a new 'parameters' column with the value 'combined'
    combined_df['parameters'] = 'combined'
    
    # Return the combined DataFrame
    return combined_df

def load_dataset() -> list:
    current_dir = pathlib.Path.cwd()
    raw_data_dir = pathlib.Path(current_dir.parent / "raw")
    college_physics_path = f'{raw_data_dir}/college_physics.csv'
    high_school_physics_path = f'{raw_data_dir}/high_school_physics.csv'
    college_biology_path = f'{raw_data_dir}/college_biology.csv'
    high_school_biology_path = f'{raw_data_dir}/high_school_biology.csv'
    college_chemistry_path = f'{raw_data_dir}/college_chemistry.csv'
    high_school_chemistry_path = f'{raw_data_dir}/high_school_chemistry.csv'
    gpqa_physics_path = f'{raw_data_dir}/gpqa_physics.csv'
    gpqa_chemistry_path = f'{raw_data_dir}/gpqa_chemistry.csv'
    gpqa_biology_path = f'{raw_data_dir}/gpqa_biology.csv'

    college_physics_df = pd.read_csv(college_physics_path)[0:100]
    high_school_physics_df = pd.read_csv(high_school_physics_path)[0:150]
    college_biology_df = pd.read_csv(college_biology_path)
    high_school_biology_df = pd.read_csv(high_school_biology_path)
    college_chemistry_df = pd.read_csv(college_chemistry_path)
    high_school_chemistry_df = pd.read_csv(high_school_chemistry_path)[0:200]
    gpqa_physics_df = pd.read_csv(gpqa_physics_path)[0:180]
    gpqa_chemistry_df = pd.read_csv(gpqa_chemistry_path)[0:180]
    gpqa_biology_df = pd.read_csv(gpqa_biology_path)[0:70]

    # Define dataset collection
    return [
        ('college_physics_df', college_physics_df),
        ('high_school_physics_df', high_school_physics_df),
        ('college_biology_df', college_biology_df),
        ('high_school_biology_df', high_school_biology_df),
        ('college_chemistry_df', college_chemistry_df),
        ('high_school_chemistry_df', high_school_chemistry_df),
        ('gpqa_physics_df', gpqa_physics_df),
        ('gpqa_chemistry_df', gpqa_chemistry_df),
        ('gpqa_biology_df', gpqa_biology_df)
    ]


class QuizLikeDatasetProcessor_DemographicClass:
    def __init__(self, dataset, race, age, gender, experiment_type, role_play=False, model='llama3-8b-8192'):
        self.dataset = dataset
        self.race = race
        self.age = age
        self.gender = gender
        self.experiment_type = experiment_type
        self.role_play = role_play
        self.model = model
        self.correct_answers_list = dataset['answer'].tolist()
        self.max_questions = self.get_max_num_questions()
        self.quizzes = self.get_quizzes()

        # Create prompts based on the experiment type
        self.prompt_quiz_list = self.create_prompts()
        self.llm_answers = []
        self.estimates = []

        self.process_dataset()

        # Create DataFrame to store estimates and LLM answers
        self.dataframe = self.create_dataframe()

    def get_max_num_questions(self):
        return (len(self.dataset['question']) // 10) * 10

    def get_quizzes(self, questions_per_quiz=10):
        num_questions = min(self.max_questions, len(self.dataset['question']))
        selected_questions = {
            'question': self.dataset['question'][:num_questions].tolist(),
            'A': self.dataset['A'][:num_questions].tolist(),
            'B': self.dataset['B'][:num_questions].tolist(),
            'C': self.dataset['C'][:num_questions].tolist(),
            'D': self.dataset['D'][:num_questions].tolist(),
        }
        quizzes = [
            {
                'question': selected_questions['question'][i:i + questions_per_quiz],
                'A': selected_questions['A'][i:i + questions_per_quiz],
                'B': selected_questions['B'][i:i + questions_per_quiz],
                'C': selected_questions['C'][i:i + questions_per_quiz],
                'D': selected_questions['D'][i:i + questions_per_quiz],
            }
            for i in range(0, num_questions, questions_per_quiz)
        ]
        return quizzes

    def create_prompts(self):
        if self.experiment_type == 'estimate_only':
            return self.prompt_create_demographic_quiz_estimation_only()
        elif self.experiment_type == 'answer_only':
            return self.prompt_create_demographic_quiz_answer_only()
        elif self.experiment_type == 'answer_and_estimate':
            return self.prompt_create_demographic_quiz_answer_estimation()
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

    def prompt_create_demographic_quiz_answer_only(self):
        front_prompt = f"You are a {self.race} {self.gender} {self.age}. Read the question and provide your answer (only the option letter). Do not provide any explanation.\nQuestions:\n"
        back_prompt = """Now, please answer these questions:
        1. <Only Option Letter >
        2. <Only Option Letter >
        3. <Only Option Letter >
        4. <Only Option Letter >
        5. <Only Option Letter >
        6. <Only Option Letter >
        7. <Only Option Letter >
        8. <Only Option Letter >
        9. <Only Option Letter >
        10. <Only Option Letter >
        \n."""
        
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def prompt_create_demographic_quiz_estimation_only(self):
        front_prompt = f"You are a {self.race} {self.gender} {self.age}. Read the question, estimate how many questions you can answer correctly (choose a number from 0-10). Do not provide any explanation.\nQuestions:\n"
        back_prompt = "Estimate how many questions you can answer correctly: <only choose one number from 0-10>.\n"
        
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def prompt_create_demographic_quiz_answer_estimation(self):
        front_prompt = f"A is a {self.race} {self.gender} {self.age}. Read the question, consider A’s information, and then guess A’s answer (provide only the option letter). Do not provide any explanation. Finally, estimate how many questions (0-10) you think A can answer correctly.\nQuestions:\n"
        back_prompt = """Now, please answer these questions in the following format:
        1. <Only Option Letter >
        2. <Only Option Letter >
        3. <Only Option Letter >
        4. <Only Option Letter >
        5. <Only Option Letter >
        6. <Only Option Letter >
        7. <Only Option Letter >
        8. <Only Option Letter >
        9. <Only Option Letter >
        10. <Only Option Letter >
        Estimate how many questions you can answer correctly: <only choose one number from 0-10>.
        \n."""
        
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def _create_quiz_prompts(self, front_prompt, back_prompt):
        quizzes = []
        for quiz in self.quizzes:
            questions = []
            for i in range(len(quiz['question'])):
                question = quiz['question'][i]
                
                # Format the question and choices
                formatted_question = f"{i + 1}. {question}\n"
                formatted_question += f"   A. {quiz['A'][i]}\n"
                formatted_question += f"   B. {quiz['B'][i]}\n"
                formatted_question += f"   C. {quiz['C'][i]}\n"
                formatted_question += f"   D. {quiz['D'][i]}\n"
                
                questions.append(formatted_question)
    
            all_questions = "\n".join(questions)
            full_prompt = front_prompt + all_questions + back_prompt
            quizzes.append(full_prompt)
    
        return quizzes

    def process_dataset(self):
        print(f'Processing dataset for {self.experiment_type}...')     
        for complete_prompt in self.prompt_quiz_list:
            print(complete_prompt)
            response = completion(complete_prompt, self.model)
            print(response)
            
            if self.experiment_type == 'answer_and_estimate':
                answers, estimate = self.extract_answers_and_estimate(response)
                self.llm_answers.extend(answers)
                self.estimates.append(estimate)
            
            elif self.experiment_type == 'answer_only':
                answers = self.extract_answers_only(response)
                self.llm_answers.extend(answers)
            
            elif self.experiment_type == 'estimate_only':
                estimate = self.extract_estimates_only(response)
                self.estimates.append(estimate)
            else:
                raise ValueError(f"Unknown experiment type: {self.experiment_type}")

    def extract_answers_only(self, response):
        lines = response.strip().split('\n')
        answers = []

        for line in lines:
            match = re.match(r'^\d+\.(.*)$', line.strip())
            if match:
                answer = match.group(1).strip()
                answers.append(answer if answer in ['A', 'B', 'C', 'D'] else 'C')  # Default to 'C' if invalid answer

        return answers

    def extract_estimates_only(self, response):
        estimate_match = re.search(r'(\d+)', response)
        return int(estimate_match.group(1)) if estimate_match else 0  # Default to 0 if no estimate is found
    
    def extract_answers_and_estimate(self, response):
        lines = response.strip().split('\n')
        
        # Initialize variables
        answers = []
        estimate = None
        
        # Extract answers
        for line in lines:
            match = re.match(r'^\d+\.\s*(.+)$', line.strip())
            if match:
                answer = match.group(1).strip()
                if answer in ['A', 'B', 'C', 'D']:
                    answers.append(answer)
                else:
                    answers.append('C')  # Default to 'C' if invalid

        # Extract estimate
        estimate_line = lines[-1]
        estimate_match = re.search(r'(\d+)', estimate_line)
        if estimate_match:
            estimate = int(estimate_match.group(1))
        
        return answers, estimate


    def calculate_quiz_accuracy(self, llm_answers, correct_answers):
        """Calculate accuracy for each quiz as the proportion of correct answers."""
        correct_count = sum([1 for llm, correct in zip(llm_answers, correct_answers) if llm == correct])
        return correct_count / len(correct_answers)
    
    def create_dataframe(self):
        num_questions = self.max_questions
        num_groups = num_questions // 10
    
        # Ensure all lists are the same length
        llm_answers_filled = (self.llm_answers + ['N/A'] * num_questions)[:num_questions]
        correct_answers_filled = (self.correct_answers_list + ['N/A'] * num_questions)[:num_questions]
        questions = self.dataset['question'][:num_questions].tolist()
    
        # Pack every 10 answers into a list
        def pack_into_lists(data, size=10):
            return [data[i:i + size] for i in range(0, len(data), size)]
    
        packed_llm_answers = pack_into_lists(llm_answers_filled) if self.experiment_type != 'estimate_only' else [['N/A'] * 10] * num_groups
        packed_correct_answers = pack_into_lists(correct_answers_filled)
        packed_questions = pack_into_lists(questions)
    
        # Calculate quiz accuracy for each quiz
        quiz_accuracies = [self.calculate_quiz_accuracy(llm_ans, correct_ans) 
                           for llm_ans, correct_ans in zip(packed_llm_answers, packed_correct_answers)]
    
        # Adjust estimates list if experiment_type is 'answer_only'
        if self.experiment_type == 'answer_only':
            estimates_filled = [None] * num_groups
            llm_confidences = [None] * num_groups
        else:
            estimates_filled = (self.estimates + [None] * num_groups)[:num_groups]
            llm_confidences = [i / 10 if i is not None else None for i in estimates_filled]
    
        data = {
            'quiz_id': [f'Quiz_{i+1}' for i in range(num_groups)],
            'question': packed_questions,
            'gender': self.gender,
            'age': self.age,
            'race': self.race,
            'llm_answers': packed_llm_answers,
            'correct_answers': packed_correct_answers,
            'estimates': estimates_filled,
            'llm_confidences': llm_confidences,
            'quiz_accuracy': quiz_accuracies,
            'parameters': self.experiment_type,
            'model_name': self.model
        }
    
        return pd.DataFrame(data)
    
def run_experiment(experiment_type: str, model: str, api_keys: list):
    # create a new folder for all results
    current_dir = pathlib.Path.cwd()
    result_dir = current_dir / "processed" / experiment_type
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    else:
        print("Directory already exist")

    datasets = load_dataset()
    initialize_client(api_keys)

    # Loop through each combination and store the data as CSV
    for dataset_name, dataset in datasets:
        for age in ages:
            for gender in genders:
                for race in races:
                    # Create an instance of QuizLikeDatasetProcessor_DemographicClass
                    processor_estimate = QuizLikeDatasetProcessor_DemographicClass(dataset, race, age, gender, 'estimate_only', model)
                    processor_answer = QuizLikeDatasetProcessor_DemographicClass(dataset, race, age, gender, 'answer_only', model)
                    
                    # Define the filename including the dataset name
                    filename = f"{dataset_name}_{race}_{age}_{gender}.csv"
                    
                    # Save DataFrame to CSV
                    combine_dataframes(processor_answer.dataframe, processor_estimate.dataframe).to_csv(result_dir / filename, index=False)
                    
                    print(f"Saved {filename} to {result_dir}")