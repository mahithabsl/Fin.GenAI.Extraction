import os
import yaml
from typing import List, Dict
from .prompt_template import BASE_TEMPLATE

def load_examples() -> List[Dict]:
    """
    Load financial examples from the YAML configuration file.
    
    Returns:
        List[Dict]: List of example dictionaries containing:
            - context: str
            - query: str
            - expected_answer: str
            
    Raises:
        FileNotFoundError: If the YAML file is not found
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, 'financial_examples.yaml')
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data['examples']

def format_examples(examples: List[Dict]) -> str:
    """
    Format a list of examples into a structured string for the prompt template.
    
    Args:
        examples (List[Dict]): List of example dictionaries containing:
            - context: str
            - query: str
            - expected_answer: str
            
    Returns:
        str: Formatted string containing all examples in the format:
            Example 1:
            Context: <context>
            Question: <query>
            Answer: <expected_answer>
    """
    formatted_examples = ""
    for i, example in enumerate(examples, 1):
        formatted_examples += f"""Example {i}:
Context: {example['context']}
Question: {example['query']}
Answer: {example['expected_answer']}

"""
    return formatted_examples

# Load examples from YAML
EXAMPLES = load_examples()

# Format the examples once
FORMATTED_EXAMPLES = format_examples(EXAMPLES)

# Create the final template with examples
QUERY_TEMPLATE = BASE_TEMPLATE.replace("{examples}", FORMATTED_EXAMPLES)

