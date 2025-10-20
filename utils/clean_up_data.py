import pandas as pd
import re
import logging
import ast
import pandas as pd
from ast import literal_eval
import seaborn as sns

class CleanData:
    def __init__(self, csv_path, logger=None):
        """
        Initialize the processor with the CSV file path.
        """
        self.csv_path = csv_path
        self.data = self.get_data()
        self.logger = None

    def get_data(self):
        """
        Reads the CSV file from the given path.
        """
        self.data = pd.read_csv(self.csv_path)
        return self.data

    def check_query_fact_in_stable_models(self):
        """
        Constructs a query fact string from 'query_relation' and 'query_edge'
        (e.g. query_relation 'father_of' and query_edge "(2,5)" become "father_of(2,5)"),
        and verifies that this query fact appears in every story of the stable_models column
        for every row of the dataframe.
        
        If the 'stable_models' column does not exist, it is computed by calling add_stable_models().
        
        Returns True if every row satisfies the constraint; otherwise, returns False.
        For every row that fails the containment check, outputs a dictionary containing:
          - row_index
          - query_edge
          - query_relation
          - story_edges
          - edge_types
        The output is logged using self.logger (if available) or printed.
        """
        # Ensure the data is loaded
        if self.data is None:
            self.get_data()
        # Ensure stable_models is present
        if 'stable_models' not in self.data.columns:
            self.add_stable_models()

        failed_rows = []

        # Iterate over each row and check if the query fact exists in every story.
        for idx, row in self.data.iterrows():
            # Construct the query fact: assume query_edge is in the form "(arg1,arg2,...)".
            query_relation = row['query_relation']
            query_edge = eval(row['query_edge'])
            query_fact = f"{query_relation}({query_edge[0]},{query_edge[1]})"
            # Retrieve the list of stories (each story is a list of fact strings)
            stable_models = row['stable_models']
            
            # Check if the query fact is in every story
            row_valid = all(query_fact in story for story in stable_models)
            if not row_valid:
                failed_info = {
                    "row_index": idx,
                    "query_edge": query_edge,
                    "query_relation": query_relation,
                    "story_edges": row.get('story_edges', None),
                    "edge_types": row.get('edge_types', None),
                    "stable_models": row.get('stable_models', None)
                }
                failed_rows.append(failed_info)
        
        # Report results
        if failed_rows:
            for info in failed_rows:
                if self.logger != None:
                    self.logger.error(info)
                    self.logger.error(f'''-------------------Next row---------------------- \n \n               ''')
                else:
                    print(info)
                    print((f'''-------------------Next row---------------------- \n \n               '''))
            return False
        return True

    def add_stable_models(self):
        """
        Processes the 'models' column to add a new column 'stable_models'.
        The 'models' column is expected to be a string representing a list of stories.
        Each story is enclosed in curly braces {}.
        Each fact in a story has the format:
            Function('function_name', [Number(x), Number(y), ...], True)
        For each fact, this method extracts the function name and arguments, converts the numbers to strings,
        and creates a fact string of the form: function_name(arg1, arg2, ...).
        If any fact ends with False, an exception is raised.
        """
        def parse_models(models_str, expected_num):
            stories = []
            # Find all story blocks enclosed by curly braces
            story_matches = re.findall(r'\{(.*?)\}', models_str, re.DOTALL)
            if len(story_matches) != expected_num:
                raise Exception(f"Expected {expected_num} stories, but found {len(story_matches)}")
            for story_text in story_matches:
                facts = []
                # Extract facts: function name, list of numbers, and truth flag.
                fact_matches = re.findall(r"Function\('([^']+)'\s*,\s*\[([^\]]*)\]\s*,\s*(True|False)\)", story_text)
                for func_name, numbers_str, truth_flag in fact_matches:
                    if truth_flag != "True":
                        raise Exception(f"Fact {func_name} with arguments [{numbers_str}] ends with False")
                    # Extract numbers from the arguments
                    num_matches = re.findall(r"Number\((\d+)\)", numbers_str)
                    if not num_matches:
                        raise Exception("No numbers found in fact arguments")
                    nums = [int(n) for n in num_matches]
                    # Transformation: use the numbers as is
                    transformed_args = [str(n) for n in nums]
                    fact_str = f"{func_name}({','.join(transformed_args)})"
                    facts.append(fact_str)
                stories.append(facts)
            return stories

        stable_models_list = []
        for idx, row in self.data.iterrows():
            models_str = row['models']
            num_ans = int(row['num_answer_sets'])
            try:
                stories = parse_models(models_str, num_ans)
            except Exception as e:
                raise Exception(f"Error processing row {idx}: {e}")
            stable_models_list.append(stories)
        self.data['stable_models'] = stable_models_list
        return self.data

    def overwrite_data(self):
        """
        Overwrites the original CSV file with the updated dataframe (which includes the new 'stable_models' column).
        """
        if self.data is None:
            raise Exception("Data is not loaded. Please run get_data() first.")
        self.data.to_csv(self.csv_path, index=False)



def log_dataframe_with_gaps(df: pd.DataFrame, logger: logging.Logger, 
                           column_list=None):
    """
    Logs a Pandas DataFrame with three line gaps between rows and one line gap 
    between columns within a row. Does not truncate any content.

    Args:
        df (pd.DataFrame): The input DataFrame.
        logger (logging.Logger): The logger instance to use.
        column_list (list, optional): List of columns to include. Defaults to None.
        level (int, optional): Logging level. Defaults to logging.DEBUG.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name or 'Series')  # Convert Series to DataFrame
    elif isinstance(df, pd.DataFrame):
        df = df
    else:
        raise TypeError("Input data must be a Pandas DataFrame or Series.")

    if column_list is not None:
        df = df[column_list]
    
    for index, row in df.iterrows():
        row_str = ""
        for col_name, value in row.items():
            row_str += f'{col_name}: {str(value)}'
            row_str += "\n"  # One line gap between columns
        logger.debug(row_str)
        logger.debug("\n\n") 


def process_program_for_places(program, exclude_n):
    """
    Finds all place IDs in the program except the excluded one.
    Returns a set of alternative place IDs.
    """
    alternative_places = set()
    lines = program.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('is_place(') and line.endswith(').'):
            try:
                n = int(line[len('is_place('):-2])
                if n != exclude_n:
                    alternative_places.add(n)
            except ValueError:
                continue
                
    return alternative_places


def get_max_branching_realized(program_str):
    """
    Finds the maximum number of alternative choices across all ambiguous facts in the program.
    Looks for {choice1, choice2,...} patterns that aren't rule heads.
    """
    max_branching = 1  # Default to 1 (no branching)
    
    # Split program into lines and process each line
    for line in program_str.split('\n'):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('%') or line.startswith('*'):
            continue
            
        # Skip rule definitions (lines with :-)
        if ':-' in line:
            continue
            
        # Find all {...} patterns in the line
        brace_parts = []
        start = 0
        while True:
            open_brace = line.find('{', start)
            if open_brace == -1:
                break
            close_brace = line.find('}', open_brace)
            if close_brace == -1:
                break
            brace_parts.append(line[open_brace+1:close_brace])
            start = close_brace + 1
            
        # Process each {...} content
        for part in brace_parts:
            # Split choices by comma and clean whitespace
            choices = [c.strip() for c in part.split(';') if c.strip()]
            # Update max branching if we found more choices
            if len(choices) > max_branching:
                max_branching = len(choices)
    
    return max_branching

def parse_fact_details(fact_str):
    '''
    Turns an added_fact into a fact_detail
    '''
    m = re.match(r"(\w+)\((\d+)(?:,(\d+))?\)\.", fact_str)
    if not m:
        raise ValueError(f"Invalid fact format: {fact_str}")
    rel = m.group(1)
    e1 = int(m.group(2))
    e2 = int(m.group(3)) if m.group(3) is not None else e1
    return (e1, e2), rel


if __name__ == "__main__":
    f_stories ='/home/anirban/projects/benchmarkStories/benchmark_builders/test_stories_bench_ASP_small.csv'
    obj =  CleanData(f_stories)
    obj.add_stable_models()
    obj.check_query_fact_in_stable_models()
    # print(obj.data.columns)
    # print(obj.data.iloc[0]['models'])
    # print(obj.data.iloc[0]['stable_models'])
