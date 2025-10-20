import re
import networkx as nx
import random 
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import time 
import ast 
import pickle
import pandas as pd
import logging

HIERARCHY_DICT = {
    # Grandparent relationships
    'paternal_grandmother_of': ['paternal_grandparent_of', 'grandmother_of', 'grandparent_of'],
    'maternal_grandmother_of': ['maternal_grandparent_of', 'grandmother_of', 'grandparent_of'],
    'paternal_grandfather_of': ['paternal_grandparent_of', 'grandfather_of', 'grandparent_of'],
    'maternal_grandfather_of': ['maternal_grandparent_of', 'grandfather_of', 'grandparent_of'],
    'grandmother_of': ['grandparent_of'],
    'grandfather_of': ['grandparent_of'],
    'paternal_grandparent_of': ['grandparent_of'],
    'maternal_grandparent_of': ['grandparent_of'],
    
    # Parent-in-law relationships
    'mother_in_law_of': ['parent_in_law_of'],
    'father_in_law_of': ['parent_in_law_of'],
    
    # Child-in-law relationships
    'son_in_law_of': ['child_in_law_of'],
    'daughter_in_law_of': ['child_in_law_of'],
    
    # Child relationships
    'son_of': ['child_of'],
    'daughter_of': ['child_of'],
    
    # Sibling relationships
    'sister_of': ['sibling_of'],
    'brother_of': ['sibling_of'],
    
    # Aunt/Uncle relationships
    'uncle_of': ['aunt_or_uncle_of'],
    'maternal_uncle_of': ['maternal_aunt_or_uncle_of', 'uncle_of', 'aunt_or_uncle_of'],
    'paternal_uncle_of': ['paternal_aunt_or_uncle_of', 'uncle_of', 'aunt_or_uncle_of'],
    'aunt_of': ['aunt_or_uncle_of'],
    'paternal_aunt_of': ['paternal_aunt_or_uncle_of', 'aunt_of', 'aunt_or_uncle_of'],
    'maternal_aunt_of': ['maternal_aunt_or_uncle_of', 'aunt_of', 'aunt_or_uncle_of'],
    'paternal_aunt_or_uncle_of': ['aunt_or_uncle_of'],
    'maternal_aunt_or_uncle_of': ['aunt_or_uncle_of'],
    
    # Nibling relationships
    'niece_of': ['nibling_of'],
    'nephew_of': ['nibling_of'],
    
    # Grandchild relationships
    'granddaughter_of': ['grandchild_of'],
    'grandson_of': ['grandchild_of'],
    
    # Spouse relationships
    'husband_of': ['spouse_of'],
    'wife_of': ['spouse_of'],
    
    # Sibling-in-law relationships
    'sister_in_law_of': ['sibling_in_law_of'],
    'brother_in_law_of': ['sibling_in_law_of'],
    ## parents
    'mother_of': ['parent_of'],
    'father_of': ['parent_of'],
    #HetionetToy
    'use_to_treat':['palliates']
}

def parse_models(models_str: str, expected_num: int) -> list:
    """Parses the models string into a list of answer sets with their facts.
    
    Extracts answer sets from the models string and converts each fact into
    a tuple of (function_name, argument_numbers). Only includes facts marked
    as True and skips any False facts.

    Args:
        models_str: String containing the model data with answer sets enclosed in {}
        expected_num: Expected number of answer sets in the models string

    Returns:
        List of answer sets, where each answer set is a list of (function_name, args) tuples

    Raises:
        Exception: If the number of found answer sets doesn't match expected_num
    """
    stories = []
    # Find all story blocks enclosed by curly braces
    story_matches = re.findall(r'\{(.*?)\}', models_str, re.DOTALL)
    if len(story_matches) != expected_num:
        raise Exception(f"Expected {expected_num} stories, but found {len(story_matches)}")
    
    for story_text in story_matches:
        facts = []
        # Extract facts: function name, list of numbers, and truth flag
        fact_matches = re.findall(
            r"Function\('([^']+)'\s*,\s*\[([^\]]*)\]\s*,\s*(True|False)\)", 
            story_text
        )
        for func_name, numbers_str, truth_flag in fact_matches:
            if truth_flag != "True":
                continue  # Skip facts that are False
            # Extract numbers from the arguments
            num_matches = re.findall(r"Number\((\d+)\)", numbers_str)
            nums = [int(n) for n in num_matches]
            facts.append((func_name, nums))
        stories.append(facts)
    return stories


def extract_other_relationships(row: pd.Series, p_logger= None) -> list:
    """Extracts relationships between query edge nodes that appear in all answer sets.
    
    For a given row, finds all relationships between the query edge nodes that:
    1. Are present in every answer set
    2. Are different from the query relation
    3. Have exactly two arguments (binary relations)

    Args:
        row: Pandas Series representing a dataframe row with columns:
            - query_edge: Tuple of two numbers representing the edge
            - query_relation: String name of the primary relationship
            - num_answer_sets: Integer count of expected answer sets
            - models: String containing the answer sets data

    Returns:
        Sorted list of relationship names that meet the criteria, or empty list if none found
    """
    query_edge = row['query_edge']
    query_rel = row['query_relation']
    num_answer_sets = row['num_answer_sets']
    models_str = row['models'] if isinstance(row['models'], str) else str(row['models'])
    
    try:
        answer_sets = parse_models(models_str, num_answer_sets)
    except Exception as e:
        p_logger.exception(f"Error parsing models for row: {e}")
        raise    
    other_rels = set()
    # if (row['story_index'] ==  9) & (query_edge[0]==5) & (query_edge[1]==11):
    #     p_logger.debug(f''' For story index {row['story_index']} , query_edge= {query_edge}, 
    #         the model \n yields after parsing {answer_sets}.''')
    # For each answer set
    for p_ind, answer_set in enumerate(answer_sets):
        current_rels = set()
        # Iterate through each fact in the answer set
        for func_name, nums in answer_set:
            # Check if it's a binary relationship with the same edge as query_edge
            if (len(nums) == 2 and 
                nums[0] == query_edge[0] and 
                nums[1] == query_edge[1] and 
                func_name != query_rel):
                current_rels.add(func_name)
        # For the first answer set, initialize other_rels
        if p_ind==0:
            other_rels = current_rels
        else:
            # Keep only relationships present in all answer sets
            other_rels.intersection_update(current_rels)
        # if (row['story_index'] ==  9) & (query_edge[0]==5) & (query_edge[1]==11):
        #     p_logger.debug(f''' For story index {row['story_index']} , query_edge= {query_edge},
        #         the answer set {answer_set} \n yields {current_rels}, other rels become {other_rels}''')
    return sorted(list(other_rels))


def extract_other_credulous_rels(row: pd.Series, p_logger=None) -> dict:
    """
    Extracts strictly credulous relationships between query_edge nodes that:
    1. Are binary relations (2 arguments)
    2. Are different from the query_relation
    3. Appear in at least one but not all answer sets
    4. Are not in 'other_relationships'

    Args:
        row: Pandas Series with the following columns:
            - query_edge: Tuple of two numbers (subject, object)
            - query_relation: Primary relationship name (string)
            - num_answer_sets: Integer number of answer sets
            - models: String with model data
            - other_relationships: List of relations common to all answer sets

    Returns:
        Dictionary where keys are strictly credulous relation names,
        and values are proportions of answer sets in which they appear (0 < p < 1).
        Returns an empty dict if no such relationships exist.
    """
    query_edge = row['query_edge']
    query_rel = row['query_relation']
    num_answer_sets = row['num_answer_sets']
    models_str = row['models'] if isinstance(row['models'], str) else str(row['models'])
    other_relationships = set(row['other_relationships']) if isinstance(row['other_relationships'], list) else set()

    try:
        answer_sets = parse_models(models_str, num_answer_sets)
    except Exception as e:
        if p_logger:
            p_logger.exception(f"Error parsing models for row: {e}")
        raise

    # Count how many answer sets each relevant relation appears in
    credulous_counter = {}

    for answer_set in answer_sets:
        seen_in_this_set = set()
        for func_name, nums in answer_set:
            if (
                len(nums) == 2 and
                nums[0] == query_edge[0] and
                nums[1] == query_edge[1] and
                func_name != query_rel and
                func_name not in other_relationships
            ):
                seen_in_this_set.add(func_name)
        for rel in seen_in_this_set:
            credulous_counter[rel] = credulous_counter.get(rel, 0) + 1

    # Keep only those that appear in at least one but not all answer sets
    strictly_credulous = {
        rel: count / num_answer_sets
        for rel, count in credulous_counter.items()
        if 0 < count < num_answer_sets
    }

    return strictly_credulous

def add_strictly_credulous_column(df: pd.DataFrame, p_logger=None) -> pd.DataFrame:
    """
    Adds two columns to the DataFrame:
    1. 'strictly_cred_rel': a dictionary of strictly credulous relationships and their proportions.
    2. 'max_cred_proportion': maximum proportion among strictly credulous rels, or 0 if none.

    Args:
        df: Input DataFrame with required columns:
            - 'query_edge', 'query_relation', 'num_answer_sets', 'models', 'other_relationships'
        p_logger: Optional logger to capture errors or debugging info.

    Returns:
        Modified DataFrame with the two new columns.
    """

    def process_row(row):
        try:
            rel_dict = extract_other_credulous_rels(row, p_logger=p_logger)
            max_val = max(rel_dict.values()) if rel_dict else 0.0
            return pd.Series({'strictly_cred_rel': rel_dict, 'max_cred_proportion': max_val})
        except Exception as e:
            if p_logger:
                p_logger.exception(f"Error processing row {row.name}: {e}")
            return pd.Series({'strictly_cred_rel': {}, 'max_cred_proportion': 0.0})
    result = df.apply(process_row, axis=1)
    df = df.copy()
    df['strictly_cred_rel'] = result['strictly_cred_rel']
    df['max_cred_proportion'] = result['max_cred_proportion']
    return df

def create_all_relationships(row: pd.Series) -> list:
    """Combines query_relation with other_implied_relationships into a sorted list.
    
    Creates a comprehensive list of all relationships by combining:
    1. The primary query_relation
    2. Any other implied relationships found in all answer sets
    
    Args:
        row: Pandas Series representing a dataframe row with columns:
            - query_relation: String name of the primary relationship
            - other_implied_relationships: List of additional relationships

    Returns:
        Alphabetically sorted list containing all relationships (primary + implied)
    """
    # Get the primary relationship
    primary_rel = row['query_relation']
    
    # Get other implied relationships (ensure it's a list)
    other_rels = row.get('other_implied_relationships', []) or []
    
    # Combine and remove duplicates by converting to set
    all_rels = set(other_rels)
    all_rels.add(primary_rel)
    
    # Convert back to list and sort alphabetically
    return tuple(sorted(list(all_rels)))

def apply_hierarchy_rules(row: pd.Series) -> pd.Series:
    """Applies hierarchy rules to determine which rows to keep and correct alternatives.
    Should be good for new worlds without hierarchy betwen relationships, where 
    """
    query_rel = row['query_relation']
    other_rels = row.get('other_implied_relationships', []) or []
    
    # Check if any implied relationship is a parent of query_relation in hierarchy
    to_delete = False
    correct_alternatives = []
    
    for rel in other_rels:
        # Check if query_rel is in hierarchy under rel (rel is parent of query_rel)
        if rel in HIERARCHY_DICT and query_rel in HIERARCHY_DICT[rel]:
            to_delete = True
        # Check if rel is NOT in hierarchy under query_rel (rel is not child of query_rel)
        elif query_rel not in HIERARCHY_DICT or rel not in HIERARCHY_DICT[query_rel]:
            correct_alternatives.append(rel)
    
    row['to_Delete'] = to_delete
    row['correct_implied_alternatives'] = correct_alternatives
    return row

def is_implied(rel: str, row: pd.Series) -> bool:
    """Determines if a relationship is implied (i.e., missing in at least one story branch).
    
    A relationship is implied if the corresponding fact `rel(source, target)` 
    is *not present* in at least one story branch in `fact_choice_branches`.

    Args:
        rel: The relationship name to check.
        row: A dataframe row containing:
             - 'query_edge': Tuple as a string representing (source, target)
             - 'fact_choice_branches': Dict as a string where each value is a string of facts

    Returns:
        True if the fact is implied (missing in at least one branch), False otherwise.
    """
    try:
        edge = row['query_edge']
        fact_choice_branches = ast.literal_eval(row['fact_choice_branches'])
    except Exception as e:
        print(f"Error parsing edge or fact_choice_branches: {e}")
        return False  # Fallback

    source, target = edge
    target_fact = re.sub(r'\s+', '', f"{rel}({source},{target})")  # Normalized form

    for branch_text in fact_choice_branches.values():
        facts = {
            re.sub(r'\s+', '', line.strip().rstrip('.'))
            for line in branch_text.strip().split('\n')
            if line.strip()
        }
        if target_fact not in facts:
            return True  # Missing in this branch ⇒ implied

    return False  # Present in all branches ⇒ not implied

def process_dataframe(input: pd.DataFrame,p_logger=None) -> None:
    """Processes the input dataframe and saves the results.
    
    Performs the following operations:
    1. Loads dataframe from pickle file
    2. Drops columns with names ending in 'full'
    Here are the descriptions for the new columns:

    other_relationships: This column captures any relationships between entities that are either explicitly stated in the story or implied by the narrative.
    other_implied_relationships: This column identifies any relationships between entities that are implied by the established rules of the story's world.
    correct_implied_alternatives: This column contains relationships from the 'other_implied_relationships' column that are not lower in the hierarchical tree.

    Important: If the 'other_implied_relationships' column contains any relationships that are higher in the hierarchy tree, then that entire row will be deleted.
    4. Saves processed dataframe to output pickle file

    Args:
        input_file: Path to input pickle file containing the dataframe
        output_file: Path where processed dataframe should be saved

    Returns:
        None
    """
    # Load the dataframe from pickle file
    df = input.copy()
    # Create new columns
    df['other_relationships'] = df.apply(extract_other_relationships, axis=1, p_logger=p_logger  )
    # p_logger.debug(f'Obtained other_relationships ')
    df['other_implied_relationships'] = df.apply(
        lambda row: [rel for rel in row['other_relationships'] if is_implied(rel, row)]
        if row['other_relationships'] else [],
        axis=1
    )    
    # Apply hierarchy rules
    df = df.apply(apply_hierarchy_rules, axis=1)
    if p_logger is not None and 'to_Delete' in df.columns and df['to_Delete'].any():
        # Get the columns where at least one row has to_Delete=True
        columns_with_deletions = df.columns[df.isin({True}).any()].tolist()
        p_logger.warning(
            f"Low-hierarchy queries detected in the dataset. "
            f"Rows marked for deletion in columns: {columns_with_deletions}"
        )
        df = df[~df['to_Delete']]
    df = df.drop(columns=[ 'to_Delete'])
    df['alternate_labels_true'] = df['correct_implied_alternatives'].apply(lambda x: len(x) > 0)
    # p_logger.debug(f'Obtained other_relationships, implied_rels and correct rels {df[df['other_relationships'].apply(lambda x: isinstance(x, list) and len(x) > 0)][['query_edge', 'query_relation','other_relationships', 'other_implied_relationships1', 'other_implied_relationships', 'correct_implied_alternatives']]} \n \n')
    return df


def merging_multi_labels(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Merges multiple relationship labels per group defined by 'query_edge' and 'story_index'.
    Keeps the row with highest OPEC_pos_refn (primary) and then highest ReasoningDepth (secondary),
    aggregates metrics and derivations, and logs errors for full domination cases.

    Args:
        df (pd.DataFrame): Input DataFrame with columns:
            ['query_edge', 'story_index', 'query_relation', 'derivation_chain']
            plus metric columns:
            ['OPEC_pos_refn', 'OPEC', 'BL_no_contradiction', 'BL',
             'ReasoningWidth', 'ReasoningDepth_only_pos_derivations', 'ReasoningDepth']
        logger (logging.Logger): Logger instance for debug/info/error messages.

    Returns:
        pd.DataFrame: Processed DataFrame with new columns:
            ['other_implied_relationships1',
             'derivations_for_other_implied_relationships',
             'metric_for_other_implied_relationships']
    """
    # Define metrics to aggregate
    metrics = [
        'OPEC_pos_refn', 'OPEC', 'BL_no_contradiction', 'BL',
        'ReasoningWidth', 'ReasoningDepth_only_pos_derivations', 'ReasoningDepth'
    ]
    new_rows: List[pd.Series] = []

    # Group by keys
    for (edge, story_idx), group in df.groupby(['query_edge', 'story_index']):

        # Singleton group: keep as-is
        if len(group) == 1:
            row = group.iloc[0].copy()
            row['other_implied_relationships1'] = []
            row['derivations_for_other_implied_relationships'] = None
            row['metric_for_other_implied_relationships'] = None
            row['branch_results_for_other_implied_relationships'] = None
            new_rows.append(row)
            continue
        # logger.debug(f"Processing group query_edge={edge}, story_index={story_idx}, size={len(group)}")
        # Step 2: Hierarchical elimination based on global HIERARCHY_DICT
        dominated_idx = set()
        for idx_i, row_i in group.iterrows():
            implied = HIERARCHY_DICT.get(row_i['query_relation'], [])
            for idx_j, row_j in group.iterrows():
                if row_j['query_relation'] in implied:
                    dominated_idx.add(idx_j)

        remaining = group.drop(index=dominated_idx)

        # Error if no remaining rows (should not occur)
        if remaining.empty:
            err_msg = (
                f"All rows dominated in group query_edge={edge}, story_index={story_idx}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Step 2: Select retained row by highest OPEC_pos_refn then highest ReasoningDepth
        retained = (
            remaining
            .sort_values(['OPEC_pos_refn', 'ReasoningDepth'], ascending=False)
            .iloc[0]
        )
        retained_idx = retained.name

        # Identify all rows to delete: everything except retained
        to_delete_idx = group.index.difference([retained_idx])

        # Step 3A: Collect all deleted relations (hierarchical + ranking drops)
        other_rels = group.loc[to_delete_idx, 'query_relation'].tolist()
        
        # Step 3B: Compute max metrics across full group
        max_metrics = group[metrics].max()

        # Step 3C: Build metric dict for all relations
        metric_dict: Dict[str, List[List[Any]]] = {
            row['query_relation']: [[m, row[m]] for m in metrics]
            for _, row in group.iterrows()
        }

        # Step 3D: Build derivation dict for all relations
        derivation_dict: Dict[str, Any] = {
            row['query_relation']: row['derivation_chain']
            for _, row in group.iterrows()
        }

        alt_branch_results: Dict[str, Any] = {
            row['query_relation']: row['branch_outcomes']
            for _, row in group.iterrows()
        }

        # Assemble final retained row
        new_row = retained.copy()
        # Update metrics to group maxima
        for m in metrics:
            new_row[m] = max_metrics[m]
        new_row['other_implied_relationships1'] = other_rels
        new_row['metric_for_other_implied_relationships'] = metric_dict
        new_row['derivations_for_other_implied_relationships'] = derivation_dict
        new_row['branch_results_for_other_implied_relationships'] = alt_branch_results
        new_rows.append(new_row)

    # Combine all processed rows
    result_df = pd.DataFrame(new_rows).reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    logger = logging.getLogger('my_app')  # Name it whatever you want
    logger.setLevel(logging.DEBUG)      # Only warnings and above
    # Add a console handler to print messages
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    #--------experiment 1 merging_multi_labels 
    # f_stories = '/home/anirban/projects/benchmarkStories/benchmark_builders/stories/test_stories_bench_ASP_genders_no_amb50.csv'
    # stories = pd.read_csv(f_stories)
    # print(stories['story_index'].value_counts())
    # print(f'before merging multilables stories has shape {stories.shape}')
    # stories2 = merging_multi_labels(stories, logger)
    # print(f'After merging multilables stories has shape {stories2.shape}')
    # stories2.to_csv('/home/anirban/projects/benchmarkStories/benchmark_builders/stories/test_stories_bench_ASP_genders_no_amb50DownSized.csv')
    #--------experiment 2
    # stories = process_dataframe(stories) 
    # print(type(stories.iloc[0]['other_implied_relationships']))
    # ##experiment 3.. see if other relationships is correct
    # f_stories ='/home/anirban/projects/benchmarkStories/benchmark_builders/stories/test_stories_bench_ASP_genders1.csv '
    # stories= pd.read_csv(f_stories)
    # filtered_df = stories[(stories['query_edge']== str(((5, 11)))) & (stories['story_index']==9)]
    # row = filtered_df.iloc[0]
    # print(extract_other_relationships(row))
    #--------test the creduluos method
    dummy_models_str = """
    {Function('uncle_of', [Number(5), Number(6)], True), Function('colleague_of', [Number(5), Number(6)], True), Function('is_person', [Number(5)], True), Function('is_person', [Number(6)], True)}
    {Function('uncle_of', [Number(5), Number(6)], True), Function('is_person', [Number(5)], True), Function('is_person', [Number(6)], True)}
    """

    # Step 2: Define dummy row data
    dummy_row = {
        'query_relation': 'uncle_of',
        'query_edge': (5, 6),
        'models': dummy_models_str,
        'other_relationships': [],
        'num_answer_sets': 2,
        'strictly_cred_rel': {},  # placeholder
        'max_cred_proportion': 0.0  # placeholder
    }
    file_asp = '/home/anirban/projects/EdgeTRansformer/data_old/ambig/ambig-split/subset2_short_rules_high_variants.pkl'
    df2 = pd.read_pickle(file_asp)
    # Append dummy row as a DataFrame
    df2 = pd.concat([df2, pd.DataFrame([dummy_row])], ignore_index=True)
    df2 = add_strictly_credulous_column(df2, logger)
    # df3 =df2[df2['num_answer_sets']==2]
    # print(df2.iloc[0][['query_relation', 'query_edge', 'models','other_relationships', 'num_answer_sets', 'strictly_cred_rel','max_cred_proportion']])
    print(df2['max_cred_proportion'].value_counts())
    print(df2['query_relation'].value_counts())
    # print(df2['num_unique_variants'].value_counts())