import pandas as pd
from ast import literal_eval
from typing import Dict, List, Set, Tuple, Any
import re
def find_rows_with_special_atoms(stories_df):
    """
    Finds rows where 'unique stable model' branches contain rules with:
    - has_property
    - belongs_to_group 
    - belongs_to
    in their rule bodies.
    
    Returns DataFrame containing only matching rows with additional columns indicating which atoms were found.
    """
    # Initialize lists to store results
    matching_rows = []
    has_property_indices = []
    belongs_to_group_indices = []
    belongs_to_indices = []
    
    for idx, row in stories_df.iterrows():
        try:
            # Parse the branch results and derivation chain
            branch_results = literal_eval(row['branch_results']) if isinstance(row['branch_results'], str) else row['branch_results']
            derivation_chain = literal_eval(row['derivation_chain']) if isinstance(row['derivation_chain'], str) else row['derivation_chain']
            
            # Flags for found atoms
            has_prop = False
            belongs_group = False
            belongs = False
            
            # Check each branch
            for branch_num, result in branch_results.items():
                if result == 'unique stable model' and branch_num in derivation_chain:
                    branch_derivations = derivation_chain[branch_num]
                    
                    for rel, deriv_str in branch_derivations.items():
                        # Split into derivation steps
                        steps = [s.strip() for s in deriv_str.split('|')]
                        
                        for step in steps:
                            if not step.startswith('fact:') and ':-' in step:
                                _, body = step.split(':-', 1)
                                body_atoms = [atom.strip() for atom in body.split(',')]
                                
                                for atom in body_atoms:
                                    if 'has_property(' in atom:
                                        has_prop = True
                                    if 'belongs_to_group(' in atom:
                                        belongs_group = True
                                    if 'belongs_to(' in atom:
                                        belongs = True
            
            # If any special atom found, add to results
            if has_prop or belongs_group or belongs:
                # Create a copy of the row with new columns
                new_row = row.copy()
                new_row['has_property_in_rules'] = has_prop
                new_row['belongs_to_group_in_rules'] = belongs_group
                new_row['belongs_to_in_rules'] = belongs
                matching_rows.append(new_row)
                
                # Track indices for each type
                if has_prop:
                    has_property_indices.append(idx)
                if belongs_group:
                    belongs_to_group_indices.append(idx)
                if belongs:
                    belongs_to_indices.append(idx)
                    
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Create DataFrame from matching rows
    if matching_rows:
        result_df = pd.DataFrame(matching_rows)
        
        # Print summary
        print(f"Found {len(result_df)} rows with special atoms in rules:")
        print(f"- has_property: {len(has_property_indices)} rows")
        print(f"- belongs_to_group: {len(belongs_to_group_indices)} rows") 
        print(f"- belongs_to: {len(belongs_to_indices)} rows")
        
        return result_df
    else:
        print("No rows found with special atoms in rules")
        return pd.DataFrame()

### ------------------------- Difficulty of derivation--------------------------------------------

def parse_derivation_dict(derivation: Dict[Any,str]) -> Tuple[Set[str], List[str]]:
    """
    From one branch’s derivation dict, extract:
      - facts: every string after 'fact:'
      - rules: every substring containing ':-'
      #TODO MISCOUNTING NUM FACTS 
    """
    facts: Set[str] = set()
    rules: List[str] = []
    for _, deriv_str in derivation.items():
        if not deriv_str:
            continue
        # split on " | " (with optional spaces)
        for part in re.split(r'\s*\|\s*', deriv_str.strip()):
            part = part.strip()
            if part.startswith('fact:'):
                facts.add(part[len('fact:'):].strip())
            elif ':-' in part:
                rules.append(part)
    return facts, rules

def extract_entities(facts: Set[str]) -> Set[str]:
    """
    From a set of fact-strings of the form predicate(arg1,arg2,...),
    return all unique entities (args), but for special binary predicates
    (has_property, belongs_to_group, belongs_to) ignore their second argument.
    """
    entities: Set[str] = set()
    special = {'has_property', 'belongs_to_group', 'belongs_to'}
    for fact in facts:
        m = re.match(r'(\w+)\((.*)\)', fact)
        if not m:
            continue
        pred, args_str = m.groups()
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        if pred in special and len(args) >= 2:
            entities.add(args[0])
        else:
            for arg in args:
                entities.add(arg)
    return entities

def group_branches_by_derivation(
    derivation_chain: Dict[int,Dict[Any,str]],
    branch_results: Dict[int,str]
) -> Dict[str,List[int]]:
    """
    Group branch indices by their stringified derivation dict.
    """
    groups: Dict[str,List[int]] = {}
    for idx, deriv in derivation_chain.items():
        key = str(deriv)
        groups.setdefault(key, []).append(idx)
    return groups

def calculate_rule_and_fact_density(
    derivation: Dict[Any,str],
    outcome: str
) -> List[float]:
    r"""
    Compute the four metrics for a single branch's derivation.

    Metrics:
      (A) outcome (str)
          - Taken directly from branch_results: either
            'unique stable model' or 'contradiction'.

      (B) rule_density (float)
          - Definition: total #rules / total #facts
          - #rules: count of all rules (any substring containing ':-')
          - #facts: count of all lines tagged with 'fact:'

      (C) non_trivial_rule_density (float)
          - Definition: #non-trivial_rules / total #facts
          - A rule is “non-trivial” if its body (right of ':-')
            has more than one top-level atom.
          - Top-level atoms are extracted via regex `r'[\w_]+\([^)]*\)'`
            to avoid splitting commas inside arguments.

      (D) rule_to_node_ratio (float)
          - Definition: total #rules / #unique_entities
          - Unique entities are all distinct argument-values appearing
            in facts (predicate(arg1,arg2,…)), except that for these
            special binary predicates:
              • has_property
              • belongs_to_group
              • belongs_to
            you **ignore** the second argument when collecting entities.

    Parameters:
    - derivation : dict[Any,str]
      One branch’s derivation mapping atom→derivation string.
    - outcome : str
      The branch result, passed through from branch_results.

    Returns:
    - List[float]: [ outcome, rule_density, non_trivial_rule_density, non_trivial_rule_to_node_ratio ]
    """
    facts, rules = parse_derivation_dict(derivation)

    # identify non-trivial rules
    non_trivial = []
    for r in rules:
        body = r.split(':-', 1)[1]
        # top-level atoms = predicate(arg...)
        atoms = re.findall(r'[\w_]+\([^)]*\)', body)
        if len(atoms) > 1:
            non_trivial.append(r)
    ## Counting the contradiction for negative refinements as a rule during calulcation of BL
    if outcome == 'unique stable model':
        num_rules = len(rules)
        num_non_trivial = len(non_trivial)
    if outcome == 'contradiction':
        num_rules = len(rules)+1
        num_non_trivial = len(non_trivial)+1
    num_facts = len(facts)
    num_nodes = len(extract_entities(facts))

    rule_density     = num_rules       / num_facts  if num_facts else 0.0
    non_triv_density = num_non_trivial / num_facts  if num_facts else 0.0
    non_triv_to_node = num_rules / num_nodes  if num_nodes else 0.0

    return [outcome, round(rule_density,2), round(non_triv_density,2), round(non_triv_to_node,2)]

def generate_fact_ratio_dict(
    grouped_branches: Dict[str,List[int]],
    derivation_chain: Dict[int,Dict[Any,str]],
    branch_results: Dict[int,str]
) -> Dict[Tuple[int,...],List[float]]:
    """
    For each unique derivation, compute metrics [A–D] once
    and assign them to its tuple of branch indices.
    """
    out: Dict[Tuple[int,...],List[float]] = {}
    for deriv_key, branches in grouped_branches.items():
        outcome = branch_results[branches[0]]
        metrics = calculate_rule_and_fact_density(
            derivation_chain[branches[0]], outcome
        )
        out[tuple(branches)] = metrics
    return out

def update_max_rule_to_fact_ratio1(
    fact_ratio_dict: Dict[Tuple[int,...],List[Any]]
) -> float:
    """
    Among only those entries whose outcome=='unique stable model',
    return the maximum of metric (D).
    """
    vals = [metrics[3] for metrics in fact_ratio_dict.values() if metrics[0]=='unique stable model']
    return max(vals) if vals else 0.0

def update_max_rule_to_fact_ratio2(
    fact_ratio_dict: Dict[Tuple[int,...],List[Any]]
) -> float:
    """
    Among only those entries,
    return the maximum of metric (D).
    """
    vals = [metrics[3] for metrics in fact_ratio_dict.values() ]
    return max(vals) if vals else 0.0


def save_dataframe_with_new_columns(input_file_path: str, output_file_path: str):
    """
    Reads a CSV with stringified 'derivation_chain' and 'branch_results' columns,
    computes two new columns via df.apply (without mutating those originals),
    and writes out a new CSV:
    
      • graph_complexity_stats      : str(fact_ratio_dict)
      • BL    : float
    
    All float metrics are rounded to 2 decimals.
    """
    df = pd.read_csv(input_file_path)

    def _compute_row_stats(row):
        # parse the two dict‐strings just for this row
        deriv_chain    = literal_eval(row['derivation_chain'])
        branch_outcomes = literal_eval(row['branch_results'])

        # group & compute
        gb  = group_branches_by_derivation(deriv_chain, branch_outcomes)
        frd = generate_fact_ratio_dict(      gb, deriv_chain, branch_outcomes)
        mx  = round(update_max_rule_to_fact_ratio(frd), 2)

        return pd.Series({
            'graph_complexity_stats'   : str(frd),
            'BL' : mx
        })

    # Apply row-wise; this does not overwrite the original columns
    df = df.join(df.apply(_compute_row_stats, axis=1))
    print(f''' Here are the columns of the dataframe {df.columns}.             ''')
    df.to_csv(output_file_path, index=False)
    print(f"Saved with new columns to {output_file_path}")


if __name__ == "__main__":
    ## Example1
    derivation_chain_1 = {
    0: {'!=(6,8)': '', 'living_in(11,6)': 'fact: living_in_same_place(7,11)  |  fact: living_in(7,6)  |  '
    '    living_in(11,6) :- living_in_same_place(7,11), living_in(7,6).', 'living_in(11,8)': 'fact: living_in(11,8)'},
    1: {'!=(6,8)': '', 'living_in(11,6)': 'fact: living_in_same_place(7,11)  |  fact: living_in(7,6)  |  '
    '    living_in(11,6) :- living_in_same_place(7,11), living_in(7,6).', 'living_in(11,8)': 'fact: living_in(11,8)'},
    2: {'nephew_of(3,9)': 'fact: maternal_aunt_of(9,3)  |  maternal_aunt_or_uncle_of(9,3) :- maternal_aunt_of(9,3).  |  '
           'aunt_or_uncle_of(9,3) :- maternal_aunt_or_uncle_of(9,3).  |  nibling_of(3,9) :- aunt_or_uncle_of(9,3).  |  '
           'fact: belongs_to_group(3,male)  |  nephew_of(3,9) :- nibling_of(3,9), belongs_to_group(3,male).'},
    3: {'nephew_of(3,9)': 'fact: maternal_aunt_of(9,3)  |  maternal_aunt_or_uncle_of(9,3) :- maternal_aunt_of(9,3).  |  '
           'aunt_or_uncle_of(9,3) :- maternal_aunt_or_uncle_of(9,3).  |  nibling_of(3,9) :- aunt_or_uncle_of(9,3).  |  '
           'fact: belongs_to_group(3,male)  |  nephew_of(3,9) :- nibling_of(3,9), belongs_to_group(3,male).'}
    }

    branch_results_1 = {0: 'contradiction', 1: 'contradiction', 2: 'unique stable model', 3: 'unique stable model'}

    # Running the function
    grouped_branches_1 = group_branches_by_derivation(derivation_chain_1, branch_results_1)
    print(grouped_branches_1)
    # fact_ratio_dict_1 = generate_fact_ratio_dict(grouped_branches_1, derivation_chain_1, branch_results_1)
    # max_rule_to_fact_ratio_1 = update_max_rule_to_fact_ratio(fact_ratio_dict_1)

    # print(f"Fact Ratio Dict: {fact_ratio_dict_1}")
    # print(f"Max Rule to Fact Ratio: {max_rule_to_fact_ratio_1}")
    ## Example 2 saving_dataset
    # file_path = '/home/anirban/BrainStorming/DatsetsGenerated/NonPathMeasurement/dataset1.csv'
    # save_dataframe_with_new_columns(file_path,file_path)
