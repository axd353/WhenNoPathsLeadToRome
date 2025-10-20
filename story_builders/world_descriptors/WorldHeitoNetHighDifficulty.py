import pandas as pd
import random
import numpy as np
import re
import sys
sys.path.append('../..')
from utils.clingo_utils import run_clingo
from story_builders.FamilyRegChecker import FamilyRegChecker
from utils.clean_up_data import parse_fact_details
from utils.wrangle_derivations import extract_fact_atoms
from collections import defaultdict
from WorldGendersLocationsNoHornAmbFactsSp import WorldGendersLocationsNoHornAmbFactsSp
from WorldBioToy import WorldBioToy
import os
import glob
import ast

class WorldHeitoNetHighDifficulty(WorldBioToy):
    """
    A world-specific subclass that recursively merges base stories into a coherent
    set of entities and facts, supporting injective predicates to add new entities.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize parent and set up the list of predicates that can inject new entities.

        Args:
            *args, **kwargs: Passed to superclass initializer.
        """
        super().__init__(*args, **kwargs)
        # Predicates that can introduce previously unseen entities via relations
        self.num_combs_tried = 0

    def set_experiment_conds(self, config, story_ind):
        """
        Extend the parent method by loading a DataFrame of base stories for merging.

        Args:
            config (dict): Configuration dictionary; must contain 'merge_stories_df'.
            story_ind (int): Index of the current experiment/story.

        Returns:
            dict: Experiment settings inherited from the parent.

        Raises:
            ValueError: If 'merge_stories_df' is not provided.
        """
        exp_settings = super().set_experiment_conds(config, story_ind)
        if "merge_stories_df" in config:
            self.base_stories = pd.read_pickle(config["merge_stories_df"])
        else:
            raise ValueError("Configuration must include 'merge_stories_df' path to base stories DataFrame.")
        if "target_metric" in config:
            self.target_metric = config["target_metric"]
        else:
            self.target_metric = 'OPEC'
        return exp_settings

    def gen_entities(self, ent_num, program):
        """
        Recursively sample and merge rows from self.base_stories until at least ent_num
        distinct entities are included. Facts are renamed to avoid ID collisions, and
        injective predicates are used to link new entities where applicable.

        Args:
            ent_num (int): Minimum required distinct entities to cover.
            program (str): Initial ASP program string to append generated facts.

        Returns:
            entities (list[int]): Sorted list of covered entity IDs.
            new_program (str): Original program plus appended fact lines.
            added_facts (list[str]): Fact strings added during merging.
            fact_details_list (list[((int,int), str)]): Parsed fact details.

        Raises:
            RuntimeError: If base_stories is not loaded via set_experiment_conds.
        """
        if not hasattr(self, 'base_stories'):
            raise RuntimeError("Base stories not loaded; call set_experiment_conds first.")
        self.logger.debug(f''' TO gEnerate {ent_num} entites       ''')
        # Start with one random base story
        indices = list(self.base_stories.index)
        random.shuffle(indices)
        first_idx = indices.pop()
        p_row = self.base_stories.loc[first_idx]
        added_facts = list(p_row['story_facts'])
        while added_facts and isinstance(added_facts[0], list):
            added_facts = added_facts[0]
        self.logger.debug(f'added_facts we started with are {added_facts}')
        stories_used = [{'source':[p_row["story_index"], p_row["query_edge"], p_row["query_relation"], p_row["file_name"] ], 'ent_maps': None, self.target_metric: p_row[self.target_metric],   
                          'linked_ents':[] ,'story_facts': p_row["story_facts"],'derivation' : p_row["derivation_chain"] }]
        entities_set = set()
        for fact in added_facts:
            args, _ = parse_fact_details(fact)
            entities_set.update(args)

        used_indices = {first_idx}
        # self.logger.debug(f''' Starting off with the base {entities_set}. and \n {added_facts}       ''')
        # Continue merging until we have enough entities
        while len(entities_set) < ent_num:
            # Identify mergeable rows: any row whose predicate appears in added_facts
            candidates = []
            for idx in indices:
                row = self.base_stories.loc[idx]
                pred = row['target_pred']
                # predicate match only (ignores argument values)
                if any(fact.startswith(f"{pred}(") for fact in added_facts):
                    candidates.append(idx)
                    break
            if not candidates:
                self.num_combs_tried += 1
                self.logger.debug(f'ent num is {ent_num}, and program is {program}')
                return self.gen_entities(ent_num, program)  # no more stories can be linked

            # Sample one candidate to merge
            idx = random.choice(candidates)
            indices.remove(idx)
            used_indices.add(idx)
            p_row = self.base_stories.loc[idx]

            # Build mapping from old to new entity IDs
            name_map = defaultdict(lambda: None)
            link_set = set()
            matched_fact = None
            # Initialize head/tail mapping from any matching added fact
            for fact in random.sample(added_facts, len(added_facts)):
                if fact.startswith(f"{p_row['target_pred']}("):
                    (h, t), _ = parse_fact_details(fact)
                    name_map[p_row['target_head_ent']] = h
                    name_map[p_row['target_tail_ent']] = t
                    link_set.update([p_row['target_head_ent'], p_row['target_tail_ent']])
                    matched_fact = fact
                    break
            # Assign new IDs for any remaining unlinked entities in the story
            for fact in p_row['story_facts']:
                (a1, a2), _ = parse_fact_details(fact)
                for ent in (a1, a2):
                    if ent not in link_set:
                        new_id = max(entities_set) + 1
                        while new_id in entities_set or new_id in name_map.values():
                            new_id += 1
                        name_map[ent] = new_id
             
            # Remap and merge facts into added_facts
            for fact in p_row['story_facts']:
                # Try to match either rel(e1).  or rel(e1,e2).
                m_unary = re.match(r"^(\w+)\((\d+)\)\.$", fact)
                if m_unary:
                    rel = m_unary.group(1)
                    e1 = int(m_unary.group(2))
                    ne1 = name_map[e1]
                    new_fact = f"{rel}({ne1})."
                    if new_fact not in added_facts:
                        added_facts.append(new_fact)
                        entities_set.add(ne1)
                    continue

                m_binary = re.match(r"^(\w+)\((\d+),(\d+)\)\.$", fact)
                if m_binary:
                    rel = m_binary.group(1)
                    e1 = int(m_binary.group(2))
                    e2 = int(m_binary.group(3))
                    ne1 = name_map[e1]
                    ne2 = name_map[e2]
                    new_fact = f"{rel}({ne1},{ne2})."
                    if new_fact not in added_facts:
                        added_facts.append(new_fact)
                        entities_set.update([ne1, ne2])
                    continue

                # If it doesn't match either pattern, raise an error (unexpected format):
                raise ValueError(f"Invalid fact format in story_facts: {fact}")
            ## make mnatched-fact an inferred fact
            if matched_fact is not None and matched_fact in added_facts:
                added_facts.remove(matched_fact)
            stories_used.append({'source':[p_row["story_index"], p_row["query_edge"], p_row["query_relation"], p_row["file_name"] ], 'ent_maps': name_map.items(), 
                self.target_metric: p_row[self.target_metric] , 'linked_ents': link_set ,'story_facts': p_row["story_facts"], 'derivation' : p_row["derivation_chain"]})
        # Build final outputs
        entities = sorted(entities_set)
        new_program = program + ''.join(f + "\n" for f in added_facts)
        fact_details_list = [(parse_fact_details(f)[0], parse_fact_details(f)[1]) for f in added_facts]
        # if self.num_combs_tried < 2:
        #     self.logger.debug(f''' Failure  {self.num_combs_tried}, After merging we wind up with {entities} and {added_facts} . \n Stories originate from \n  {'\n'.join(str(d) for d in stories_used)}         ''')
            # self.logger.debug(f''' \n  the new program is {new_program}''')
            # self.logger.debug(f''' \n  the fact details are {fact_details_list}''')
        # ASP consistency check: retry if no models
        temp_models = run_clingo(new_program)
        if not temp_models:
            self.logger.debug(f''' \n The story {added_facts}  is raising contradictons ..retry.               ''')
            self.num_combs_tried += 1
            return self.gen_entities(ent_num, program)
        self.logger.debug(f'''FInal return after {self.num_combs_tried} failed tries for story: {self.story_number}.  After merging we wind up with {entities} and {added_facts} . \n Stories originate from \n 
                           {'\n'.join(str(d) for d in stories_used)}         ''')
        # self.logger.debug(f''' \n  the new program is {new_program}''')
        # self.logger.debug(f''' \n  the fact details are {fact_details_list}''')
        self.build_entity_lists(fact_details_list)
        return entities, new_program, added_facts, fact_details_list
    
def build_base_stories(
    results_dir: str,
    metric_filter_values: list[int],
    output_file: str,
    target_metric: str = 'OPEC',
    p_logger=None
) -> pd.DataFrame:
    """
    Traverse all CSVs in `results_dir` of the form `result_*.csv`, filter them by `target_metric`,
    and build a single DataFrame (`base_stories`) with one row per ASP “story.”  
    Finally, pickle the DataFrame to `output_file`. dIFFERENT TO THE ONE IN nORA, HERE TYPE DFINING FACTS ARE BINARY PREDS. 

    Args:
        results_dir (str):
            Directory containing ASP output CSVs named like `result_*.csv`.
        metric_filter_values (list[int]):
            List of difficulty metric values to keep.  Any row whose `target_metric`
            is not in this list will be dropped.
        output_file (str):
            Path to which the final DataFrame is saved via `pd.to_pickle(...)`.
        target_metric (str):
            Difficulty metric column name to filter by (default “OPEC”).
        p_logger:
            Optional logger (unused here).

    Returns:
        pd.DataFrame:
            A DataFrame with columns:
              - `target_pred` (str)
              - `target_head_ent` (int)
              - `target_tail_ent` (int)
              - `story_facts` (list[str])
              - `story_index` (int)
              - `query_edge` (str)
              - `query_relation` (str)
              - `file_name` (str)
              - `derivation_chain` (str)
              - `<target_metric>` (int)
    """
    pattern = os.path.join(results_dir, "result_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No files matching 'result_*.csv' in {results_dir!r}")

    rows = []
    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        usecols = ["query_edge", "query_relation", target_metric, "story_index",'OPEC', 'BL', 'ReasoningDepth',
                   "derivation_chain", "fact_choice_branches"]
        df = pd.read_csv(csv_path, usecols=usecols)
        df = df[df[target_metric].isin(metric_filter_values)]
        if df.empty:
            continue

        for _, row in df.iterrows():
            # parse query_edge "(h, t)"
            try:
                h_t = ast.literal_eval(row["query_edge"])
                target_head_ent, target_tail_ent = h_t
            except Exception:
                raise ValueError(f"Could not parse query_edge {row['query_edge']!r} in {file_name}")

            target_pred = row["query_relation"]

            # extract atoms from derivation_chain
            deriv_dict = ast.literal_eval(row["derivation_chain"])[0]
            atoms = []
            for rule_str in deriv_dict.values():
                atoms.extend(extract_fact_atoms(rule_str))
            derivation_facts = [f"{atom}." for atom in atoms]

            # monitored entities from derivation + target
            monitored_entities = {
                int(m.group(1)) for atom in atoms
                for m in [re.match(r"\w+\((\d+),(\d+)\)", atom)]
                if m
            }
            # Always include the target_head_ent & target_tail_ent
            monitored_entities.add(target_head_ent)
            monitored_entities.add(target_tail_ent)

            # extract type‐defining facts from fact_choice_branches
            fc_text = ast.literal_eval(row["fact_choice_branches"])[0]
            type_facts = []
            for line in fc_text.splitlines():
                line = line.strip()
                # match is_gene(i,i), is_compound(j,j), or is_disease(k,k)
                m = re.match(r"^(is_gene|is_compound|is_disease)\((\d+),\s*(\d+)\)\.$", line)
                if m:
                    ent1, ent2 = int(m.group(2)), int(m.group(3))
                    # only keep self‐loops for monitored entities
                    if ent1 == ent2 and ent1 in monitored_entities:
                        type_facts.append(line)

            # combine
            story_facts = derivation_facts + type_facts

            rows.append({
                "target_pred":      target_pred,
                "target_head_ent":  target_head_ent,
                "target_tail_ent":  target_tail_ent,
                "story_facts":      story_facts,
                "story_index":      int(row["story_index"]),
                "query_edge":       row["query_edge"],
                "query_relation":   target_pred,
                "file_name":        file_name,
                "derivation_chain": row["derivation_chain"],
                target_metric:      row[target_metric],
                'BL':      row['BL'],
                'OPEC':      row['OPEC'],
                'ReasoningDepth': row['ReasoningDepth'],
            })

    if rows:
        base_stories = pd.DataFrame(rows)
    else:
        cols = ["target_pred","target_head_ent","target_tail_ent",
                "story_facts","story_index","query_edge",
                "query_relation","file_name"]
        base_stories = pd.DataFrame(columns=cols)
    base_stories = base_stories.loc[:, ~base_stories.columns.duplicated()]
    print(f'Saving {base_stories.shape[0]} stories to {output_file} , {base_stories[target_metric].value_counts()} ')
    base_stories.to_pickle(output_file)
    return base_stories



def balance_gene_disease(base_data, random_seed=42):
    """
    Balances a dataframe by equalizing the counts of rows containing 'is_gene' and 'is_disease' predicates,
    while preserving all rows that contain both. Maintains reproducibility through explicit random seeding.
    
    Parameters:
    -----------
    base_data : pandas.DataFrame
        Input dataframe containing a 'story_facts' column with biological predicates.
        Expected format: Each row's 'story_facts' should be a list of strings containing
        predicates like 'is_gene(...)', 'is_disease(...)', etc.
        
    random_seed : int, optional (default=42)
        Seed value for all random operations (Python random, numpy, and pandas).
        Set for reproducible sampling. Change for different randomizations.
    
    Returns:
    --------
    pandas.DataFrame
        A new dataframe with balanced representation where:
        - All rows containing both 'is_gene' AND 'is_disease' are kept
        - The remaining rows are subsampled such that:
          * Count of 'is_gene'-only rows equals count of 'is_disease'-only rows
          * Total gene-related rows ≈ total disease-related rows
    
    Effects:
    --------
    - Prints before/after counts of biological predicates
    - Preserves compound information (though doesn't actively balance it)
    - Maintains all columns from original dataframe
    
    Example:
    --------
    >>> balanced_df = balance_gene_disease(raw_data, random_seed=42)
    Initial counts:
    Rows with gene: 1500
    Rows with disease: 1800  
    Rows with compound: 1200
    Rows with both gene and disease: 300
    
    Balanced counts (gene and disease):
    Rows with gene: 900
    Rows with disease: 900
    Rows with compound: 850
    Rows with both gene and disease: 300
    
    Notes:
    -----
    - Sampling is performed without replacement
    - Final dataframe is shuffled while maintaining reproducibility
    - Compound counts may decrease proportionally with subsampling
    - For complete reproducibility, set random_seed to a fixed value
    """
    # Set all random seeds for reproducibility
    random.seed(random_seed)  # NEW: For Python's random
    np.random.seed(random_seed)  # NEW: For numpy's random
    # pandas uses numpy's random under the hood
    # Create masks for each predicate
    has_gene = base_data['story_facts'].apply(lambda facts: any('is_gene(' in fact for fact in facts))
    has_disease = base_data['story_facts'].apply(lambda facts: any('is_disease(' in fact for fact in facts))
    has_compound = base_data['story_facts'].apply(lambda facts: any('is_compound(' in fact for fact in facts))
    
    # Initial counts
    print("Initial counts:")
    print(f"Rows with gene: {has_gene.sum()}")
    print(f"Rows with disease: {has_disease.sum()}")
    print(f"Rows with compound: {has_compound.sum()}")
    print(f"Rows with both gene and disease: {(has_gene & has_disease).sum()}\n")
    
    # Get rows that have both gene and disease
    both_gene_disease = has_gene & has_disease
    both_rows = base_data[both_gene_disease]
    
    # Get rows that have only gene or only disease
    only_gene = has_gene & ~has_disease
    only_disease = has_disease & ~has_gene
    
    # Determine which group is larger and needs subsampling
    gene_count = only_gene.sum()
    disease_count = only_disease.sum()
    
    # Target size is the count of rows with both plus the smaller group's count
    target_size = both_rows.shape[0] + min(gene_count, disease_count)
    
    if gene_count > disease_count:
        # Subsample only_gene to match disease_count
        subsampled_gene = base_data[only_gene].sample(n=disease_count, random_state=random_seed)
        balanced_data = pd.concat([both_rows, subsampled_gene, base_data[only_disease]])
    elif disease_count > gene_count:
        # Subsample only_disease to match gene_count
        subsampled_disease = base_data[only_disease].sample(n=gene_count, random_state=random_seed)
        balanced_data = pd.concat([both_rows, base_data[only_gene], subsampled_disease])
    else:
        # Counts are equal, no subsampling needed
        balanced_data = pd.concat([both_rows, base_data[only_gene], base_data[only_disease]])
    
    # Shuffle the final dataset
    balanced_data = balanced_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Updated counts
    has_gene_bal = balanced_data['story_facts'].apply(lambda facts: any('is_gene(' in fact for fact in facts))
    has_disease_bal = balanced_data['story_facts'].apply(lambda facts: any('is_disease(' in fact for fact in facts))
    has_compound_bal = balanced_data['story_facts'].apply(lambda facts: any('is_compound(' in fact for fact in facts))
    
    print("Balanced counts (gene and disease):")
    print(f"Rows with gene: {has_gene_bal.sum()}")
    print(f"Rows with disease: {has_disease_bal.sum()}")
    print(f"Rows with compound: {has_compound_bal.sum()}")
    print(f"Rows with both gene and disease: {(has_gene_bal & has_disease_bal).sum()}")
    
    return balanced_data



def build_base_stories_add_preds(
    results_dir: str,
    metric_filter_values: list[int],
    output_file: str,
    boost_preds: list[str],
    target_metric: str = 'OPEC'
) -> pd.DataFrame:
    """
    Traverse all CSVs in `results_dir` of the form `result_*.csv`, filter them by `target_metric`,
    and build a single DataFrame (`base_stories`) with one row per ASP “story.”  
    Finally, pickle the DataFrame to `output_file`. dIFFERENT TO THE ONE IN nORA, HERE TYPE DFINING FACTS ARE BINARY PREDS. 

    Args:
        results_dir (str):
            Directory containing ASP output CSVs named like `result_*.csv`.
        metric_filter_values (list[int]):
            List of difficulty metric values to keep.  Any row whose `target_metric`
            is not in this list will be dropped.
        output_file (str):
            Path to which the final DataFrame is saved via `pd.to_pickle(...)`.
        target_metric (str):
            Difficulty metric column name to filter by (default “OPEC”).
        p_logger:
            Optional logger (unused here).

    Returns:
        pd.DataFrame:
            A DataFrame with columns:
              - `target_pred` (str)
              - `target_head_ent` (int)
              - `target_tail_ent` (int)
              - `story_facts` (list[str])
              - `story_index` (int)
              - `query_edge` (str)
              - `query_relation` (str)
              - `file_name` (str)
              - `derivation_chain` (str)
              - `<target_metric>` (int)
    """
    pattern = os.path.join(results_dir, "result_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No files matching 'result_*.csv' in {results_dir!r}")

    rows = []
    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        usecols = ["query_edge", "query_relation", target_metric, "story_index",'OPEC', 'BL', 'ReasoningDepth',
                   "derivation_chain", "fact_choice_branches"]
        df = pd.read_csv(csv_path, usecols=usecols)
        df = df[df[target_metric].isin(metric_filter_values)]
        df = df[df["query_relation"].isin(boost_preds)]
        if df.empty:
            continue

        for _, row in df.iterrows():
            # parse query_edge "(h, t)"
            try:
                h_t = ast.literal_eval(row["query_edge"])
                target_head_ent, target_tail_ent = h_t
            except Exception:
                raise ValueError(f"Could not parse query_edge {row['query_edge']!r} in {file_name}")

            target_pred = row["query_relation"]

            # extract atoms from derivation_chain
            deriv_dict = ast.literal_eval(row["derivation_chain"])[0]
            atoms = []
            for rule_str in deriv_dict.values():
                atoms.extend(extract_fact_atoms(rule_str))
            derivation_facts = [f"{atom}." for atom in atoms]

            # monitored entities from derivation + target
            monitored_entities = {
                int(m.group(1)) for atom in atoms
                for m in [re.match(r"\w+\((\d+),(\d+)\)", atom)]
                if m
            }
            # Always include the target_head_ent & target_tail_ent
            monitored_entities.add(target_head_ent)
            monitored_entities.add(target_tail_ent)

            # extract type‐defining facts from fact_choice_branches
            fc_text = ast.literal_eval(row["fact_choice_branches"])[0]
            type_facts = []
            for line in fc_text.splitlines():
                line = line.strip()
                # match is_gene(i,i), is_compound(j,j), or is_disease(k,k)
                m = re.match(r"^(is_gene|is_compound|is_disease)\((\d+),\s*(\d+)\)\.$", line)
                if m:
                    ent1, ent2 = int(m.group(2)), int(m.group(3))
                    # only keep self‐loops for monitored entities
                    if ent1 == ent2 and ent1 in monitored_entities:
                        type_facts.append(line)

            # combine
            story_facts = derivation_facts + type_facts

            rows.append({
                "target_pred":      target_pred,
                "target_head_ent":  target_head_ent,
                "target_tail_ent":  target_tail_ent,
                "story_facts":      story_facts,
                "story_index":      int(row["story_index"]),
                "query_edge":       row["query_edge"],
                "query_relation":   target_pred,
                "file_name":        file_name,
                "derivation_chain": row["derivation_chain"],
                target_metric:      row[target_metric],
                'BL':      row['BL'],
                'OPEC':      row['OPEC'],
                'ReasoningDepth': row['ReasoningDepth'],
            })

    if rows:
        base_stories = pd.DataFrame(rows)
    else:
        cols = ["target_pred","target_head_ent","target_tail_ent",
                "story_facts","story_index","query_edge",
                "query_relation","file_name"]
        base_stories = pd.DataFrame(columns=cols)
    
    ##drop columns witrh same name 
    base_stories = base_stories.loc[:, ~base_stories.columns.duplicated()]
    print(f'Saving {base_stories.shape[0]} stories to {output_file} , {base_stories[target_metric].value_counts()} ')
    base_stories.to_pickle(output_file)
    return base_stories




if __name__ == "__main__":
    # --------------------HeitoNetHighOpec Base stories
    base_stories = '/home/anirban/BrainStorming/DatsetsGenerated/HetioNetToyR2Aug2025/datawrangling_benchmarks_project/data_output'
    OPEC_values =  [1,2]
    target_metric = 'OPEC'
    # output_file =  '../configs/test_HeitoNetHiGHoPEC.pkl'
    # build_base_stories(results_dir=base_stories, metric_filter_values=OPEC_values,output_file=output_file, target_metric= target_metric)
    ##too few are looking at genes.. HAND CRAFTED SOLIUTION -- first see output before moving forward
    # base_stories =  pd.read_pickle(output_file)
    # final_df = balance_gene_disease(base_stories)
    #### final_df.to_pickle(output_file)
    ##sample some predicates from those that are not included 
    # # output_file =  '../configs/test_HeitoNetHiGHoPECRarePreds.pkl'
    # # rare_preds = ['has_side_effect', 'no_side_effect', 'palliates', 'do_not_use_to_treat', 'no_palliate', 'ss1', 'ss2', 'ss3','ss4']
    # # build_base_stories_add_preds(results_dir=base_stories, metric_filter_values=OPEC_values, output_file=output_file, target_metric= target_metric, boost_preds=rare_preds)
    # df1 = pd.read_pickle('../configs/test_HeitoNetHiGHoPECRarePreds.pkl')
    # df2 = pd.read_pickle('../configs/test_HeitoNetHiGHoPEC.pkl')
    # sample_size = int(len(df1) * 0.2)
    # df2_sampled = df2.sample(n=sample_size, random_state=42)
    # combined_df = pd.concat([df1, df2_sampled])
    # combined_df.to_pickle('../configs/test_HeitoNetHiGHoPECRarePreds2.pkl')
    # -------------------- Base stories High RD
    # RD_values =  [4,5,6]
    # output_file =  '../configs/test_HeitoNetHighRD.pkl'
    # target_metric = 'ReasoningDepth'
    # # build_base_stories(results_dir=base_stories, metric_filter_values=RD_values,output_file=output_file, target_metric= target_metric)
    # base_stories =  pd.read_pickle(output_file)
    # final_df = balance_gene_disease(base_stories)
    # final_df.to_pickle(output_file)
 
    # --------------------HeitoNetHighOpec Base stories High BL
    base_stories = '/home/anirban/BrainStorming/DatsetsGenerated/HetioNetToyR2Aug2025/datawrangling_benchmarks_project/data_output'
    BL_values =  [1.25,1.33,1.20, 1]
    # #output_file =  '../configs/test_HeitoNetHiGHBL.pkl'
    target_metric = 'BL'
    # # build_base_stories(results_dir=base_stories, metric_filter_values=BL_values,output_file=output_file, target_metric= target_metric)
    # ##downsampling to get good stories ##too few are looking at genes.. HAND CRAFTED SOLIUTION -- first see output before moving forward
    # # base_stories =  pd.read_pickle(output_file)
    # # final_df = balance_gene_disease(base_stories)
    # # final_df.to_pickle(output_file)
    # ### downsampling some Bl 1 examples
    # balanced_data = pd.read_pickle(output_file)
    # print(balanced_data.shape)
    # bl_1_indices = balanced_data[balanced_data['BL'] == 1.00].index
    # # Randomly select 50% of these indices to remove
    # remove_count = int(len(bl_1_indices) * 0.5)
    # remove_indices = np.random.choice(bl_1_indices, size=remove_count, replace=False)
    # # Create new dataframe without these rows
    # final_df = balanced_data.drop(remove_indices)
    # print(final_df.shape)
    # final_df.to_pickle(output_file)
    
    ##sample some predicates from those that are not included 
    BL_values =  [1.25,1.33,1.20, 1]
    target_metric = 'BL'
    output_file =  '../configs/test_HeitoNetHiGHBLRarePreds.pkl'
    rare_preds = ['has_side_effect',  'do_not_use_to_treat', 'ss3']
    build_base_stories_add_preds(results_dir=base_stories, metric_filter_values=OPEC_values, output_file=output_file, target_metric= target_metric, boost_preds=rare_preds)
    df1 = pd.read_pickle('../configs/test_HeitoNetHiGHBLRarePreds.pkl')
    ss3_mask = df1['query_relation'] == 'ss3'
    ss3_rows = df1[ss3_mask]
    # Calculate 80% of ss3 rows to drop
    num_to_drop = int(len(ss3_rows) * 0.7)
    # Randomly select 80% of ss3 rows to drop
    rows_to_drop = ss3_rows.sample(n=num_to_drop, random_state=42)
    # Remove the selected rows from df1
    df1 = df1.drop(rows_to_drop.index)
    print(df1['query_relation'].value_counts())
    df2 = pd.read_pickle('../configs/test_HeitoNetHiGHBL.pkl')
    print(f'rare pred numbers {df1.shape[0]}')
    df2_sampled = df2[df2['BL'] > 1]
    print(f'really high BL: {df2_sampled.shape[0]}')
    combined_df = pd.concat([df1, df2_sampled])
    print(combined_df['query_relation'].value_counts())
    combined_df.to_pickle('../configs/test_HeitoNetHiGHBLRarePreds2.pkl')
