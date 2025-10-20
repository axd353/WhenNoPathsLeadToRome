import pandas as pd
import random
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

class WorldNoRAHighOPEC(WorldGendersLocationsNoHornAmbFactsSp):
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
        self.injective_predicates = [
            'paternal_grandmother_of', 'maternal_grandmother_of',
            'paternal_grandfather_of', 'maternal_grandfather_of',
            'mother_of', 'father_of', 'spouse_of',
            'mother_in_law_of', 'father_in_law_of'
        ]
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
            # Loop to propagate via injective predicates without early exit.. figure out what entites are linked between stories
            changed = True
            while changed:
                changed = False
                for fact in p_row['story_facts']:
                    (a1, a2), rel = parse_fact_details(fact)
                    if rel in self.injective_predicates and a2 in link_set:
                        mapped = name_map[a2]
                        # find matching existing fact to infer a1's mapped ID
                        for existing in added_facts:
                            (e1, e2), ex_rel = parse_fact_details(existing)
                            if ex_rel == rel and (e1, e2) == (e1, mapped):
                                # either assign or validate mapping
                                if a1 not in link_set:
                                    link_set.add(a1)
                                    name_map[a1] = e1
                                    changed = True
                                else:
                                    # ensure consistent mapping if already present
                                    if name_map[a1] != e1:
                                        self.logger.debug(
                                            f"Inconsistent mapping for entity {a1} in present story {self.story_number} :"
                                            f''' was mapped to {name_map[a1]}, now we found {e1}.. we are working through {existing} 
                                            in added facts. And {fact} in the new base story, {a2} in present story already mapped to {mapped} that we attempt to merge.\n
                                            We are trying to merge {matched_fact} in added_facts with {p_row['query_relation']} ,  {p_row['query_edge']} in new base story. 
                                            \n see {stories_used}. \n the new base story-facts are {p_row['story_facts']} and
                                            \n derivation chain is {p_row['derivation_chain']} . present added facts are {added_facts}.     '''
                                        )
                                        self.num_combs_tried += 1
                                        return self.gen_entities(ent_num, program)                      

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
        # Regularity check: retry if program fails regularity criteria
        if not self.p_regularity_checker.is_regular(new_program, logger=self.logger):
            self.num_combs_tried += 1
            return self.gen_entities(ent_num, program)
        # ASP consistency check: retry if no models
        temp_models = run_clingo(new_program)
        if not temp_models:
            self.logger.debug(f''' \n The story {added_facts}  is raising contradictons ..retry.               ''')
            self.num_combs_tried += 1
            return self.gen_entities(ent_num, program)
        self.logger.debug(f'''FInal return after {self.num_combs_tried} failed tries for story: {self.story_number}.  After merging we wind up with {entities} and {added_facts} . \n Stories originate from \n  {'\n'.join(str(d) for d in stories_used)}         ''')
        # self.logger.debug(f''' \n  the new program is {new_program}''')
        # self.logger.debug(f''' \n  the fact details are {fact_details_list}''')
        self.build_entity_lists(fact_details_list)
        self.logger.debug(f''' for story: {self.story_number} we have {self.people} and {self.places}      ''')
        return entities, new_program, added_facts, fact_details_list






def build_base_stories(
    results_dir: str,
    metric_filter_values: list[int],
    output_file: str,
    target_metric= 'OPEC',
    p_logger= None
) -> pd.DataFrame:
    """
    Traverse all CSVs in `results_dir` of the form `result_*.csv`, filter them by OPEC,
    and build a single DataFrame (`base_stories`) with one row per ASP “story.”  
    Finally, pickle the DataFrame to `output_file`.

    Args:
        results_dir (str):
            Path to a directory containing ASP output CSVs named like `result_*.csv`.
            (No recursive lookup; only files directly under this directory are scanned.)
        metric_filter_values (list[int]):
            List of difficulty metric  values to keep.  Any row whose `OPEC` is NOT in this list
            will be dropped.
        output_file (str):
            Path (including filename) to which the final `base_stories` DataFrame is saved
            via `pd.to_pickle(...)`.

    Returns:
        pd.DataFrame:
            A DataFrame with columns:
              - `target_pred`        (str): same as `query_relation`
              - `target_head_ent`    (int): head entity from `query_edge`
              - `target_tail_ent`    (int): tail entity from `query_edge`
              - `story_facts`        (list[str]): list of fact‐strings, each ending with “.”,
                                           combining atoms from the derivation chain plus any
                                           unary facts (rel(ent).) from the choice‐branches for 
                                           monitored entities.
              - `story_index`        (int): carried over from the ASP CSV
              - `query_edge`         (str): carried over from the ASP CSV
              - `query_relation`     (str): carried over from the ASP CSV
              - `file_name`          (str): just the CSV basename where this row originated

    Detailed Steps:
      All csv files must be stories without any ambiguity for this to work
    Returns:
        pd.DataFrame: The completed `base_stories` DataFrame (then also pickled to `output_file`).

    Raises:
        FileNotFoundError: If `results_dir` does not exist or has no matching CSVs.
        ValueError: If any `query_edge` fails to parse as a two‐element tuple, or if `derivation_chain`
                    / `fact_choice_branches` aren’t valid single‐element lists of dicts.
    """
    # 1. Gather all matching CSV filenames (no recursion)
    pattern = os.path.join(results_dir, "result_*.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        raise FileNotFoundError(f"No files matching 'result_*.csv' in {results_dir!r}")
    print(f' will work off {csv_files}')
    rows: list[dict] = []
    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)

        # 2.a) Read only relevant columns and filter target_metric
        usecols = ["query_edge","query_relation",target_metric,"story_index","derivation_chain","fact_choice_branches"]
        df = pd.read_csv(csv_path, usecols=usecols)

        # 2.b) Keep only rows whose target_metric ∈ metric_filter_values
        df = df[df[target_metric].isin(metric_filter_values)]
        if df.empty:
            continue

        # 3. Process each filtered row
        for _, row in df.iterrows():
            # 3.a) Parse the query_edge "(h, t)" → tuple(int, int)
            try:
                h_t: tuple[int, int] = ast.literal_eval(row["query_edge"])
                if not (isinstance(h_t, tuple) and len(h_t) == 2 and
                        all(isinstance(x, int) for x in h_t)):
                    raise ValueError
            except Exception:
                raise ValueError(f"Could not parse query_edge {row['query_edge']!r} in {file_name}")

            target_head_ent, target_tail_ent = h_t
            target_pred = row["query_relation"]

            # 3.b.1) Extract atoms from derivation_chain
            try:
                deriv_dict = ast.literal_eval(row["derivation_chain"])[0]
                if not isinstance(deriv_dict, dict):
                    raise ValueError
            except Exception:
                raise ValueError(f'''Invalid derivation_chain in row {row.name} of {file_name!r} ,\n {row['story_index']}, {row['query_edge']},  {row['query_relation']}
                                  already added {len(rows)} base_stories\n .. see {row["derivation_chain"]}''')

            # 'atoms' are things like ["paternal_grandparent_of(12,7)", ...]
            atoms: list[str] = []
            for rule_str in deriv_dict.values():
                # extract_fact_atoms(rule_str) should return a list of strings like "predName(arg1,arg2)"
                extracted = extract_fact_atoms(rule_str)
                atoms.extend(extracted)

            # Each atom → add trailing period, e.g. "predName(12,7)."
            derivation_facts = [f"{atom}." for atom in atoms]

            # Build a set of monitored entity IDs (from both derivation_facts and the target tuple)
            monitored_entities = set()
            # parse each atom‐string as "(e1,e2)"
            for atom in atoms:
                m = re.match(r"\w+\((\d+),(\d+)\)", atom)
                if m:
                    monitored_entities.add(int(m.group(1)))
                    monitored_entities.add(int(m.group(2)))
            # Always include the target_head_ent & target_tail_ent
            monitored_entities.add(target_head_ent)
            monitored_entities.add(target_tail_ent)

            # 3.b.2) Extract unary facts from fact_choice_branches
            try:
                fc_dict = ast.literal_eval(row["fact_choice_branches"])[0]
                if not isinstance(fc_dict, str):
                    raise ValueError
            except Exception:
                raise ValueError(f"Invalid fact_choice_branches in row {row.name} of {file_name!r} \n ... see {row["fact_choice_branches"]}")

            unary_facts: list[str] = []
            for line in fc_dict.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Only keep unary facts of the form: rel(ent).
                m = re.match(r"^(\w+)\((\d+)\)\.$", line)
                if m:
                    ent_id = int(m.group(2))
                    if ent_id in monitored_entities:
                        unary_facts.append(line)

            # 3.b.3) Combine derivation_facts + unary_facts
            story_facts = derivation_facts + unary_facts

            # 3.b.4) Build the single dictionary for this story
            single_row = {
                "target_pred":      target_pred,
                "target_head_ent":  target_head_ent,
                "target_tail_ent":  target_tail_ent,
                "story_facts":      story_facts,
                "story_index":      int(row["story_index"]),
                "query_edge":       row["query_edge"],
                "query_relation":   target_pred,
                "file_name":        file_name,
                "derivation_chain":  row["derivation_chain"],
                target_metric: row[target_metric],
            }
            rows.append(single_row)

    # 5. Concatenate into a single DataFrame
    if not rows:
        # If no rows survived filtering, return an empty DataFrame with the right columns
        columns = [
            "target_pred", "target_head_ent", "target_tail_ent",
            "story_facts", "story_index", "query_edge", "query_relation", "file_name"
        ]
        base_stories = pd.DataFrame(columns=columns)
    else:
        base_stories = pd.DataFrame(rows)

    # 6. Pickle & return
    print(f'Saving {base_stories.shape[0]} stories to {output_file} , {base_stories[target_metric].value_counts()} ')
    base_stories.to_pickle(output_file)
    return base_stories


if __name__ == "__main__":
    #-----------------------creating high test-OPEC-NA
    # base_stories = '/home/anirban/BrainStorming/DatsetsGenerated/NoRaExploring/datawrangling_benchmarks_project/data_output'
    # OPEC_values =  [1,2,3]
    # output_file =  '../configs/test_NORaHiGHoPEC.pkl'
    # build_base_stories(results_dir=base_stories, metric_filter_values=OPEC_values,output_file=output_file)
    # output stories from training wiht high reasoning depth 
    # # ------------------creating high test-D-NA
    # base_stories = '/home/anirban/BrainStorming/DatsetsGenerated/NoRaExploring/datawrangling_benchmarks_project/data_output'
    # depth_values =  [4,5,6]
    # output_file =  '../configs/test_NORaHiGHReasoningDepth.pkl'
    # build_base_stories(results_dir=base_stories, metric_filter_values=depth_values,output_file=output_file, target_metric= 'ReasoningDepth')
    # -------------------creating high test-BL-NA
    # base_stories = '/home/anirban/BrainStorming/DatsetsGenerated/NoRaExploring/datawrangling_benchmarks_project/data_output'
    # BL_values =  [1.25,1.20,1.33,1.5]
    # output_file =  '../configs/test_NORaHiGHBL.pkl'
    # build_base_stories(results_dir=base_stories, metric_filter_values=BL_values,output_file=output_file, target_metric= 'BL')
    # --------------------HeitoNetHighOpec Base stories
    base_stories = '/home/anirban/BrainStorming/DatsetsGenerated/HetioNetToy/datawrangling_benchmarks_project/data_output'
    OPEC_values =  [1,2]
    output_file =  '../configs/test_HeitoNetHiGHoPEC.pkl'
    build_base_stories(results_dir=base_stories, metric_filter_values=OPEC_values,output_file=output_file)
    