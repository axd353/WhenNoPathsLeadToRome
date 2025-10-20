import random
import re
import itertools
import sys
sys.path.append('../..')
from utils.clingo_utils import run_clingo
from story_builders.FamilyRegChecker import FamilyRegChecker
import logging
import math
from WorldGendersLocationsNoHorn import WorldGendersLocationsNoHorn
from utils.FindDerivationForPositiveProgram import PositiveProgramTP
from utils.clean_up_data import log_dataframe_with_gaps, get_max_branching_realized
from utils.wrangle_derivations import get_total_cluttr_hops, get_max_cluttr_hops, get_branch_cluttr_hops, get_non_path_atom_stats
from utils.wrangle_derivations2 import  group_branches_by_derivation, generate_fact_ratio_dict, update_max_rule_to_fact_ratio1, update_max_rule_to_fact_ratio2 
import time

class WorldGendersLocationsNoHornAmbFacts(WorldGendersLocationsNoHorn):
    def __init__(self, universal_rules, logger, person_percent=0.85, male_prob=0.5, assign_loc_prob=0.07, prob_living_in_same_place=0.1, ambig_fact_prob=0.1):
        """
        Initialize with parent parameters plus:
          ambig_fact_prob: probability with which an ambiguous fact is generated.
        """
        super().__init__(universal_rules, logger, person_percent, male_prob, assign_loc_prob, prob_living_in_same_place)
        self.ambig_fact_prob = ambig_fact_prob
        self.num_amb_nodes = 0
        self.max_ambiguity = 2 
        self.amb_par_lower_bound_on = False
        self.num_branches_in_current_story = 1
        self.max_num_branches = 0 # todo make this an experiment parameter, this =0 means no ambiguity 
        self.p_branch_multiplier = 1
        self.p_max_branching_realized = 0
        self.last_save_time = time.time()

    def set_experiment_conds(self, config, story_ind):
        """
        Sets experimental conditions by randomly sampling parameter values from a configuration dictionary.
        In addition to setting person_percent, male_prob, assign_loc_prob, etc. as defined in the parent,
        this method also sets the ambiguous fact probability.
        
        Expected keys in config:
          - "person_percent_range": [low, high]
          - "male_prob_range": [low, high]
          - "assign_loc_prob_range": [low, high]
          - "prob_living_in_same_place_range": [low, high]
          - "no_gender_assign": [low, high]
          - "proportion_of_underage": [low, high]
          - "ambig_fact_prob_range": [low, high]
        
        Returns:
            A dictionary with the experimental settings.
        """        
        # Use the parent's set_experiment_conds to set common parameters.
        exp_settings = super().set_experiment_conds(config, story_ind)
        if "ambig_fact_prob_range" in config:
            low, high = config["ambig_fact_prob_range"]
            self.ambig_fact_prob = random.uniform(low, high)
            exp_settings["ambig_fact_prob"] = self.ambig_fact_prob
        else:
            self.ambig_fact_prob = 0.15
        if "max_ambiguity_range" in config:
            low, high = config["max_ambiguity_range"]
            self.max_ambiguity = random.randint(low, high)
            exp_settings["max_ambiguity"] = self.max_ambiguity
        if "max_num_refinements" in config:
            self.max_num_branches = int(config["max_num_refinements"])
            exp_settings["max_num_refinements"] = self.max_num_branches
        ### resetting stuff at the begiining of new story
        self.num_amb_nodes = 0
        self.num_branches_in_current_story = 1

        self.logger.info("Experimental conditions set for ambiguous fact probability: " + str(exp_settings))
        return exp_settings

    def generate_random_fact(self, entities, program):
        """
        Overrides fact generation so that with probability self.ambig_fact_prob an ambiguous fact is produced,
        while respecting the max_num_branches constraint.
        
        Branching logic:
        - Starts with self.num_branches_in_current_story = 1
        - When ambiguous fact is generated (l alternatives, u=u, v=v):
            - If v==u: self.num_branches_in_current_story *= comb(l, u)
            - If v==l: self.num_branches_in_current_story *= sum(comb(l, r) for u <= r <= v)
        - If self.num_branches_in_current_story > self.max_num_branches, falls back to parent's method
        
        Returns:
            Generated fact string and fact details list
        """
        # Check if we've exceeded max branches
        self.p_branch_multiplier = 1
        if hasattr(self, 'num_branches_in_current_story') and \
        hasattr(self, 'max_num_branches') and \
        self.num_branches_in_current_story > math.ceil(self.max_num_branches/2):
            self.logger.debug(f''' Num branches already at {self.num_branches_in_current_story} .. no more   ''')
            return super().generate_random_fact(entities, program)
        
        if random.random() < self.ambig_fact_prob:
            updated_rels = [rel for rel in self.plausible_relations if rel[0] not in self.exclude_preds_during_gen]
            relation, num_args = random.choice(updated_rels)
            
            # First, sample l from [2, self.max_ambiguity]
            l = random.randint(2, self.max_ambiguity)
            if int(num_args) != 2 or len(entities) < l + 1:
                return super().generate_random_fact(entities, program)
                
            # Sample l+1 distinct entities: x1, x2, ..., xₗ₊₁
            sample_entities = random.sample(entities, l + 1)
            x1 = sample_entities[0]
            alternatives = sample_entities[1:]
            
            # Sample u and v
            if self.amb_par_lower_bound_on:
                u = random.randint(1, l-1)
            else:
                u = 1
            v = random.choices([u, l], k=1)[0]
            
            # Calculate branch multiplier
            from math import comb
            if v == u:
                self.p_branch_multiplier = comb(l, u)
            else:  # v == l
                self.p_branch_multiplier = sum(comb(l, r) for r in range(u, l+1))
            
            # Update branch count
            if not hasattr(self, 'num_branches_in_current_story'):
                self.num_branches_in_current_story = 1
            
            ##these branches might not get accepted by verify_facts, we will do this step in verify 
            #self.num_branches_in_current_story *= self.p_branch_multiplier

            # Construct the ambiguous fact string
            alt_strings = [f"{relation}({x1},{alt})" for alt in alternatives]
            amb_fact = f"{u}{{" + "; ".join(alt_strings) + f"}}{v}."
            
            # Build fact details list
            amb_node = f'amb_node{self.num_amb_nodes}'
            fact_detail = [((x1, amb_node), relation)]
            for alt in alternatives:
                if v == u:
                    fact_detail.append(((amb_node, alt), f'amb_exactly_{u}_true'))
                else:
                    fact_detail.append(((amb_node, alt), f'amb_atleast_{u}_true'))
            
            self.num_amb_nodes += 1
            return amb_fact, fact_detail
        else:
            return super().generate_random_fact(entities, program)
        

    def get_program_variants(self, facts_part):
        """
        Given the facts part (i.e. generated facts without the universal rules),
        returns a list of concrete variants by resolving ambiguous choice rules.
        
        An ambiguous rule has the form:
            u{fact1; fact2; ...; factn}v.
        meaning that between u and v of the facts are true.
        
        This method enumerates all valid subsets of alternatives (i.e. of sizes between
        u and min(v, n)) and replaces the ambiguous rule with the corresponding chosen facts
        (each fact ending with a period). If no ambiguous rule is found, returns a one-element list.
        """
        # Pattern: one or more digits, a '{', a non-greedy capture of everything up to '}', then a '}', one or more digits, a dot.
        pattern = r'(\d+)\{([^}]+)\}(\d+)\.'
        matches = list(re.finditer(pattern, facts_part))
        if not matches:
            return [facts_part]
        
        # For each ambiguous fact occurrence, compute the valid replacements.
        # Each replacement is a string built by concatenating a valid subset of alternatives,
        # with each alternative ending with a period.
        replacements = []
        for match in matches:
            full_text = match.group(0)  # the entire ambiguous rule string.
            lower_bound = int(match.group(1))
            alternatives_str = match.group(2)
            upper_bound = int(match.group(3))
            # Split alternatives on semicolon.
            alternatives = [alt.strip() for alt in alternatives_str.split(";")]
            # Ensure each alternative ends with a period.
            alternatives = [alt if alt.endswith('.') else alt + '.' for alt in alternatives]
            valid_subsets = []
            # The valid subset sizes range from the lower bound to min(upper_bound, number of alternatives)
            max_size = min(upper_bound, len(alternatives))
            for r in range(lower_bound, max_size + 1):
                for subset in itertools.combinations(alternatives, r):
                    # Join the chosen facts with a newline separator.
                    replacement_text = "\n".join(subset)
                    valid_subsets.append(replacement_text)
            replacements.append((full_text, valid_subsets))
        
        # Construct all variants by replacing each ambiguous rule with each valid replacement.
        variants = [facts_part]
        for (amb_text, replacement_options) in replacements:
            new_variants = []
            for variant in variants:
                for rep in replacement_options:
                    new_variant = variant.replace(amb_text, rep, 1)
                    new_variants.append(new_variant)
            variants = new_variants
        return variants


    def disentangle_chain(self, chain):
        """
        Given a derivation chain string (which includes both facts and rules),
        splits it into two sets:
        - rules_set: tokens that do not start with "fact:".
        - facts_set: tokens that start with "fact:" (with the "fact:" prefix removed).
        Returns a tuple: (rules_set, facts_set)
        """
        # self.logger.debug(f' --- Disentangling derivation chain: {chain} ')
        # Split the chain on " | " to get all segments.
        segments = [seg.strip() for seg in chain.split("|") if seg.strip()]
        rules_set = set()
        facts_set = set()
        for seg in segments:
            if seg.startswith("fact:"):
                fact = seg[len("fact:"):].strip()
                facts_set.add(fact)
            else:
                # Treat the entire segment as a rule.
                rules_set.add(seg)
        # self.logger.debug(f' Derived rules: {rules_set} and facts: {facts_set} ')
        return rules_set, facts_set
    
    def process_variant(self, complete_program, query_norm, precomputed_models):
        """
        For a given complete program (universal rules + one variant of generated facts) and a normalized query,
        process the variant using the (precomputed) solver results.
        
        First, we always set up the engine:
            engine = PositiveProgramTP(complete_program, logger=self.logger)
            engine.parse_program()
            final_model, step_count = engine.compute_least_model()
        
        In the case where no models are returned (i.e. precomputed_models is falsy), engine.busted_result is
        processed row by row. For each row, the derived atom (normalized) is used as the key in a dictionary whose
        value is a dict containing:
            {"rules": [list of grounded rules], "facts": [list of grounded facts]}.
        
        Similarly, if a stable model exists, then the derivations DataFrame is filtered by query_norm and processed
        similarly.
        
        Returns a tuple: (outcome, rules_set_total, facts_set_total, derivations)
        where outcome is either "contradiction" or "unique stable model".
        """
        # Always set up the engine.
        engine = PositiveProgramTP(complete_program, logger=self.logger)#TODO: YOU COULD AVOID RECOMPUTING THSI FOR EACH QUERY
        engine.parse_program()
        final_model, step_count = engine.compute_least_model()

        rules_set_total = set()
        facts_set_total = set()
        derivations = {}  # dictionary mapping normalized derived_atom -> {"rules": [...], "facts": [...]}

        # Contradiction branch: no models (precomputed_models falsy)
        if not precomputed_models:
            # With compute_least_model() called, engine.busted_result is now available.
            if engine.busted_result is None or "derivation_chain" not in engine.busted_result.columns:
                raise ValueError(
                    "Inconsistency detected: run_clingo returned no models and engine.compute_least_model() did not yield a busted_result with derivation_chain. "
                    f"Program: {complete_program}"
                )
            # Process each row of the busted_result DataFrame.
            for idx, row in engine.busted_result.iterrows():
                # Normalize the derived atom (remove spaces)
                derived_atom = row["derived_atom"].replace(" ", "")
                chain = row["derivation_chain"]
                r_set, f_set = self.disentangle_chain(chain)
                rules_set_total.update(r_set)
                facts_set_total.update(f_set)
                derivations[derived_atom] = row["derivation_chain"]
            return "contradiction", rules_set_total, facts_set_total, derivations
        # Stable model branch:
        else:
            if final_model is None:
                raise ValueError(
                    "Inconsistency detected: run_clingo returned models, but engine.compute_least_model() returned None. "
                    f"Program: {complete_program}, Query: {query_norm}"
                )
            deriv_df = engine._build_derivations_df(final_model)
            deriv_df['derived_atom_str'] = deriv_df['derived_atom'].apply(lambda x: x.replace(' ', ''))
            query_deriv = deriv_df[ deriv_df['derived_atom_str'] == query_norm ]
            if query_deriv.empty:
                self.logger.error(f''' PositiveProgramTP yeilds {final_model} , and 
                                  clingo returns the stable model {precomputed_models} , this branch yields is {complete_program}. ''')
                raise ValueError(
                    "Inconsistency detected: _build_derivations_df did not yield any derivation for query "
                    f"'{query_norm}'. Derived DF: {deriv_df.to_dict('records')}, Program: {complete_program}"
                )
            for idx, row in query_deriv.iterrows():
                derived_atom = row["derived_atom_str"]
                chain = row["derivation_chain"]
                r_set, f_set = self.disentangle_chain(chain)
                rules_set_total.update(r_set)
                facts_set_total.update(f_set)
                derivations[derived_atom] = row["derivation_chain"]
            return "unique stable model", rules_set_total, facts_set_total, derivations

    def calc_diff(self, df, output_file=None):
        """
        Analyze derivation difficulty for each story in the DataFrame using PositiveProgramTP.
        Input DataFrame (df) contains the columns:
        ['entities', 'story_edges', 'edge_types', 'query_edge', 'query_relation', 
        'program', 'models', 'num_answer_sets', 'story_index'].
        Additional columns added:
        - derivation_chain: a dictionary mapping branch indices (fact choices) to derivation chains.
        - chain_len: total number of distinct grounded rules (aggregated across branches).
        - num_facts_required: total number of distinct grounded facts (aggregated across branches).
        - sum_facts_world_rules: sum of chain_len and num_facts_required.
        - fact_choice_branches: dictionary mapping branch indices to their variant facts.
        - branch_results: dictionary mapping branch indices to the outcome ('contradiction' or 'unique stable model').
        - unique_rules: the aggregated set of rules as a string.
        - unique_facts: the aggregated set of facts as a string.
        - num_variants: total number of variants generated.
        - ReasoningWidth: number of unique derivation variants (after aggregating over branch derivations).
        - branch_cluttr_hops: {branch_index: number_of_horn_rules} — the count of horn rules in the derivation for each branch.
        - total_cluttr_hops: Sum of all horn rules across unique derivation variants.
        - max_cluttr_hops: [max_all, max_unique_models, max_contradictions]
                      max_all: highest horn rule count in any variant. max_unique_models: highest among only the 'unique stable model' branches.
                                                max_contradictions: highest among contradiction branches.
            Note: The program is composed as universal_rules + "\n" + facts.
                    All rows sharing the same 'story_index' must also share the same facts part.
        - non_path_atom_counts: dict with branch_id tuple → num non-path atoms
        - OPEC_pos_refn: maximum across all derivations
        Returns:
            A new DataFrame with the added derivation metrics.
        """
        new_df = df.copy()
        new_df['derivation_chain'] = None
        new_df['chain_len'] = None
        new_df['num_facts_required'] = None
        new_df['sum_facts_world_rules'] = None
        new_df['fact_choice_branches'] = None
        new_df['branch_results'] = None
        new_df['branch_outcomes'] = None
        new_df['unique_rules'] = None
        new_df['unique_facts'] = None
        new_df['num_variants'] = None
        new_df['ReasoningWidth'] = None
        new_df['branch_cluttr_hops'] = None  # Dict mapping branch → horn rule count
        new_df['total_cluttr_hops'] = None   # Sum across all unique derivation variants
        new_df['max_cluttr_hops'] = None     # [max_all, max_model, max_contra] tuple
        new_df['ReasoningDepth_only_pos_derivations'] = None
        new_df['graph_complexity_stats'] = None
        new_df['BL'] = None
        new_df['BL_no_contradiction'] = None
        new_df['non_path_atom_counts'] = None
        new_df['OPEC_pos_refn'] = None
        new_df['OPEC'] = None
        
        prefix = self.universal_rules + "\n"
        def get_facts_part(program):
            if program.startswith(prefix):
                return program[len(prefix):]
            return program

        new_df['facts_part'] = new_df['program'].apply(get_facts_part)

        def build_query(row):
            q = f"{row['query_relation']}({row['query_edge'][0]},{row['query_edge'][1]})"
            return q.replace(" ", "")
        new_df['query_str'] = new_df.apply(build_query, axis=1)

        # Verify that grouping by facts_part is equivalent to grouping by story_index.
        for facts, group in new_df.groupby('facts_part'):
            if len(group['story_index'].unique()) > 1:
                for  st_ind in group['story_index'].unique():
                    temp_df = new_df[new_df['story_index'] == st_ind]
                    self.logger.exception(f' for story index {st_ind} \n program is  {temp_df.iloc[0]['program']} \n')
                raise ValueError(
                    f"Inconsistent grouping: facts_part [{facts}] appears in multiple story_index values: {group['story_index'].unique()}"
                )
            
        total_stories = len(new_df['story_index'].unique())
        story_counter = 0

        # Group by facts_part (which aligns with story_index).
        grouped = new_df.groupby('facts_part')
        for facts, group in grouped:
            # self.logger.debug(f' Begin caluclating difficulty of story {self.story_number} which has facts part \n {facts} \n ')
            # Generate all variants for this facts part.
            self.p_max_branching_realized = get_max_branching_realized(group.iloc[0]['program'])
            variants = self.get_program_variants(facts)
            fact_choice_branches = {}  # branch index -> facts variant string
            branch_results = {}        # branch index -> outcome from processing this branch
            complete_variants = {}     # branch index -> complete program (universal + facts variant)
            variant_solver_results = {}  # branch index -> precomputed run_clingo result
            for idx, variant in enumerate(variants):
                fact_choice_branches[idx] = variant
                complete_prog = self.universal_rules + "\n" + variant
                complete_variants[idx] = complete_prog
                # Call run_clingo once per variant.
                solver_result = run_clingo(complete_prog)
                # If run_clingo returns a falsy value, treat it as contradiction.
                variant_solver_results[idx] = solver_result if solver_result is not None else None

            # Store branch mappings for the group.
            group_idx = group.index
            new_df.loc[group_idx, 'fact_choice_branches'] = str(fact_choice_branches)

            for query in group['query_str'].unique():
                branch_info = {}
                for b_idx, complete_prog in complete_variants.items():
                    outcome, r_set, f_set, derivs = self.process_variant(complete_prog, query, variant_solver_results[b_idx])
                    branch_info[b_idx] = {"outcome": outcome, "rules": r_set, "facts": f_set, "derivations": derivs}
                # Aggregate results across branches for this query.
                agg_rules = set()
                agg_facts = set()
                agg_derivations = {}
                for b_idx, info in branch_info.items():
                    agg_rules.update(info["rules"])
                    agg_facts.update(info["facts"])
                    agg_derivations[b_idx] = info["derivations"]
                    branch_results[b_idx] = info["outcome"]
                chain_len = len(agg_rules)
                num_facts_required = len(agg_facts)
                sum_facts_world_rules = chain_len + num_facts_required
                ## Counting the contradiction for negative refinements as a rule during calulcation of ReasoningDepth
                max_rule_len_non_contraction_chain = max(len(info["rules"])  for info in branch_info.values() if info["outcome"] == 'unique stable model') ## longest chain for positive derivations  
                max_rule_len_contradiction_refn = max((len(info["rules"])+1  for info in branch_info.values() if info["outcome"] == 'contradiction'), default=0)
                ReasoningDepth = max([max_rule_len_non_contraction_chain, max_rule_len_contradiction_refn]) ##longest chain across all versions
                unique_variant_set = set()
                # Here, for each branch, convert derivations (which is a dict) to a string (or tuple) and add to set.
                for deriv in agg_derivations.values():
                    unique_variant_set.add(str(deriv))
                ReasoningWidth = len(unique_variant_set)
                # --- Per-branch horn rule analysis ---
                branch_cluttr_hops = get_branch_cluttr_hops(agg_derivations)
                # --- Across-variant clutter hops ---
                variant_branch_outcomes = {}
                for b_idx, info in branch_info.items():
                    s = str(info["derivations"])
                    variant_branch_outcomes[s] = info["outcome"]

                total_cluttr = get_total_cluttr_hops(unique_variant_set)
                max_all, max_model, max_contra = get_max_cluttr_hops(unique_variant_set, variant_branch_outcomes)
                ## back-forth reasoning realted to non path                
                # 1. group branches by identical derivations
                gb = group_branches_by_derivation(agg_derivations, branch_results)

                # 2. compute the metrics for each unique-derivation group
                frd = generate_fact_ratio_dict(gb, agg_derivations, branch_results)
                idxs = group[group['query_str'] == query].index
                # 3. find the max‐node ratio among unique-stable-model branches, across everything
                max_ratio_stab = update_max_rule_to_fact_ratio1(frd)
                max_ratio = update_max_rule_to_fact_ratio2(frd)
                # 🔥 Non-path atom stats
                idxs = group[group['query_str'] == query].index
                query_edge = new_df.loc[idxs[0], 'query_edge']
                # build mappings
                variant_to_branch = {}
                variant_branch_outcomes = {}
                for b_idx, info in branch_info.items():
                    vs = str(info["derivations"])
                    variant_to_branch.setdefault(vs, []).append(b_idx)
                    variant_branch_outcomes[vs] = info["outcome"]

                # compute 'non_path_atom_counts', 'OPEC_pos_refn'
                non_path_counts, max_non_path = get_non_path_atom_stats(unique_variant_set,query_edge,
                    variant_to_branch,variant_branch_outcomes)

                # Update new_df rows corresponding to this query.
                new_df.loc[idxs, 'derivation_chain'] = str(agg_derivations)
                new_df.loc[idxs, 'chain_len'] = chain_len
                new_df.loc[idxs, 'num_facts_required'] = num_facts_required
                new_df.loc[idxs, 'sum_facts_world_rules'] = sum_facts_world_rules
                new_df.loc[idxs, 'branch_results'] = str(branch_results)
                new_df.loc[idxs, 'branch_outcomes'] = str({key:val[0] for key,val in frd.items()})## groups branches/refinements into equivalent classes according to derivations and show outcome of each 
                new_df.loc[idxs, 'unique_rules'] = str(agg_rules)
                new_df.loc[idxs, 'unique_facts'] = str(agg_facts)
                new_df.loc[idxs, 'num_variants'] = len(variants)
                new_df.loc[idxs, 'ReasoningWidth'] = ReasoningWidth
                new_df.loc[idxs, 'ReasoningDepth'] = ReasoningDepth ##across all branches
                new_df.loc[idxs, 'ReasoningDepth_only_pos_derivations'] =  max_rule_len_non_contraction_chain
                new_df.loc[idxs, 'max_branching_realized'] = self.p_max_branching_realized
                new_df.loc[idxs, 'branch_cluttr_hops'] = str(branch_cluttr_hops)
                new_df.loc[idxs, 'total_cluttr_hops'] = total_cluttr
                new_df.loc[idxs, 'max_cluttr_hops'] = str([max_all, max_model, max_contra])
                new_df.loc[idxs, 'graph_complexity_stats']       = str(frd)
                new_df.loc[idxs, 'BL_no_contradiction'] = max_ratio_stab
                new_df.loc[idxs, 'BL'] = max_ratio
                new_df.loc[idxs, 'non_path_atom_counts'] = str(non_path_counts)
                new_df.loc[idxs, 'OPEC_pos_refn'] = max_non_path ## Then max_non_path is taken only over variants with outcome 'unique stable model'. includes 'no_son'm male etc. function called add_contradiction_stats_to_df calculates for contradiction branches
                new_df.loc[idxs, 'OPEC'] = max(non_path_counts.values())

            self.logger.info(f'Finished calculating difficulties of {story_counter} out of {total_stories} stories.')
            story_counter += 1

            # --- TIMER CHECK: Save intermediate results if one hour has passed ---
            if output_file is not None and (time.time() - self.last_save_time) >= 1800:
                # Create a copy of new_df without any rows containing None values.
                temp_df = new_df.dropna()
                self.save_dataset(temp_df, output_file)
                self.logger.info(f"Intermediate dataset saved to {output_file} after one hour. it has shape {temp_df.shape}")
                self.last_save_time = time.time()

        new_df = new_df.drop(columns=["facts_part", "query_str"])
        new_df = new_df.sort_values(by='chain_len', ascending=False)
        if output_file is not None:
            self.save_dataset(new_df, output_file)
        return new_df
    
