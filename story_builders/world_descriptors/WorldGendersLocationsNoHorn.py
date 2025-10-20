import sys
import random
import re
sys.path.append('../..')
from utils.clingo_utils import run_clingo
from story_builders.FamilyRegChecker import FamilyRegChecker
import logging
import math
from WorldSpecifics import WorldSpecifics
from utils.FindDerivationForPositiveProgram import PositiveProgramTP
from utils.clean_up_data import log_dataframe_with_gaps


class WorldGendersLocationsNoHorn(WorldSpecifics):
    def __init__(self, universal_rules, logger, person_percent=.85, male_prob=.5, assign_loc_prob =.07, prob_living_in_same_place =.1 ):
        """
        plausible_relations: list of possible relationship predicates.
        p_regularity_checker: an object with a method is_regular(program, logger=logger)
        logger: a logger object.
        """
        super().__init__()
        self.logger = logger
        self.universal_rules =  universal_rules
        self.plausible_relations = self.extract_plausible_relations(universal_rules)
        self.exclude_preds_during_gen = ["is_person" , "is_place", 'living_in' , 'is_male', 'is_female', 'is_underage' , 'no_siblings', 'no_children',
                                         'no_brothers','no_sisters', 'no_daughters', 'no_sons',  ] # living in requires very specific entities, so these are generated randomly 
        self.p_regularity_checker = FamilyRegChecker()
        self.print_counter = 0
        self.print_max_times = 0
        self.person_percent = person_percent
        self.male_prob = male_prob
        self.assign_loc_prob = assign_loc_prob
        self.prob_living_in_same_place = prob_living_in_same_place
        self.no_gender_assign = 0
        self.proportion_of_underage = .06
        self.story_number = 0 # how many stories have been generated with it
        self.places = list()
        self.people = list()

    def set_experiment_conds(self, config, story_ind):
        """
        Sets the experimental conditions by randomly sampling parameter values.
        Expects the configuration dictionary to have keys:
          - "person_percent_range": [low, high]
          - "male_prob_range": [low, high]
          - "assign_loc_prob_range": [low, high]
        Updates self.person_percent, self.male_prob, self.assign_loc_prob accordingly,
        logs and returns a dictionary with the values used.
        """
        exp_settings = {}
        if "person_percent_range" in config:
            low, high = config["person_percent_range"]
            self.person_percent = random.uniform(low, high)
            exp_settings["person_percent"] = self.person_percent
        if "male_prob_range" in config:
            low, high = config["male_prob_range"]
            self.male_prob = random.uniform(low, high)
            exp_settings["male_prob"] = self.male_prob
        if "assign_loc_prob_range" in config:
            low, high = config["assign_loc_prob_range"]
            self.assign_loc_prob = random.uniform(low, high)
            exp_settings["assign_loc_prob"] = self.assign_loc_prob
        if "prob_living_in_same_place_range" in config:
            low, high = config["prob_living_in_same_place_range"]
            self.prob_living_in_same_place = random.uniform(low, high)
            exp_settings["prob_living_in_same_place"] = self.prob_living_in_same_place
        if "no_gender_assign" in config:
            low, high = config["no_gender_assign"]
            self.no_gender_assign = random.uniform(low, high)
            exp_settings["no_gender_assign"] = self.no_gender_assign
        if "proportion_of_underage" in config:
            low, high = config["proportion_of_underage"]
            self.proportion_of_underage = random.uniform(low, high)
            exp_settings["proportion_of_underage"] = self.proportion_of_underage
        if "min_entailed_atoms_per_story" in config:
            self.min_entailed_atoms_per_story = config["min_entailed_atoms_per_story"]
            exp_settings["min_entailed_atoms_per_story"] = self.min_entailed_atoms_per_story
        else:
            self.min_entailed_atoms_per_story = 1
            exp_settings["min_entailed_atoms_per_story"] = self.min_entailed_atoms_per_story
        self.logger.info("Experimental conditions set: " + str(exp_settings))
        return exp_settings

    def gen_entities(self, ent_num, program):
        """
        Generates entities and initial facts for each entity.
        
        For each entity (identified by p_num):
         - With probability self.person_percent, the entity is considered a person:
             * Generates the fact "is_person(p_num)."
             * Generates a membership fact "is_male(p_num, p_num)." with probability self.male_prob,
               otherwise "is_female(p_num, p_num)."
         - Otherwise, the entity is considered a place:
             * Generates the fact "is_place(p_num)."
        
        Returns:
            entities: list of entity indices [0, 1, ..., ent_num-1].
            program: the original program string appended with the generated facts.
            added_facts: a list of fact strings that were added.
            fact_details_list: a list of tuples, each containing the fact detail, e.g.
                               ((p_num, p_num), "is_person") or ((p_num, p_num), 'is_male').
        """
        self.places = []
        self.people = []
        entities = list(range(ent_num))
        added_facts = []
        fact_details_list = []
        for p_num in entities:
            if random.random() < self.person_percent:
                # Generate a person.
                p_fact1 = f"is_person({p_num})."
                p_fact_detail1 = ((p_num, p_num), "is_person")
                added_facts.append(p_fact1)
                fact_details_list.append(p_fact_detail1)
                self.people.append(p_num)
                # # Add reflexive living_in_same_place relationship.. too keep queries interesting. 
                # living_fact = f"living_in_same_place({p_num}, {p_num})."
                # added_facts.append(living_fact)
                # fact_details_list.append(((p_num, p_num), "living_in_same_place")) 
                # Determine gender.
                if random.random() > self.no_gender_assign:
                    if random.random() < self.male_prob:
                        p_fact2 = f"is_male({p_num},{p_num})."
                        p_fact_detail2 = ((p_num, p_num), "is_male")
                    else:
                        p_fact2 = f"is_female({p_num},{p_num})."
                        p_fact_detail2 = ((p_num, p_num), "is_female")
                    added_facts.append(p_fact2)
                    fact_details_list.append(p_fact_detail2)
                if random.random() < self.proportion_of_underage:
                    p_fact2 = f"is_underage({p_num},{p_num})."
                    p_fact_detail2 = ((p_num, p_num), "is_underage")
                    added_facts.append(p_fact2)
                    fact_details_list.append(p_fact_detail2)
            else:
                # Generate a place.
                p_fact1 = f"is_place({p_num})."
                p_fact_detail1 = ((p_num, p_num), "is_place")
                added_facts.append(p_fact1)
                fact_details_list.append(p_fact_detail1)
                self.places.append(p_num)
        
        # self.logger.debug(f'The story {self.story_number} has places: {self.places} ')
        # self.logger.debug(f'The story {self.story_number} has people: {self.people}')
        # Append all generated facts to the program.
        for fact in added_facts:
            program += fact + "\n"
        return entities, program, added_facts, fact_details_list

    def generate_random_fact(self, entities, program):
        """
        Generates one random fact using a randomly chosen relation from plausible_relations
        and two distinct entities from the list.
        Returns:
          - fact (str): e.g. "father_of(0,3)."
          - fact_details: a list of tuples ((entity1, entity2), relation)
        """
        if (random.random() < self.assign_loc_prob) & (len(self.people)!=0) & (len(self.places)!=0):
            p_pers = random.sample(self.people,1)[0]
            p_place = random.sample(self.places,1)[0]
            fact = f"living_in({p_pers},{p_place})."
            # self.logger.debug(f'Attempted to sample fact {fact}')
            return fact, [((p_pers, p_place), 'living_in')]
        # Then try to generate living_in_same_place if applicable
        if (random.random() < self.prob_living_in_same_place) and (len(self.people) >= 2):
            e1, e2 = random.sample(self.people, 2)
            fact = f"living_in_same_place({e1},{e2})."
            # self.logger.debug(f'Attempted to sample living_in_same_place fact {fact}')
            return fact, [((e1, e2), 'living_in_same_place')]
        updated_rels = [rel for rel in self.plausible_relations if rel[0] not in self.exclude_preds_during_gen ]
        self.logger.debug(f'We will only sample predicates from {updated_rels} for vanilla story.')
        relation, num_args = random.choice(updated_rels)
        num_args = int(num_args)
        ##TODO for speed perhaps consider only sampling entities frm people 
        if num_args == 2:
            if len(entities) < 2:
                raise ValueError("Need at least two entities to generate a fact.")
            e1, e2 = random.sample(entities, 2)
            fact = f"{relation}({e1},{e2})."
            return fact, [((e1, e2), relation)]
        elif num_args == 1:
            if len(entities) < 1:
                raise ValueError("Need at least one entity to generate a fact.")
            e1 = random.sample(entities, 1)[0]
            fact = f"{relation}({e1})."
            return fact, [((e1,e1), relation)]
        else:
            raise ValueError(f"Oops something went wrong, we sampled: {relation}, {num_args}")

    def _verify_fact(self, fact, program, consecutive_contradictions, too_many_consecutive_contradictions):
        """
        Verifies a candidate fact by testing if adding it to the program yields at least one answer set
        and passes the regularity check. Updates the consecutive_contradictions counter.
        
        Returns a tuple:
          (accepted, updated_program, models, consecutive_contradictions, break_flag)
        """ 
        temp_program = program + fact + "\n"
        if self.print_counter < self.print_max_times:
            self.logger.debug(f"Running clingo with the program: {temp_program}")
        temp_models = run_clingo(temp_program)
        self.print_counter += 1
        if not temp_models:
            consecutive_contradictions += 1
            # self.logger.info(f"Fact {fact} contradicted existing facts. Consecutive contradictions: {consecutive_contradictions}.")
            if consecutive_contradictions >= too_many_consecutive_contradictions:
                self.logger.info("Too many consecutive contradictions. Exiting fact generation early.")
                return (False, program, temp_models, consecutive_contradictions, True)
            return (False, program, temp_models, consecutive_contradictions, False)
        # Reset counter since fact did not cause contradiction.
        consecutive_contradictions = 0
        # Regularity check.
        if not self.p_regularity_checker.is_regular(temp_program, logger=self.logger):
            # self.logger.debug(f"Fact {fact} caused irregularity. Discarding this fact.")
            return (False, program, temp_models, consecutive_contradictions, False)
        # if 'living_in(' in fact:
        #     self.logger.debug(f'adding the fact {fact} to the story \n.')
        ## in the event that there are ambiguous facts, this is too keep a bound on the number of branches, TODO : correct coding would be to define this method in WorldGendersLocationsNoHornAmbFacts
        if hasattr(self, 'num_branches_in_current_story') & hasattr(self, 'p_branch_multiplier'):
            self.num_branches_in_current_story *= self.p_branch_multiplier
        return (True, temp_program, temp_models, consecutive_contradictions, False)

    def calc_diff(self, df,output_file=None):
        """
        Analyzes the derivation difficulty of query edges in the dataframe by using PositiveProgramTP.
        
        Args:
            df: DataFrame with columns ['entities', 'story_edges', 'edge_types', 'query_edge', 
                                    'query_relation', 'program', 'models', 'num_answer_sets',
                                    'story_index']
        
        Returns:
            DataFrame with added columns for derivation metrics:
            ['derivation_chain', 'chain_len', 'num_facts_required', 'sum_facts_world_rules']
        """
        # Make a copy of the input dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Initialize new columns with empty values
        result_df['derivation_chain'] = None
        result_df['chain_len'] = None
        result_df['num_facts_required'] = None
        result_df['sum_facts_world_rules'] = None
        
        # Group by story_index to process each story separately
        for story_idx, group in df.groupby('story_index'):
            self.logger.debug(f'beginning understanding difficulty of story {story_idx}')
            # All rows in this group share the same program, story_edges, edge_types
            program_str = group['program'].iloc[0]
            
            try:
                # Create PositiveProgramTP engine and compute least model
                engine = PositiveProgramTP(program_str, logger=self.logger)
                engine.parse_program()
                final_model, step_count = engine.compute_least_model()
                if final_model == None:
                    self.logger(f'''Program has no stable_model, \n ..\n derivation for failed model is {engine.busted_result}.
                                  \n Steps taken to reach contradiction are {engine.busted_result['chain_len'].sum()}.    ''')
                    raise ValueError(f'''  {program_str} contradicts itself. ''')
                # Convert final_model to string format for comparison
                final_model_str = {
                    f"{pred}({','.join(map(str, args))})" 
                    for (pred, args) in final_model
                    if isinstance(args, tuple)  # Handle n-ary predicates
                }
                # Also include unary predicates (where args is not a tuple)
                final_model_str.update({
                    f"{pred}({args})" 
                    for (pred, args) in final_model
                    if not isinstance(args, tuple)
                })
                
                # Get derivation details for all atoms in the model
                derivations_df = engine._build_derivations_df(final_model)
                # Convert derivations_df to use string format for comparison 
                derivations_df['derived_atom_str'] = derivations_df['derived_atom'].apply(
                    lambda x: x.replace(' ', '')  # Remove any spaces in the atom string
                )
                
                # Process each row in the original dataframe (using original indices)
                for orig_idx in group.index:
                    row = df.loc[orig_idx]
                    query_edge = row['query_edge']
                    query_relation = row['query_relation']
                    
                    # Construct the atom string in consistent format
                    atom_str = f"{query_relation}({query_edge[0]},{query_edge[1]})".replace(' ', '')
                    
                    # Verify the query is in the stable model
                    if atom_str not in final_model_str:
                        self.logger.debug(f'About to error here are derivations {derivations_df.columns}')
                        log_dataframe_with_gaps(derivations_df, self.logger, column_list = ['derived_atom', 'derivation_chain', 'chain_len', 'num_facts_required',
       'sum_facts_world_rules', 'story_facts', 'derived_atom_str'])
                        raise ValueError(
                            f"Story {story_idx} (from row story index {row['story_index']}): Query {atom_str} from {row['query_edge']} // {row['query_relation']} not in stable model. \n "
                            f"Model contains (from checker): {final_model_str} \n"
                            f'program in this row is {row['program']} \n \n '
                            f'program string the program ran {program_str} \n'
                            f'the answer set (from clingo) for the row is  {row['models']} \n'
                        )
                    
                    # Get derivation details for this atom
                    atom_details = derivations_df[derivations_df['derived_atom_str'] == atom_str]
                    
                    if len(atom_details) == 0:
                        # Shouldn't happen since we checked it's in final_model
                        raise ValueError(f"Story {story_idx}: No derivation details found for {atom_str}")
                    
                    # Update the result dataframe with derivation metrics USING ORIGINAL INDEX
                    result_df.at[orig_idx, 'derivation_chain'] = atom_details['derivation_chain'].values[0]
                    result_df.at[orig_idx, 'chain_len'] = atom_details['chain_len'].values[0]
                    result_df.at[orig_idx, 'num_facts_required'] = atom_details['num_facts_required'].values[0]
                    result_df.at[orig_idx, 'sum_facts_world_rules'] = atom_details['sum_facts_world_rules'].values[0]
                self.logger.debug(f'done understanding difficulty of {story_idx} \n')    
            except Exception as e:
                self.logger.error(f"Error processing story {story_idx}: {str(e)}")
                raise
        result_df = result_df.sort_values(by='chain_len', ascending=False)
        if output_file != None:
            self.save_dataset(result_df,output_file)
        return result_df

    

    def build_entity_lists(self, fact_details_list):
        """
        Populate self.places and self.people based on fact_details_list.

        Args:
            fact_details_list (list[((int, int), str)]): List of ((e1, e2), relation) tuples.
        """
        self.places = []
        self.people = []
        for (e1, e2), rel in fact_details_list:
            # For unary facts, e1 == e2; only those with rel 'is_place' or 'is_person'
            if e1 == e2:
                if rel == 'is_place':
                    self.places.append(e1)
                elif rel == 'is_person':
                    self.people.append(e1)