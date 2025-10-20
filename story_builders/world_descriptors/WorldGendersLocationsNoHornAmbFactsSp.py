import random
import re
import itertools
import sys
import math
import pandas as pd
sys.path.append('../..')
from WorldGendersLocationsNoHorn import WorldGendersLocationsNoHorn
from WorldGendersLocationsNoHornAmbFacts import WorldGendersLocationsNoHornAmbFacts
from utils.clean_up_data import process_program_for_places,  log_dataframe_with_gaps
from utils.HandleMultiLabels import merging_multi_labels, process_dataframe 
import ast
from ast import literal_eval

'''
Very specialized world, only works for this worlds regularity
'''

pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  #prevent line breaks
pd.set_option('display.max_colwidth', None) #prevent truncation of strings

class WorldGendersLocationsNoHornAmbFactsSp(WorldGendersLocationsNoHornAmbFacts):
    """
    Extended version that modifies ambiguous fact generation with special handling for location assignments.
    When branch limits are exceeded, calls grandparent's (WorldGendersLocationsNoHorn) method directly.
    """
    def set_experiment_conds(self, config, story_ind):
        """
        Extends parent's method by sampling additional probabilities for no-sibling properties if provided.
        """
        # Inherit and set common parameters
        exp_settings = super().set_experiment_conds(config, story_ind)

        # Sample no-sibling property probabilities
        # Configuration keys expected (low, high):
        # - "prob__range_property_no_bros"
        # - "prob__range_property_no_sis"
        # - "prob__range_property_no_dghter"
        # - "prob__range_property_no_son"
        siblings = {
            'no_bros': 'prob__range_property_no_bros',
            'no_sis': 'prob__range_property_no_sis',
            'no_dghter': 'prob__range_property_no_dghter',
            'no_son': 'prob__range_property_no_son'
        }
        for prop, key in siblings.items():
            if key in config:
                low, high = config[key]
                value = random.uniform(low, high)
            else:
                # default to zero probability if not provided
                value = 0.0
            setattr(self, f'prob_property_{prop}', value)
            exp_settings[f'prob_property_{prop}'] = value

        return exp_settings

    def gen_entities(self, ent_num, program):
        """
        Overrides entity generation to include no-sibling properties for persons.
        """
        # Generate base entities and facts
        entities, program, added_facts, fact_details_list = super().gen_entities(ent_num, program)

        # After base generation, append no-sibling facts for each person
        for p_num in getattr(self, 'people', []):
            # no brothers
            if hasattr(self, 'prob_property_no_bros') and random.random() < self.prob_property_no_bros:
                fact = f"no_brothers({p_num},{p_num})."
                added_facts.append(fact)
                fact_details_list.append(((p_num, p_num),'no_brothers' ))
                program += fact + "\n"
                # self.logger.debug(f'Added no brother fact {fact} to {self.story_number}')
            # no sisters
            if hasattr(self, 'prob_property_no_sis') and random.random() < self.prob_property_no_sis:
                fact = f"no_sisters({p_num},{p_num})."
                added_facts.append(fact)
                fact_details_list.append(((p_num, p_num), 'no_sisters'))
                program += fact + "\n"
                # self.logger.debug(f'Added no sister fact {fact} to {self.story_number}')
            # no daughters
            if hasattr(self, 'prob_property_no_dghter') and random.random() < self.prob_property_no_dghter:
                fact = f"no_daughters({p_num},{p_num})."
                added_facts.append(fact)
                fact_details_list.append(((p_num, p_num), 'no_daughters' ))
                program += fact + "\n"
                # self.logger.debug(f'Added no daughter fact {fact} to {self.story_number}')
            # no sons
            if hasattr(self, 'prob_property_no_son') and random.random() < self.prob_property_no_son:
                fact = f"no_sons({p_num},{p_num})."
                added_facts.append(fact)
                fact_details_list.append(((p_num, p_num), 'no_sons'))
                program += fact + "\n"

        return entities, program, added_facts, fact_details_list


    def generate_random_fact(self, entities, program):
        """
        Modified version that:
        1. With probability self.assign_loc_prob, generates 'living_in' relations between people and places
           - x1 is from self.people
           - alternatives are from self.places
        2. Otherwise generates normal ambiguous facts between people
           - x1 and alternatives are from self.people
           - relation is from plausible_relations (excluding excluded predicates)
        3. When branch limits exceeded, calls grandparent's non-ambiguous fact generation
        """
        # Check if we've exceeded max branches
        self.p_branch_multiplier = 1
        if hasattr(self, 'num_branches_in_current_story') and \
           hasattr(self, 'max_num_branches') and \
           self.num_branches_in_current_story > math.ceil(self.max_num_branches/2):
            # self.logger.debug(f'''Specialized Num branches already at {self.num_branches_in_current_story}.. no more''')
            # Call grandparent's method directly
            return WorldGendersLocationsNoHorn.generate_random_fact(self, entities, program)
        
        if random.random() < self.ambig_fact_prob:
            # Decide if we're generating location facts or regular facts
            if hasattr(self, 'assign_loc_prob') and random.random() < self.assign_loc_prob:
                # Location assignment case - living_in between people and places
                if len(self.people) < 1 or len(self.places) < 2:  # Need at least 1 person and 2 places
                    return WorldGendersLocationsNoHorn.generate_random_fact(self, entities, program)
                
                relation = 'living_in'
                l = random.randint(2, self.max_ambiguity)
                if len(self.places) < l:
                    return WorldGendersLocationsNoHorn.generate_random_fact(self, entities, program)
                # Sample 1 person and l places
                x1 = random.choice(self.people)
                alternatives = random.sample(self.places, l)
                # self.logger.debug(f'''generating ambiguous (not yet accepted) fact with living_in relationship for story number {self.story_number} \n 
                #                   sampled person {x1} from {self.people} \n sampled {alternatives} (places) from {self.places}   ''')
            else:
                # Regular case - relations between people
                updated_rels = [rel for rel in self.plausible_relations if rel[0] not in self.exclude_preds_during_gen]
                if not updated_rels:
                    return WorldGendersLocationsNoHorn.generate_random_fact(self, entities, program)
                
                relation, num_args = random.choice(updated_rels)
                l = random.randint(2, self.max_ambiguity)
                if int(num_args) != 2 or len(self.people) < l+1:  # Need at least 3 people for l=2
                    return WorldGendersLocationsNoHorn.generate_random_fact(self, entities, program)
                            
                # Sample l+1 people
                sample_entities = random.sample(self.people, l + 1)
                x1 = sample_entities[0]
                alternatives = sample_entities[1:]
                # self.logger.debug(f'''generating ambiguous (not yet accepted) fact with {relation} relationship for story_number {self.story_number} \n 
                #                   sampled person {x1} from {self.people} \n sampled {alternatives} (pleople) from {self.people}   ''')
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
            # For non-ambiguous facts, call grandparent's method
            return WorldGendersLocationsNoHorn.generate_random_fact(self, entities, program)
    
    
    def save_dataset(self, df, file_path):
        """
        Saves the dataset DataFrame with additional 'Not_living_in' rows for alternative locations. Also merges multiple relationships 
        """
        df = merging_multi_labels(df, self.logger)
        df =  process_dataframe(df,self.logger)
        self.check_relationships(df)
        df.drop('other_implied_relationships1',axis=1, inplace=True)
        try:
            # Filter rows with 'living_in' relation
            living_in_rows = df[df['query_relation'] == 'living_in']
            new_rows = []
            for _, p_row in living_in_rows.iterrows():
                # Process the program to find alternative places
                program = p_row['program']
                query_edge = p_row['query_edge']
                
                # Handle case where query_edge might be string or tuple
                if isinstance(query_edge, str):
                    try:
                        query_edge = ast.literal_eval(query_edge)
                    except:
                        continue
                
                n1, n2 = query_edge
                alternative_places = process_program_for_places(program, n2)
                
                if not alternative_places:
                    self.logger.debug(f"No alternative places found in program:\n{program}")
                    continue
                # Pick one random alternative place
                # self.logger.debug(f"No alternative places found for {(n1,n2)} which are {alternative_places}")
                # In WorldGendersLocationsNoHornAmbFactsSp.save_dataset()
                n3 = random.choice(list(alternative_places))
                new_row = self.create_new_row(p_row, n1, n2, n3) 
                new_rows.append(new_row)
 
            # Add new rows to the dataframe if any were created
            if new_rows:
                new_rows_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_rows_df], ignore_index=True)
            df['query_label'] = df.apply(lambda r: r['other_relationships'] + [r['query_relation']],axis=1)
            # Call parent's save_dataset
            super().save_dataset(df, file_path)
            
        except Exception as e:
            self.logger.error("Error in enhanced save_dataset: " + str(e))
            raise


    def create_new_row(self, p_row, n1, n2, n3):
        """
        Creates a new row based on the parent row and alternative place. Adds Not_living_in DATA WHENEVER POSSIBLE.
        Parameters:
        - p_row: The parent row (living_in relation)
        - n1: The person ID
        - n2: The original place ID (from living_in(n1,n2))
        - n3: The alternative place ID (for Not_living_in(n1,n3))
        """
        new_row = p_row.copy()
        # self.logger.debug(f''' person {n1} lives in {n2} and not in {n3}. ''')
        # Update basic fields
        new_row['query_edge'] = (n1, n3)
        new_row['query_relation'] = 'Not_living_in'  # Fixed typo from previous version
        new_row['chain_len'] = p_row['chain_len'] + 1
        
        try:
            # Process branch results and derivation chain
            branch_results = literal_eval(p_row['branch_results']) if isinstance(p_row['branch_results'], str) else p_row['branch_results']
            derivation_chain = literal_eval(p_row['derivation_chain']) if isinstance(p_row['derivation_chain'], str) else p_row['derivation_chain']
            
            new_derivation_chain = {}
            max_rules = int(p_row['ReasoningDepth'])
            
            for b, result in branch_results.items():
                if b in derivation_chain:
                    if result == 'contradiction':
                        # Keep contradiction branches as-is
                        new_derivation_chain[b] = derivation_chain[b]
                    elif result == 'unique stable model':
                        # Update derivation for successful branches
                        orig_deriv = derivation_chain[b]
                        new_deriv = {}
                        for rel, deriv_str in orig_deriv.items():
                            if rel.startswith('living_in('):
                                new_rel = f'Not_living_in({n1},{n3})'
                                new_deriv_str = deriv_str + f'  |  Not_living_in({n1},{n3}) :- living_in({n1},{n2}), {n2} != {n3}.'
                                new_deriv[new_rel] = new_deriv_str
                                
                                # Calculate rule chain length
                                steps = [s.strip() for s in new_deriv_str.split('|')]
                                rule_steps = [s for s in steps if not s.startswith('fact:')]
                                current_branch_rules = len(rule_steps)
                                # self.logger.debug(f''' For the story {self.story_number} :branch number {b} \n
                                #                         the derivation for {new_rel} is {new_deriv_str} of rule-length {current_branch_rules}, 
                                #                         and current max_rules is {max_rules}  ''')
                                max_rules = max(max_rules, current_branch_rules)
                        
                        new_derivation_chain[b] = new_deriv
            
            new_row['derivation_chain'] = str(new_derivation_chain)
            new_row['ReasoningDepth'] = max(p_row['ReasoningDepth'], max_rules)
            
            # Update unique rules
            unique_rules = literal_eval(p_row['unique_rules']) if isinstance(p_row['unique_rules'], str) else p_row['unique_rules']
            new_rule = f'Not_living_in({n1},{n3}) :- living_in({n1},{n2}), {n2} != {n3}.'
            unique_rules.add(new_rule)
            new_row['unique_rules'] = str(unique_rules)
            
        except Exception as e:
            # If processing fails, use original values
            new_row['derivation_chain'] = p_row['derivation_chain']
            new_row['ReasoningDepth'] = p_row['ReasoningDepth']
            new_row['unique_rules'] = p_row['unique_rules']
            import traceback
            traceback.print_exc()  # For debugging
        
        return new_row
    
    def check_relationships(self, df):
        # Get mask of mismatched rows
        mismatch_mask = df.apply(
            lambda row: set(row['other_implied_relationships']) != set(row['other_implied_relationships1']),
            axis=1
        )
        
        if mismatch_mask.any() and self.logger:
            # Get first mismatched row properly
            bad_row = df[mismatch_mask].iloc[0]
            self.logger.error(
                f"Mismatch found in row {bad_row.name}:\n"
                f"query_edge: {bad_row['query_edge']}\n"
                f"query_relation: {bad_row['query_relation']}\n"
                f"story_index: {bad_row['story_index']}\n"
                f"other_implied_relationships: {bad_row['other_implied_relationships']}\n"
                f"other_implied_relationships1: {bad_row['other_implied_relationships1']}"
                f"total_mismatches: {len(mismatch_mask)}"
            )
            return False
        return True