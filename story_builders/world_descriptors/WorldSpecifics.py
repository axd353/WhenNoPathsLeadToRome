import sys
import random
import re
sys.path.append('../..')
from utils.clingo_utils import run_clingo
from story_builders.FamilyRegChecker import FamilyRegChecker
import logging


class WorldSpecifics:
    def __init__(self):
        pass

    def gen_entities(self, ent_num, program):
        """
        Default implementation: entities is a simple list from 0 to ent_num-1;
        program remains unchanged.
        """
        return list(range(ent_num)), program, []

    def generate_random_fact(self, entities, program):
        """
        Should be implemented in subclasses.
        """
        raise NotImplementedError

    def _verify_fact(self, fact, program, consecutive_contradictions, too_many_consecutive_contradictions):
        """
        Should be implemented in subclasses.
        """
        raise NotImplementedError
    
    def extract_plausible_relations(self, universal_rules):
        """
        Extracts all predicate names from the universal rules with their arity.
        Returns a list of tuples (predicate_name, arity).
        Handles both n-ary and unary predicates.
        """
        # Pattern to match predicates with arguments
        pattern = r'(?<!\w)([a-z][a-zA-Z0-9_]*)\s*\(([^)]*)\)'
        
        # Pattern for standalone facts (0-ary)
        standalone_pattern = r'(?<!\w)([a-z][a-zA-Z0-9_]*)(?=\s*\.)'
        
        relations = set()
        # Find all matches in the rules with arguments
        for match in re.finditer(pattern, universal_rules):
            pred_name = match.group(1)
            args = match.group(2)
            # Count arguments (split by commas, ignoring whitespace and variables)
            arity = len([arg for arg in re.split(r'\s*,\s*', args) if arg.strip()])
            relations.add((pred_name, arity))
        
        # Handle standalone facts (0-ary)
        for match in re.finditer(standalone_pattern, universal_rules):
            pred_name = match.group(1)
            relations.add((pred_name, 0))
        
        # Remove any built-in predicates or special terms
        exclude = {'show', 'not', 'sum', 'count', 'min', 'max', 'in', 'rel_star'}
        plausible = {rel for rel in relations if rel[0] not in exclude}
        
        # Sort by predicate name
        plausible_list = sorted(plausible, key=lambda x: x[0])
        
        self.logger.info("Extracted plausible relations with arity: " + 
                        ", ".join([f"{name}/{arity}" for name, arity in plausible_list]))
        return plausible_list
    
    def save_dataset(self, df, file_path):
        """
        Saves the dataset DataFrame to the specified file path (inferred format, e.g., CSV).
        """
        try:
            df.to_csv(file_path, index=False)
            self.logger.info(f"Dataset of size {df.shape[0]} successfully saved to {file_path}. :)")
        except Exception as e:
            self.logger.error("Error saving dataset: " + str(e))
            

class WorldSpecificsClutter(WorldSpecifics):
    def __init__(self, universal_rules, logger):
        """
        plausible_relations: list of possible relationship predicates.
        p_regularity_checker: an object with a method is_regular(program, logger=logger)
        logger: a logger object.
        """
        super().__init__()
        self.logger = logger
        self.plausible_relations = self.extract_plausible_relations(universal_rules)
        self.p_regularity_checker = FamilyRegChecker()
        

    def extract_plausible_relations(self, universal_rules):
        """
        Extracts all predicate names from the universal rules.
        Captures:
        1. Predicates in head of rules (before :-)
        2. Predicates in body of rules (after :-)
        3. Constraints (after :-)
        Excludes variables (uppercase tokens) and special terms like #show
        """
        # Pattern to match predicates in:
        # 1. Rule heads: predicate(args) :- 
        # 2. Rule bodies: :- ... predicate(args) ...
        # 3. Constraints: :- predicate(args)
        pattern = r'(?<!\w)([a-z][a-zA-Z0-9_]*)\s*\([^)]*\)'
        
        # Also match standalone predicates without parentheses for facts
        standalone_pattern = r'(?<!\w)([a-z][a-zA-Z0-9_]*)(?=\s*\.)'
        
        relations = set()
        
        # Find all matches in the rules
        for match in re.finditer(pattern, universal_rules):
            relations.add(match.group(1))
        
        for match in re.finditer(standalone_pattern, universal_rules):
            relations.add(match.group(1))
        
        # Remove any built-in predicates or special terms
        exclude = {'show', 'not', 'sum', 'count', 'min', 'max', 'in', 'rel_star'}
        plausible = {rel for rel in relations if rel not in exclude}
        
        plausible_list = sorted(plausible)
        self.logger.info("Extracted plausible relations: " + ", ".join(plausible_list))
        return sorted(plausible_list)

    def gen_entities(self, ent_num, program):
        """
        Returns entities as list(range(ent_num)) and leaves the program unchanged.
        """
        return list(range(ent_num)), program, [] , []

    def generate_random_fact(self, entities, program):
        """
        Generates one random fact using a randomly chosen relation from plausible_relations
        and two distinct entities from the list.
        Returns:
          - fact (str): e.g. "father_of(0,3)."
          - fact_detail: tuple ((entity1, entity2), relation)
        """
        if len(entities) < 2:
            raise ValueError("Need at least two entities to generate a fact.")
        e1, e2 = random.sample(entities, 2)
        relation = random.choice(self.plausible_relations)
        fact = f"{relation}({e1},{e2})."
        return fact, ((e1, e2), relation)

    def _verify_fact(self, fact, program, consecutive_contradictions, too_many_consecutive_contradictions):
        """
        Verifies a candidate fact by testing if adding it to the program yields at least one answer set
        and passes the regularity check. Updates the consecutive_contradictions counter.
        
        Returns a tuple:
          (accepted, updated_program, models, consecutive_contradictions, break_flag)
        """
        temp_program = program + fact + "\n"
        self.logger.debug(f"Running clingo with the program: {temp_program}")
        temp_models = run_clingo(temp_program)
        if not temp_models:
            consecutive_contradictions += 1
            self.logger.info(f"Fact {fact} contradicted existing facts. Consecutive contradictions: {consecutive_contradictions}.")
            if consecutive_contradictions >= too_many_consecutive_contradictions:
                self.logger.info("Too many consecutive contradictions. Exiting fact generation early.")
                return (False, program, temp_models, consecutive_contradictions, True)
            return (False, program, temp_models, consecutive_contradictions, False)
        # Reset counter since fact did not cause contradiction.
        consecutive_contradictions = 0
        # Regularity check.
        if not self.p_regularity_checker.is_regular(temp_program, logger=self.logger):
            self.logger.debug(f"Fact {fact} caused irregularity. Discarding this fact.")
            return (False, program, temp_models, consecutive_contradictions, False)
        return (True, temp_program, temp_models, consecutive_contradictions, False)
