import random
import re
import logging
import pandas as pd
import functools
import os 
from datetime import datetime
import numpy as np
import glob
import sys
import json
import math
# Add the project directory to path for importing utilities and WorldSpecifics.
sys.path.append('..')
sys.path.append('world_descriptors')
from utils.clingo_utils import run_clingo


class BenchmarkDatasetBuilderASP:
    def __init__(self,config_file,  seed=None, output_file=None):
        """
        Initialize with a directory of universal rules files, a Regularity_checker object,
        and an optional seed.
        Loads the universal rules from all text files in the directory, extracts plausible relation names,
        and instantiates a world object (here, WorldSpecificsClutter).
        """
        self.config = self._parse_config(config_file)
        self.world_rules_dir = self.config["world_rules_dir"]
        self.output_file =  output_file
        world_obj_class = self.config["world_obj_class"]
        np.random.seed(seed)
        random.seed(seed)
        self.logger = self._create_logger()
        self.calc_difficulty = eval(self.config["calculate_difficulty"]) ### be careful with this
        self.universal_rules_filepaths = BenchmarkDatasetBuilderASP.get_universal_rules_filepaths(self.world_rules_dir)
        self.logger.info(f''' FIles to read rules from are   {self.universal_rules_filepaths}.                  ''')
        self.universal_rules = self.load_universal_rules()
        self.logger.info("Initialized BenchmarkDatasetBuilderASP.")
        # You must register new worlds here.
        if world_obj_class == "WorldSpecificsClutter":
            from WorldSpecifics import WorldSpecificsClutter
            self.world_obj = WorldSpecificsClutter(self.universal_rules, self.logger)
        elif  world_obj_class == 'WorldGendersLocationsNoHorn':
            from WorldGendersLocationsNoHorn import WorldGendersLocationsNoHorn
            self.world_obj = WorldGendersLocationsNoHorn(self.universal_rules, self.logger)
        elif  world_obj_class == 'WorldGendersLocationsNoHornAmbFacts':
            from WorldGendersLocationsNoHornAmbFacts import WorldGendersLocationsNoHornAmbFacts
            self.world_obj = WorldGendersLocationsNoHornAmbFacts(self.universal_rules, self.logger)
        elif  world_obj_class == 'WorldGendersLocationsNoHornAmbFactsSp':
            from WorldGendersLocationsNoHornAmbFactsSp import WorldGendersLocationsNoHornAmbFactsSp
            self.world_obj = WorldGendersLocationsNoHornAmbFactsSp(self.universal_rules, self.logger)
        elif  world_obj_class == 'WorldNoRAHighOPEC':
            from WorldNoRAHighOPEC import WorldNoRAHighOPEC
            self.world_obj = WorldNoRAHighOPEC(self.universal_rules, self.logger)
        elif  world_obj_class == 'WorldBioToy':
            from WorldBioToy import WorldBioToy
            self.world_obj = WorldBioToy(self.universal_rules, self.logger)
        elif  world_obj_class == 'WorldHeitoNetHighDifficulty':
            from WorldHeitoNetHighDifficulty import WorldHeitoNetHighDifficulty
            self.world_obj = WorldHeitoNetHighDifficulty(self.universal_rules, self.logger)
        else:
            raise ValueError("Unsupported world object class: " + world_obj_class)
        self.logger.debug(f' Using the world object class {self.config["world_obj_class"]} . ')
        self.plausible_relations = self.world_obj.plausible_relations

    def _parse_config(self, config_file):
        """
        Parses a JSON configuration file and returns a dictionary.
        Expected keys include "world_rules_dir" and "world_obj_class".
        """
        filtered_lines = []
        with open(config_file, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                filtered_lines.append(line)        
        config = json.loads(''.join(filtered_lines))
        return config

    def _create_logger(self):
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logger_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if logger.hasHandlers():
            logger.handlers.clear()
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s  - %(filename)s - %(module)s.%(funcName)s -  %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @staticmethod
    def get_universal_rules_filepaths(world_rules_dir):
        """
        Given a directory containing text files, returns a list of full file paths for all text files.
        """
        pattern = os.path.join(world_rules_dir, "*.txt")
        return glob.glob(pattern)

    def load_universal_rules(self):
        """
        Loads universal rules from all provided text files. If a file’s first line is
        '##in clingo lingo', the file is processed by _load_rules_from_clingo_file;
        otherwise, _load_rules_from_file is used.
        """
        all_rules = []
        for file_path in self.universal_rules_filepaths:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            if first_line.startswith("##in clingo lingo"):
                self.logger.info(f"File {file_path} is in clingo lingo format.")
                rules_from_file = self._load_rules_from_clingo_file(file_path)
            else:
                rules_from_file = self._load_rules_from_file(file_path)
            all_rules.extend(rules_from_file)
        final_rules = "\n".join(all_rules)
        self.logger.info("Loaded universal rules from files.")
        return final_rules

    def _load_rules_from_clingo_file(self, file_path):
        """
        Loads rules from a file in 'clingo lingo' format.
        Lines starting with '##' are ignored.
        """
        rules = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("##"):
                    continue
                if line:
                    rules.append(line)
        return rules

    def _load_rules_from_file(self, file_path):
        """
        Loads and converts rules from a text file (with quantifiers) into clingo rules.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        converted_rules = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            match = re.match(r'^(all\s+.*?\.)\s*(.*)$', line)
            if match:
                quantifiers, rule_body = match.groups()
                variables = re.findall(r'all\s+(\w+)', quantifiers)
                var_map = {var: var.upper() for var in variables}
                pattern = r'^\(\s*\((.*)\)\s*->\s*(not\s*\(.*\)|\(.*\))\s*\)$'
                m = re.match(pattern, rule_body)
                if m:
                    antecedent = m.group(1).strip()
                    consequent = m.group(2).strip()
                    if consequent.lower().startswith("not"):
                        m_not = re.match(r'not\s*\((.*)\)', consequent, re.IGNORECASE)
                        if m_not:
                            consequent_clean = "not " + m_not.group(1).strip()
                        else:
                            consequent_clean = consequent
                    else:
                        consequent_clean = re.sub(r'^\(|\)$', '', consequent).strip()
                else:
                    parts = rule_body.split("->", 1)
                    if len(parts) == 2:
                        antecedent = parts[0].strip().strip("()")
                        consequent_clean = parts[1].strip().strip("()")
                    else:
                        self.logger.error("Rule parsing failed for line: " + line)
                        continue
            else:
                pattern = r'^\(\s*\((.*)\)\s*->\s*(not\s*\(.*\)|\(.*\))\s*\)$'
                m = re.match(pattern, line)
                if m:
                    antecedent = m.group(1).strip()
                    consequent = m.group(2).strip()
                    if consequent.lower().startswith("not"):
                        m_not = re.match(r'not\s*\((.*)\)', consequent, re.IGNORECASE)
                        if m_not:
                            consequent_clean = "not " + m_not.group(1).strip()
                        else:
                            consequent_clean = consequent
                    else:
                        consequent_clean = re.sub(r'^\(|\)$', '', consequent).strip()
                    var_map = {}
                else:
                    self.logger.error("Rule parsing failed for line: " + line)
                    continue
            antecedent = antecedent.replace("&", ",")
            for var, Var in var_map.items():
                antecedent = re.sub(r'\b' + re.escape(var) + r'\b', Var, antecedent)
                consequent_clean = re.sub(r'\b' + re.escape(var) + r'\b', Var, consequent_clean)
            rule = f"{consequent_clean} :- {antecedent}."
            converted_rules.append(rule)
        return converted_rules


    def build_story_benchmark(self, ent_num, max_facts, too_many_consecutive_contradictions=50000):
        """
        Generates facts one at a time and then computes the non-trivial entailed facts.
        Delegates entity generation, fact generation, and fact verification to self.world_obj.
        Returns a list of story dictionaries (one per non-trivial inferred fact) or None if generation fails.
        """
        if self.config["world_obj_class"] == 'WorldNoRAHighOPEC': 
            ent_num = int(.63*ent_num)
        entities, program, added_facts, fact_details_list = self.world_obj.gen_entities(ent_num, self.universal_rules + "\n")
        self.logger.debug(f''' After entity generation ({self.batch_counter}) we have  {len(added_facts)} added_facts:  {added_facts}. \n 
                   We also have   {len(fact_details_list)} added_facts:  {fact_details_list}.              ''')
        self.logger.debug(f"Possible entities: {entities}, possible relationships: {self.plausible_relations}. \n")
        added_fact_set = set(added_facts)
        self.print_once = True
        fact_count = max(int(len(added_fact_set)/2), 0)
        consecutive_contradictions = 0
        while fact_count < max_facts:
            fact, fact_details = self.world_obj.generate_random_fact(entities, program)
            # self.logger.debug(f"Candidate fact {fact} generated; {fact_count} facts already accepted.")
            if fact in added_fact_set:
                continue
            accepted, new_program, temp_models, consecutive_contradictions, break_flag = \
                self.world_obj._verify_fact(fact, program, consecutive_contradictions, too_many_consecutive_contradictions)
            if break_flag:
                break
            if not accepted:
                continue
            program = new_program
            added_facts.append(fact)
            added_fact_set.add(fact)
            fact_count += 1
            fact_details_list.extend(fact_details)
        self.logger.debug(f'''\n ------------------- Generated story with index: {self.batch_counter } Checking for inferred facts after generating {len(added_facts)} random facts. \n 
                             program is \n {program}. ''')
        models = run_clingo(program)
        if not models:
            self.logger.info("No answer sets after adding facts, restarting story generation.")
            return None
        model_fact_sets = [{str(atom) + "." for atom in model} for model in models]
        intersection_facts = functools.reduce(lambda a, b: a & b, model_fact_sets)
        self.logger.debug(f"{len(intersection_facts)} facts entailed by all answer sets; {len(models)} answer sets; explicitly added: {len(added_fact_set)}.")
        non_trivial_entailed = intersection_facts - added_fact_set
        self.logger.debug(f"story {self.batch_counter} Non-trivial entailed facts: {non_trivial_entailed}.")
        if len(non_trivial_entailed) < self.world_obj.min_entailed_atoms_per_story:
            self.logger.info(f"{len(non_trivial_entailed)} non-trivial entailed facts found. Require at least {self.world_obj.min_entailed_atoms_per_story}.Restarting story generation.")
            return None
        stories = []
        for fact in non_trivial_entailed:
            #TODO entity names have to be ints here---giving statements like 
            m = re.match(r'^(not\s+)?(\w+)\((\d+),\s*(\d+)\)\.$', fact)
            if m:
                neg, predicate, arg1, arg2 = m.groups()
                query_relation = (("not " if neg else "") + predicate)
                query_edge = (int(arg1), int(arg2))
            else:
                self.logger.debug(f"Could not parse inferred fact: {fact}")
                continue
            story = {
                "entities": entities,
                "story_edges": [detail[0] for detail in fact_details_list],
                "edge_types": [detail[1] for detail in fact_details_list],
                "query_edge": query_edge,
                "query_relation": query_relation,
                "program": program,
                "models": models,
                "num_answer_sets": len(models)
            }
            stories.append(story)
        self.logger.debug(f"For story {self.batch_counter}: Returning {len(stories)} query examples from this generation.")
        return stories

    def build_dataset(self, total_num_stories, ent_num, max_facts, too_many_consecutive_contradictions=2521):
        """
        Builds a dataset of multiple stories by repeatedly generating story batches until the desired total is reached.
        Uses corresponding ent_num and max_facts values from the input lists for each story.
        
        Args:
            total_num_stories: Total number of stories to generate
            ent_num: List of entity counts to use (length must equal total_num_stories)
            max_facts: List of max fact counts to use (length must equal total_num_stories)
            too_many_consecutive_contradictions: Maximum allowed consecutive contradictions
            
        Returns:
            pandas DataFrame where each row corresponds to one story
        """
        # Validate input lists
        if len(ent_num) != total_num_stories or len(max_facts) != total_num_stories:
            raise ValueError("ent_num and max_facts lists must have length equal to total_num_stories")
        stories = []
        self.batch_counter = 0
        story_counter = 0  # Number of examples or instances generated
        experiment_flag = self.config.get("experiment_with_params", False)
        self.logger.debug(f'exp flag is {experiment_flag}, and the type is {type(experiment_flag)}.')
        ## here story means what in the neurips paper is example_instances. 
        while story_counter < total_num_stories:
            self.world_obj.story_number = self.batch_counter
            self.logger.info(f"\n ---- Begin  Generating story {self.batch_counter} (example:{story_counter+1}/{total_num_stories})...")
            if experiment_flag:
                experimental_setting = self.world_obj.set_experiment_conds(self.config, self.batch_counter)
                self.logger.info(f'\n Story batch {self.batch_counter} will have {experimental_setting} ')
            # Get parameters for this story
            current_ent_num = ent_num[self.batch_counter]
            current_max_facts = max_facts[self.batch_counter]
            
            self.logger.debug(f"Using ent_num={current_ent_num}, max_facts={current_max_facts}")
            
            # Generate story batch (may contain multiple stories)
            story_batch = self.build_story_benchmark(
                current_ent_num, 
                current_max_facts, 
                too_many_consecutive_contradictions
            )
            
            if story_batch is None:
                self.logger.warning("Story batch generation failed, retrying...")
                continue
            
            # Add story index and increment counters, a story batch contains different entailed facts from the same story 
            for story in story_batch:
                story['story_index'] = self.batch_counter
                story['ent_num_used'] = current_ent_num  # Store parameters used
                story['max_facts_used'] = current_max_facts
                if experiment_flag:
                    story['experimental_setting'] = experimental_setting
                stories.append(story)
                story_counter += 1#COUNTS THE TOTAL NUMBER OF QUERIES IN THE DATASET
                    
            self.batch_counter += 1
        
        # Create DataFrame
        df = pd.DataFrame(stories)
        self.logger.info(f"Dataset generation complete. Generated {len(df)} stories.")
        
        # Calculate derivation difficulties if enabled
        if self.calc_difficulty:        
            df = self.world_obj.calc_diff(df, self.output_file)## gets saved        
        return df

    def save_dataset(self, df, file_path):
        """
        Saves the dataset DataFrame to the specified file path (inferred format, e.g., CSV).
        """
        self.world_obj.save_dataset(df,file_path)

if __name__ == "__main__":
    ## ============================================Example NoRa 1.1   ============================================
    # seed = 268
    # output_file = "../output_stories/test_stories_bench_ASP_Nora1.1.csv"
    # # output_file = "stories/test_stories_bench_ASP_genders40k.csv" #turn on ambiguity 
    # builder = BenchmarkDatasetBuilderASP('configs/config_NoRA1.1.json', seed=seed, output_file=output_file)
    # total_num_stories = 160   # Number of problem instances you want to generate
    # ent_nums = np.random.choice([22,24,26,27,28,30], size=total_num_stories) #number of nodes in the stories will be sampled from this 
    # max_facts_list = np.random.randint(40, 55, size=total_num_stories)##number of edges in the story will be sampled from this 
    # num_tries = 1000
    # dataset_df = builder.build_dataset(total_num_stories, ent_nums, max_facts_list, too_many_consecutive_contradictions=num_tries)
    # print(f"Generated Dataset: {output_file} ")
    # print(dataset_df.shape)

    ## ============================================ Example NoRa ============================================
    # seed = 1990
    # output_file = "../output_stories/test_stories_bench_ASP_Nora.csv"
    # builder = BenchmarkDatasetBuilderASP('configs/config_NoRA.json', seed=seed, output_file=output_file)
    # total_num_stories = 100   # Number of problem instances you want to generate 
    # ent_nums = np.random.choice([20,23,24,25], size=total_num_stories)#number of nodes in the stories will be sampled from this 
    # max_facts_list = np.random.randint(33, 46, size=total_num_stories)##number of edges in the story will be sampled from this 
    # num_tries = 2000
    # dataset_df = builder.build_dataset(total_num_stories, ent_nums, max_facts_list,too_many_consecutive_contradictions=num_tries)
    # print("Generated Dataset:")
    # print(dataset_df.shape)

    ##------------------------------------------Example Hetionet =================================================
    # seed = 10
    # output_file = "../output_stories/test_stories_bench_ASP_Nora.csv"
    # builder = BenchmarkDatasetBuilderASP('configs/config_HetioNet.json', seed=seed, output_file=output_file)
    # total_num_stories = 120   # Number of problem instances to genrate 
    # ent_nums = np.random.choice([18,19,20,21], size=total_num_stories) #number of nodes in the stories will be sampled from this 
    # max_facts_list = np.random.randint(40, 44, size=total_num_stories) #number of edges in the stories will be sampled from this 
    # num_tries = 2000
    # dataset_df = builder.build_dataset(total_num_stories, ent_nums, max_facts_list, too_many_consecutive_contradictions=num_tries)
    # print(f"Generated Dataset: {output_file} ")
    # print(dataset_df.shape)
    