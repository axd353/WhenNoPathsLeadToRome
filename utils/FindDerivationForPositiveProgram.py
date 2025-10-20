import logging
from typing import Dict, List, Tuple, Set, Optional, Union
import pickle
import pandas as pd
import sys
sys.path.append('..')
from utils.clean_up_data import log_dataframe_with_gaps
###todod , cannot handle != operator in ehad of program
class PositiveProgramTP:
    """
    A class for:
      1) Parsing a *positive* (safe) ASP program (no 'not').
      2) Iterating T_P (the one-step consequence operator) to find the least model.
      3) Tracking a minimal (non-cyclic) derivation for each derived atom.
      4) Stopping if a constraint is violated (no stable model).
      5) Printing final iteration stats plus a derivation table for each atom.
    """

    def __init__(self, program_str: str, logger: Optional[logging.Logger] = None):
        self.program_str = program_str
        self.logger = logger or logging.getLogger(__name__)

        # Program data
        self.facts: List[Tuple[str, Tuple]] = []   # e.g. [("mother_of", (2,1)), ...]
        self.rules: List[Tuple[Tuple[str, List[str]], List[Tuple[str, List[str]]]]] = []
        self.constraints: List[List[Tuple[str, List[str]]]] = []

        # For derivations:  a map from an atom (pred, (args...)) -> (chain_of_strings, chain_length)
        # Example: derivations[("isfemale",(2,))] = (["fact: isfemale(2)"], 0) or
        #          (["mother_of(2,1) :- .", "isfemale(2) :- mother_of(2,1)."], 2)
        # We aim to keep the chain with the fewest rules.
        self.derivations: Dict[Tuple[str, Tuple], Tuple[List[str], int]] = {}

    # ----------------------------------------------------------------------
    # 1) PARSE PROGRAM
    # ----------------------------------------------------------------------
    def parse_program(self) -> None:
        """
        Very naive parsing of:
          - Facts: "p(1,2)."
          - Horn rules: "head(...) :- body1(...), body2(...)."
          - Constraints: ":- some_atom(...), other_atom(...)."
        We skip lines with 'not'.
        """
        self.facts.clear()
        self.rules.clear()
        self.constraints.clear()
        self.derivations.clear()  # reset

        lines = self.program_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if not line.endswith('.'):
                self.logger.debug("Skipping malformed line (no period): %s", line)
                continue

            # Remove trailing period
            line = line[:-1].strip()

            # If 'not' is present, skip (purely positive approach)
            if ' not ' in line:
                self.logger.debug("Skipping line with 'not': %s", line)
                continue

            # Check constraint
            if line.startswith(':-'):
                constraint_body_str = line[2:].strip()  # remove ':-'
                body_atoms_str = self._split_body_literals(constraint_body_str)
                body_atoms = []
                for atom_str in body_atoms_str:
                    pred_b, vars_b = self._parse_atom(atom_str)
                    body_atoms.append((pred_b, vars_b))
                self.constraints.append(body_atoms)
                continue

            # Check if it's a rule
            if ':-' in line:
                head_part, body_part = line.split(':-')
                head_part = head_part.strip()
                body_part = body_part.strip()

                # parse head
                head_pred, head_vars = self._parse_atom(head_part)
                # parse body
                body_atoms_str = self._split_body_literals(body_part)
                body_atoms = []
                for atom_str in body_atoms_str:
                    pred_b, vars_b = self._parse_atom(atom_str)
                    body_atoms.append((pred_b, vars_b))

                self.rules.append(((head_pred, head_vars), body_atoms))
            else:
                # It's a fact
                pred, args = self._parse_atom(line)
                ground_args = [self._convert_if_int(a) for a in args]
                self.facts.append((pred, tuple(ground_args)))
        self._reorder_neq_in_all_bodies()
        # self.logger.debug(f"Facts: {self.facts}, \n Rules: {self.rules},\n Constraints: {self.constraints}")

    def _reorder_neq_in_all_bodies(self):
        """
        For every rule and constraint body, place any '!=' predicates at the end of the body.
        """
        # Reorder rule bodies
        for i, ((head_pred, head_vars), body) in enumerate(self.rules):
            new_body = self._move_neq_literals_to_end(body)
            self.rules[i] = ((head_pred, head_vars), new_body)

        # Reorder constraint bodies
        for j, cbody in enumerate(self.constraints):
            self.constraints[j] = self._move_neq_literals_to_end(cbody)


    def _move_neq_literals_to_end(self, body: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
        """
        Given a body list of (predicate, [args]), move any items whose predicate is '!=' to the end.
        """
        normal_lits = []
        neq_lits = []
        for (pred, args) in body:
            if pred == '!=':
                neq_lits.append((pred, args))
            else:
                normal_lits.append((pred, args))
        return normal_lits + neq_lits

    def _split_body_literals(self, body_part: str) -> List[str]:
        """
        Split by top-level commas, ignoring commas within parentheses.
        e.g. "p(X,Y), q(Z)" -> ["p(X,Y)", "q(Z)"]
        """
        result = []
        current = []
        depth = 0
        for ch in body_part:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                literal = ''.join(current).strip()
                if literal:
                    result.append(literal)
                current = []
            else:
                current.append(ch)
        # last literal
        literal = ''.join(current).strip()
        if literal:
            result.append(literal)
        return result

    def _parse_atom(self, text: str) -> Tuple[str, List[str]]:
        """
        e.g. "p(1,2)" -> ("p", ["1","2"]), or "fact" -> ("fact", [])
        """
        text = text.strip()
        # 1) Check for the special case "U != V" (with no parentheses)
        if '!=' in text and '(' not in text:
            parts = text.split('!=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                return ("!=", [left, right])
            else:
                raise ValueError(f"Could not parse '!=' atom: {text}")
        idx_open = text.find('(')
        if idx_open == -1:
            # 0-arity
            return (text, [])
        pred = text[:idx_open].strip()
        idx_close = text.rfind(')')
        if idx_close == -1:
            raise ValueError("Missing closing parenthesis in: " + text)
        inside = text[idx_open+1:idx_close].strip()
        if not inside:
            return (pred, [])
        args = [x.strip() for x in inside.split(',')]
        return (pred, args)

    def _convert_if_int(self, s: str) -> Union[int, str]:
        try:
            return int(s)
        except ValueError:
            return s

    def _is_variable(self, x: str) -> bool:
        return len(x) > 0 and x[0].isupper()

    # ----------------------------------------------------------------------
    # 2) COMPUTE LEAST MODEL WITH DERIVATIONS
    # ----------------------------------------------------------------------
    def _ground_atom(
        self,
        pred: str,
        arglist: List[str],
        subs: Dict[str, int]
    ) -> Tuple[str, Tuple]:
        """
        Apply the (var->value) substitution to the head arguments
        and produce a ground atom:  (pred, (val1, val2, ...)).
        """
        grounded_args = []
        for a in arglist:
            if self._is_variable(a):
                grounded_args.append(subs[a])
            else:
                grounded_args.append(self._convert_if_int(a))
        return (pred, tuple(grounded_args))
 
    def compute_least_model(self) -> Union[Tuple[Set[Tuple[str, Tuple]], int], Tuple[str,int]]:
        """
        Iteratively apply T_P from the empty set until fixpoint,
        check constraints, and gather derivations.

        After success, build a derivation DataFrame and optionally print or save it.

        Returns either (final_atoms, iteration_count) or ("No Stable Model", iteration_of_violation).
        """
        step_count = 0
        A_old: Set[Tuple[str, Tuple]] = set()

        # Set up derivations for facts (chain_len=0)
        for f in self.facts:
            if f not in self.derivations:
                self.derivations[f] = ([f"fact: {self._atom_to_str(f)}"], 0)
            A_old.add(f)

        iteration_data = []
        cumul_fires = 0

        while True:
            step_count += 1
            A_new, iteration_fired_count = self._immediate_consequence_operator(A_old)

            violated, which_c,busted_atoms = self._check_constraints_violated(A_new)
            
            if violated:
                new_atoms = A_new - A_old
                iteration_data.append((
                    step_count,
                    len(new_atoms),
                    iteration_fired_count,
                    cumul_fires + iteration_fired_count,
                    len(A_new),
                    which_c
                ))

                # self.logger.info(f'''Constraint violated  at iteration {step_count}. => No Stable Model \n the contsraint is {which_c} \n the old stable modle was {A_old} \n 
                #                  new model is {A_new} \n grounding for the busted violation is {busted_atoms}''')
                self.busted_result = self._build_derivations_df(busted_atoms)
                # log_dataframe_with_gaps(self.busted_result, self.logger, column_list = ['derived_atom', 'derivation_chain', 'chain_len', 'num_facts_required',
                #         'sum_facts_world_rules', 'story_facts'])
                # self.logger.debug(f''' Steps taken to reach contradiction are {self.busted_result['chain_len'].sum()}                   ''')
                return (None, step_count)

            new_atoms = A_new - A_old
            # print(f'During {step_count}, new atoms fired is  {new_atoms} \n and the derivations are {self.derivations} \n')
            n_new_atoms = len(new_atoms)
            cumul_fires += iteration_fired_count

            iteration_data.append((
                step_count,
                n_new_atoms,
                iteration_fired_count,
                cumul_fires,
                len(A_new),
                ""
            ))
            # self.logger.debug(f"Iteration {step_count}: #newAtoms={n_new_atoms}, #ruleFires={iteration_fired_count}, size={len(A_new)}")

            if A_new == A_old:
                # fixpoint => stable model
                # self._print_iteration_table(iteration_data)

                # Build and display the derivation DataFrame
                deriv_df = self._build_derivations_df(A_new)
                # For demonstration, we can print it here, or you can remove the print if you like:
                # print("\n--- Derivation DataFrame ---")
                # print(deriv_df)

                return (A_new, step_count)

            A_old = A_new

    def _immediate_consequence_operator(
        self,
        current_atoms: Set[Tuple[str, Tuple]]
    ) -> Tuple[Set[Tuple[str, Tuple]], int]:
        """
        T_P step: add facts + all new heads derived from rules
        Return (new_atom_set, iteration_fired_count)
        Also update derivations for newly derived atoms.
        """
        new_atoms = set(current_atoms)
        # Make sure we include all known facts
        for f in self.facts:
            new_atoms.add(f)

        iteration_fired_count = 0

        # Fire rules
        for (head, body) in self.rules:
            (head_pred, head_vars) = head
            # unify body with current new_atoms
            all_subs = self._unify_body_with_facts(body, new_atoms)
            for subs in all_subs:
                # produce the ground head
                grounded_head = self._ground_atom(head_pred, head_vars, subs)
                if grounded_head not in new_atoms:
                    new_atoms.add(grounded_head)
                    iteration_fired_count += 1
                    # build the minimal chain for grounded_head
                    new_chain, new_chain_len = self._build_chain_for_head(grounded_head, body, subs)
                    pred = grounded_head[0]
                    if pred == "!=":
                        # Manually set chain = [] and chain_len=0 for '!=' checks
                        old_chain =  [[],0]
                    ##TODO, for head with != predicate that is elading toa  contradiction, this will result in error.
                    old_chain = self._derivation_for_atom(grounded_head)
                    if new_chain_len < old_chain[1]:
                        self.derivations[grounded_head] = (new_chain, new_chain_len)
                else:
                    new_chain, new_chain_len = self._build_chain_for_head(grounded_head, body, subs)
                    if new_chain_len < self.derivations[grounded_head][1]:
                        self.derivations[grounded_head] = (new_chain, new_chain_len)


        return (new_atoms, iteration_fired_count)

    def _build_chain_for_head(self, head_atom: Tuple[str, Tuple], body_atoms: List[Tuple[str, List[str]]], subs: Dict[str,int]) -> Tuple[List[str], int]:
        """
        Construct the minimal rule chain that yields 'head_atom' via the current rule.
        Steps:
          1) Merge the chains of all body atoms.
          2) Add the newly grounded rule for head_atom.
        Return (chain_of_strings, chain_length).
        We treat chain_length as the # of actual rules in the chain (facts count as 0).
        """
        # Merge chain sets from each body atom
        merged_chain: List[str] = []
        used_rules = set()  # to avoid duplicates if multiple body atoms share some rules
        max_body_len = 0

        for (b_pred, b_vars) in body_atoms:
            # ground the body atom
            b_atom = self._ground_atom(b_pred, b_vars, subs)
            if b_atom in self.derivations:
                (chain_b, len_b) = self.derivations[b_atom]
                # For each rule in chain_b, if not seen, append it
                for rule_str in chain_b:
                    if rule_str not in used_rules:
                        used_rules.add(rule_str)
                        merged_chain.append(rule_str)

        # now add the newly grounded rule
        grounded_rule_str = self._make_rule_str(head_atom, body_atoms, subs)
        merged_chain.append(grounded_rule_str)
        chain_length = 0
        # Count how many are actual "rule" lines. If we used the label "fact: ..." for facts, we skip those in length
        for item in merged_chain:
            if item.startswith("fact:"):
                # doesn't count as a rule
                continue
            # else it is a real rule
            chain_length += 1

        return (merged_chain, chain_length)

    def _make_rule_str(self, head_atom: Tuple[str,Tuple], body_atoms: List[Tuple[str,List[str]]], subs: Dict[str,int]) -> str:
        """
        Build a string "HEAD(...) :- BODY1(...), BODY2(...)."
        for the newly grounded rule.
        """
        head_str = self._atom_to_str(head_atom)
        body_str_list = []
        for (bp, bvars) in body_atoms:
            b_atom = self._ground_atom(bp, bvars, subs)
            body_str_list.append(self._atom_to_str(b_atom))
        joined_body = ", ".join(body_str_list)
        return f"{head_str} :- {joined_body}."

    def _atom_to_str(self, atom: Tuple[str, Tuple]) -> str:
        """
        e.g. ("ismale", (2,)) -> "ismale(2)"
             ("mother_of", (2,1)) -> "mother_of(2,1)"
             0-arity -> "foo"
        """
        (pred, args) = atom
        if not args:
            return pred
        joined = ",".join(str(a) for a in args)
        return f"{pred}({joined})"

    def _unify_body_with_facts(self, body: List[Tuple[str,List[str]]], known_atoms: Set[Tuple[str,Tuple]]) -> List[Dict[str,int]]:
        """
        Return all variable substitutions that make every body literal appear in known_atoms.
        -        example1:  For body: [('parent_of', ['X', 'Y']), ('isfemale', ['Y'])], 
        and current_atoms: {('mother_of', (2, 1)), ('isfemale', (1,)), ('isfemale', (2,)), ('parent_of', (2, 1))} ,
        the list of subs is [{'X': 2, 'Y': 1}].
        
        example2: For body: [('parent_of', ['X', 'Y']), ('parent_of', ['X', 'Z'])], 
        and current_atoms: {('isfemale', (1,)), ('sibling_of', (1, 1)), ('isfemale', (2,)), ('parent_of', (2, 1)), 
        ('mother_of', (2, 1))} ,
        the list of subs is [{'X': 2, 'Y': 1, 'Z': 1}]
        """
        if not body:
            return [dict()]
        substitutions = [dict()]
        for (b_pred, b_vars) in body:
            new_subs = []
            if b_pred == "!=":
                # Special logic for distinctness, We unify b_vars with an implicit "distinctness" condition
                for subs in substitutions:
                    if len(b_vars) != 2:
                        continue
                    # Apply the partial subs if it binds these variables
                    left_val = self._get_bound_value(b_vars[0], subs)
                    right_val = self._get_bound_value(b_vars[1], subs)

                    # If left or right are still unbound variables, we can bind them to a symbolic "distinct set" approachbut simplest is: if both are ground and equal => fail,
                    # if both are ground and distinct => keep,
                    # if one is unbound => we can unify it, but ensure distinctness from the other if it is bound.
                    maybe_subs = self._distinctness_unify(b_vars[0], left_val, b_vars[1], right_val, subs)
                    if maybe_subs is not None:
                        new_subs.append(maybe_subs)
                # *** IMPORTANT: Reassign here ***
                substitutions = new_subs
                if not substitutions:
                    break
            else:
                for subs in substitutions:
                    for (g_pred, g_args) in known_atoms:
                        if g_pred != b_pred:
                            continue
                        maybe_subs = self._unify_args(b_vars, g_args, subs)
                        if maybe_subs is not None:
                            new_subs.append(maybe_subs)
                substitutions = new_subs
                if not substitutions:
                    break
        return substitutions

    def _unify_args(self, rule_args: List[str], fact_args: Tuple, current_subs: Dict[str,int]) -> Optional[Dict[str,int]]:
        """
        Attempt to unify 'rule_args' with 'fact_args', given partial substitution 'current_subs'.
        Return new extended subs if consistent, else None.
        """
        if len(rule_args) != len(fact_args):
            return None
        updated = dict(current_subs)
        for (r_arg, f_arg) in zip(rule_args, fact_args):
            if self._is_variable(r_arg):
                if r_arg in updated:
                    if updated[r_arg] != f_arg:
                        return None
                else:
                    updated[r_arg] = f_arg
            else:
                # must match exactly
                if str(r_arg) != str(f_arg):
                    return None
        return updated
    
    def _distinctness_unify(
        self,
        varLeft: str, left_val: Union[int,str,None],
        varRight: str, right_val: Union[int,str,None],
        current_subs: Dict[str,int]
    ) -> Optional[Dict[str,int]]:
        # If both are ground
        if left_val is not None and right_val is not None:
            # If they're equal, fail
            if left_val == right_val:
                return None
            # Distinct => we accept the current_subs as is
            return current_subs

        # neqs are at the end of the body, ungrounded vars mean theya are meaningless. assume infinite supply of constants. 
        return current_subs
    
    def _get_bound_value(self, arg: str, subs: Dict[str,int]) -> Optional[Union[int,str]]:
        # If it's a variable and in subs, return subs[arg]
        if self._is_variable(arg):
            if arg in subs:
                return subs[arg]
            else:
                return None  # unbound
        else:
            # It's a constant or integer
            return self._convert_if_int(arg)

    # ----------------------------------------------------------------------
    # 3) CONSTRAINT CHECK
    # ----------------------------------------------------------------------
    def _check_constraints_violated(self, atoms: Set[Tuple[str, Tuple]]) -> Tuple[bool,str]:
        """
        For each constraint :- body, check if there's a unifier that satisfies body.
        If yes, return (True, "some text about which constraint triggered").
        Otherwise (False,"").
        """
        for i, cbody in enumerate(self.constraints, start=1):
            if not cbody:
                return (True, f"Constraint {i} has empty body => trivially violated.")
            # unify entire body
            sublist = [dict()]
            violated = True
            for (pred_b, vars_b) in cbody:
                new_subs = []
                if pred_b == "!=":
                    # handle your distinctness logic
                    for subs_ in sublist:
                        if len(vars_b) != 2:
                            continue
                        left_val = self._get_bound_value(vars_b[0], subs_)
                        right_val= self._get_bound_value(vars_b[1], subs_)
                        maybe_subs = self._distinctness_unify(vars_b[0], left_val, vars_b[1], right_val, subs_)
                        if maybe_subs is not None:
                            new_subs.append(maybe_subs)
                    sublist = new_subs
                    if not sublist:
                        violated = False
                        break
                else:
                    for subs in sublist:
                        for (gp, ga) in atoms:
                            if gp != pred_b:
                                continue
                            maybe_subs = self._unify_args(vars_b, ga, subs)
                            if maybe_subs is not None:
                                new_subs.append(maybe_subs)
                    if not new_subs:
                        violated = False
                        break
                    sublist = new_subs
            if violated and sublist:
                # Let's pick the first unifier for simplicity (we want a violating grounding) 
                final_sub = sublist[0]
                # Ground every literal in cbody
                grounded_atoms = []
                for (pb, vlist) in cbody:
                    if pb == "!=":
                        left_val = self._get_bound_value(vlist[0], final_sub)
                        right_val= self._get_bound_value(vlist[1], final_sub)
                        grounded_atoms.append(("!=", (left_val, right_val)))
                    else:
                        # normal
                        grounded_atoms.append(self._ground_atom(pb, vlist, final_sub))
                # build a textual representation
                c_text = ":- " + ", ".join(
                    self._atom_to_str((p, tuple(a))) for (p,a) in cbody
                ) + "."
                return (True, f"Constraint {i} triggered => {c_text} ", grounded_atoms)
        ## if rules have != as pred in head, and they have been unduelya ctivated, we will have a recourse
        for (pred, args) in atoms:
            if pred == "!=":
                if len(args) == 2:
                    (left, right) = args
                    if left == right:
                        # We found an atom "!= (x, x)" in the stable model => contradiction
                        # We'll call this "Constraint - '!= contradiction'"
                        # The 'busted_atoms' is just this single one
                        return (True,
                                f"Derived an impossible '!=({left},{right})'",
                                [(pred, args)]
                            )
        return (False, "", [])

    # ----------------------------------------------------------------------
    # 4) PRINTING TABLES
    # ----------------------------------------------------------------------
    def _print_iteration_table(self, iteration_data: List[Tuple[int,int,int,int,int,str]]):
        """
        Print columns: Iter, #NewAtoms, #RuleFires, CumulFires, ModelSize, ConstraintFired
        """
        print("\nIteration Results:")
        header = (
            f"{'Iter':>4} "
            f"{'#NewAtoms':>10} "
            f"{'#RuleFires':>11} "
            f"{'CumulFires':>11} "
            f"{'ModelSize':>10} "
            f"{'ConstraintFired':>20}"
        )
        print(header)
        print("-"*len(header))
        for (iter_no, newA, rf, cumRf, sz, cstr) in iteration_data:
            short_c = (cstr[:20]+"...") if len(cstr)>23 else cstr
            print(f"{iter_no:4d} {newA:10d} {rf:11d} {cumRf:11d} {sz:10d} {short_c:>20}")
        print()

    def _print_derivation_table(self, final_model: Set[Tuple[str, Tuple]]):
        """
        For each atom in the final model, print:
          Atom, Minimal Derivation (list of grounded rules/fact lines), #RulesInChain
        Sorted for consistency (e.g. by name).
        """
        print("Derivation Table (for each atom in final stable model):")
        atoms_sorted = sorted(list(final_model), key=lambda x: (x[0], x[1]))
        # columns: Atom, Derivation, #Rules
        colH = f"{'Atom':>20}  {'#Rules':>6}  {'Derivation'}"
        print(colH)
        print("-"*80)
        for atm in atoms_sorted:
            deriv_chain, chain_len = self._derivation_for_atom(atm)
            # Build a single string for the chain lines
            chain_text = "; ".join(deriv_chain)
            # Possibly truncate if huge
            print(f"{self._atom_to_str(atm):>20}  {chain_len:6d}  {chain_text}")
        print()


    # (All your existing helper methods: _immediate_consequence_operator, _unify_body_with_facts, etc.)

    # NEW METHOD to build the derivations DataFrame
    def _build_derivations_df(self,
                              final_model: Set[Tuple[str, Tuple]],
                              file_path: Optional[str] = None
                              ) -> pd.DataFrame:
        """
        Builds a pandas DataFrame with columns:

          derived_atom          (str)
          derivation_chain      (list or str)
          chain_len             (int, # rules in chain)
          num_facts_required    (int, # of facts in chain)
          sum_facts_world_rules (int, chain_len + num_facts_required)
          story_facts           (list of all facts from the program)

        If file_path is given, we also save this DataFrame to that path (e.g., CSV).
        """
        rows = []
        # Let's store a textual version of the entire set of facts
        story_facts_txt = [ self._atom_to_str(f) for f in self.facts ]
        # self.logger.debug(f'LOoking to find derivations of  {final_model}.  ')
        for atm in sorted(final_model, key=lambda x: (x[0], x[1])):
            # e.g. atm = ("ismale",(2,))
            deriv_chain, chain_len = self._derivation_for_atom(atm)

            # Count how many lines in deriv_chain are "fact:" lines
            num_facts_required = sum(1 for d in deriv_chain if d.startswith("fact:"))
            # sum_facts_world_rules = chain_len + num_facts_required
            sum_facts_world_rules = chain_len + num_facts_required

            # We can store derivation_chain as a single string or a list. Let's do a single string:
            derivation_chain_str = "  |  ".join(deriv_chain)

            # Build a row
            row = {
                "derived_atom": self._atom_to_str(atm),
                "derivation_chain": derivation_chain_str,
                "chain_len": chain_len,
                "num_facts_required": num_facts_required,
                "sum_facts_world_rules": sum_facts_world_rules,
                "story_facts": story_facts_txt
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if file_path:
            # Save to CSV or any format. e.g.:
            df.to_csv(file_path, index=False)
            self.logger.info(f"Derivations DataFrame saved to {file_path} it has columns {df.columns}, is of shape {df.shape}")

        return df

    def _derivation_for_atom(self, atom: Tuple[str, Tuple]) -> Tuple[List[str], int]:
        """
        Custom 'getter' for derivations. If 'atom' is '!=', we check that it has exactly two
        distinct arguments. If they differ, we return ([], 0). Otherwise, raise an error.
        For normal atoms, we return self.derivations.get(atom, (["(no deriv)"], 999999)).
        """
        pred, args = atom

        if pred == "!=":
            # if len(args) != 2:
            #     raise ValueError(f"Bad '!=' atom with {len(args)} args: {atom}")
            # # Check if the two arguments differ
            # left, right = args
            # if left == right:
            #     raise ValueError(f"Distinctness check failed: {atom} has the same args.")
            # If distinct, we say 'no chain' and length=0
            return [], 0
        else:
            # Normal atom
            return self.derivations.get(atom, (["(no deriv)"], 999999))

def read_clingo_program(file_path):
    """
    Reads a Clingo program file and returns it as a single string.
    Skips blank lines and lines starting with ##.
    Each rule must end with a period on its own line.
    """
    with open(file_path, 'r') as f:
        lines = []
        for line in f:
            line = line.strip()
            # Skip blank lines and comment lines
            if not line or line.startswith('##'):
                continue
            # Ensure the line ends with a period
            if not line.endswith('.'):
                raise ValueError(f"Rule doesn't end with period: {line}")
            lines.append(line)
    # Join with newlines and ensure final newline
    program_str = '\n'.join(lines) + '\n'
    return program_str



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    my_logger = logging.getLogger("TPOperatorDemo")

    # #example 1
    # program_string = r'''
    # parent_of(2,1).
    # isfemale(1).
    # father_of(2,3).
    # isfemale(X) :- daughter_of(X,Y).
    # isfemale(X) :- mother_of(X,Y).
    # ismale(X)   :- son_of(X,Y).
    # ismale(X)   :- father_of(X,Y).
    # :- ismale(X), isfemale(X).
    # parent_of(X,Y) :- mother_of(X,Y).
    # parent_of(X,Y) :- father_of(X,Y).
    # sibling_of(Y,Z) :- mother_of(X,Y) , mother_of(X,Z).
    # sibling_of(Y,Z) :- father_of(X,Y) , father_of(X,Z).
    # mother_of(X,Y) :- parent_of(X,Y) , isfemale(X).
    # father_of(X,Y) :- parent_of(X,Y) , ismale(X).
    # '''
    # engine = PositiveProgramTP(program_string, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # print("Computed fixpoint (least model) =", final_model)
    # print("Number of T_P iterations =", step_count)
    # engine._print_derivation_table(final_model)

    # #example 2
    # print(f'''------------New example -------------------- \n \n       ''')
    # program_string = r'''
    # mother_of(2,1).
    # isfemale(1).
    # father_of(2,3).
    

    # isfemale(X) :- daughter_of(X,Y).
    # isfemale(X) :- mother_of(X,Y).
    # ismale(X)   :- son_of(X,Y).
    # ismale(X)   :- father_of(X,Y).
    # :- ismale(X), isfemale(X).
    # parent_of(X,Y) :- mother_of(X,Y).
    # parent_of(X,Y) :- father_of(X,Y).
    # sibling_of(Y,Z) :- mother_of(X,Y) , mother_of(X,Z).
    # sibling_of(Y,Z) :- father_of(X,Y) , father_of(X,Z).
    # mother_of(X,Y) :- parent_of(X,Y) , isfemale(X).
    # father_of(X,Y) :- parent_of(X,Y) , ismale(X).
    # '''

    # engine = PositiveProgramTP(program_string, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # print("Computed fixpoint (least model) =", final_model)
    # print("Number of T_P iterations =", step_count)

    #example 3
    # program_string = read_clingo_program("example_prog1.txt")
    # engine = PositiveProgramTP(program_string, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # print("Computed fixpoint (least model) =", final_model)
    # print("Number of T_P iterations =", step_count)
    # engine._print_derivation_table(final_model)
    # engine._build_derivations_df(final_model,file_path='test_derivation_world_gender_locs_no_horn.csv')

    # #example 4 != body is activated
    # program_string ='''
    #         parent_of(1,2).
    #         child_of(3,1).
    #         sibling_of(Y, X) :- parent_of(Z1,X), Y != X, child_of(Y,Z1).
    #         :- sibling_of(X, X).
    #     '''
    # engine = PositiveProgramTP(program_string, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # my_logger.info(f"---Computed fixpoint (least model) = {final_model}")
    # my_logger.info(f"Number of T_P iterations ={step_count} \n")
    # engine._print_derivation_table(final_model)
    # derivations_df = engine._build_derivations_df(final_model,file_path='test_derivation_world_gender_locs_no_horn.csv')
    # my_logger.debug(derivations_df.columns)
    # log_dataframe_with_gaps(derivations_df, my_logger, column_list = ['derivation_chain','chain_len', 'derived_atom'])

    # ## program 5 contradiction , reports stuff about contradiction correctly
    # my_logger.info(f'\n \n begin story 5')
    # program_str = '''
    # colleague_of(a,b1).
    # living_in(b1,cardiff).
    # living_in(b2,bristol).
    # living_in(b3,cardiff).
    # is_child(b1).

    # living_in(Y,Z) :- living_in_same_place(X,Y), living_in(X,Z).
    # living_in_same_place(X,Y) :- living_in_same_place(Y,X).
    # living_in_same_place(Y,X) :- colleague_of(Y,X).
    # colleague_of(X,Y) :- colleague_of(Y,X).
    # colleague_of(X,Y) :- colleague_of(X,Z), colleague_of(Z,Y).
    # :- colleague_of(X,Y), is_child(X).
    # '''
    # engine = PositiveProgramTP(program_str, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # my_logger.info(f"---Computed fixpoint (least model) = {final_model}")
    # my_logger.info(f"Number of T_P iterations ={step_count}, busted results are: \n \n")
    # log_dataframe_with_gaps(engine.busted_result, my_logger)
    
    #program 6 contradciton, body of contradicting rule has !=
    # program_str = '''
    # mother_of(2,1).
    # father_of(3,1).
    # father_of(4,1).
    # parent_of(X,Y) :- mother_of(X,Y).
    # parent_of(X,Y) :- father_of(X,Y).
    # :- parent_of(U,X), parent_of(V,X), parent_of(W,X), U != V, W != V, U != W.
    # '''
    # engine = PositiveProgramTP(program_str, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # my_logger.info(f"---Computed fixpoint (least model) = {final_model}")
    # my_logger.info(f"Number of T_P iterations ={step_count} \n")
   
    # #program 7 heads of rule have !=, and they get unduely activated
    # my_logger.info(f'\n \n begin story 7')
    # program_str = '''
    # spouse_of(2,2).
    # X != Y:- spouse_of(X,Y).
    # '''
    # engine = PositiveProgramTP(program_str, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()
    # log_dataframe_with_gaps(engine.busted_result, my_logger)
    # my_logger.info(f"---Computed fixpoint (least model) = {final_model}")
    # my_logger.info(f"Number of T_P iterations ={step_count} \n")

    # paper program
    # program_str = '''
    # living_in_same_place(X,Z) :- living_in_same_place(X,Y), living_in_same_place(Y,Z).
    # living_in_same_place(Y,X) :- school_mates_with(Y,X).
    # living_in_same_place(Y,X) :- belongs_to(X, underage), parent_of(Y,X).
    # living_in_same_place(Y,X) :- living_in_same_place(X,Y).
    # :- belongs_to(X, underage), parent_of(X,Y).
    # living_in(Y,Z) :- living_in_same_place(X,Y), living_in(X,Z).
    # belongs_to(X, underage) :- school_mates_with(X,U).
    # is_agegroup(underage).
    # school_mates_with(ram,irfan).
    # parent_of(lola,ram).
    # living_in(irfan,calcutta).
    # '''
    # engine = PositiveProgramTP(program_str, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # my_logger.info(f"---Computed fixpoint (least model) = {final_model}")
    # my_logger.info(f"Number of T_P iterations ={step_count} \n")
    # engine._print_derivation_table(final_model)
    ##example##illustrate noisinbess of number of derivation steps
    # program_string = read_clingo_program("example_prog1.txt")
    # engine = PositiveProgramTP(program_string, logger=my_logger)
    # engine.parse_program()  # parse the rules/facts
    # final_model, step_count = engine.compute_least_model()

    # print("Computed fixpoint (least model) =", final_model)
    # print("Number of T_P iterations =", step_count)
    # engine._print_derivation_table(final_model)
    # engine._build_derivations_df(final_model,file_path='NoRATesting.csv')
    #program 7
    ### explore this one ------------------------------------------------------------------------------------------
    ### find solutions of ['mother_of(0,13).', 'sister_of(3,0).', 'is_female(0,0).', 'is_person(0).', 'is_person(3).', 'is_person(13).']
    ###  {0: {'sister_of(0,3)': 'fact: mother_of(0,13)  |  fact: sister_of(3,0)  |  
    # sibling_of(3,0) :- sister_of(3,0).  |  maternal_aunt_or_uncle_of(3,13) :- mother_of(0,13), sibling_of(3,0).  |  
    # sibling_of(0,3) :- mother_of(0,13), maternal_aunt_or_uncle_of(3,13).  |  fact: is_female(0,0)  |  
    # sister_of(0,3) :- sibling_of(0,3), is_female(0,0).'}}
    program_string = read_clingo_program("example_prog1.txt")
    engine = PositiveProgramTP(program_string, logger=my_logger)
    engine.parse_program()  # parse the rules/facts
    final_model, step_count = engine.compute_least_model()

    print("Computed fixpoint (least model) =", final_model)
    print("Number of T_P iterations =", step_count)
    engine._print_derivation_table(final_model)
    engine._build_derivations_df(final_model,file_path='test_derivation_world_gender_locs_no_horn.csv')