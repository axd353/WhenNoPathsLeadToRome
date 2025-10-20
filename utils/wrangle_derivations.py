import re
import networkx as nx
import random 
from typing import Dict, List, Optional, Set, Tuple
import time 
import ast 
import pandas as pd
def extract_vars(atom):
    """
    Extract terms from rel(a,b), including constants or variables (e.g., '7', 'X')

    Example:
        Input: 'sibling_of(17,8)'
        Output: ['17', '8']
    """
    inside = atom[atom.find("(")+1:atom.find(")")]
    return [v.strip() for v in inside.split(",") if v.strip()]


def ensure_list(x):
    if isinstance(x, str):
        try:
            # try to parse a Python literal list
            return ast.literal_eval(x)
        except Exception:
            # fallback: split on commas
            return [item.strip() for item in x.split(',') if item.strip()]
    return x


def is_horn_rule(rule_str):
    """
    Determines whether a rule is a binary horn rule based on graph structure.

    A rule is considered a horn rule if:
      - It contains exactly two atoms in the body.
      - The graph constructed from body atoms has a path of length 2
        connecting the two terms in the head.

    This version treats all terms (variables or constants) as nodes.

    Returns:
        True if the rule meets the criteria, False otherwise.
    """
    if ":-" not in rule_str:
        return False

    try:
        head, body = rule_str.split(":-")
        # Use regex to extract atoms from body (e.g., rel(1,2))
        body_atoms = re.findall(r"\w+\([^\)]+\)", body)

        if len(body_atoms) != 2:
            return False

        head_terms = extract_vars(head)
        if len(head_terms) != 2:
            return False
        x, y = head_terms

        # Build undirected graph from body atoms
        G = nx.Graph()
        for atom in body_atoms:
            terms = extract_vars(atom)
            if len(terms) == 2:
                G.add_edge(terms[0], terms[1])

        return nx.has_path(G, x, y) and nx.shortest_path_length(G, x, y) == 2
    except Exception as e:
        return False



def get_num_cluttr_hops(derivation_dict):
    """
    Calculates the number of horn rules in a single derivation dictionary.

    Input:
        derivation_dict (dict): Mapping from atoms to derivation strings.

    Example:
        {
          "grandson_of(4,8)": "fact: ... | rule1 :- ... | rule2 :- ... | ..."
        }

    Output:
        Integer count of horn rules found in the rules of this derivation.

    Notes:
        - Ignores segments labeled 'fact:'.
        - Uses graph-based horn rule detection.
    """
    count = 0
    for deriv_str in derivation_dict.values():
        rules = [seg.strip() for seg in deriv_str.split("|") if ":-" in seg and not seg.strip().startswith("fact:")]
        count += sum(is_horn_rule(r) for r in rules)
    return count

def get_branch_cluttr_hops(derivation_chain):
    """
    Computes the number of horn rules for each branch in a derivation_chain dict.

    Input:
        derivation_chain (dict): Mapping from branch index → derivation dict

    Output:
        branch_cluttr_dict (dict): Mapping from branch index → number of horn rules

    Example:
        Input:
            {
              0: { 'goal_atom': 'fact: ... | rule1 :- ..., ... | rule2 :- ..., ...' },
              1: { 'other_atom': 'fact: ... | rule3 :- ..., ...' }
            }

        Output:
            { 0: 2, 1: 1 }
    """
    return {b: get_num_cluttr_hops(deriv) for b, deriv in derivation_chain.items()}

def get_total_cluttr_hops(unique_variant_set):
    """
    Sum of clutter hops (horn rules) across all unique derivation variants.

    Input:
        unique_variant_set (set): Set of str() of derivation dicts
            Each str is like: "{...}" (stringified dictionary)

    Output:
        Integer sum of clutter hops across all variants.
    """
    total = 0
    for s in unique_variant_set:
        deriv_dict = eval(s)
        total += get_num_cluttr_hops(deriv_dict)
    return total

def get_max_cluttr_hops(unique_variant_set, variant_branch_outcomes):
    """
    Calculates the maximum clutter hops across all derivation variants,
    and splits it by outcome type.

    Inputs:
        unique_variant_set (set): set of str(derivation_dict)
        variant_branch_outcomes (dict): maps str(derivation_dict) → 'unique stable model' or 'contradiction'

    Outputs:
        - max_all: maximum across all branches
        - max_model: max of only those with 'unique stable model'
        - max_contra: max of only those with 'contradiction'

    Example:
        Input:
            unique_variant_set = {"{...}", "{...}"}
            variant_branch_outcomes = {
                "{...}": "unique stable model",
                "{...}": "contradiction"
            }

        Output:
            (5, 4, 3)
    """
    max_all = max_model = max_contra = 0
    for s in unique_variant_set:
        deriv_dict = eval(s)
        hops = get_num_cluttr_hops(deriv_dict)
        max_all = max(max_all, hops)
        outcome = variant_branch_outcomes.get(s, None)
        if outcome == "unique stable model":
            max_model = max(max_model, hops)
        elif outcome == "contradiction":
            max_contra = max(max_contra, hops)
    return max_all, max_model, max_contra

#-------------------------- non-path edges

def extract_fact_atoms(deriv_str: str) -> List[str]:
    # … your existing code …
    atoms = []
    for segment in deriv_str.split("|"):
        seg = segment.strip()
        if seg.startswith("fact:"):
            atom = seg[len("fact:"):].strip()
            if re.match(r"[^\s\|:]+\([^\)]+\)", atom):
                atoms.append(atom)
    return atoms

def atom_to_edge(atom: str) -> Tuple[str,str] or None:
    # … your existing code …
    inside = atom[atom.find("(")+1:atom.rfind(")")]
    terms = [t.strip() for t in inside.split(",") if t.strip()]
    if len(terms) == 1:
        return (terms[0], terms[0])
    if len(terms) == 2:
        a, b = terms
        return (min(a,b), max(a,b))
    return None

def count_non_path_atoms_in_branch(
    derivation_dict: Dict,
    query_edge: Tuple[str,str]
) -> int:
    """
    Count how many unique fact‑atoms e=(n1,n2) in derivation_dict are *not*
    on any simple path between src and tgt in the derivation-only graph.

    We:
      1) Extract unique fact‑edges.
      2) Build G from those edges.
      3) Always add src and tgt as nodes so path checks never NodeNotFound.
      4) For each edge e:
         - Remove e
         - If removing e disconnects src→tgt, then e is on every s–t path ⇒ on_path=True
         - Else attempt two “forced” subpaths in G−e (src→n1 + n2→tgt or src→n2 + n1→tgt)
         - If neither works, on_path=False ⇒ non_path +=1
    """
    src, tgt = str(query_edge[0]), str(query_edge[1])

    # 1) gather unique fact-atoms
    atoms = []
    for s in derivation_dict.values():
        atoms.extend(extract_fact_atoms(s))
    unique_atoms = list(dict.fromkeys(atoms))

    # 2) build derivation-only graph
    fact_edges = []
    for atom in unique_atoms:
        e = atom_to_edge(atom)
        if e is not None:
            fact_edges.append(e)

    G = nx.Graph()
    G.add_edges_from(fact_edges)
    # 3) ensure src and tgt exist
    G.add_node(src)
    G.add_node(tgt)
    if set(G.nodes()) <= {src, tgt}:
        return len([e for e in fact_edges if e[0]==e[1]])

    if not nx.has_path(G, src, tgt):
        return len(fact_edges)

    non_path = 0
    for atom, e in zip(unique_atoms, fact_edges):
        n1, n2 = e
        if n1 == n2:
            non_path += 1 # self -edges are always considered non-path
            continue
        # remove the edge before testing
        G.remove_edge(n1, n2)
        on_path = False
        try:
            # a) if no path at all, then e was a bridge on every path
            if not nx.has_path(G, src, tgt):
                on_path = True
            else:
                # b) forced-inclusion test
                if nx.has_path(G, src, n1) and nx.has_path(G, n2, tgt):
                    on_path = True
                elif nx.has_path(G, src, n2) and nx.has_path(G, n1, tgt):
                    on_path = True
        except nx.NetworkXNoPath:
            # means no path in G−e, so e is on every path
            on_path = True
        except nx.NodeNotFound:
            # if either src/tgt not in G (shouldn't now), treat as no path
            on_path = False

        # restore the edge for next iteration
        G.add_edge(n1, n2)

        if not on_path:
            non_path += 1

    return non_path

def get_non_path_atom_stats(
    unique_variant_set: Set[str],
    query_edge: Tuple[any, any],
    variant_to_branch: Dict[str, List[int]],
    variant_branch_outcomes: Dict[str, str]
) -> Tuple[Dict[Tuple[int, ...], int], int]:
    """
    For each unique derivation variant-string vs in unique_variant_set:
      - If outcome is 'unique stable model', parse vs→derivation_dict and
        cnt = count_non_path_atoms_in_branch(derivation_dict, query_edge).
      - If outcome is 'contradiction', vs→full_dict with multiple entries;
        split into single-entry dicts, call count_non_path_atoms_in_branch on each,
        sum counts.
    Return:
      counts: mapping (sorted tuple of branch IDs) → total non-path atoms
      max_non: maximum over only 'unique stable model' variants
    """
    counts: Dict[Tuple[int, ...], int] = {}
    max_non = 0

    for vs in unique_variant_set:
        # get list of branch indices for this variant
        branches = tuple(sorted(variant_to_branch[vs]))
        outcome = variant_branch_outcomes.get(vs)
        # parse the variant-string to a dict
        try:
            full_dict = ast.literal_eval(vs)
        except Exception:
            # skip unparsable
            continue

        total_cnt = 0
        if outcome == 'unique stable model':
            # single dict → direct
            total_cnt = count_non_path_atoms_in_branch(full_dict, query_edge)
            max_non = max(max_non, total_cnt)
        else:  # contradiction: derive per-atom query_edge
            for atom_key, deriv_str in full_dict.items():
                # match pred(arg1[,arg2])
                m = re.match(r"^(\w+)\(([^,()]+)(?:,([^,()]+))?\)$", atom_key)
                if not m:
                    continue
                pred, a1, a2 = m.group(1), m.group(2), m.group(3)
                if pred == '!=':
                    continue
                if a2 is None:
                    qedge = (a1, a1)
                else:
                    qedge = (a1, a2)
                sub_dict = {atom_key: deriv_str}
                total_cnt += count_non_path_atoms_in_branch(sub_dict, qedge)

        counts[branches] = total_cnt

    return counts, max_non




if __name__ == "__main__":
    # # --- Sample horn rule: path of length 2 between head terms 7 and 8
    # rule1 = "sibling_in_law_of(7,8) :- sibling_of(17,8), spouse_of(7,17)."
    # print("Rule 1 is horn:", is_horn_rule(rule1))  # ✅ Expected: True

    # # --- Not a horn rule: no connection between head vars
    # rule2 = "rel1(A,B) :- rel2(C,D), rel3(E,F)."
    # print("Rule 2 is horn:", is_horn_rule(rule2))  # ❌ Expected: False

    # # --- Direct connection: not length 2
    # rule3 = "rel1(X,Y) :- rel2(X,Y), rel3(Y,Z)."  # Just 1 edge X-Y
    # print("Rule 3 is horn:", is_horn_rule(rule3))  # ❌ Expected: False

    # # --- Another valid horn rule
    # rule4 = "uncle_of(1,3) :- brother_of(1,2), parent_of(2,3)."
    # print("Rule 4 is horn:", is_horn_rule(rule4))  # ✅ Expected: True

    ### NO pathlike behavior outcome tests
    # print("Ex1: small numeric derivation graph")
    # # derivation edges: (1,2),(4,5),(2,2)
    # story_edges = []  # ignored in this new def
    # query_edge = (1,2)
    # deriv1 = {
    #     "b": (
    #       "fact: foo(1,2) | "       # edge (1,2) → on-path
    #       "fact: baz(4,5) | "       # (4,5) → off-path
    #       "fact: solo(2) | "        # (2,2) → on-path
    #       "x(1,1) :- foo(1,2)"      # rule, ignored
    #     )
    # }
    # vs = {str(deriv1)}
    # v2b = {str(deriv1): [0,1]}
    # v2o = {str(deriv1): "unique stable model"}
    # npc, mxc = get_non_path_atom_stats(vs, query_edge, v2b, v2o)
    # print(" non_path_atom_counts:", npc, " max:", mxc)
    # # → {(0,1): 1}, max=1

    # print("\nEx2: symbolic/unary mix")
    # deriv2 = {
    #   "c": (
    #     "fact: a(A,B) | "        # (A,B) on-path?
    #     "fact: b(X,Y) | "        # off-path
    #     "fact: self(Z) | "       # (Z,Z) off-path
    #     "foo(A,A) :- a(A,B)"     # rule
    #   )
    # }
    # vs2 = {str(deriv2)}
    # v2b2 = {str(deriv2): [5]}
    # v2o2 = {str(deriv2): "contradiction"}
    # q2 = ("A","B")
    # npc2, mxc2 = get_non_path_atom_stats(vs2, q2, v2b2, v2o2)
    # print(" non_path_atom_counts:", npc2, " max:", mxc2)
    # # → {(5,): 2}, max=0 (no unique models to consider)

    # print("\nExample 3 (closer to reality):")
    # query3 = (8,7)
    # branch0 = {
    #   'belongs_to(2,underage)': (
    #     "fact: school_mates_with(8,2)  |  "
    #     "school_mates_with(2,8) :- ...  |  "
    #     "belongs_to(2,underage) :- ..."
    #   ),
    #   'spouse_of(2,6)': "fact: spouse_of(2,6)"
    # }
    # branch1 = {
    #   'brother_in_law_of(8,7)': (
    #     "fact: brother_of(8,17)  |  "
    #     "sibling_of(8,17) :- ...  |  "
    #     "sibling_of(17,8) :- ...  |  "
    #     "fact: spouse_of(7,17)  |  "
    #     "sibling_in_law_of(7,8) :- ...  |  "
    #     "sibling_in_law_of(8,7) :- ...  |  "
    #     "fact: belongs_to_group(8,male)  |  "
    #     "brother_in_law_of(8,7) :- ..."
    #   )
    # }
    # branch2 = {
    #   'belongs_to(2,underage)': (
    #     "fact: school_mates_with(8,2)  |  "
    #     "school_mates_with(2,8) :- ...  |  "
    #     "fact: school_mates_with(8,5)  |  "
    #     "school_mates_with(2,5) :- ...  |  "
    #     "belongs_to(2,underage) :- ..."
    #   ),
    #   'spouse_of(2,6)': "fact: spouse_of(2,6)"
    # }

    # vs3 = {str(branch0), str(branch1), str(branch2)}
    # v2b3 = {
    #   str(branch0): [0],
    #   str(branch1): [1],
    #   str(branch2): [2]
    # }
    # v2o3 = {
    #   str(branch0): "contradiction",
    #   str(branch1): "unique stable model",
    #   str(branch2): "contradiction"
    # }

    # npc3, mxc3 = get_non_path_atom_stats(vs3, query3, v2b3, v2o3)
    # print(" counts:", npc3, " max:", mxc3)
    ## example contradiction branches
    # query3 = (11,3)
    # branch1 = {'aunt_or_uncle_of(13,2)': 'fact: aunt_or_uncle_of(13,2)', 'parent_in_law_of(13,2)': 'fact: parent_in_law_of(13,2)'}
    # vs3 = {str(branch1)}
    # v2b3 = {str(branch1): [0]}
    # v2o3 = {str(branch1): "contradiction"}
    # npc3, mxc3 = get_non_path_atom_stats(vs3, query3, v2b3, v2o3)
    # print(" counts:", npc3, " max:", mxc3)
    ## example: For a branch withoout ambiguity, how to use extract_fact_atoms , count_non_path_atoms_in_branch
    f_path = '/home/anirban/BrainStorming/DatsetsGenerated/NoRaExploring/result_59502805_4_seed158.csv'
    df_ASP =  pd.read_csv(f_path)
    print(df_ASP['OPEC'].value_counts())
    df_ASP = df_ASP[df_ASP['OPEC']==3]
    
    print(df_ASP[['query_edge', 'query_relation', 'OPEC', 'story_index', 'derivation_chain' ]].dtypes)
    print(df_ASP.iloc[0][['query_edge', 'query_relation', 'OPEC', 'story_index', 'derivation_chain' ]])
    dict_to_eval =  ast.literal_eval(df_ASP.iloc[0]['derivation_chain'])[0]
    print(dict_to_eval)
    atoms = []
    for s in dict_to_eval.values():
        atoms.extend(extract_fact_atoms(s))
    print(atoms)
    print(df_ASP.iloc[0]['fact_choice_branches'])
    print(type(df_ASP.iloc[0]['fact_choice_branches']))