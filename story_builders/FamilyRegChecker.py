import re
import networkx as nx
import sys
sys.path.append('..')
from RegularityChecker import RegularityChecker
import logging


class FamilyRegChecker(RegularityChecker):
    def __init__(self, flat_rels=None, ascending_rels=None, descending_rels=None):
        """
        Initialize with three lists of relationships.
        Default lists are:
          - flat_rels: ['brother_of', 'husband_of', 'sister_of', 'wife_of', 'sibling_of', 'spouse_of']
          - ascending_rels: ['aunt_of', 'father_in_law_of', 'father_of', 'grandfather_of','grandparent_of','parent_in_law_of',
                             'grandmother_of', 'mother_in_law_of', 'mother_of', 'uncle_of', 'parent_of', 'aunt_uncle_of']
          - descending_rels: ['daughter_in_law_of', 'daughter_of', 'granddaughter_of','grandchild_of','nibling_of','child_of',
                              'grandson_of', 'neice_of', 'nephew_of', 'son_in_law_of', 'son_of']
        """
        if flat_rels is None:
            flat_rels = ['brother_of', 'husband_of', 'sister_of', 'wife_of', 'sibling_of', 'spouse_of']
        if ascending_rels is None:
            ascending_rels = ['aunt_of', 'father_in_law_of', 'father_of', 'grandfather_of','grandparent_of', 'parent_in_law_of',
                             'grandmother_of', 'mother_in_law_of', 'mother_of', 'uncle_of', 'parent_of', 'aunt_uncle_of',
                             'paternal_aunt_or_uncle_of', 'maternal_aunt_or_uncle_of', 'paternal_aunt_of', 'paternal_uncle_of', 'maternal_aunt_of', 
                             'maternal_uncle_of', 'maternal_grandparent_of', 'maternal_grandmother_of',  'maternal_grandfather_of',
                              'paternal_grandparent_of', 'paternal_grandmother_of',  'paternal_grandfather_of' ]
        if descending_rels is None:
            descending_rels = ['daughter_in_law_of', 'daughter_of', 'granddaughter_of','grandchild_of','nibling_of','child_of',
                              'grandson_of', 'neice_of', 'nephew_of', 'son_in_law_of', 'son_of']
        super().__init__()
        self.flat_rels = flat_rels
        self.ascending_rels = ascending_rels
        self.descending_rels = descending_rels
        
    def is_regular(self, program: str, logger=  None) -> bool:
        res1 = self.is_regular1(program, logger)
        res2 = self.is_regular2(program, logger)
        return res1 & res2

    def is_regular1(self, program: str, logger=  None) -> bool:
        """
        Checks if the logic program satisfies regularity conditions.
        It extracts all grounded facts (lines without ":-") whose predicate belongs to either
        flat_rels or ascending_rels. For each such fact an edge is recorded (from source to target)
        in a mapping.
        
        A directed graph is constructed (ignoring edge labels) and all cycles are enumerated using
        NetworkX's simple_cycles.
        
        For each cycle, the original mapping is consulted: if any edge in the cycle has a predicate
        that is in ascending_rels, then this cycle is considered irregular and the method returns False.
        Otherwise, it returns True.
        """
        self.logger = logger
        # Split program into lines and extract grounded facts.
        lines = program.strip().splitlines()
        fact_lines = [line.strip() for line in lines if ":-" not in line and line.strip()]
        
        # Build edge mapping: (src, dst) -> list of predicates.
        edge_dict = {}
        nodes = set()
        for fact in fact_lines:
            # Remove trailing period.
            if fact.endswith('.'):
                fact = fact[:-1]
            # Expect a fact in the form "predicate(arg1,arg2)" or "not predicate(arg1,arg2)".
            m = re.match(r'^(not\s+)?(\w+)\((\d+),\s*(\d+)\)$', fact)
            if m:
                neg, predicate, arg1, arg2 = m.groups()
                # We consider only facts for predicates in flat_rels or ascending_rels.
                if predicate not in self.flat_rels and predicate not in self.ascending_rels:
                    continue
                src = int(arg1)
                dst = int(arg2)
                nodes.add(src)
                nodes.add(dst)
                edge_dict.setdefault((src, dst), []).append(predicate)
        
        # Build a directed graph (ignoring edge labels).
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for (src, dst) in edge_dict:
            G.add_edge(src, dst)
        
        # Use NetworkX to enumerate all simple cycles.
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            # For a cycle, derive the list of edges (u, v) along the cycle.
            n = len(cycle)
            cycle_edges = []
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                cycle_edges.append((u, v))
            # Check each edge for an ascending relationship.
            for (u, v) in cycle_edges:
                predicates = edge_dict.get((u, v), [])
                for p in predicates:
                    if p in self.ascending_rels:
                        if self.logger != None:
                            self.logger.debug(f'''  IN The cycle {cycle} in the edge {(u,v)} has type {p}. So rejecting fact.             ''')
                        return False
        return True
    
    def is_regular2(self, program: str, logger=  None) -> bool:
        """
        Checks if the logic program satisfies regularity conditions.
        It extracts all grounded facts (lines without ":-") whose predicate belongs to either
        flat_rels or descending_rels. For each such fact an edge is recorded (from source to target)
        in a mapping.
        
        A directed graph is constructed (ignoring edge labels) and all cycles are enumerated using
        NetworkX's simple_cycles.
        
        For each cycle, the original mapping is consulted: if any edge in the cycle has a predicate
        that is in descending_rels, then this cycle is considered irregular and the method returns False.
        Otherwise, it returns True.
        """
        self.logger = logger
        # Split program into lines and extract grounded facts.
        lines = program.strip().splitlines()
        fact_lines = [line.strip() for line in lines if ":-" not in line and line.strip()]
        
        # Build edge mapping: (src, dst) -> list of predicates.
        edge_dict = {}
        nodes = set()
        for fact in fact_lines:
            # Remove trailing period.
            if fact.endswith('.'):
                fact = fact[:-1]
            # Expect a fact in the form "predicate(arg1,arg2)" or "not predicate(arg1,arg2)".
            m = re.match(r'^(not\s+)?(\w+)\((\d+),\s*(\d+)\)$', fact)
            if m:
                neg, predicate, arg1, arg2 = m.groups()
                # We consider only facts for predicates in flat_rels or descending_rels.
                if predicate not in self.flat_rels and predicate not in self.descending_rels:
                    continue
                src = int(arg1)
                dst = int(arg2)
                nodes.add(src)
                nodes.add(dst)
                edge_dict.setdefault((src, dst), []).append(predicate)
        
        # Build a directed graph (ignoring edge labels).
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for (src, dst) in edge_dict:
            G.add_edge(src, dst)
        
        # Use NetworkX to enumerate all simple cycles.
        cycles = list(nx.simple_cycles(G))
        if (self.logger != None) & (len(cycles)>0):
            self.logger.debug(f'''  The cycles in the entity rel graph are {cycles}  .              ''')
        for cycle in cycles:
            # For a cycle, derive the list of edges (u, v) along the cycle.
            n = len(cycle)
            cycle_edges = []
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                cycle_edges.append((u, v))
            # Check each edge for an ascending relationship.
            for (u, v) in cycle_edges:
                predicates = edge_dict.get((u, v), [])
                for p in predicates:
                    if p in self.descending_rels:
                        if self.logger != None:
                            self.logger.debug(f'''  In The cycle {cycle} in the edge {(u,v)} has type {p}. So rejecting fact.             ''')
                        return False
        return True
    
if __name__ == "__main__":
    # Create a FamilyRegChecker instance.
    checker = FamilyRegChecker()
    # Set up a basic logger.
    logger = logging.getLogger("FamilyRegChecker")
    logging.basicConfig(level=logging.DEBUG)


    # Example 1: No cycles (acyclic program).
    program1 = """
    mother_of(1,2).
    sister_of(0,1).
    """
    result1 = checker.is_regular(program1, logger=logger)
    print("Example 1 (No cycles) -> is_regular:", result1)
    # Expected: True

    # Example 2: Cycles exist, but only with flat relationships.
    # 'brother_of' is in flat_rels, so the cycle should be allowed.
    program2 = """
    brother_of(0,1).
    brother_of(1,2).
    brother_of(2,0).
    """
    result2 = checker.is_regular(program2, logger=logger)
    print("Example 2 (Cycle with flat_rels only) -> is_regular:", result2)
    # Expected: True

    # Example 3: Two cycles present, one of which includes an ascending relationship.
    # In this program, the cycle {0->1, 1->0} is formed by 'aunt_of' and 'mother_of' (both in ascending_rels),
    # so is_regular should return False.
    program3 = """
    aunt_of(0,1).
    mother_of(1,0).
    brother_of(2,3).
    brother_of(3,2).
    """
    result3 = checker.is_regular(program3, logger=logger)
    print("Example 3 (Cycle with an ascending relation) -> is_regular:", result3)
    # Expected: False