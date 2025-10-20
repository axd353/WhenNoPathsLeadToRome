import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class StoryGraphPlotter:
    def __init__(self):
        pass

    def parse_model_edges(self, model):
        """
        Given a model (a set of clingo symbols represented as strings), returns a set of edges.
        Each edge is represented as a tuple: (source, target, predicate).
        """
        edges = set()
        for symbol in model:
            s = str(symbol)
            m = re.search(r"Function\('(\w+)',\s*\[Number\((\d+)\),\s*Number\((\d+)\)\]", s)
            if m:
                pred = m.group(1)
                src = int(m.group(2))
                tgt = int(m.group(3))
                edges.add((src, tgt, pred))
            else:
                m = re.match(r'(\w+)\((\d+),\s*(\d+)\)', s)
                if m:
                    pred = m.group(1)
                    src = int(m.group(2))
                    tgt = int(m.group(3))
                    edges.add((src, tgt, pred))
        return edges

    def _compute_edge_curvature(self, u, v, pos, union_edges_simple):
        """
        Computes the curvature (rad parameter) for an edge from u to v.
        If a reverse edge exists (ignoring predicate), assign opposite curvatures.
        Otherwise, use a default: 0.2 if pos[u][0] < pos[v][0] else -0.2.
        """
        default_rad = 0.2 if pos[u][0] < pos[v][0] else -0.2
        # Check if a reverse edge (v,u) exists in union_edges_simple.
        rev_exists = any(True for (x, y, _) in union_edges_simple if x == v and y == u)
        if rev_exists:
            return default_rad if u < v else -default_rad
        return default_rad

    def _bezier_midpoint(self, P0, P2, rad):
        """
        Given endpoints P0 and P2 (numpy arrays) and curvature rad,
        compute the quadratic Bézier midpoint as follows:
            Let M = (P0 + P2) / 2.
            The control point is P1 = M + rad * np.array([-d_y, d_x]), where d = P2-P0.
            Then B(0.5) = 0.25 * P0 + 0.5 * P1 + 0.25 * P2.
        """
        M = (P0 + P2) / 2.0
        d = P2 - P0
        P1 = M - rad * np.array([-d[1], d[0]])
        B_mid = 0.25 * P0 + 0.5 * P1 + 0.25 * P2
        return B_mid

    def plot_story(self, story, save_path=None, no_show=False):
        """
        Plots the directed graph for a given story dictionary.
        
        - Nodes come from story["entities"].
        - Given edges (from story_edges/edge_types) are drawn as green solid edges.
        - Inferred edges (from model(s) but not explicitly given) are drawn as black dotted edges.
        - Reverse edges are drawn with opposite curvatures.
        - For each edge, the label is placed at the Bézier midpoint of the curved edge.
          Labels for given edges use "lightcoral" and for inferred edges use "darkred".
        - An explanatory title is added.
        - If save_path is provided, the plot is saved.
        """
        # Build the directed graph.
        G = nx.DiGraph()
        for node in story["entities"]:
            G.add_node(node)
        
        # Add given edges.
        given_edges = set()
        print(f'story edges are {story["story_edges"]}')
        print(f'story edges types are {story["edge_types"]}')
        for (src, tgt), rel in zip(story["story_edges"], story["edge_types"]):
            given_edges.add((src, tgt, rel))
            G.add_edge(src, tgt, label=rel, edge_type="given")
        
        # Determine inferred edges.
        # TODO CLEAN modesl, strng smight be given as list. 
        models = story["models"]
        if len(models) == 1:
            model_edges = self.parse_model_edges(models[0])
        else:
            model_edges = None
            for m in models:
                edges = self.parse_model_edges(m)
                if model_edges is None:
                    model_edges = edges
                else:
                    model_edges = model_edges.intersection(edges)
        inferred_edges = model_edges - given_edges if model_edges is not None else set()
        for (src, tgt, rel) in inferred_edges:
            G.add_edge(src, tgt, label=rel, edge_type="inferred")
        
        # Compute layout.
        pos = nx.spring_layout(G, seed=42)
        
        # Prepare union_edges for reverse edge checking.
        union_edges = given_edges.union(inferred_edges)
        union_edges_simple = {(u, v, None) for (u, v, p) in union_edges}
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw nodes.
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Iterate over graph edges directly.
        for u, v, data in G.edges(data=True):
            p = data["label"]
            etype = data["edge_type"]
            color = "green" if etype == "given" else "black"
            style = "solid" if etype == "given" else "dotted"
            text_color = "lightcoral" if etype == "given" else "darkred"
            
            # Compute curvature for this edge.
            rad = self._compute_edge_curvature(u, v, pos, union_edges_simple)
            
            # Draw the edge with specified curvature.
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                ax=ax,
                edge_color=color,
                style=style,
                width=2,
                connectionstyle=f"arc3, rad={rad}"
            )
            
            # Compute label position using the Bézier midpoint.
            P0 = np.array(pos[u])
            P2 = np.array(pos[v])
            B_mid = self._bezier_midpoint(P0, P2, rad)
            label_x, label_y = B_mid[0], B_mid[1]
            
            ax.text(label_x, label_y, p, fontsize=6, color=text_color,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        
        explanation = (
            "GREEN EDGES are GIVEN (from the story); DOTTED EDGES are INFERRED (present in all models).\n"
            "Reverse edges are drawn with opposite curvatures so that labels do not overlap.\n"
            "Edge labels are centered on the curved edge.\n"
            "For example: an edge 3 → 5 labeled 'father_of' means 3 is the father of 5."
        )
        ax.set_title(explanation)
        ax.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Graph saved to {save_path}")
        if no_show==False:
            plt.show()


# --- Example Story Dictionaries ---

def example_story_single_model():
    story = {
        "entities": [0, 1, 2, 3, 4, 5],
        "story_edges": [(3, 5), (2, 3), (4, 1), (2, 5), (5, 0), (4, 0), (0, 3), (4, 5), (5, 4), (5, 3)],
        "edge_types": ['nephew_of', 'father_of', 'brother_of', 'father_of', 'father_of', 'brother_of', 'neice_of', 'son_of', 'daughter_of', 'mother_of'],
        "query_edge": (3, 2),
        "query_relation": "son_of",
        "program": (
            "neice_of(Y, X) :- brother_of(Z1, X) , daughter_of(Y, Z1).\n"
            "father_of(Y, X) :- brother_of(Z1, X) , father_of(Y, X).\n"
            "mother_of(Y, X) :- brother_of(Z1, X) , mother_of(Y, X).\n"
            "nephew_of(Y, X) :- brother_of(Z1, X) , son_of(Y, X).\n"
            "not father_of(Y, X) :- nephew_of(Y, X).\n"
            "not brother_of(Y, X) :- nephew_of(Y, X).\n"
            "not neice_of(Y, X) :- nephew_of(Y, X).\n"
            "not son_of(Y, X) :- nephew_of(Y, X).\n"
            "not daughter_of(Y, X) :- nephew_of(Y, X).\n"
            "not mother_of(Y, X) :- nephew_of(Y, X).\n"
            "son_of(X,Y) :- father_of(Y, X).\n"
            "nephew_of(3,5).\n"
            "father_of(2,3).\n"
            "brother_of(4,1).\n"
            "father_of(2,5).\n"
            "father_of(5,0).\n"
            "brother_of(4,0).\n"
            "neice_of(0,3).\n"
            "son_of(4,5).\n"
            "daughter_of(5,4).\n"
            "mother_of(5,3).\n"
        ),
        "models": [
            {
                "Function('brother_of', [Number(4), Number(0)], True)",
                "Function('son_of', [Number(5), Number(2)], True)",
                "Function('son_of', [Number(0), Number(5)], True)",
                "Function('son_of', [Number(3), Number(2)], True)",
                "Function('nephew_of', [Number(3), Number(5)], True)",
                "Function('daughter_of', [Number(5), Number(4)], True)",
                "Function('neice_of', [Number(0), Number(3)], True)",
                "Function('brother_of', [Number(4), Number(1)], True)",
                "Function('neice_of', [Number(5), Number(1)], True)",
                "Function('neice_of', [Number(5), Number(0)], True)",
                "Function('father_of', [Number(2), Number(3)], True)",
                "Function('father_of', [Number(2), Number(5)], True)",
                "Function('father_of', [Number(5), Number(0)], True)",
                "Function('mother_of', [Number(5), Number(3)], True)",
                "Function('son_of', [Number(4), Number(5)], True)"
            }
        ],
        "num_answer_sets": 1
    }
    return story

def example_story_multiple_models():
    model1 = {
        "Function('brother_of', [Number(4), Number(0)], True)",
        "Function('son_of', [Number(3), Number(2)], True)",
        "Function('father_of', [Number(2), Number(3)], True)",
        "Function('father_of', [Number(2), Number(5)], True)",
        "Function('father_of', [Number(5), Number(0)], True)",
        "Function('mother_of', [Number(5), Number(3)], True)",
        "Function('son_of', [Number(4), Number(5)], True)"
    }
    model2 = {
        "Function('brother_of', [Number(4), Number(0)], True)",
        "Function('son_of', [Number(3), Number(2)], True)",
        "Function('father_of', [Number(2), Number(3)], True)",
        "Function('father_of', [Number(2), Number(5)], True)",
        "Function('father_of', [Number(5), Number(0)], True)",
        "Function('mother_of', [Number(5), Number(3)], True)",
        "Function('son_of', [Number(4), Number(5)], True)",
        "Function('neice_of', [Number(0), Number(3)], True)"
    }
    story = {
        "entities": [0, 1, 2, 3, 4, 5],
        "story_edges": [(3, 5), (2, 3), (4, 1), (2, 5), (5, 0), (4, 0), (0, 3), (4, 5), (5, 4), (5, 3)],
        "edge_types": ['nephew_of', 'father_of', 'brother_of', 'father_of', 'father_of', 'brother_of', 'neice_of', 'son_of', 'daughter_of', 'mother_of'],
        "query_edge": (3, 2),
        "query_relation": "son_of",
        "program": (
            "neice_of(Y, X) :- brother_of(Z1, X) , daughter_of(Y, Z1).\n"
            "father_of(Y, X) :- brother_of(Z1, X) , father_of(Y, X).\n"
            "mother_of(Y, X) :- brother_of(Z1, X) , mother_of(Y, X).\n"
            "nephew_of(Y, X) :- brother_of(Z1, X) , son_of(Y, X).\n"
            "not father_of(Y, X) :- nephew_of(Y, X).\n"
            "not brother_of(Y, X) :- nephew_of(Y, X).\n"
            "not neice_of(Y, X) :- nephew_of(Y, X).\n"
            "not son_of(Y, X) :- nephew_of(Y, X).\n"
            "not daughter_of(Y, X) :- nephew_of(Y, X).\n"
            "not mother_of(Y, X) :- nephew_of(Y, X).\n"
            "son_of(X,Y) :- father_of(Y, X).\n"
            "nephew_of(3,5).\n"
            "father_of(2,3).\n"
            "brother_of(4,1).\n"
            "father_of(2,5).\n"
            "father_of(5,0).\n"
            "brother_of(4,0).\n"
            "neice_of(0,3).\n"
            "son_of(4,5).\n"
            "daughter_of(5,4).\n"
            "mother_of(5,3).\n"
        ),
        "models": [model1, model2],
        "num_answer_sets": 2
    }
    return story


def plot_experimental_results(df_path):
    """
    Groups the final dataset (DataFrame) by 'story_index'. 
    For each story (i.e. group), it:
        - Extracts the experimental settings from the 'experimental_setting' column
        (the same for all rows within the group).
        - Computes three statistics from the numeric column 'chain_len':
            qt1: maximum chain_len,
            qt2: median chain_len,
            qt3: second largest chain_len (if only one value exists, then that value).
        - Then, for each experimental parameter (e.g., person_percent, male_prob, assign_loc_prob)
        found in the experimental_setting, it creates a scatter plot of the chain-length 
        metrics vs. the parameter value.
        
    Returns:
        A new DataFrame where each row corresponds to one story and includes the computed qt1, qt2, qt3
        as well as the experimental settings.
    """
    df= pd.read_csv(df_path)
    # Group by story_index.
    grouped = df.groupby('story_index')
    results = []
    # Iterate through each group (each story).
    for story_index, group in grouped:
        # Extract experimental settings from the group (if available)
        exp_setting = group['experimental_setting'].iloc[0] if 'experimental_setting' in group.columns else {}
        exp_setting = eval(exp_setting)## risky, todo
        
        # Works for WorldGendersLOcationNoHorn  Use the constant parameters stored in these columns.
        ent_used = group['ent_num_used'].iloc[0] if 'ent_num_used' in group.columns else None
        facts_used = group['max_facts_used'].iloc[0] if 'max_facts_used' in group.columns else None
        
        # Get the chain lengths.
        chain_lengths = group['chain_len'].dropna().astype(float)
        if chain_lengths.empty:
            continue
        qt1 = chain_lengths.max()
        sorted_lengths = chain_lengths.sort_values(ascending=False)
        qt3 = sorted_lengths.iloc[1] if len(sorted_lengths) > 1 else sorted_lengths.iloc[0]
        
        # Build a result row.
        row = {
            "story_index": story_index,
            "qt1": qt1,
            "ent_num_used": ent_used,
            "max_facts_used": facts_used
        }
        # Add experimental settings.
        for key, value in exp_setting.items():
            row[key] = value
        results.append(row)
        
    result_df = pd.DataFrame(results)
    
    # List of parameters to plot (includes experimental settings and constant parameters).
    exp_params = ['person_percent', 'prob_living_in_same_place', 'no_gender_assign', 'assign_loc_prob', 'ent_num_used', 'max_facts_used']
    
    for param in exp_params:
        if param in result_df.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(result_df[param], result_df['qt1'], label='max chain_len', color='red')
            plt.xlabel(param)
            plt.ylabel('Chain Length Metrics')
            plt.title(f'Chain Length Metrics vs {param}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    return result_df
    
    

if __name__ == "__main__":
    plotter = StoryGraphPlotter()
    
    # # Plot single-model story.
    # story1 = example_story_single_model()
    # print("Plotting single-model story...")
    # plotter.plot_story(story1, save_path="single_model_story.png")
    
    # # Plot multiple-model story.
    # story2 = example_story_multiple_models()
    # print("Plotting multiple-model story...")
    # plotter.plot_story(story2, save_path="multiple_model_story.png")

    path_res = '/home/anirban/projects/benchmarkStories/benchmark_builders/stories/test_stories_bench_ASP_genders2000k.csv'
    plot_experimental_results(path_res)