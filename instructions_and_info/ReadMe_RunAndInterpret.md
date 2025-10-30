# Running the Code and Interpreting Results

This repository accompanies the **2025 NeurIPS paper**  
**_“When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning.”_**  
Please cite this work if you use this code.

---

## Overview

Using this code, you can generate many **random stories** from a specified set of **world rules**.

- Each story is a set of **binary relationships** between entities (which can be visualized as a **graph**, called the *story graph*).
- Each such binary relationship between entities is called a **story fact**.

The generated stories have two key properties:

1. **Constraint Satisfaction** — They do not violate any constraints in the world rules.  
   For example, if you generate stories using the *family-based* world rules **NoRA** or **NoRA 1.1**, the story will never contain both  the facts
   `father_of(Ben, Susan)` and `husband_of(Ben, Susan)` since the NoRA rules state that a parent cannot also be a spouse.
2. **At Least One Entailed Fact** — Every story includes at least one *entailed fact*—a fact not explicitly given in the story but derivable from the world rules and story facts.

---

## How to Run the Code

The main entry point is:

```
story_builders/BenchmarkDatasetBuilderASP.py
```

In the `main()` section of this script, there are **three example setups** for generating stories with different sets of world rules.

To run:

```bash
cd story_builders
python BenchmarkDatasetBuilderASP.py
```

Comment out all but one of the examples before running.

---

## Output Description

The output is a **CSV file** saved to your chosen directory.  
Each row corresponds to a **problem instance** derived from one entailed fact.

### Columns in the Output CSV

- **`edge_types`**: The relation types for the story facts.  
- **`story_edges`**: The entity pairs for the story facts.  
  → You can zip `edge_types` and `story_edges` to reconstruct the full story facts.  
- **`query_edge`**: The pair of entities (source, target) under query.  
- **`query_label`**: A list of all relationships that hold between the source and target.  
  It is guaranteed that at least one relationship `rel*` in `query_label` satisfies `rel*(source, target)` as an *entailed fact*.
- **`query_relation`**: The most difficult relationship between source and target to infer from the story.
- **`derivation_chain`**: A proof showing how the `query_relation` is derived from story facts and world rules.
- **`derivations_for_other_implied_relationships`**: If more than one relationship between the source and target is entailed (i.e., not directly given as a story fact), those derivations appear here.

These rows represent problem instances for  **downstream reasoning tasks**, where models learn to infer the `query_label` from  
`edge_types`, `story_edges`, and `query_edge`, effectively having to induce the world rules in the process.

### Difficulty Metrics

Each problem instance/row is annotated with multiple difficulty metrics:

- **`BL`**
- **`OPEC`**
- **`ReasoningDepth`**
- **`ReasoningWidth`**

These metrics quantify different aspects of reasoning difficulty needed to infer the query labels between source and target from the story. See paper for more info

---

## Configuration and World Rules

When running:

```bash
python BenchmarkDatasetBuilderASP.py
```

the user specifies how many problem instances to generate.  
The code automatically determines how many stories are required to meet this goal.

A **crucial input file** is the **configuration file** passed to `BenchmarkDatasetBuilderASP`.  
It specifies the **`world_rules_dir`**, which contains all the world rules used to generate stories.

The directory structure is:

```
world_rules/
    ├── HetioNet/
    ├── NoRA/
    └── NoRA1.1/
```

Each `world_rules_dir` may contain one or more `.txt` files listing the world rules.  
The configuration file also specifies:

- **`world_obj_class`**:  
  The main Python class responsible for generating stories for a given set of world rules.  
  - *NoRA* and *NoRA 1.1* use the same class.
  - *HetioNet* uses a different `world_obj_class`.
- **`Othr Hyperparameters`**:  
  In *NoRA 1.1*, person_percent_range is a hyperparameter. Here entities include both persons and places. This hyperparameter controls what percentage of entities are persons.

Several example config files are provided in:

```
story_builders/configs/
    ├── config_HetioNet.json
    ├── config_NoRA.json
    └── config_NoRA1.1.json
```

---

## Ambiguous Story Facts

To generate **ambiguous story facts**, set `max_num_refinements` in the config file to a non-zero value.  
Higher values introduce more ambiguity. Ambiguous story generation has been tested **only for NoRA**.

For each generated problem instance with ambiguity, the following columns are relevant:

- **`branch_results`**: Indicates which refinements (branches) lead to contradictions and which lead to unique sets of facts (explicit + entailed).  
- **`branch_outcomes`**: Groups refinements that share the same proof of the query relation.  
- **`fact_choice_branches`**: Shows how each ambiguous fact is resolved for each branch/refinement index (0, 1, …).  
- **`branch_results_for_other_implied_relationships`**: Same as `branch_outcomes`, but for other entailed relationships.

---


---
## Video Links

- Walkthrough on YouTube: https://www.youtube.com/watch?v=Aj_3kiWTrbs
---
---
## Credits

This code uses the **ASP compiler** developed by  
👉 [https://github.com/potassco](https://github.com/potassco)

---

