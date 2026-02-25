- `llm/`: dataset construction code and outputs

  Results
  - `multi_target_candidate/`: multi-target candidate extraction results
  - `multi_target_selection/`: multi-target selection results
  - `single_target_selection/`: single-target selection results
  - `prompt/`: prompts for building multi-/single-target datasets

  Process
  1. `generate_query.py`: generate queries for multi-target candidate extraction
  2. `extract_mt_candidates_by_*`: scripts to extract multi-target candidates
  3. `calculate_mt_candidates_confidence.py`: evaluate multi-target candidates
  4. `select_mt_by_threshold.py`: select multi-targets by threshold
  5. `make_single_target.py`: generate new single-targets from multi-targets

- `MultiTargetDataset/`
  - `datasets.py`: dataset class for loading the dataset