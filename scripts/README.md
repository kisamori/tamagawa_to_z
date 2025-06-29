# Scripts Documentation

This directory contains scripts for executing a series of analyses to discover and evaluate archaeological site candidates from place name data.

## Workflow Overview

This project consists of four main steps. Each script functions like a specialist responsible for a specific role in each step.

1.  **Preparation & Dictionary Building**
    - Creates foundational data and dictionaries for interpreting place names during analysis.
2.  **Candidate Exploration**
    - Cross-references geospatial data with place names to explore potential archaeological sites.
3.  **Candidate Evaluation & Analysis**
    - Uses AI and machine learning to score discovered candidates and rank them by promise.
4.  **Workflow Automation & Optimization**
    - Orchestrates the entire analysis flow and automatically explores optimal analysis conditions.

---

## 1. Preparation & Dictionary Building

### `run_root_extraction.py`

#### Role
**【Word Dictionary Creator】**
Identifies words (roots) related to specific categories (water, terrain, etc.) such as "river" and "waterside" from place names, and creates/updates word lists that form the foundation of analysis.

#### Main Functions
- Collects place names from OSM (OpenStreetMap) data.
- Uses AI (LLM) to analyze collected place names and discover new root candidates.
- Automatically or manually adds/updates new roots to existing root dictionaries (`data/dict/*.csv`).
- Also has functionality to integrate category-specific root dictionaries into `all_roots.csv`.

#### Usage
- **Basic execution (extract water-related roots from Acre region)**
  ```bash
  python scripts/run_root_extraction.py --region acre --visualize
  ```
- **Integrate category-specific CSVs to create `all_roots.csv`**
  ```bash
  python scripts/run_root_extraction.py --create-all-roots
  ```
- **Main arguments**
  - `--region`: Specify target region (`acre`, `marajo`, etc.).
  - `--bbox`: Manually specify target BBox (coordinate range).
  - `--pbf-path`: Specify the OSM PBF file path to use.
  - `--output-dir`: Specify directory to output results.
  - `--sample-size`: Specify number of place name samples for LLM analysis to reduce costs.
  - `--visualize`: Save visualization images such as maps generated during processing.
  - `--create-all-roots`: Skip other processing and merge category-specific CSVs into `all_roots.csv`.

### `run_split.py`

#### Role
**【Data Organization Specialist】**
Divides known archaeological site data into groups such as "training," "validation," and "test" so that machine learning models can learn and evaluate fairly and accurately.

#### Main Functions
- Divides archaeological site data based on discovery year and geographical location.
  - **Train:** Data for model training
  - **Validation:** Data for evaluating model performance during training
  - **Test-time:** Temporally newer data (for evaluating performance on future data)
  - **Test-region:** Data from geographically different regions (for evaluating generalization performance on unknown regions)
- Outputs divided data as separate GIS files (`.gpkg`).

#### Usage
- **Basic execution**
  ```bash
  python scripts/run_split.py --config configs/dataset_split.yaml --sites data/known/known_acre.kmz --output data/known/split
  ```
- **Main arguments**
  - `--config`: Path to configuration file (`.yaml`) defining division rules.
  - `--sites`: Path to archaeological site data to be divided (`.kmz`, `.csv`, `.gpkg`).
  - `--output`: Path to directory for outputting divided files.
  - `--dry-run`: Display only statistical information of division results without actually outputting files.

---

## 2. Candidate Exploration

### `run_site_identification.py`

#### Role
**【Main Candidate Search Specialist】**
Searches for "**places that don't have rivers now but have 'river' in their names**." Such places have a high possibility of being traces of ancient rivers (paleochannels) and become archaeological site candidates.

#### Main Functions
- Comprehensively analyzes place name data, current river data (HydroRIVERS), and historical water occurrence probability data (GSW).
- Scores and lists candidate sites based on distance from rivers and water occurrence frequency.
- Core analysis pipeline that is the target of Optuna optimization.

#### Usage
- **Basic execution (Acre region)**
  ```bash
  python scripts/run_site_identification.py --region acre --visualize
  ```
- **Execution with specified parameters**
  ```bash
  python scripts/run_site_identification.py --region acre --dist-threshold 2.5 --occ-threshold 15.0
  ```
- **Main arguments**
  - `--region`: Specify target region (`acre`, `marajo`, etc.).
  - `--rivers-path`, `--gsw-path`, `--pbf-path`: Individually specify data paths.
  - `--output-path`: Specify output destination for final candidate site list (`.csv`).
  - `--dist-threshold`: Distance threshold (km) for "farther than this from rivers."
  - `--occ-threshold`: Water occurrence frequency threshold (%) for "water probability below this."
  - `--visualize`: Save candidate site distribution maps and other visualizations as images.

### `run_analyze_site.py`

#### Role
**【Known Site Analyst】**
Analyzes in detail how place names are distributed in patterns (distance and direction) around already discovered archaeological sites. This serves as a standard for judging whether candidate site patterns are "site-like."

#### Main Functions
- Extracts place names around known archaeological sites from OSM.
- Converts extracted place names to polar coordinates (distance and angle) centered on sites.
- Calculates distances between place names and nearest rivers.
- Outputs analysis results to CSV files and optionally performs visualization.

#### Usage
- **Basic execution**
  ```bash
  python scripts/run_analyze_site.py --region acre --radius 5.0 --visualize
  ```
- **Main arguments**
  - `--region`: Specify analysis target region (`acre`, `marajo`, etc.).
  - `--radius`: Specify radius (km) for searching place names from each site.
  - `--visualize`: Save analysis results (place name distribution maps, polar coordinate plots, etc.) as images.
  - `--similarity-analysis`: Continue with similarity analysis using extracted data.

---

## 3. Candidate Evaluation & Analysis

### `run_similarity_analysis.py`

#### Role
**【AI Appraiser (Scoring)】**
Uses machine learning to **score** how similar discovered candidates are to "place name patterns of known archaeological sites" and creates rankings in order of promise. Additionally, AI explains the reasons "why they are similar."

#### Main Functions
- Generates features (distance, density, direction, etc.) from place name distributions of known sites.
- Builds machine learning models (kNN, clustering, etc.) and scores similarity between candidates and known sites.
- Ranks candidates by similarity and outputs results in CSV and KMZ formats.
- Uses OpenAI's LLM to automatically generate textual explanations of similarity rationale for each candidate.

#### Usage
- **Basic execution (create candidate rankings)**
  ```bash
  python scripts/run_similarity_analysis.py --region acre --mode candidate_ranking
  ```
- **Main arguments**
  - `--region`: Specify analysis target region (`acre`, `marajo`, etc.).
  - `--mode`: Choose `candidate_ranking` (evaluate candidates) or `similarity_only` (evaluate only similarity between known sites).
  - `--output-dir`: Specify directory to output reports and ranking results.
  - `--config`: Specify configuration file (`.yaml`) defining data paths for each region.

### `run_inspector.py`

#### Role
**【AI Auditor】**
Checks overall analysis results and **proposes improvements** such as "Is this analysis working well?" and "Which parameters should be adjusted to improve accuracy?"

#### Main Functions
- Compares candidate data with known archaeological site data and calculates evaluation metrics such as Recall and mAP (mean Average Precision).
- Generates improvement proposals such as parameter adjustments based on analysis results.
- Outputs analysis reports (Markdown format) and improvement plans (YAML format).

#### Usage
- **Basic execution**
  ```bash
  python scripts/run_inspector.py --candidates data/output/candidates/paleochannel_candidates.csv --known data/known/known_acre.kmz
  ```
- **Main arguments**
  - `--candidates`: Path to candidate data for evaluation (`.csv`).
  - `--known`: Path to known archaeological site data as comparison standard (`.kmz`, `.gpkg`, etc.).
  - `--output`: Specify directory to output reports and plan files.

### `run_researcher.py`

#### Role
**【AI Researcher】**
A consultant that further explores the auditor's reports and **creates research plans** with more specific questions like "Which improvement proposal is most effective?" and "What should we try next?"

#### Main Functions
- Receives output from `run_inspector.py` (analysis reports and improvement plans) as input.
- Uses AI (LLM) to conduct deeper analysis and prioritize multiple improvement proposals.
- Outputs research reports and more detailed improvement plans.

#### Usage
- **Basic execution (automatically loads latest Inspector report)**
  ```bash
  python scripts/run_researcher.py
  ```
- **Main arguments**
  - `--artefacts`: Path to directory containing Inspector output. If not specified, the latest one is automatically selected.
  - `--output`: Specify directory to output research reports and other files.

---

## 4. Workflow Automation & Optimization

### `run_optuna.py`

#### Role
**【Optimization Specialist】**
Automatically tries hundreds of analysis conditions (parameters) such as "How many km away from rivers should we search?" and finds the **"golden parameters"** that produce the best results.

#### Main Functions
- Repeatedly executes the analysis pipeline of `run_site_identification.py` while changing parameters.
- Uses the optimization library Optuna to search for parameter combinations that yield the best scores.
- Found optimal parameters are saved as JSON files and used by `run_best_params.py`.

#### Usage
- **Basic execution**
  ```bash
  python scripts/run_optuna.py --region acre --trials 50 --sites data/known/split/val.gpkg
  ```
- **Main arguments**
  - `--region`: Specify region for optimization target (`acre`, `marajo`, etc.).
  - `--trials`: Specify number of optimization trials.
  - `--sites`: Path to validation site data (`.gpkg`, etc.) serving as evaluation standard.
  - `--timeout`: Specify maximum execution time (seconds) per trial.
  - `--resume`: Resume interrupted optimization.

### `run_best_params.py`

#### Role
**【Final Executor】**
Uses the "golden parameters" found by the optimization specialist (`run_optuna.py`) to **execute final analysis under optimal conditions** and evaluate/visualize results.

#### Main Functions
- Loads optimal parameters (JSON file) output by Optuna.
- Uses those parameters to identify and score archaeological site candidates.
- Calculates scores against evaluation datasets (Validation/Test) and evaluates performance.
- Optionally provides detailed visualization of each analysis step and final candidate distribution.

#### Usage
- **Basic execution**
  ```bash
  python scripts/run_best_params.py --region acre --params data/output/optuna/best_params.json --sites data/known/known_acre.kmz --visualize
  ```
- **Main arguments**
  - `--region`: Specify target region (`acre`, `marajo`, etc.).
  - `--params`: JSON file path of best parameters output by Optuna.
  - `--sites`: Path to archaeological site data used for evaluation.
  - `--output`: Specify directory to output evaluation results and other files.
  - `--visualize`: Detailed visualization of final results and analysis process.

### `run_harmonizer.py`

#### Role
**【Project Manager】**
A coordinator that **executes multiple scripts (dictionary creation, candidate search, etc.) in the proper order**. Orchestrates the entire project workflow.

#### Main Functions
- Internally calls `run_root_extraction.py` and `run_site_identification.py`.
- The `--mode` option allows selection of executing both tasks or just one of them.

#### Usage
- **Execute both tasks**
  ```bash
  python scripts/run_harmonizer.py --mode both
  ```
- **Execute dictionary management only**
  ```bash
  python scripts/run_harmonizer.py --mode root-extraction
  ```
- **Execute site identification only**
  ```bash
  python scripts/run_harmonizer.py --mode site-identification
  ```
- **Main arguments**
  - `--mode`: Select execution mode from `both`, `root-extraction`, `site-identification`.