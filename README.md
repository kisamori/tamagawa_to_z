# Tamagawa to Z: Unraveling the Mysteries of Lost Rivers and Ancient Ruins in the Amazon with AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.4.0+-blue.svg)](https://python-poetry.org/)

## Introduction: In Search of Lost River Memories

Rivers are the cradles of civilization. However, their flow is not eternal.
Across the vast expanse of time, rivers meander and dramatically change their form. Water may have long vanished from places where civilizations once flourished.

But what if those memories have been passed down to the present day as "place names"?

Our adventure began with such a romantic hypothesis. We wondered if the names and legends of lands might be time capsules of the soul, rooted in places even after people have left and rivers have dried up.

The name of this project, "Tamagawa to Z," derives from the "Tamagawa River" in Japan, where our team members live. This river, which repeatedly flooded in the past, left the same place name "Todoroki" on both its banks. Even now, place names connect these two lands that were divided by the river and separated by time. Fascinated by the storytelling power of these place names, we decided to embark on a journey to search for the same miracle in the distant lands of the Amazon.

### Our Compass: LLM and Multilingual Toponymy

The history of the Amazon is a crossroads of diverse languages. Indigenous languages, languages of settlers. They mixed together, changed over time, and hid the true nature of the land behind a complex veil.

Our greatest weapon is the power of LLMs to break through these language barriers and the knowledge of "toponymy (the study of place names)".

We're not simply looking for words that mean "water." We had LLMs interpret countless place names scattered throughout the Amazon in multilingual contexts, comprehensively judging their etymology, semantic similarities, and historical backgrounds, thereby practicing what could be called "linguistic taxonomy." This is the work of excavating the "memories of the land" that lie dormant within words.

Through this process, we created the world's only **"Multilingual Toponymic Dictionary."** This is our unique and irreplaceable compass in our long and arduous quest.

### Route to Discovery

We chose Acre State, Brazil, full of potential for unseen discoveries, as the stage for our adventure. This land, where "new surfaces" are continuously appearing due to deforestation and development, is truly a frontier where numerous geoglyphs have been discovered in satellite images in recent years.

Our route is as follows:

1.  **Listening to Voices Sleeping in Maps:** Armed with our created "Multilingual Toponymic Dictionary," we picked up place names in Acre State from OSM (OpenStreetMap) and listed places that might be related to water.

2.  **Setting Course for the Past:** Our purpose is the exploration of "lost rivers." Therefore, we narrowed down to lands with "low wetness" that are at a certain distance from current rivers and suggest the possibility of former residential areas. To improve the accuracy of this narrowing down, we adjusted parameters while dialoguing with the LLM (O3 Pro) itself, using how well we could reproduce known ruins as an indicator. It was like having repeated discussions about unknown sea charts with an experienced and seasoned navigator called AI.

3.  **Relying on the Stars:** Eight selected candidate sites. We scored them from multiple perspectives, including feature matching combining topography and toponymy at known ruins, possibility of destruction/burial by artifacts, unexplored survey areas, topographical advantages from satellite images, and mythological traditions.

### The Promised Land: Ramal Olho D'água

After countless analyses and dialogues with AI, we finally arrived at one place.

| Site Name | Ramal Olho D'água |
| :--- | :--- |
| **Coordinates** | **-9.839247, -68.498725** |

**View on Map:**

<a href="https://www.google.com/maps?q=-9.839247,-68.498725">
  <img src="docs/images/8.gif" alt="Ramal Olho D'água" width="500">
</a>

- 🌍 [Open in Google Maps](https://www.google.com/maps?q=-9.839247,-68.498725)

This place is located in a true "blank zone," away from known archaeological sites.
Satellite data suggests an ideal environment where "micro-highlands" and "low wetlands" spread in a mosaic pattern, making it easy for ancient people to build settlements.

And when we investigated folk tales related to this land using the place name "Olho D'Água (Eye of Water)" as a clue, we found in the materials of an Indigenous protection organization in Acre State a description stating that **Indigenous people once lived around this very "Olho D'Água."**

Following the memories of place names excavated by AI, various perspectives strongly suggest that ruins lie dormant in this place.

---


## Quick Start

### Setup

**Prerequisites:**
- Python 3.10 or higher is required (does not work with 3.8 or 3.9)
- Geospatial libraries (especially pyproj, geopandas) require Python 3.10 or higher

```bash
# 1. Clone the repository
git clone https://github.com/username/tamagawa_to_z.git
cd tamagawa_to_z

# 2. Install using Poetry
poetry install
```

### 3. Place Required Data

#### Required Data Files

| File Name | Content | Source | Role |
|-----------|---------|--------|------|
| `HydroRIVERS_SA.shp` | South America river network (~95MB) | [HydroSHEDS](https://hydrosheds.org) | Distance calculation from current rivers |
| `GSW_occurrence.tif` | Water surface occurrence frequency data (1984-2021) | [GSW Portal](https://global-surface-water.appspot.com) | Water area frequency determination |
| OSM data files | Northern South America & BR regional OSM data | [Geofabrik](https://download.geofabrik.de/south-america.html) | Place name data extraction |

**Note:** Individual download required due to large file sizes.

### Manual Download Procedure

#### 1. Obtaining HydroRIVERS
```bash
# Download and extract
wget -O hydrorivers_sa.zip "https://data.hydrosheds.org/file/HydroRIVERS/SA_HydroRIVERS_v10_shp.zip"
unzip hydrorivers_sa.zip -d data/raw/hydrorivers_sa
```

#### 2. Obtaining GSW occurrence
1. Download tiles for 80°W~40°W / 0°~20°S from [GSW Portal](https://global-surface-water.appspot.com)
2. Place downloaded files in GSW_occurrence directory:
```bash
mkdir -p data/raw/GSW_occurrence
# Move downloaded tile files to this directory
mv occurrence_*.tif data/raw/GSW_occurrence/
```

#### 3. Obtaining OSM Data
```bash
# Create OSM directory
mkdir -p data/raw/osm

# Download OSM data for northern South America and various Brazilian regions
wget -O data/raw/osm/bolivia-latest.osm.pbf "https://download.geofabrik.de/south-america/bolivia-latest.osm.pbf"
wget -O data/raw/osm/peru-latest.osm.pbf "https://download.geofabrik.de/south-america/peru-latest.osm.pbf"
wget -O data/raw/osm/centro-oeste-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/centro-oeste-latest.osm.pbf"
wget -O data/raw/osm/nordeste-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/nordeste-latest.osm.pbf"
wget -O data/raw/osm/norte-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/norte-latest.osm.pbf"
wget -O data/raw/osm/sudeste-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/sudeste-latest.osm.pbf"
wget -O data/raw/osm/sul-latest.osm.pbf "https://download.geofabrik.de/south-america/brazil/sul-latest.osm.pbf"
```

#### 4. Placement Verification
```bash
ls data/raw/hydrorivers_sa/   # Confirm HydroRIVERS files exist
ls data/raw/GSW_occurrence/   # Confirm GSW occurrence tiles exist
ls data/raw/osm/      # Confirm OSM files for each region exist
```
### Running with Jupyter Notebook

```bash
# Launch Jupyter Notebook
poetry run jupyter notebook notebooks/01_harmonizer.ipynb
```

## Installation

**Note:** This project requires Python 3.10 or higher. Please use pyenv or similar to set the appropriate Python version.

### Using Poetry (Recommended)

```bash
# Clone repository
git clone https://github.com/username/tamagawa_to_z.git
cd tamagawa_to_z

# Install using Poetry
poetry install
```

### Using with Kaggle

**Note:** Please select Python 3.10 or higher runtime in Kaggle as well.

```bash
# Install using requirements.txt
pip install -r requirements.txt

# Or write at the beginning of Kaggle notebook as follows
!pip install git+https://github.com/kisamori/tamagawa_to_z.git
```

## Usage

This project explores Amazon ancient ruin candidate sites by combining AI and geospatial data.

**For detailed usage instructions, please refer to [scripts/README.md](scripts/README.md).**

> ⚠️ **Important Note:** Running all the scripts in sequence takes a very long time and reproducibility cannot be guaranteed due to LLM uncertainty. We have created a presentation notebook that loads dumped results from each pipeline stage to demonstrate the complete workflow efficiently.
>
> 📊 **Quick Demo:** For a comprehensive overview of the entire pipeline with pre-computed results, see:  
> **[notebooks/00_presentation.ipynb](notebooks/00_presentation.ipynb)**

### Basic Workflow

1. **Data Preparation** - Split known ruin data (train/validation/test)
2. **Parameter Optimization** - Optimization using Bayesian Optimization + LLM
3. **Candidate Site Prediction** - Extract candidate sites with optimal parameters
4. **Evaluation & Analysis** - Result evaluation and improvement suggestions by AI agents

### Quick Execution

```bash
# 1. Data splitting
python scripts/run_split.py --config configs/dataset_split.yaml

# 2. Parameter optimization
python scripts/run_optuna.py --config configs/optuna_run.yaml --trials 50

# 3. Run prediction (evaluation & analysis also run automatically)
python scripts/run_best_params.py --params data/output/optuna/.../best_params.json --run-analysis
```

### Analysis Workflow

For detailed analysis flow, please refer to the following Jupyter Notebook:

**📊 Main Workflow (Planned):** `notebooks/analysis_workflow.ipynb`

This notebook explains the following analysis procedures step by step:
- Building a multilingual toponymic dictionary
- Collection and preprocessing of place name data
- Extraction of candidate sites using machine learning
- Evaluation and visualization of results

## Project Structure

```
tamagawa_to_z/
├── README.md              # This file
├── LICENSE                # MIT License  
├── pyproject.toml         # Poetry configuration (Python 3.10+ required)
├── requirements.txt       # Kaggle dependencies
│
├── scripts/               # CLI execution scripts
│   ├── README.md             # Detailed usage instructions
│   ├── run_split.py          # Data splitting
│   ├── run_optuna.py         # Hyperparameter optimization
│   ├── run_best_params.py    # Best parameter execution
│   ├── run_inspector.py      # Evaluation analysis
│   └── run_researcher.py     # Improvement suggestions
│
├── configs/               # Configuration files
│   ├── optuna_space.yaml     # Optimization parameter space definition
│   └── dataset_split.yaml    # Data splitting configuration
│
├── notebooks/             # Experiments, visualization, demos
│   ├── 01_harmonizer.ipynb      # Main processing demo
│   └── 99_kaggle_demo.ipynb     # Kaggle demo
│
├── src/                   # Package main body
│   └── tamagawa_to_z/
│       ├── harmonizer/           # Multilingual toponymic analysis
│       ├── tuning/               # Hyperparameter optimization
│       ├── inspector_agent/      # Evaluation & analysis agent
│       ├── researcher_agent/     # Improvement suggestion agent
│       ├── site_analysis/        # Site analysis tools
│       └── utils/                # Utilities
│
├── tests/                 # Tests
│
└── data/               # Data (not included in Git)
    ├── raw/            # Input data
    │   ├── hydrorivers_sa/         # HydroRIVERS data
    │   │   └── HydroRIVERS_v10_sa.shp  # South America river network
    │   ├── GSW_occurrence/         # Global Surface Water data
    │   │   └── occurrence_*.tif        # Water surface occurrence frequency tiles
    │   └── osm/                    # OSM data
    │       ├── bolivia-latest.osm.pbf     # Bolivia
    │       ├── peru-latest.osm.pbf        # Peru
    │       ├── centro-oeste-latest.osm.pbf # Brazil Central-West
    │       ├── nordeste-latest.osm.pbf    # Brazil Northeast
    │       ├── norte-latest.osm.pbf       # Brazil North
    │       ├── sudeste-latest.osm.pbf     # Brazil Southeast
    │       └── sul-latest.osm.pbf         # Brazil South
    ├── known/          # Known site data
    │   └── known_acre.kmz          # Training known sites (KMZ format)
    ├── splits/         # Data splitting results
    │   ├── train.gpkg  # Training data
    │   ├── val.gpkg    # Validation data
    │   └── test.gpkg   # Test data
    └── output/         # Output results
        └── optuna/     # Optimization results
            ├── 20250623_152052/    # Timestamp directory
            │   ├── best_params.json
            │   ├── best_run_val_candidates.csv
            │   └── best_run_test_time_candidates.csv
            └── optuna.db           # Optuna SQLite DB
```

## License

This project is published under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

When citing this project, please use the following format:

```
@software{tamagawa_to_z,
  author = {tamagawa_to_z Contributors},
  title = {tamagawa_to_z: Amazon Ancient River Channel and Settlement Exploration Framework},
  year = {2025},
  url = {https://github.com/username/tamagawa_to_z}
}
```
