# PanPhage Model Training & Testing

## Overview

PanPhage is a machine-learning framework for predicting phage–bacterium infection outcomes from pangenome data. Given a bacterial isolate and a phage isolate, the model uses the presence/absence of gene clusters in each organism's pangenome as features to predict whether infection will occur. It is designed to scale across large collections of isolates without requiring direct experimental testing of every combination.

This repository contains all code and instructions needed to train new models and test novel bacterial or phage isolates against an existing trained model.


## Citation

If you use PanPhage in your research, please cite:

[UNDER REVIEW - PLACEHOLDER]
> [Author(s)]. [Title]. *[Journal]*, [Year]. DOI: [https://doi.org/XXXXXX](https://doi.org/XXXXXX)


## Get Started

1. Download and set up [git](https://git-scm.com/) and [git lfs](https://git-lfs.com/).

2. Clone the repository:

```bash
git clone https://github.com/Antonia-Chalka/PanPhage_Models_Reference.git
```

3. Set up the following conda environments:

```bash
conda create -n prokka_env -c conda-forge -c bioconda -c defaults prokka

conda create -n panaroo_env -c conda-forge -c bioconda -c defaults python=3.9 'panaroo>=1.3'

conda create -n pipe_traintest_v2_env -c conda-forge pandas numpy matplotlib scikit-learn=1.5.1 networkx
```

> **Note:** Some steps below include R code for creating training data. This can be ported to Python — the required libraries are listed within each script.

---


## Data

### Reference File

| File | Description |
|------|-------------|
| `data/0.ref/ecoli_trusted_uniprot.fasta` | Curated *E. coli* protein sequences from UniProt, used as a reference database for Prokka annotation. |

### Bacterial Data

| File / Folder | Description |
|---------------|-------------|
| `data/0.bacterial_assemblies/` | 301 bacterial genome assemblies (FASTA format). |
| `data/1.bacterial_annotations/` | Prokka annotation outputs for each assembly. |
| `data/2.bacterial_pangenome/` | Reference bacterial pangenome generated with Panaroo. Contains `gene_presence_absence.Rtab`, `gene_data.csv`, `final_graph.gml`, and supporting files. The pangenome comprises 17,978 gene clusters (3,228 core, 186 soft-core, 2,610 shell, 11,954 cloud). |
| `data/mlst.tsv` | Multi-locus sequence type (MLST) assignments for 302 bacterial isolates. |

### Phage Data

| File / Folder | Description |
|---------------|-------------|
| `data/0.phage_annotations/` | Pharokka/Phold annotation outputs (GFF format) for 31 phages. |
| `data/2.phage_pangenomes/panaroo_default/` | Reference phage pangenome generated with Panaroo. Contains `gene_presence_absence.Rtab`, `gene_data.csv`, `final_graph.gml`, and supporting files. The pangenome comprises 1,779 gene clusters (295 shell, 1,484 cloud; no core genes identified at the 99% threshold). |

### Interaction Scores

| File / Folder | Description |
|---------------|-------------|
| `data/0.interaction_scores_ignacio/` | Per-phage CSV files (62 files, one per phage) containing bacterial isolate × phage infection scores measured at 18 h. Each file is named by phage identifier. |

### Training Data

| File | Description |
|------|-------------|
| `data/training_data.tsv` | Pre-built training dataset combining bacterial pangenome, phage pangenome, and interaction scores (9,331 rows × 18,666 features). Used directly by the model training notebook. |
| `data/feature_distances.csv` | Per-phage nearest-neighbour and mean distances in feature space, plus an isolation-by-distance metric (IMID), used for model evaluation. |

### Trained Models

| File / Folder | Description |
|---------------|-------------|
| `data/lgo_bacteria/` | 305 leave-genome-out (LGO) models trained by withholding one bacterial isolate at a time (`.pkl` format). |
| `data/lgo_phage/` | 3 leave-genome-out models trained by withholding one phage at a time. |
| `data/lgo_mlst/` | 35 leave-one-sequence-type-out (LOSTO) models, one per MLST group. |
| `data/stat_sel_features/` | Per-fold outputs from recursive feature elimination (RFE), including selected feature lists (`rfe_features.txt`, `rfecv_features.txt`), train/test splits, and cross-validation performance statistics. |

### Model Outputs

| File / Folder | Description |
|---------------|-------------|
| `data/output_panphage_Ignacio_251119/` | Prediction outputs from the full PanPhage model on the test dataset (November 2019). |
| `data/output_singlephage_Ignacio_251128/` | Prediction outputs from single-phage baseline models on the test dataset (November 2019). |


## Test Models

When testing new data, name the folder by `year_month` (e.g. `2024_03`). A different naming scheme is fine, but keep it documented.

> **Note:** Paths in the scripts below are examples — adjust them to match your data structure.

### Test New Bacterial Isolates

1. Collect assemblies into one folder. Double-check that the filenames are correct.

2. Annotate (replace `<test_folder>` with your folder name):

```bash
conda activate prokka_env

for file in ./data/5.testing/<test_folder>/0.assemblies/*.fasta; do
    filename=$(basename "$file" .fasta)
    prokka --prefix "${filename}" --force --proteins ./data/0.ref/ecoli_trusted_uniprot.fasta --centre X --compliant --outdir ./data/5.testing/<test_folder>/1.annotations/"${filename}" ${file}
done
```

3. Integrate with the existing pangenome. Rather than generating a new pangenome from scratch, we add our bacterial sequences to the existing bacterial pangenome. This produces an updated pangenome whose gene cluster names may differ slightly — these are reconciled in the next step.

```bash
conda activate panaroo_env

# Hard-link annotation (*.gff) files into the panaroo input directory
find ./data/5.testing/<test_folder> -path '*/*.gff' -exec sh -c 'ln -s $(realpath "{}") ./data/5.testing/<test_folder>/2.pangenome/0.panaroo_in/$(basename "{}")' \;

# Integrate each annotation into the reference bacterial pangenome
for file in ./data/5.testing/<test_folder>/2.pangenome/0.panaroo_in/*.gff; do
    filename=$(basename "$file" .gff)
    panaroo-integrate -d ./data/2.bacterial_pangenome/ -i $file -t 8 -o ./data/5.testing/<test_folder>/2.pangenome/1.panaroo_merge_out/"${filename}"
done
```

4. Match pangenome names. We now have two pangenomes — the reference and the updated one containing our isolate. Gene cluster names differ between them, but the gene content is equivalent (except for genes belonging to the isolate being tested). The script `panaroo_merge_process.py` matches cluster names by comparing gene content:

```bash
conda activate pipe_traintest_v2_env

python ./code/panaroo_merge_process.py \
    --gene_data_file_ref  ./data/2.bacterial_pangenome/gene_data.csv \
    --gml_file_ref ./data/2.bacterial_pangenome/final_graph.gml \
    --model_features_file ./data/4.3.panphage_model_out_v3/feature_bacteria.txt \
    --test_dirs ./data/5.testing/<test_folder>/2.pangenome/1.panaroo_merge_out \
    --out_dir ./data/5.testing/<test_folder>/2.pangenome/2.panaroo_match/
```

5. Predict. Run `phage_test.py`. The script builds the test data by combining bacterial and phage gene clusters, then passes them through the model to generate predictions.

---

### Test New Phage Isolates

Follow the same procedure as for bacterial isolates: assembly → annotation → `panaroo-integrate` → match pangenome names → generate predictions.

---

## Training Models

### Bacterial Data

1. Place all bacterial assemblies in `./data/0.bacterial_assemblies`.

2. Annotate with Prokka:

```bash
conda activate prokka_env

for file in ./data/0.bacterial_assemblies/*.fasta; do
    filename=$(basename "$file" .fasta)
    prokka --prefix "${filename}" --force --proteins ./data/0.ref/ecoli_trusted_uniprot.fasta --centre X --compliant --outdir ./data/1.bacterial_annotations/"${filename}" ${file}
done
```

3. Create the bacterial pangenome:

```bash
conda activate panaroo_env

panaroo -o ./data/2.bacterial_pangenome/ --clean-mode moderate -t 10 --remove-invalid-genes --merge_paralogs -i ./data/1.bacterial_annotations/*/*.gff
```

---

### Phage Data

1. Annotate phages using [pharokka](https://github.com/gbouras13/pharokka) and [phold](https://github.com/gbouras13/phold).

2. Generate the phage pangenome:

```bash
conda activate panaroo_env

panaroo -i ./data/1.phage_annotations -o ./data/2.phage_pangenome --clean-mode moderate -t 10 --remove-invalid-genes --merge_paralogs
```

---

### Create Training Data

The code below is written in R but can be ported to Python:

1. Load both pangenome files and the interaction data.
2. Append `_bacteria` or `_phage` to feature names so their origin is identifiable.
3. Join the bacterial and phage pangenomes against the interaction scores to create the training dataset.

```R
library(tidyverse)
library(data.table)
set.seed(100)

# Load Bacteria Pangenome & pivot
panaroo_bacteria <- read.table("./data/2.bacterial_pangenome/gene_presence_absence.Rtab", header=TRUE) %>%
  pivot_longer(cols = -Gene, names_to = "Isolate", values_to = "Value") %>%
  pivot_wider(names_from = Gene, values_from = Value)

# Append '_bacteria' to gene columns
colnames(panaroo_bacteria) <- ifelse(
  colnames(panaroo_bacteria) == "Isolate",
  "Isolate",
  paste0(colnames(panaroo_bacteria), "_bacteria")
)

# Load Phage Pangenome & pivot
panaroo_phage <- read.table("./data/2.phage_pangenome/gene_presence_absence.Rtab", header=TRUE) %>%
  pivot_longer(cols = -Gene, names_to = "Phage", values_to = "Value") %>%
  pivot_wider(names_from = Gene, values_from = Value)

# Append '_phage' to gene columns
colnames(panaroo_phage) <- ifelse(
  colnames(panaroo_phage) == "Phage",
  "Phage",
  paste0(colnames(panaroo_phage), "_phage")
)

# Load Bacteria x Phage Interaction Scores and combine into a dataframe
phage_files <- list.files("./data/0.interaction_scores", pattern = "\\.tsv$", full.names = TRUE)
phage_data <- lapply(phage_files, function(file) {
  read.table(file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
})

# Filter out low-quality bacterial sequences and reshape to long format
interaction_data <- reduce(phage_data, function(x, y) { merge(x, y, by = "Isolate", all = TRUE) }) %>%
  filter(!Isolate %in% c("CAN130", "CAN198", "CAN99", "HB20", "HU33", "HU34", "HU36",
                          "HU38", "HU48", "HU49", "HU51", "HU56", "HU96"))

interaction_data_long <- interaction_data %>%
  pivot_longer(cols = -Isolate, names_to = "Phage", values_to = "Score")

# Join interaction data with both pangenomes.
# Exclude phages not in the phage pangenome or unsuitable for training.
# Cap scores at 100 and create a combined Isolate_Phage index.
input_data <- left_join(interaction_data_long, panaroo_bacteria, by = "Isolate") %>%
  left_join(panaroo_phage, by = "Phage") %>%
  filter(!Phage %in% c("A01", "E", "F1", "GEO", "P", "TB54")) %>%
  mutate(Isolate_Phage = paste0(Isolate, "_", Phage)) %>%
  column_to_rownames(var = "Isolate_Phage") %>%
  select(-Phage, -Isolate)

# Remove columns with no variance
input_data <- input_data[, sapply(input_data, function(col) length(unique(col)) > 1)]

# Normalise scores (cap at 100)
input_data <- input_data %>%
  mutate(Score = if_else(Score > 100, 100, Score))

write.table(input_data, "./data/3.trainingdata.tsv", sep = "\t")
```

---

### Train Model

See `code/pythonnotebooks/single_panphage_model_training.ipynb` and run the code blocks in order. Comments throughout the notebook explain how models are generated.
