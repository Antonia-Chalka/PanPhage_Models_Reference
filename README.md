
# PanPhage Model Training & Testing

## Get started

Set up the following conda environments

``` bash 
conda create -n prokka_env -c conda-forge -c bioconda -c defaults prokka

conda create -n panaroo_env -c conda-forge -c bioconda -c defaults python=3.9 'panaroo>=1.3'

conda create -n  pipe_traintest_v2_env -c conda-forge pandas numpy matplotlib scikit-learn=1.5.1 networkx
```

You will need additional tools like pharokka for phage annotation and assembly (long reads) - ask Talal for more info

The code below includes R code that can be ported into python - the libraries needed are in the script.

## Training Models

**DISCLAIMER: Check all paths before you run something as things may have moved.**

**The code below may be inefficient to buggy. Again please feel free to change it in its entirety.** 

### Bacterial Data

1. Put all your bacterial assemblies in a folder (`./data/0.bacterial_assemblies`)

2. Annotate with prokka

``` bash
conda activate prokka_env

for file in ./data/0.bacterial_assemblies/*.fasta; do 
    prokka --prefix "${filename}" --force --proteins ./data/0.ref/ecoli_trusted_uniprot.fasta --centre X --compliant --outdir ./data/1.bacterial_annotations/"${filename}" ${file} 
done
```

3. Create Bacterial Pangenome

``` bash
conda activate panaroo_env

panaroo -o ./data/2.bacterial_pangenome/ --clean-mode moderate -t 10 --remove-invalid-genes --merge_paralogs -i ./data/1.bacterial_annotations/*/*.gff

```

### Phage Data


1. Generate Phage Annotations (PARTIAL)

The phages were assembled and annotated by Talal so you'll have to ask him for the specific code. I also used the code below to get & rename all the gff files from pharokka into 1 folder for the pangenome generation:


``` bash
find ./data/0.phage_pharokka -path './BAD' -prune -o -type f -name '*.gff' ! -name '*aragorn*' ! -name '*minced*' ! -name 'trnascan*' -print | xargs -I {} cp {} ./data/1.phage_annotations

cd ./data/1.phage_annotations

rename 's/^.{5}_//' *.gff
```

2. Generate Phage Pangenome

``` bash
conda activate panaroo_env

panaroo -i ./data/1.phage_annotations -o ./data/2.phage_pangenome --clean-mode moderate -t 10 --remove-invalid-genes --merge_paralogs
```

### Create Training Data

I wrote the code below in R but you can port it into python. What you basically need to do is:

1. Load both pangenome files and the interaction data

2. Append bacteria/phage to the feature names so you can tell where they came from

3. Match the bacterial and phage pangenomes to the scores to create the training data.

``` R
# R CODE
library(tidyverse)
library(data.table)
set.seed(100)

# Load Bacteria Pangenome & pivot
panaroo_bacteria <- read.table("./data/2.bacterial_pangenome/gene_presence_absence.Rtab", header=TRUE) %>%
  pivot_longer(cols = -Gene, names_to = "Isolate", values_to = "Value") %>%
  pivot_wider(names_from = Gene, values_from = Value)

## Append '_bacteria' to gene columns
colnames(panaroo_bacteria) <- ifelse(
  colnames(panaroo_bacteria) == "Isolate",
  "Isolate",            
  paste0(colnames(panaroo_bacteria), "_bacteria") 
)

# Load Phage Genomes & pivot
panaroo_phage <- read.table("./data/2.phage_pangenome/gene_presence_absence.Rtab", header=TRUE) %>%
  pivot_longer(cols = -Gene, names_to = "Phage", values_to = "Value") %>%
  pivot_wider(names_from = Gene, values_from = Value)

## Append '_phage' to gene columns
colnames(panaroo_phage) <- ifelse(
  colnames(panaroo_phage) == "Phage",
  "Phage",            
  paste0(colnames(panaroo_phage), "_phage") 
)

# Load Bacteria x Phage Interaction Scores and combine into a dataframe
phage_files <- list.files("./data/0.interaction_scores", pattern = "\\.tsv$", full.names = TRUE)
phage_data <- lapply(phage_files, function(file) {  read.table(file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)})

## Process interaction data to filter low quality bacterial sequences (after chatting to Marianne) and transform into long format
interaction_data <- reduce(phage_data, function(x, y) {  merge(x, y, by = "Isolate", all = TRUE)})%>%
  filter(!Isolate %in% c("CAN130", "CAN198", "CAN99", "HB20", "HU33", "HU34", "HU36", "HU38", "HU48", "HU49", "HU51", "HU56", "HU96" ))

interaction_data_long <- interaction_data %>% pivot_longer(  cols = -Isolate, names_to = "Phage",   values_to = "Score" )

# Combine interaction, bacteria and phage pangenome. Remove phages that are not in the phage pangenome or should not be included in training, and create an index based on the bacteria x phage combo

input_data <- left_join(interaction_data_long, panaroo_bacteria, by="Isolate") %>%
  left_join(panaroo_phage, by="Phage") %>%
  filter(!Phage %in% c("A01","E","F1","GEO","P","TB54"))%>%
  mutate(Isolate_Phage = paste0(Isolate, "_", Phage)) %>%
  column_to_rownames(var="Isolate_Phage") %>%
  select(-Phage,-Isolate) 
 
input_data <- input_data[, sapply(input_data, function(col) length(unique(col)) > 1)]

# Normalise scores so that the maximum is 100 (again feel free to change)
input_data <- input_data %>% 
  mutate(Score=if_else(Score >100, 100, Score))

write.table(input_data, "./data/3.trainingdata.tsv",sep="\t")
```

### Train Model

**See `code/phage_train.py` for the code.** Again feel free to change as you see fit. The code has been adapted from a python notebook and changed into a script but I have not tested that extensively so there may be some bugs.

See the comments for more detail on how models are generated. 

The code will output the following:

* `best_xgb_model.pkl`: The model file saved via pickle.

* `feature_reduction.png`: The model accuracy (higher is better) as you reduce features

* `feature_importances.csv`: Important features w metrics

* `feature_phage.txt` & `feature_bacteria.txt`: The important features split into the phage/bacteria (ie a filtered version of `feature_importances.csv`)

* `predictions.csv`: Self explanatory, used to generate figures and calculate RMSE. Also shows the train/test split.

When I create a new model I tend to increment the folder name (i.e. the next model would be places in `4.4.panphage_model_out_v4`) and make a note. Here is info on the previous versions:

* V1 is an early version where I didn't keep track of the feature names so I had to rerun it.

* V2 is the new version but that had some bad data that Marianne identified.

* V3 is the current version.

### Generate Figures

The R code below is used to generate the PanPhage Quadrant plots and calculate metrics. I usually only run parts of this code so the v3 folder may not have everything in it. Again, feel free to import this to python:

``` R
# R Code
library(tidyverse)
library(data.table)
library(irr)
library(plotly)
set.seed(100)

predictions <- read.table(sep=",",file="./data/4.3.panphage_model_out_v3/predictions.csv", header=TRUE)%>%
  separate(X, into = c("Isolate", "Phage"), sep = "_") %>%
  mutate(Predicted_Classification=ifelse(predicted>60, "Infection","No Infection"),
         Observed_Classification=ifelse(observed>60, "Infection","No Infection"),
         Accuracy=ifelse(Predicted_Classification==Observed_Classification, "TRUE","FALSE"))

# Calculate model stats
stats <- predictions %>% group_by(Phage) %>%
  summarise(rmse = sqrt(mean((observed-predicted)**2)),# Accuracy
            accuracy = mean(Predicted_Classification == Observed_Classification),
            
            # False Positives and False Negatives
            false_positives = sum(Predicted_Classification == "Infection" & Observed_Classification == "No Infection"),
            false_negatives = sum(Predicted_Classification == "No Infection" & Observed_Classification == "Infection"),
            true_positives=sum(Predicted_Classification == "Infection" & Observed_Classification == "Infection"),
            true_negatives=sum(Predicted_Classification == "No Infection" & Observed_Classification == "No Infection"),
            
            # Precision, Recall, F1 Score
            precision = sum(Predicted_Classification == "Infection" & Observed_Classification == "Infection") / sum(Predicted_Classification == "Infection"),
            recall = sum(Predicted_Classification == "Infection"& Observed_Classification == "Infection") / sum(Observed_Classification == "Infection"),
            f1 = 2 * (precision * recall) / (precision + recall),
            
            # Kappa (Cohen's Kappa)
            kappa =kappa2(cbind(Predicted_Classification, Observed_Classification))$value ,
            
            .groups = "drop")
write.csv(stats, file="./data/4.3.panphage_model_out_v3/1.model_stats.csv", row.names = FALSE)

# Create quadrant prediction plot for everything, showing test/train split
panphage<-ggplot(data=predictions, aes(x=predicted,y=observed, colour=Phage,shape=dataset )) +
  geom_point() +
  geom_vline(xintercept=70)+
  geom_hline( yintercept=70) +
  theme_bw() +
  labs(title = "Predicted vs Observed Interaction Scores for Phage", 
       x="Predicted",y="Observed", shape="Dataset"
       )+
  annotate("rect", xmin = 70, xmax = Inf, ymin = 70, ymax = Inf, fill= "deepskyblue",alpha = .2)  + 
  annotate("rect", xmin = -Inf, xmax = 70, ymin = -Inf, ymax = 70 , fill= "deepskyblue",alpha = .2) +
  annotate("rect", xmin = 70, xmax = Inf, ymin = -Inf,ymax = 70,  fill= "red",alpha = .2) +
  annotate("rect", xmin = -Inf, xmax = 70, ymin = 70,ymax = Inf,  fill= "red",alpha = .2) +
  coord_cartesian(xlim = c(0,100), ylim = c(0,100))+
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))+
  geom_abline(intercept = 0,slope = 1)
panphage
ggsave(filename = "./data/4.3.panphage_model_out_v3/2.panphagemodel.svg", device="svg", width = 8, height = 6, dpi = 300)

## Save interactive version of above plot
htmlwidgets::saveWidget(ggplotly(panphage), "./data/4.3.panphage_model_out_v3/2.panphagemodel.html")


# Create quadrant plot but for individual phages (used it to visually compare interaction with single phage model approach)
phage_groups <- predictions %>% group_split(Phage)

## Create a list of plots, one for each phage
plots <- lapply(phage_groups, function(group) {
  ggplot(data=group, aes(x=predicted,y=observed, shape=dataset)) +
    geom_point() +
    geom_vline(xintercept=70)+
    geom_hline( yintercept=70) +
    theme_bw() +
    labs(title = paste("Predicted vs Observed Interaction Scores for Phage ",  unique(group$Phage)),
         x="Predicted",y="Observed", shape="Dataset"
    )+
    annotate("rect", xmin = 70, xmax = Inf, ymin = 70, ymax = Inf, fill= "green",alpha = .2)  + 
    annotate("rect", xmin = -Inf, xmax = 70, ymin = -Inf, ymax = 70 , fill= "green",alpha = .2) +
    annotate("rect", xmin = 70, xmax = Inf, ymin = -Inf,ymax = 70,  fill= "red",alpha = .2) +
    annotate("rect", xmin = -Inf, xmax = 70, ymin = 70,ymax = Inf,  fill= "red",alpha = .2) +
    coord_cartesian(xlim = c(0,100), ylim = c(0,100))+
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0))+
    geom_abline(intercept = 0,slope = 1)
})
plots
## Save each plot to a separate file
for (i in seq_along(plots)) {
  ggsave(
    filename = paste0("./data/4.3.panphage_model_out_v3/3.single_phage_", unique(phage_groups[[i]]$Phage), ".svg"),
    plot = plots[[i]],
    width = 6, height = 4, device="svg"
  )
  htmlwidgets::saveWidget(ggplotly(plots[[i]]), file=paste0("./data/4.3.panphage_model_out_v3/3.single_phage_", unique(phage_groups[[i]]$Phage), ".html"))
}
```

## Test Models

When testing new data, I tend to name the folder based on year_month. Feel free to use a different naming scheme, but just keep it documented. Some folders may contain additional files (eg folders named old, which ususally conitain predictions from older models). 

* 2025_april: V2 predictions

* 2025_march: V2 predictions in `old` folder, v3 predictions appear as `5.predictions_fixedmodel.csv`

* 2025_may: V3 predictions

### Test New Bacterial Isolates

1. Collect assemblies into 1 folder. May seem redudant but it's important to doublecheck the names are correct etc.

2. Annotate (REPLACE \<test_folder\> with your name)

``` bash
conda activate prokka_env

for file in ./data/5.testing/<test_folder>/0.assemblies/*.fasta; do 
    prokka --prefix "${filename}" --force --proteins ./data/0.ref/ecoli_trusted_uniprot.fasta --centre X --compliant --outdir ./data/5.testing/<test_folder>/1.annotations/"${filename}" ${file} 
done
```

3. Integrate with existing Pangenome. Instead of generating a new pangenome we use the existing bacteria pangenome and add our bacterial sequences. This will generate a 'new' pangenome that, unfortunately, may have some different names but we will match these in the next step.

``` bash
conda activate panaroo_env

# Hard-link )ie make a shortcut) of annotaion (*.gff) files
find ./data/5.testing/<test_folder> -path '*/*.gff' -exec sh -c 'ln -s $(realpath "{}") ./data/5.testing/<test_folder>/2.pangenome/0.panaroo_in/$(basename "{}")' \;

# Integrate each annotaion into the reference bacterial pangenome
for file in ./data/5.testing/<test_folder>/2.pangenome/0.panaroo_in/*.gff; 
do 
    filename=$(basename "$file" .gff)
    panaroo-integrate -d ./data/2.bacterial_pangenome/ -i $file -t 8 -o ./data/5.testing/<test_folder>/2.pangenome/1.panaroo_merge_out/"${filename}"
done
```

4. Match Pangenome Names: This is where it gets a bit tricky. We have 2 pangenomes, 1 reference, and one new one that has our isolate. The names of the gene clusters are not identical across our two pangenomes, but the gene content is (with exception of the genes of the bacteria we are testing), therefore what we have to do is look at the gene content of each cluster and match the names that way. I have made a python script `panaroo_merge_process.py` that does as such. See below for examples on how to run it:

``` bash
conda activate pipe_test_v2_env

python ./code/panaroo_merge_process.py \
    --gene_data_file_ref  ./data/2.bacterial_pangenome/gene_data.csv \
    --gml_file_ref ./data/2.bacterial_pangenome/final_graph.gml \
    --model_features_file ./data/4.3.panphage_model_out_v3/feature_bacteria.txt \
    --test_dirs ./data/5.testing/<test_folder>/2.pangenome/1.panaroo_merge_out \
    --out_dir ./data/5.testing/<test_folder>/2.pangenome/2.panaroo_match/
```

5. Predict

See `phage_test.py`. Again this is code from a python notebook that I'm turning into a script. The code will generate the testing data by combining the bacterial gene clusters with each of the phage ones and then run it through our model for testing.


### Test New Phage Isolates

Not yet performed but will follow prcedure above (i.e. assembly -> annotation -> panaroo_interate -> match pangenome names -> create predictions).

