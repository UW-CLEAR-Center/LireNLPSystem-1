---
# LireNLPSystem package documentation
---
## Table of Contents

* [Installation](#Installation)
    + [Troubleshoot: Configuring rJava on Mac](#Troubleshoot)
* [Overview](#Overview)
* [Preprocessing](#Preprocessing)
    + [SectionSegmentation](#SectionSegmentation)
* [Featurization](#Featurization)
    + [Rules](#Rules)
        + [RuleBasedNLP_JavaSentence](#RuleBasedNLP_JavaSentence)
        + [RuleBasedNLP](#RuleBasedNLP)
    + [N-Grams](#Ngrams)
        + [CreateTextFeatures](#CreateTextFeatures)
    + [Document Embeddings](#DocumentEmbeddings)
    + [Controlled Vocabulary](#ControlledVocabulary)
* [Train/Test](#TrainTest)
    + [runMlMethod](#runMlMethod)
* [Apply Weights](#Weights)
    + [MachineLearningNLP](#MachineLearningNLP)
* [Advanced usage](#AdvancedUsage)
    + [Modifying source Java code](#ModifyJavaCode)
    + [Updating R package](#UpdateRPackage)
    + [User-supplied feature weights](#UserMLWeights)
* [GitHub folder structure](#GitHubStructure)

<a name="Installation"></a>

## Installation

To install, run the following code:
```{r}
install.packages("devtools")
library("devtools")
devtools::install_github("chethanjjj/LireNLPSystem")
library(LireNLPSystem)
```

For the install, you may need to use the updated version of tidyselect (1.1.0) and recipes (0.1.15) if you're coming across installation errors.

For the demo, you should also have the following installed:
```{r}
install.packages("readr")
install.packages("dplyr")
install.packages("tidyr")
install.packages("caret")

library(LireNLPSystem)
library(readr)
library(dplyr)
library(tidyr)
library(caret)

```
<a name="Troubleshoot"></a>

### Troubleshoot: Configuring rJava on Mac
Installation should be straight-forward; however if you are using MacOS the dependency package rJava could throw errors. This is because R needs to know where Java is on the machine. In that case open Terminal and do the following which is adapted from [this source:](https://github.com/snowflakedb/dplyr-snowflakedb/wiki/Configuring-R-rJava-RJDBC-on-Mac-OS-X)

First, install Xcode (if not already installed):
```
xcode-select --install
```

Then install Java8 (newer versions may exist but this should be sufficient):
```
curl -#ROL -b "oraclelicense=a"  http://download.oracle.com/otn-pub/java/jdk/8u60-b27/jdk-8u60-macosx-x64.dmg
open jdk-8u60-macosx-x64.dmg
```

Now, we need to tell R to use the installed Java 8 as ```JAVA_HOME```
```
R CMD javareconf
```

Finally, install ```rJava``` from source and compile it against the Java 8 JDK.
```
R --quiet -e 'install.packages("rJava", type="source", repos="http://cran.us.r-project.org")'
```

If you run into an error try:
```
curl -#ROL https://www.rforge.net/rJava/snapshot/rJava_0.9-8.tar.gz
R CMD INSTALL rJava_0.9-8.tar.gz
```

You may also try searching for "configuring rJava on Mac" on the internet.

<a name="Overview"></a>

## Overview

The purpose of the pipeline is to classify your reports for the 26 findings described in [Tan et. al](https://www.academicradiology.org/article/S1076-6332(18)30121-1/fulltext), as well as 10 additional ```rare and serious``` findings. Please see File ```lire_finding_matrix.xlsx``` file, Sheet ```finding_matrix``` Column ```finding_string``` for the complete list. The pipeline will *preprocess* and *featurize* your reports. However, after these steps there are two possible paths to go down. You can either *train/test* your own model with it's own feature weights or *apply our weights*.

![](images/Pipeline.png "Pipeline")

There are six main R functions in this package:  

* [SectionSegmentation](#SectionSegmentation)
* [RuleBasedNLP_JavaSentence](#RuleBasedNLP_JavaSentence)
* [RuleBasedNLP](#RuleBasedNLP)
* [CreateTextFeatures](#CreateTextFeatures)
* [MachineLearningNLP](#MachineLearningNLP)
* [runMlMethod](#runMlMethod)

These functions together creates the workflow for the NLP system for the LIRE project. 

In this tutorial we will through each part of the pipeline.

<a name="Preprocessing"></a>

## Preprocessing

The purpose of this section is isolate the different sections in a radiology report, so that we can use the Finding and Impression sections for the rest of the pipeline.

<a name="SectionSegmentation"></a>

### SectionSegmentation

This function takes in a data frame, and segments the ```imagereporttext``` column into the following sections: ```History```, ```Exam```, ```Comparison```, ```Technique```, ```Body```, ```Impression```, ```Datetime```. Some of these sections may be empty. This code is developed and validated ONLY for the four LIRE sites:

* site = 1: Kaiser Permanente Washington (previously Group Health)
* site = 2: Kaiser Permanente Northen California
* site = 3: Henry Ford
* site = 4: Mayo Clinic

The default is ```site = 2```. If you are NOT using LIRE reports, the algorithm may be inaccurate.

Example usage:
```{r}

### This is fake data.
text.df <- data.frame(imageid = c("W231", "W2242", "W452", "5235"),
                           examID = c("631182", "1226", "2090", "1939"),
                           siteID = c(2,2,2,2),
                           imageTypeID = c(1,3,1,3),
                           imagereporttext = c("** HISTORY **: Progressive radicular symptoms for 8 weeks Comparison study: None 
                                               ** FINDINGS **: Dextroconvex scoliosis with apex at L3. 
                                               ** IMPRESSION **: Scoliosis present.",
                                               "** FINDINGS **: Disk height loss is present focally. 
                                               ** IMPRESSION **:  Mild to moderate broad-based disc bulge.",
                                               "** FINDINGS **: canal stenosis and mild narrowing of the left latera. 
                                               ** IMPRESSION **: mild bilateral foraminal narrowing.",
                                               "** FINDINGS **: Disc unremarkable. 
                                               ** IMPRESSION **:  Foraminal stenosis."))


segmented.reports <- SectionSegmentation(text.df, site = 2)
View(segmented.reports)

```

<a name="Featurization"></a>

## Featurization

The purpose of this section is to convert our free-text information into a format that our ML model can use. For this pipeline, we convert our text into rule results and n-grams.

<a name="Rules"></a>

### Rules

<a name="RuleBasedNLP_JavaSentence"></a>

#### RuleBasedNLP_JavaSentence

This function takes in a data frame, with at the minimum these columns:

* ```imageid```: Unique identifier of report
* ```bodyText```: Column of text of report body
* ```impressionText```: Column of text of report impression
* ```findings_longstring```: String of findings separated with ;, for example disc_degeneration;endplate_edema. Full finding list available in the ```lire_finding_matrix.xlsx``` file on GitHub page.

This function assumes reports are sectioned into body and impression. To use this function if only one column of report text is available, please create an extra column for impression and fill it with ```NA```. 

The function passes the data frame to the ```RuleBasedNLP.jar``` file in ```inst/java```, and outputs sentence level prediction of the following:

* Sentence: The exact sentence from report.
* Section of sentence: Which section (body or impression) sentence is from.
* regex: If a related keyword to the finding is present.
* negex: If the keyword is negated.
* keyword: The regular expression that triggered the algorithm.

Note that negex can only be 1 if regex is 1.

Example usage:
```{r}

### This is the list of LIRE findings
finding.list <- c("spondylolisthesis",
                  "annular_fissure",
                  "disc_bulge",
                  "disc_degeneration",
                  "disc_desiccation",
                  "disc_height_loss",
                  "disc_protrusion",
                  "facet_degeneration")
                      
regex.df.java <- RuleBasedNLP_JavaSentence(segmented.reports,
                               imageid = "imageid",
                               bodyText = "body",
                               impressionText = "impression",
                               findings_longstring = paste(finding.list,collapse = ";")
                               )
```

<a name="RuleBasedNLP"></a>

#### RuleBasedNLP

This function takes in the sentence-by-sentence output of the ```RuleBasedNLP_JavaSentence``` function, and aggregates over all sentences to get a report level prediction. For every finding, the logic is as follows:

* Report level prediction: See lookup table below.
* Section level prediction:  
    + 1, if at least one sentence with a non-negated keyword.
    + -1, if all keywords are negated.
    + 0, otherwise.
  
Example usage:
```{r}

rbnlp.tb <- data.frame(body = c(1,1,1,0,0,0,-1,-1,-1),
impression = c(1,0,-1,1,0,-1,1,0,-1),
rules_nlp = c(1,1,-1,1,0,-1,1,-1,-1))

View(rbnlp.tb)                       
regex.df.list <- RuleBasedNLP(regex.df.java) 

# This is the "wide" data frame to be used in the machine-learning predictions
regex.df.wide <- regex.df.list$regex.df.wide

# This is the data frame of rules NLP prediction
rules.nlp.df <- regex.df.list$rules.nlp.df

```
<a name="Ngrams"></a>

### N-Grams

<a name="CreateTextFeatures"></a>

#### CreateTextFeatures

This function takes a data frame with at the minimum these columns:

* ```imageid```: Unique identifier of report
* ```text.cols```: Columns of text of report body

It will create binary indicator of N-gram (unigrams, bigrams, trigrams) features separately for each column of text. However, to pre-process data so that it is compatible with the machine-learning feature weights, ```text.cols``` should be a vector of length 2 corresponding to the body and impression columns.

It will return a data frame based on the document-feature matrix (dfm) object: rows are reports and columns are features.

Example usage:
```{r}
unigrams = CreateTextFeatures(as.data.frame(segmented.reports),
                               id_col = "imageid",
                               text.cols = c("body","impression"),
                               n_gram_length = 1)
bigrams = CreateTextFeatures(as.data.frame(segmented.reports),
                              id_col = "imageid",
                              text.cols = c("body","impression"),
                              n_gram_length = 2)
trigrams = CreateTextFeatures(as.data.frame(segmented.reports),
                               id_col = "imageid",
                               text.cols = c("body","impression"),
                               n_gram_length = 3)
ngrams = unigrams %>%
                    inner_join(bigrams, by = "imageid") %>%
                    inner_join(trigrams, by = "imageid")

```

<a name="DocumentEmbeddings"></a>

### Document Embeddings

Unlike ngrams, where we represent reports with a one-hot encoding of their words, we instead represent with an embedding, by training a simple neural network to associate a document to it's words. In this pipeline, we have already created two 2 pretrained doc2vec models that a user can use. 1) [MIMIC](https://www.nature.com/articles/sdata201635) radiology reports and 2) other reports in the LIRE study. Check the `/python` directory for these pretrained models.

To apply do2vec models to our data, we need to combine the `body` and `impression` together and write to a file.
```{r}
segmented.reports$text = segmented.reports$text = paste0(segmented.reports$body, " " ,segmented.reports$impression)
write.csv(segmented.reports, "reports.csv")
```

We then load this report into python to run the doc2vec model
```{python}
# import libraries
import gensim
import pandas as pd
import numpy as np
from /python/documentVector import documentVectorFeatures

# load the reports
reports = pd.read_csv("reports.csv")

# load model
model = gensim.models.doc2vec.Doc2Vec.load("/python/mimicDocumentEmbedding.model")

# get the embeddings for our reports
embeddingTable = documentVectorFeatures(reports)

# write
embeddingTable.to_csv("reports.csv", index=False)
```

We then load this table back into R

```{r}
documentEmbeddings = read.csv("reports.csv")
```

<a name="ControlledVocabulary"></a>

### Controlled Vocabulary

https://github.com/vpejaver/UMLS-concept-assertion-mapper

<a name="TrainTest"></a>

## Train/Test

Now that we have our different representations of the radiology reports. We can choose with representation to go ahead and use for training. For this example, let's use the rules (report level regex and negex results) and n-garms

```{r}
# combine the rules and n-grams together
featureMatrix = rules.nlp.df %>% inner_join(ngrams, by="imageid")

# ML METHOD SETUP
MLMETHOD = "glmnet" # elastic net logistic regression
METRIC = "auc" # Can also use "f1" for optimization using F1-score
myControl = trainControl(method = "cv", 
                         number = 10,
                         search = "random",
                         verboseIter = TRUE, # Change to FALSE if don't want output training log
                         returnData = TRUE,
                         returnResamp = "final",
                         savePredictions = "final",
                         classProbs = TRUE,
                         summaryFunction = aucSummary, # custom AUC loss function instead of default misclassification error
                         selectionFunction = "best",
                         preProcOptions = c("center", "scale"),
                         predictionBounds = rep(FALSE, 2),
                         seeds = NA,
                         trim = TRUE,
                         allowParallel = TRUE)

# split data into train and test (using a random split in this example)
set.seed(1)
n = nrow(featureMatrix)
testSample = sample(1:n, 0.2*n) # 20% held out for testing
devSample = setdiff(1:n, testSample)
trainID = featureMatrix$imageid[devSample]
testID = featureMatrix$imageid[testSample]

# finding of interest
finding = "disc_height_loss"


# Remove regex features that are not for the specified finding name
colsRemove = names(featureMatrix)[which(grepl(paste("(?<!",finding,")_(r|n)egex", sep=""), names(featureMatrix), perl = TRUE))]
X = featureMatrix[, !names(featureMatrix) %in% colsRemove]

# run the model
myResult = runMLMethod(finding = finding, 
                       featureMatrix = X, 
                       outcome = outcome.df, 
                       trainID = trainID, 
                       testID = testID, 
                       metric = METRIC, 
                       mlmethod = MLMETHOD,
                       myControl = myControl,
                       "path/to/output/results")

# vi


```

<a name="runMlMethod"></a>

### runMlMethod

-INSERT INFO-

<a name="Weights"></a>

## Apply Weights

The purpose of this section is to apply weights to your feature matrix and classify each report.

<a name="MachineLearningNLP"></a>

### MachineLearningNLP

This function takes in features and outputs machine-learning NLP predictions. Three types of features are required:

* N-grams in section: This can be obtained using the ```CreateTextFeatures``` function.
* Report level regex and negex: This can be obtained using the ```RuleBasedNLP``` function (the ```.$regex.df.wide``` data frame).
* LIRE study site and imaging modality: These need to be formatted as indicator matrices (see example below).

At the minimum, report level regex/negex and N-grams in section are required. The algorithm will still run if study site and imaging modality information is not available, however the accuracy is unknown. Since this algorithm was developed specific for reports from LIRE study sites, please use it on other report types with awareness.

Example usage:
```{r}
### This is the same text.dfm in the demo.
### Need to make sure that the correct prefixes BODY and IMP are used!
colnames(text.dfm) <- gsub("IMPRESSION", "IMP", colnames(text.dfm))

#### This series of code transform numeric site and imageTypeID (modality) into indicator matrices
ftr <- segmented.reports %>% 
  dplyr::select(imageid, siteID, imageTypeID) %>%
  dplyr::rename(site = siteID,
                modality = imageTypeID) %>%
  dplyr::mutate(modality = ifelse(modality == 1, "XR", "MR")) %>%
  dplyr::mutate_all(as.factor)

site.and.modality <- predict(dummyVars(imageid ~ ., data = ftr), newdata = ftr) %>%
  as.data.frame() %>%
  dplyr::mutate(imageid = ftr$imageid)
colnames(site.and.modality) <- gsub("\\.", "_", colnames(site.and.modality))

text.dfm <- text.dfm %>%
  left_join(site.and.modality, by = "imageid")

### Apply machine-learning model tuned parameters
data(ml_feature_weights)

ml.nlp.df <- MachineLearningNLP(finding.list, 
                         text.dfm, 
                         regex.df.wide,
                         ml_feature_weights,
                         grouping_var = "imageid")
                            
### Combine rules and ML into single data.frame
nlp.df <- segmented.reports %>% dplyr::select(imageid, siteID, imageTypeID) %>%
  left_join(rules.nlp.df, by = "imageid") %>%
  left_join(ml.nlp.df, by = "imageid") %>%
  separate(imageid, into = c("patientID", "examID"), sep = "_")

View(nlp.df)
```

<a name="AdvancedUsage"></a>

## Advanced Usage

Read this section if you are interested in modifying the source code for your projects. [This reference](https://datawarrior.wordpress.com/2016/09/10/rjava-running-java-from-r-and-building-r-packages-wrapping-a-jar/) may be helpful.

<a name="ModifyJavaCode"></a>

### Modifying source Java code

If the source Java code needs to be modified for any reason (e.g. adding new keywords or findings), note that this needs to happen in two parts:

* [Loading Java source code](#LoadJava): The source Java code is configured as a [Maven project](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). The easiest way to change source code is to use Eclipse, but you can use any other IDE of your choice.
* [Packaging and exporting source code](#ExportJava): The modified source code needs to be packaged and exported into a JAR file so that R functions built on ```rJava``` can access it.

<a name="LoadJava"></a>

#### Loading Java source code

1. Download [Eclipse](https://www.eclipse.org/downloads/) - the IDE for JavaEE option. Also make sure the GitHub folder is downloaded (click on ```clone/download``` on project page - the green button).
2. Start Eclipse (any workspace is fine).
3. Click ```File``` --> ```Import``` --> ```Maven``` --> ```Existing Maven Projects```. 
4. Set root directory to ```~path_to/LireNLPSystem/java``` (```~path_to``` is wherever you stored the downloaded GitHub folder).
5. Verify that file ```pom.xml``` is in the project list, and load the project.

Then you should be good to go to modify source code to tailor to project needs.

<a name="ExportJava"></a>

#### Packaging and exporting source code

1. Compile code, make sure it runs.
2. In Eclipse, click ```File``` --> ```Export``` --> ```Java``` --> ```Runnable JAR file```.
    + Launch configuration: ```RuleBasedNLP```.
    + Export destination is ```~path_to/LireNLPSystem/inst/java```; save as ```RuleBasedNLP.jar```.
    + Library handling: ```Package required libraries into generated jar```.
3. Click ```Finish```.

<a name="UpdateRPackage"></a>

### Updating R package

The R package is created using [roxygen](https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html). Therefore documentation files (```.Rd``` files in the ```man``` folder) are automatically generated. You may modify following snippet for package updates:

```{r}
remove.packages("LireNLPSystem")
library("devtools")
library(roxygen2)
setwd("~path_to_github_folder")
setwd("./LireNLPSystem")
document()
setwd("..")
install("LireNLPSystem")
```

Then, follow normal Git/GitHub procedures to update the package remotely.

<a name="UserMLWeights"></a>

### User-supplied feature weights
To use user-supplied feature weights (instead of the ones tuned with LIRE training data), replace ```ml_feature_weights``` with a data frame having three columns: ```finding_name | feature_name | feature_weight```. Note:

* All unique elements in ```finding_name``` column must be consistent with elements in ```finding.list``` vector.
* All entries in ```feature_name``` must be consistent with those created with the ```CreateTextFeatures``` function (otherwise the cross-product will throw an error).


<a name="GitHubStructure"></a>

## GitHub folder structure

### Basic ```roxygen``` package components

* ```DESCRIPTION```: Description of the ```R``` package.
* ```NAMESPACE```: Generated by ```roxygen``` (do NOT edit by hand).
* ```README.md``` and ```README.html```: Wiki page for GitHub.
* ```inst/java```: Contains the ```RuleBasedNLP.jar``` and other ```.jar``` files which is read by functions written with ```rJava``` interface.
* ```man```: Contains ```.Rd``` documentation files (do NOT edit by hand).
* ```R```: Contains the described 5 ```.R``` files as well as an ```onLoad.R``` file to load the JVM.
* ```python```: Contains pre-trained MIMIC and Unannotated LIRE Document Vector models along with a python function to get the document vectors a given document using these models.

### Additional Components

* ```data``` and ```data-raw```: Contains files to load machine-learning feature weights. On package loading, file ```process_data.R``` reads in ```ml_feature_weights.csv``` and stores in ```data/ml_feature_weights.rda```. The object ```ml_feature_weights``` is automatically loaded into the ```R``` environment.
* ```documentation```: Contains  
    + ```ARAD_NLP_LIRE.pdf```: Accepted version of the paper.
    + ```lire_finding_matrix.xlsx```: Detailed information on finding string names and data dictionaries.
* ```java```: Typical [Maven project structure](http://www.java2s.com/Tutorials/Java/Maven_Tutorial/1030__Maven_Directory_Structure.htm):
    + ```pom.xml```: File that contains configuration information about the project; here the POM is rather [minimal](https://maven.apache.org/guides/introduction/introduction-to-the-pom.html#Minimal_POM).
    + ```src/main/java```: contains all the ```.java``` source files.
    + ```src/main/resources``` contains the ```context.csv``` and ```stopwords.txt``` files.
    + ```src/test/``` and ```src/bin/``` were auto-generated by Eclipse; not directly used by the project.
    + ```target``` is created by Maven and contains all the compiled classes, JAR files etc.

