AIUkraine 2018

October 13, 2018

# LINGUISTICS IN NLP: WHY SO COMPLEX?

## Mariana Romanyshyn

Text simplification

1. syntactic
1. lecsical
1. explanation generation

IBM content clarifier

complex word identification - CWI

data: Newsela, simple english wiki

traditional ML > DL

F-measure is not enough

## Features and problems: 

1. Frequency: Word form freq. vs. word freq., inflectional morphology
	- solution: lemmatizing
2. word length: derivational morphology
	- morphological analyzer
3. subword features: ngrams, cplx words have rare letter combinations
4. phonetic features: 
	- cplx words have higher C-V ratio
	- number of C, V, etc.
5. semantic features: 
	- number of senses in WordNet, 
	- lex-semant. relations (# hypernym, meronym, etc.)
6. Psycholinguistic features: MRC Psycholi... db
	- concreteness
	- imageability
	- familiarity
	- age of acquisition

## How to rank replacements

- simpler?
- not too simple?
	- polysemy?
	- freq?
- grammatically correct?
	- correct noun form
	- article
	- degrees of comparison
	- other inflections
	- verb governing with prepositions, adverbial particles

## Lang. models

- statistical
	- Markov assumption
	- smoothing techniques
	- don't generalize
	- don't capture long-range dependencies
	- long ngrams are expensive
- neural
	- RNN
	- generalize too much
	- long time to train
	- more expensive
	- not too much improvement, but SOTA


## when ML doesn't outperform

- more data
- simplify model
- read arxive
- study problem more

PPDB - paraphrase DB

	
# Synthetic Data and Generative Models in Deep Learning

Sergey Nikolenko, Neuromation, PDMI RAS 


Much cheeper then manual labelling
Virtually unexplored before 2016

Synthetic data for indoor navigation: SunCG

- city scenes
- humans
- MINOS: 3d environment
- House3D for embodied QA
- NLP for navigation
- CV for baby monitoring
- medical imaging
- gaze estimation
- hand movement recognition

	
# Target Leakage in Machine Learning

Yuriy Guts, DataRobot 

Contamination of training data with test data

Leads to ov erly optimistic expectation about model performnce in production.

random k-fold cross-validation

leakage can occure during the project lifecycle

## Data collection

- feature is a target in different units or formatting

## Data preparation

- model is sensitive to the feature scaling
- scaling sholud be applied AFTER splitting to train and test sets
- save mean and std from train set and use on predictions
- text: learn vocab, DTM from TRAINING set, not from whole dataset
- categorical: counting frequencies, or otherwise encoding on whole dataset

## Feature engineering

- more photos than patients. patients cross-used in train and test.
- cold start problem
- leakage in oversampling, augmentation. Partition first, then oversample or augment.
- time series: use out-of time validation/backtest.


## Training and tuning

- using the same CV split for multiple tasks: feature selection, hyperparam tuning, model selection. Treat them as a separate tasks.
- model stacking on in-sample prediction: first-level model should have the same CV-split

## Leakage in competitive ML

- removing user id != anonymization
- removing column names != anonymization
- target can be recovered from the metadata (file date creation) or external database
- overrepresented minority class in pairwise data enables reverse engineering


## Prevention measurements

- split immediately and do not use test set until testing model
- have dictionary and understanding of each column, unusual values (negative prices), outliers
- for each feature: will I have the value for this column in production? what values can it have?
- preprocessing params figured on training set and frozen
- check feature importance and prediction explanation: do top features make sense? 



# Recent Advances in Deep Learning (remote)

Ruslan Salakhutdinov, Carnegie Mellon University 


- Multimodal learning
- reasoning, attention, memory,
- NLU
- deep RL

Gated attention mechanism

Multi-hot architecture

graph-convolution networks (GCN)

relational GCN

graph propagation / graph convolution

RL for financial markets, chatbots

Neural map: location-avare memory

Deep SLAM


# Analysing Microscopy Images with Deep Learning

Dmytro Fishman, University of Tartu

Eroom's law - how many drugs can be found per year per mln $

perkin elmer harmony

	
# Billions-scale recommendations

Ivan Lobov, Criteo 

Skip-gram model for market basket

QR-decomposition for map-reduce on Spark
 
Spark on SVD

