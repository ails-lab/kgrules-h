# KGRules-H
Repository for Python implementations of the algorithms outlined in 
[Searching for explanations of black-box classifiers in the space of semantic 
queries](http://www.semantic-web-journal.net/content/searching-explanations-black-box-classifiers-space-queries) 
by Jason Liartis, Edmund Dervakos, Orfeas Menis-Mastromichalakis, Alexandros Chortaras and Giorgos Stamou, under review 
for the [Special Issue on The Role of Ontologies and Knowledge in Explainable 
AI](http://www.semantic-web-journal.net/blog/call-papers-special-issue-role-ontologies-and-knowledge-explainable-ai) 
of the Semantic Web Journal.

## Usage
```
python3 kgrules_h.py --dataset {mnist, clevrhans, mushrooms} --merge-operation {greedy-mathcing, qlcs} --ontology-fname /path/to/ontology --positives-fname /path/to/positive/examples --output /path/to/write/output/queries
```

Add parameter ```threshold``` to run the KGrules-HT variation:
```
python3 kgrules_h.py --dataset {mnist, clevrhans, mushrooms} --merge-operation {greedy-mathcing, qlcs} --ontology-fname /path/to/ontology --positives-fname /path/to/positive/examples --output /path/to/write/output/queries --threshold 20
```
