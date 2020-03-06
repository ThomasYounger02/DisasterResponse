# DisasterResponse
Build a model for an API that classifies disaster messages on the disaster data from Figure Eight

# Libraries
- numpy
- pandas
- matplotlib
- sqlite3
- sqlalchemy
- sys
- re
- nltk
- sklearn

# Motivation
Classify the different information correctly, and let the corresponding disposal personnel deal with their respective responsible parts. Help with disaster response.

# Files
- app：the templates files and HTML files
- data: the original data and data preprocess file
- models: the model training file

# How to run?
- git clone the whole file
- open terminal
- preprocess the data: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- train the model: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- run the app: pyhon app/run.py

# Notes
## the best performance of the model(RandomForestClassifier)
the average f1_score of the original model is 0.45.
the average f1_score of the original model with GridSearchCV is XXX.
the average f1_score of the original model with GridSearchCV and feature union is XXX.
