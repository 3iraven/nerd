import spacy
import json
import random

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.initialize()
    for itn in range(iterations):
        print("Training starting iteration", itn)
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = spacy.training.Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
        print(losses)
    return nlp

#This function is called by the optimizer to train the model

TRAIN_DATA = load_data("data/train_data.json")
nlp = train_spacy(TRAIN_DATA, 30)
nlp.to_disk("rad_ner_models")
