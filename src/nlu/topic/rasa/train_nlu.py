#!/usr/bin/env python
"""
Script used to train a rasa model in python
File taken from F21CA course instructions
https://github.com/HWUConvAgentsProject/CA2020_instructions/blob/master/rasa_tutorial/nlu/train_nlu.py
"""


from rasa.nlu.model import Trainer
from rasa.nlu import config
from rasa.shared.nlu.training_data.loading import load_data

# loading training data
training_data = load_data('./data/nlu.yml')

# initialising the trainer
trainer = Trainer(config.load("config.yml"))

# training
trainer.train(training_data)

# saving the model in the specified directory
trainer.persist('./models/')
