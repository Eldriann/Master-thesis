#!/usr/bin/env python
"""
Script used to give a console to the rasa model
"""
# pylint: disable=bare-except

import json
from os import listdir
from os.path import isdir, join
from rasa.nlu.model import Interpreter

MODEL_PATH = './models/'
dirs = [f for f in listdir(MODEL_PATH) if isdir(join(MODEL_PATH, f))]
dirs.sort(reverse=True)
model = join(MODEL_PATH, dirs[0])
print('Using model %s', model)
interpreter = Interpreter.load(model)
print('Model %s ready', model)

try:
    query = input("> ")
    while query is not None and len(query):
        interpretation = interpreter.parse(query)
        print(json.dumps(interpretation, indent=4))
        query = input("> ")
except:
    print('Stopping...')
print('Stopped')
