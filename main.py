from nn import *
import numpy as np
import pickle
from layered import *
import atexit
from sty import fg, rs
from os import listdir
from os.path import isfile, join
import json


# TODO: Publish as a lib.
# TODO: Improve training proccess
# TODO: reset AI with improved training & longer character length.
# TODO: cannot decrease setInOut values, training breaks with increase.
# Maybe try out a more complex nn
# Add confidence and fix DLMA.

# Let nn's see the previous nn's output.

ai = None
mil = 20  # max input length use 30 for ai2
mol = 20  # max output length
bpc = 8  # bits per character use 32 for ai2

models = [f for f in listdir("models") if isfile(join("models", f))]


saveFile = "ai2"  # ai2 uses input/output length 30, and 32 bpc


def save():
    print("Saved!")
    pickle.dump(ai, open(join("models", saveFile), "wb"))


def loadTrainingData(x):
    with open(f"training_data/{x}.json") as f:
        data = json.load(f)
    return list(data.keys()), list(data.values())


while True:
    print(f'Avialiable models: {", ".join(models)}')
    saveFile = input("Which model should be loaded? ")
    if saveFile in models:
        try:
            ai = pickle.load(open(join("models", saveFile), "rb"))
            bpc = ai.bytes_per_character
            mil = int(ai.max_input_length / bpc)
            mol = int(ai.max_input_length / bpc)
            t = "DLMA" if type(ai) == DeepLearningModelAdvanced else "DLM"
            print(f'Loaded {t} "{saveFile}" {mil}:{mol}@{bpc} = {(mol*bpc)*(mil*bpc)}')
            break
        except Exception as e:
            print("Error loading save file:", dir(e), e.__cause__, e.args, e.__traceback__, e)
    
            
    else:
        x = input("Would you like to create a new model? ").lower()
        if x.startswith("y"):
            try:
                mil = int(input("Max input length? "))
            except:
                continue
            try:
                mol = int(input("Max output length? "))
            except:
                continue
            try:
                bpc = int(input("Bits per character? "))
            except:
                continue
            print(
                "\nDLM is a simple system composed of a NN for each output bit.\nDLM Advanced is a variation of DLM where each NN receives the output of the preceding NN as input.\n"
            )
            t = input("Enter DLM, DLMA or cancel: ").lower()
            if t == "dlm":
                ai = DeepLearningModel(mil * bpc, mol * bpc, 0, save, bpc)
            elif t == "dlma":
                ai = DeepLearningModelAdvanced(mil * bpc, mol * bpc, 0, save, bpc)
            else:
                continue
            print("Model created.")
            save()
            break

# ai.setInOut(mil * bpc, mol * bpc)

# ai.bytes_per_character = bpc

def train(amt=100, file="basic"):
    if amt <= 0:
        return
    print(f'Training with {amt} iterations on "{file}" dataset...')
    try:
        ai.train(*loadTrainingData(file), amt)
        print("Training Complete")
    finally:
        save()


atexit.register(save)

# train()
decode(ai.think([1, 0, 1, 0, 1]), bpc)


while True:
    inp = input(fg(70, 70, 255) + "You: " + fg(150, 180, 255))
    print(rs.all, end="")
    if inp.startswith("$train"):
        x = inp.split(" ")[1:]
        if len(x) < 1 or x[0] == "":
            amt = 100
        else:
            amt = int(x[0])
        if len(x) < 2 or x[1] == "":
            f = "basic"
        else:
            f = x[1]
        train(amt, f)
    else:
        print(
            fg(255, 70, 70) + "AI:" + fg(255, 150, 150),
            decode(ai.think(inp), bpc),
            rs.all,
        )
