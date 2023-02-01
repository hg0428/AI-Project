from nn import *
import numpy as np
import pickle
from layered import *
import atexit
from sty import fg, rs


# TODO: Publish as a lib.
# TODO: Improve training proccess
# TODO: reset AI with improved training & longer character length.
# TODO: cannot decrease setInOut values, training breaks with increase.
# Maybe try out a more complex nn
# Move to ipad.


# Let nn's see the previous nn's output.

ai = None
mil = 20  # max input length use 30 for ai2
mol = 20  # max output length
bpc = 8  # bits per character use 32 for ai2
models = [
    "ai", 
    "ai2"
]  # 20, 20, 8  # 30, 30, 32
saveFile = "ai2"  # ai2 uses input/output length 30, and 32 bpc
while True:
    model = input("Which model should be loaded? ")
    if model in models:
        saveFile = model
        break
    else:
        print(f'Avialiable models: {", ".join(models)}')


def save():
    print("Saved!")
    pickle.dump(ai, open(saveFile, "wb"))


try:
    ai = pickle.load(open(saveFile, "rb"))
    mil = ai.max_input_length
    mol = ai.max_input_length
    bpc = ai.bytes_per_character
    print(f"Loaded {saveFile}.")
except:
    print("Error loading save file, creating new model...")
    ai = DeepLearningModel(mil * bpc, mol * bpc, 0, save, bpc)

# ai.setInOut(mil * bpc, mol * bpc)
# Sets the AI to have 64 inputs and 64 outputs

# ai.bytes_per_character = bpc


def train(amt=100):
    if amt <= 0:
        return
    print(f"Training with {amt} iterations...")
    try:
        ai.train(
            [
                "hello ",
                "Hello ",
                "hello",
                "Hello",
                "Hello!",
                "Who are you?",
                "hi",
                "dumb",
                "my name is",
                "no",
                "שלום" if bpc >= 16 else "peace",
            ],
            [
                "world",
                "World",
                " world",
                " World",
                "Hi!",
                "An AI!",
                "hey",
                "I dumb",
                "idc",
                "why",
                "שלום" if bpc >= 16 else "peace",
            ],
            amt,
        )
        print("Training Complete")
    finally:
        save()


atexit.register(save)

# train()
decode(ai.think([1, 0, 1, 0, 1]), bpc)


while True:
    inp = input(f"{fg(70, 70, 255)}You: {fg(150, 180, 255)}")
    print(rs.all, end="")
    if inp.startswith("$train"):
        x = inp.split(" ")
        x = 100 if len(x) != 2 or x[1] == "" else int(x[1])
        train(min(x, 99999))
    else:
        print(
            f"{fg(255, 70, 70)}AI:{fg(255, 150, 150)}",
            decode(ai.think(inp), bpc),
            rs.all,
        )
