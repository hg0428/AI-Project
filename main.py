from nn import NeuralNetwork, decode
from pickle import dump, load as pload
from atexit import register
from sty import fg, rs
from os import listdir
from os.path import isfile, join
from json import load


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
    dump(ai, open(join("models", saveFile), "wb"))


def loadTrainingData(x):
    with open(f"training_data/{x}.json") as f:
        data = load(f)
    return list(data.keys()), list(data.values())


while True:
    print(f'Avialiable models: {", ".join(models)}')
    saveFile = input("Which model should be loaded? ")
    if saveFile in models:
        try:
            ai = pload(open(join("models", saveFile), "rb"))
            bpc = ai.bits_per_character
            mil = int(ai.input_length / bpc)
            mol = int(ai.input_length / bpc)
            t = "NN"
            ai.type = "NN"
            print(f'Loaded {t} "{saveFile}" {mil}:{mol}@{bpc} = {(mol*bpc)*(mil*bpc)}')
            break
        except Exception as e:
            print(
                "Error loading save file:",
                dir(e),
                e.__cause__,
                e.args,
                e.__traceback__,
                e,
            )

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
            print("\nNN is a basic Neural Network.")
            t = input("Enter nn or cancel: ").lower()
            if t == "nn":
                ai = NeuralNetwork(mil * bpc, mol * bpc, bpc)
            else:
                continue
            print("Model created.")
            save()
            break

# ai.setInOut(mil * bpc, mol * bpc)

# ai.bits_per_character = bpc


def train(amt=100, file="basic"):
    if amt <= 0:
        return
    print(f'Training with {amt} iterations on "{file}" dataset...')
    try:
        ai.train(*loadTrainingData(file), amt)
        print("Training Complete")
    finally:
        save()


register(save)

# train()
decode(ai.think([1, 0, 1, 0, 1]), bpc)
# ai.train(
#     [[1, 1, 1, 1]],
#     [[0, 0, 0, 0]],
#     2
# )

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
