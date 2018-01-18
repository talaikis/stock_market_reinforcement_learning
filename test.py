from sys import argv
import codecs
from os.path import join, dirname
from keras.models import model_from_json
from numpy import where

from environment import get_data


BASE_DIR = dirname(__file__)


def load_model(model_filename):
    json_file = open(join(BASE_DIR, "models", model_filename + ".json"), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(join(BASE_DIR, "models", model_filename + ".h5"))
    print("Model loaded.")
    
    # evaluate loaded model on test data
    loaded_model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])

    return loaded_model


def test(loaded_model, X, Y):
    score = loaded_model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


if __name__ == "__main__":
    portfolio_filename = argv[1]
    model_filename = argv[2]

    instruments = {}
    f = codecs.open(portfolio_filename, "r", "utf-8")

    for line in f:
        if line.strip() != "":
            tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
            instruments[tokens[0]] = tokens[1]
    f.close()

    model = load_model(model_filename=model_filename)

    env = MarketEnv(target_symbols=list(instruments.keys()), input_symbols = [], 
        start_date="2015-08-26", 
        end_date="2016-09-26", 
        sudden_death=-1.0):

    data = env.get_data(symbol=symbol)
    print(data)

    for item in data:
        targets = where(data.Close.pct_change() > 0, 1, 0)

        test(loaded_model=model, X=data, Y=targets)
