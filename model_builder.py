from os.path import isfile


class AbstractModelBuilder(object):
    def __init__(self, weights_path = None):
        self.weights_path = weights_path

    def getModel(self):
        weights_path = self.weights_path
        model = self.buildModel()

        if weights_path and isfile(weights_path):
            try:
                model.load_weights(weights_path)
            except Exception as e:
                print(e)

        return model


    def buildModel(self):
        raise NotImplementedError("You need to implement your own model.")
