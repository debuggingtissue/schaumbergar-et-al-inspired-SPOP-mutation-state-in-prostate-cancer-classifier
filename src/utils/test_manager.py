from fastai.vision import *

class TestManager:

    def __init__(self, test_data_path, ensembles, transforms):
        self.test_data_path
        self.ensembles = ensembles

    def get_predictions(self):
        for ensemble, model in self.ensembles.iteritems():
            print(ensemble, model)
            learner = load_learner(model, test=ImageList.from_folder(self.test_data_path))
            predictions, y = learner.get_preds(ds_type=DatasetType.Test)
            predictions[:5]

            print("\n \n")





