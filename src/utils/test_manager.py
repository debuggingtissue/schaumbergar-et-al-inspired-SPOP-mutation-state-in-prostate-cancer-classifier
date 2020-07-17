from fastai.vision import *

class TestManager:

    def __init__(self, test_data_path, ensembles):
        self.test_data_path = test_data_path
        self.ensembles = ensembles

    def get_predictions(self):
        print(self.ensembles)
        for ensemble, models_in_ensemble in self.ensembles.items():
            for model in models_in_ensemble:

                print(ensemble, model)
                print(self.test_data_path)
                learner = load_learner(model, test=ImageList.from_folder(self.test_data_path))
                predictions, y = learner.get_preds(ds_type=DatasetType.Test)
                print(predictions[:5])

                print("\n \n")





