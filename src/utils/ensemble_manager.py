from utils import model_trainer


def generate_ensembles(ensemble_count, model_count_in_each_ensemble, data_manager):
    # data_sets, output_classes_count, dataset_sizes, device
    ensemble_dictionary = {}
    for ensemble_index in range(ensemble_count):
        ensemble_dictionary[f"ensemble_{ensemble_index}"] = []
        for model_index in range(model_count_in_each_ensemble):
            model_in_ensemble_save_name = model_trainer.train_model_in_ensamble(ensemble_index, model_index, data_manager)
            models_saved = ensemble_dictionary[f"ensemble_{ensemble_index}"]
            models_saved.append(model_in_ensemble_save_name)
    return ensemble_dictionary
