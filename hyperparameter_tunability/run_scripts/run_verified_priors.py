import pandas as pd
import os
import sys
import json
import openml
dir = os.getcwd()
# dir+="/hyperparameter_tunability"
sys.path.append(dir)
from hyperparameter_tunability.tools import hpi_verification_priors

if __name__ == '__main__':
    # get all datasets in the folder.
    path = "hyperparameter_tunability/datasets/"
    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    all_datasets = len(all_files)

    uniform_losses_dataset = {}
    modified_losses_dataset = {}

    # for each datset in the loop, run optimization proccess.
    for index, file in enumerate(all_files):
        print('Dataset {}({}) out of {} \n'.format(index + 1, file, all_datasets), flush=True)

        file_path = path+file

        df = pd.read_csv(file_path, index_col=None, header=0)

        df.dropna(inplace=True)
        # excluding the response variable
        X = df.iloc[:, :-1]

        # selecting the response variable
        y = df.iloc[:, -1]
        cls = "AdaBoost"
        hpiverfication_priors = hpi_verification_priors.HPIVerification_Priors(
            n_iter_opt=10,
            X=X,
            y=y,
            algorithm = cls,
            )

        uniform_best_loss, modified_best_loss = hpiverfication_priors.opt_run()

        uniform_losses_dataset[file] = uniform_best_loss
        modified_losses_dataset[file] = modified_best_loss

    with open(f"uniform_losses_datasets_{cls}.json", 'w') as f:
        json.dump(uniform_losses_dataset, f)

    with open(f"modified_losses_datasets_{cls}.json", 'w') as f:
        json.dump(modified_losses_dataset, f)

    # avg_losses = hpiverfication_priors.hpi_avg_lossess(best_losses_dataset, all_files)

    # with open("avg_losses.json", 'w') as f:
    #     json.dump(avg_losses, f)

    # ranked_hpis, hpi_names = hpiverfication.rank_hpi(avg_losses)
    # hpiverfication.plot(ranked_hpis, hpi_names)
    # print("done!")
