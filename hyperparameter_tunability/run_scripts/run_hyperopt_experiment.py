import pandas as pd
import os
import sys
import json
dir = os.getcwd()
sys.path.append(dir)
from tools import hpi_verification

if __name__ == '__main__':
    # get all datasets in the folder.
    path = "./datasets/"
    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    all_datasets = len(all_files)

    ranked_hpi_datasets = {}
    avg_losses_dataset = {}
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

        hpiverfication = hpi_verification.HPIVerification(
            n_iter_opt=3,
            X=X,
            y=y,
            algorithm="random_forest",
            )

        hp_trials_dict = hpiverfication.hyper_opt_run()

        avg_losses_dataset[file] = hpiverfication.hpi_avg_lossess_dataset(hp_trials_dict)

    with open("avg_losses_dataset.json", 'w') as f:
        json.dump(avg_losses_dataset, f)

    avg_losses = hpiverfication.hpi_avg_lossess(avg_losses_dataset, all_files)

    with open("avg_losses.json", 'w') as f:
        json.dump(avg_losses, f)

    ranked_hpis, hpi_names = hpiverfication.rank_hpi(avg_losses)
    hpiverfication.plot(ranked_hpis, hpi_names)
    print("done!")
