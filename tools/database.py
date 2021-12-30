import numpy as np
import pandas as pd


class Database(object):
    def __init__(self):
        self.algorithms = ['AB', 'ET', 'RF', 'DT', 'GB', 'SVM']
        self.database = self.import_data()
        self.dataset_names = np.unique(self.database['AB'].dataset)
        self.per_dataset_accuracies = None
        self.ranked_datasets = None
        self.metafeatures = None

    def import_data(self):
        out = {}
        algorithms = self.algorithms
        for algorithm in algorithms:
            out[algorithm] = pd.read_csv(
                'PerformanceData/{0}_results_total.csv'.format(algorithm), low_memory=False
            )
            out['fanova_{}'.format(algorithm)] = pd.read_csv(
                'PerformanceData/{0}_fANOVA_results.csv'.format(algorithm), low_memory=False
            )
        out['metafeatures'] = pd.read_csv('PerformanceData/metafeatures.csv')
        return out

    def get_per_dataset_accuracies(self, by='CV_accuracy', agg_func=max):
        if self.per_dataset_accuracies is not None:
            return self.per_dataset_accuracies
        else:
            datasets = self.dataset_names
            algorithms = self.algorithms
            database = self.database
            out = pd.DataFrame({'dataset': datasets})
            for algorithm in algorithms:
                out[algorithm] = self.score_per_dataset(
                    database[algorithm], datasets, by=by, agg_func=agg_func
                )

            out = self.row_mean_imputation(out)
            self.per_dataset_accuracies = out
            return out

    def get_above_median_performers(self):
        combined = self.get_per_dataset_accuracies()
        algorithms = self.algorithms
        median_accuracies = combined.median(axis=1)

        out = dict()
        out['dataset'] = combined.dataset
        for i in algorithms:
            out[i] = (combined.loc[:, i] >= median_accuracies) * 1

        out = pd.DataFrame(out)

        out = out.melt(id_vars=['dataset'],
                       var_name='classifier',
                       value_name='label')
        out = out.sort_values(['dataset'])
        return out

    def get_metafeatures(self):
        if self.metafeatures is not None:
            return self.metafeatures
        else:
            metafeatures = self.database['metafeatures']
            metafeatures.num_imput_type = metafeatures.num_imput_type.fillna(
                'none')
            metafeatures = metafeatures.loc[~np.isin(metafeatures.num_imput_type,
                                                     ['median', 'mode']), :]
            self.metafeatures = metafeatures
            return metafeatures

    def get_ranked_datasets(self, verbose=False):
        if self.ranked_datasets is not None:
            return self.ranked_datasets
        else:
            combined_data = self.get_per_dataset_accuracies()
            out = pd.DataFrame({'dataset': self.dataset_names})
            ranked = combined_data.rank(
                axis=1, method='average', ascending=False)
            normal_case = np.std(np.arange(1, 7, 1), ddof=1)
            non_ties = ranked.std(axis=1) == normal_case
            ties = ~non_ties
            resolve_ties = self.custom_rank(
                combined_data.loc[ties, :], verbose)
            ranked_merged = out.merge(
                ranked.loc[non_ties, :], left_index=True, right_index=True)
            self.ranked_datasets = pd.concat(
                [resolve_ties, ranked_merged], axis=0)
            return self.ranked_datasets

    def get_best_per_dataset(self):
        ranked_data = self.get_ranked_datasets()
        types = np.array(self.algorithms)
        out = pd.DataFrame({'dataset': ranked_data.dataset})
        out['best_type'] = types[np.where(ranked_data.iloc[:, 1:] == 1)[1]]
        return out

    def get_rank_frequencies(self):
        ranked_data = self.get_ranked_datasets()
        return ranked_data.iloc[:, 1:].apply(pd.value_counts).fillna(0)

    def ranker_per_row(self, row_values, dataset_name):
        names = np.array(row_values.index)
        values = np.array(row_values.values)
        uniq_values = np.unique(values)
        freq_values = self.freq(uniq_values, values)

        ranks = np.arange(6, 0, -1)  # ranking from 6,5,...,1
        final_rank = np.zeros_like(ranks)

        for i, un_val in enumerate(uniq_values):
            selected = values == un_val
            name = names[selected]
            n = freq_values[i]
            if n == 1:
                final_rank[selected] = ranks[:n]
            else:
                final_rank[selected] = self.compare_algorithms(
                    name, ranks[:n], dataset_name)
            ranks = ranks[n:]
        return final_rank

    def compare_algorithms(self, algorithm_names, ranks, dataset_name):
        database = self.database

        out = dict()
        for algorithm in algorithm_names:
            data = database[algorithm]
            selected = data.dataset == dataset_name
            filtered_data = data.loc[selected, :]
            out[algorithm] = self.metrics_per_dataset(filtered_data)

        metrics = pd.DataFrame(out)

        # ranking is different for maximizable and minimizable metrics
        nr_min_metric = 8
        min_metric_ranks = metrics.iloc[:nr_min_metric].rank(axis=1,
                                                             method='average',
                                                             ascending=True)
        max_metric_ranks = metrics.iloc[nr_min_metric:].rank(axis=1,
                                                             method='average',
                                                             ascending=False)

        metric_ranks = pd.concat([min_metric_ranks, max_metric_ranks], axis=0)
        limit = metric_ranks.shape[0]
        nr_criteria = 2
        avg_ranks = metric_ranks.iloc[:nr_criteria, :].mean(axis=0)
        rank_freq = self.freq(avg_ranks)

        while any(rank_freq > 1) and nr_criteria != limit:
            nr_criteria += 1
            avg_ranks = metric_ranks.iloc[:nr_criteria, :].mean(axis=0)
            rank_freq = self.freq(avg_ranks)

        ranks = self.arg_sort(avg_ranks.values) + min(ranks)
        return ranks

    def custom_rank(self, data, verbose):
        ranked_data = data.copy()
        n = data.shape[0]
        data.index = np.arange(n)
        for i in range(n):
            if verbose:
                print(data.dataset[i])
            ranked_data.iloc[i, 1:] = self.ranker_per_row(
                data.iloc[i, 1:], data.dataset[i])
        return ranked_data

    @staticmethod
    def merge(dataset1, dataset2, on='dataset'):
        return dataset1.merge(dataset2, on)

    @staticmethod
    def arg_sort(input_vec):
        sorted_vec = sorted(input_vec)
        rank = np.arange(len(input_vec))
        out = []
        for i in rank:
            selected = sorted_vec == input_vec[i]
            out.append(int(rank[selected]))
        return out

    @staticmethod
    def freq(a_list, b_list=None):
        if b_list is None:
            b_list = a_list.copy()

        out = []
        for a in a_list:
            element_count = sum(np.array(b_list) == a)
            out.append(element_count)
        out = np.array(out)
        return out

    @staticmethod
    def score_per_dataset(data, datasets, by, agg_func):
        out = []
        for dat in datasets:
            selected = data.dataset == dat
            if np.any(selected):
                out.append(agg_func(data.loc[selected, by]))
            else:
                out.append(np.nan)
        return out

    @staticmethod
    def row_mean_imputation(combined_data):
        contains_nan = combined_data.isna().sum()
        contains_nan = list(contains_nan.index[contains_nan != 0])
        out = combined_data.copy()
        for col in contains_nan:
            selected = out.loc[:, col].isna()
            filtered = out.loc[selected, :]
            minn = filtered.iloc[:, 1:].min(axis=1)
            out.loc[selected, col] = minn - 0.1
        return out

    @staticmethod
    def metrics_per_dataset(data):
        cv_accuracy = data.CV_accuracy
        which_max = cv_accuracy == max(cv_accuracy)
        std_max_cv_accuracy = data.Std_accuracy[which_max].mean()
        time_max_cv_accuracy = data.Mean_Train_time[which_max].mean()
        std_time_max_cv_accuracy = data.Std_Train_time[which_max].mean()
        cv_f1 = data.CV_f1_score[which_max].mean()
        cv_recall = data.CV_recall[which_max].mean()
        cv_precision = data.CV_precision[which_max].mean()
        cv_auc = data.CV_auc[which_max].mean()
        overfitting = (cv_accuracy[which_max] <
                       data.CV_accuracy_train[which_max]) * 1

        median_accuracy = cv_accuracy.median()
        std_accuracy = cv_accuracy.std()
        mean_time = data.Mean_Train_time.mean()
        std_time = data.Mean_Train_time.std()

        return [std_max_cv_accuracy, time_max_cv_accuracy,
                std_time_max_cv_accuracy, std_accuracy, overfitting.mean(),
                std_accuracy, std_time, mean_time,
                cv_f1, cv_recall, cv_precision, cv_auc,
                median_accuracy]
