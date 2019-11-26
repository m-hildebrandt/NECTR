import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from utils import generate_range, extract_metrics, visualize_item_embeddings
import models
import pickle
import random
from enum import Enum
import json


class Mask(Enum):
    """
    Class used to represent the various mechanisms for masking solutions while generating test samples

    RANDOM: In each solution, 50% of the configured items are randomly chosen for masking.
    ODD: In each solution, every second item configured is masked.
    ODD_IN_CATEGORY: In each solution, 50% of the configured items whose categories are known are chosen for masking.
    CUSTOM: Custom masking where the explicitly specified items are masked (e.g., in the zero-shot scenario).
    """

    RANDOM = 1
    ODD = 2
    ODD_IN_CATEGORY = 3
    CUSTOM = 4


class CrossValidation:
    """
    Class responsible for the cross-validation pipeline, i.e., splitting the dataset into training, validation, and test
    sets, performing the cross-validation, i.e., running the model for each hyper-parameter combination, choosing the
    model generating the best results on the validation set, evaluating the best model on the test set, and producing
    the final recommendation/ranking of items
    """

    def __init__(self, datahelper=None, parameters=None):
        self.parameters = parameters
        self.train, self.valid, self.test = None, None, None
        self.valid_masked, self.test_masked = None, None
        self.valid_mask, self.test_mask = None, None
        self.valid_solutions, self.test_solutions = None, None
        self.train_solutions, self.train_solutions_masked = None, None
        self.DataHelper = datahelper
        self.config = self.Config(self.DataHelper)

    class Config(object):
        """
        Class that acts as a wrapper for all configurable (hyper-)parameters
        """

        def __init__(self, datahelper=None):
            # Register to temporarily hold values of parameters
            self.temp_register = {}

            self.n_batch = 100
            self.batch_size = 100
            self.negative2positive_ratio = 1

            self.zero_shot = False
            self.cross_validation = True
            self.mask = Mask.RANDOM

            self.n_entity = datahelper.get_entity_count()
            self.n_item = datahelper.get_item_count()
            self.n_relation = datahelper.get_relation_count()
            self.n_category = datahelper.get_category_count()
            self.category2item = {k: datahelper.get_items_in_category(k) for k in range(self.n_category)}
            self.n_items_with_category = datahelper.get_items_with_category_count()

            self.n_epoch = 75  # training epochs
            self.n_epoch_max = 6
            self.early_stopping = False
            self.learning_rate = 0.01
            self.loss_type = 'l2'
            self.regularization = 0
            self.margin = 0.0001
            self.mean_loss = False
            self.adaptive_learning_rate = True  # adaptive learning rate

            self.retrain_model = False
            self.path_pretrained_ent_embs = None
            self.track_history = False

            self.hidden_size = 15  # number of hidden layers/latent dimensions
            self.hidden_sizeE = 15
            self.hidden_sizeR = 15
            self.clipE = False  # non-negative entities
            self.clipR = False  # non-negative relations
            self.clipCategory = False  # non-negative categories
            self.diagonalR = False  # relations -> diagonal core tensor
            self.identityR = False  # relations -> identity core tensor
            self.initR = None
            self.initE = None

            # Parameters used in NECTR
            self.nectr_train_tf_on_solutions = True
            self.nectr_training_mode = models.TrainingMode.SIMULTANEOUS
            self.nectr_n_hidden_layers = 2
            self.nectr_n_neurons = 15
            self.nectr_lambda_completion = 1
            self.nectr_nn_regularization_type = 'l1'
            self.nectr_nn_regularization = 1e-2
            self.nectr_learning_rate = 0.1
            self.nectr_poisson = False
            self.nectr_item_counts = False
            self.nectr_n_epoch_completion = 10  # training epochs of the completion loss alone

        def register(self, parameter):
            """Temporarily stores the value of the specified parameter in a temporary register
            :param parameter: Name of the parameter
            :return: None
            """
            self.temp_register[parameter] = getattr(self, parameter)

        def recall(self, parameter):
            """Loads the value of the specified parameter from the temporary register
            :param parameter: Name of the parameter
            :return: None
            """
            setattr(self, parameter, self.temp_register[parameter])

    def train_valid_test_split_zero_shot(self, data, test_size=0.2, n=100, non_linear=False):
        """Splits the data into training, validation, and test sets to simulate the zero-shot scenario
        :param data: Dataset to split
        :param test_size: Size of the test set as a ratio
        :param n: Number of "new" items to be used for the zero-shot scenario
        :param non_linear: Parameter that indicates that a non-linear model will be used on the data (e.g., NECTR)
        :return: None
        """

        # Split the triples into training, validation and test sets (only for the Solutions data)
        contains = data[data['r'] == 0]
        non_contains = data[data['r'] > 0]

        # Randomly choosing items i.e., not picking a specific category
        new_item_ids = random.sample(self.DataHelper.get_items(), n)
        print('Number of new items chosen for cold start: {}'.format(len(new_item_ids)))

        # A. Sample n new items
        # B. Get solutions containing new items
        # C. Split these solutions into train:test
        # D. Train: from solutions, pick those triples where t not in new items. Test: from solutions, mask new items

        new_item_solutions = contains[contains['t'].isin(new_item_ids)]['h'].unique()
        print('Number of solutions containing the new items: {}'.format(len(new_item_solutions)))

        others = contains[~contains['h'].isin(new_item_solutions)]
        new_item_triples = contains[contains['h'].isin(new_item_solutions)]
        temp = new_item_triples.sort_values(['h', 't']).reset_index(drop=True)
        temp['t'] = temp['t'].astype(str)
        # Group by the head i.e., solutions to get Solutions X Items
        solutions = temp.groupby('h')['t'].apply('\t'.join).reset_index(name='t')

        train, self.test = train_test_split(solutions, test_size=test_size, shuffle=True, random_state=42)

        # Reset the indices caused by shuffling
        train = train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)

        print('Total number of triples in the dataset: {}'.format(len(data)))
        print('Number of features in the dataset: {} triples'.format(len(non_contains)))
        print('Number of solutions in the dataset: {} triples'.format(len(contains)))

        train = self.DataHelper.sol2triple(train)
        train = train[~train['t'].isin(new_item_ids)]

        # Mask entries(i.e., only those corresponding to solutions) in the training set
        if non_linear:
            temp = train.sort_values(['h', 't']).reset_index(drop=True)
            temp['t'] = temp['t'].astype(str)
            # Group by the head i.e., solutions to get Solutions X Items
            train_solutions = temp.groupby('h')['t'].apply('\t'.join).reset_index(name='t')

            temp = train_solutions.copy()
            if self.config.nectr_item_counts:
                self.train_solutions, sol2index = self.DataHelper.get_solution_matrix(temp['h'].tolist())
                self.train_solutions_masked, _ = self.mask_test_set(temp, self.config.mask, as_mat=(self.train_solutions, sol2index))
            else:
                self.train_solutions_masked, _ = self.mask_test_set(temp, self.config.mask)
                self.train_solutions_masked = self.DataHelper.sol2mat(self.train_solutions_masked, sparse=True)
                self.train_solutions = self.DataHelper.sol2mat(temp[['h', 't']], sparse=True)

            if self.config.nectr_train_tf_on_solutions:
                self.train = pd.concat((train, others, non_contains))
            else:
                self.train = non_contains
        else:
            self.train = pd.concat((train, others, non_contains))

        print('Training set (solutions): {} triples'.format(len(self.train[self.train['r'] == 0])))
        print('Training set (features): {} triples'.format(len(self.train[self.train['r'] != 0])))
        print('Test set (solutions): {} triples'.format(len(self.DataHelper.sol2triple(self.test))))

        # Mask entries in the validation and test sets
        if non_linear and self.config.nectr_item_counts:
            self.test_solutions, sol2index = self.DataHelper.get_solution_matrix(self.test['h'].tolist())
            self.test_masked, self.test_mask = self.mask_test_set(self.test, new_item_ids, as_mat=(self.test_solutions, sol2index))
        else:
            self.test_masked, self.test_mask = self.mask_test_set(self.test, new_item_ids)

    def train_valid_test_split(self, data, valid_size=0.2, test_size=0.1, as_mat=False, non_linear=False):
        """Splits the data into training, validation, and test sets
        :param data: Dataset to split
        :param valid_size: Size of the validation set as a ratio
        :param test_size: Size of the test set as a ratio
        :param as_mat: Parameter that indicates that the training set should be in the form of a matrix rather than
        triples (e.g., used with NMF)
        :param non_linear: Parameter that indicates that a non-linear model will be used on the data (e.g., NECTR)
        :return: None
        """

        # Split the triples into training, validation and test sets (only for the Solutions data)
        contains = data[data['r'] == 0]
        non_contains = data[data['r'] > 0]

        temp = contains.sort_values(['h', 't']).reset_index(drop=True)
        temp['t'] = temp['t'].astype(str)
        # Group by the head i.e., solutions to get Solutions X Items
        solutions = temp.groupby('h')['t'].apply('\t'.join).reset_index(name='t')

        non_train_size = valid_size + test_size
        train, non_train = train_test_split(solutions, test_size=non_train_size, shuffle=True, random_state=42)
        self.valid, self.test = train_test_split(non_train, test_size=test_size / non_train_size, shuffle=True, random_state=42)

        # Reset the indices caused by shuffling
        train = train.reset_index(drop=True)
        self.valid = self.valid.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)

        if as_mat:
            self.train = self.DataHelper.sol2mat(train)
            print('Training set (solutions): {} triples'.format(len(self.DataHelper.sol2triple(train))))

        else:
            print('Total number of triples in the dataset: {}'.format(len(data)))
            print('Number of features in the dataset: {} triples'.format(len(non_contains)))
            print('Number of solutions in the dataset: {} triples'.format(len(contains)))
            # Mask entries(i.e., only those corresponding to solutions) in the training set
            if non_linear:
                temp = train.copy()
                if self.config.nectr_item_counts:
                    self.train_solutions, sol2index = self.DataHelper.get_solution_matrix(temp['h'].tolist())
                    self.train_solutions_masked, _ = self.mask_test_set(temp, self.config.mask, as_mat=(self.train_solutions, sol2index))
                else:
                    self.train_solutions_masked, _ = self.mask_test_set(temp, self.config.mask)
                    self.train_solutions_masked = self.DataHelper.sol2mat(self.train_solutions_masked, sparse=True)
                    self.train_solutions = self.DataHelper.sol2mat(temp[['h', 't']], sparse=True)

                if self.config.nectr_train_tf_on_solutions:
                    self.train = pd.concat((self.DataHelper.sol2triple(train), non_contains))
                else:
                    self.train = non_contains
            else:
                self.train = pd.concat((self.DataHelper.sol2triple(train), non_contains))

            print('Training set (solutions): {} triples'.format(len(self.train[self.train['r'] == 0])))
            print('Training set (features): {} triples'.format(len(self.train[self.train['r'] != 0])))

        print('Validation set (solutions): {} triples'.format(len(self.DataHelper.sol2triple(self.valid))))
        print('Test set (solutions): {} triples'.format(len(self.DataHelper.sol2triple(self.test))))

        # Mask entries in the validation and test sets
        if non_linear and self.config.nectr_item_counts:
            self.valid_solutions, sol2index = self.DataHelper.get_solution_matrix(self.valid['h'].tolist())
            self.valid_masked, self.valid_mask = self.mask_test_set(self.valid, self.config.mask, as_mat=(self.valid_solutions, sol2index))
            self.test_solutions, sol2index = self.DataHelper.get_solution_matrix(self.test['h'].tolist())
            self.test_masked, self.test_mask = self.mask_test_set(self.test, self.config.mask, as_mat=(self.test_solutions, sol2index))
        else:
            self.valid_masked, self.valid_mask = self.mask_test_set(self.valid, self.config.mask)
            self.test_masked, self.test_mask = self.mask_test_set(self.test, self.config.mask)

    def run_pipeline(self, model, path_to_results='', plot=False):
        """
        Responsible for running the complete cross-validation pipeline, i.e., running the model for each hyper-parameter
        combination, choosing the model generating the best results on the validation set, evaluating the best model
        on the test set, storing the results, and visualizing them.
        :param model: Specific model to use
        :param path_to_results: Path to store the resultant files (e.g., TensorFlow model, cross-validation results)
        :param plot: Parameter that indicates whether or not to generate visualizations (e.g., item embeddings)
        :return: None
        """

        results = []
        best_model = None

        for parameter in self.grid_search():
            for p_name, p_value in parameter.items():
                setattr(self.config, p_name, p_value)

            rs = model(datahelper=self.DataHelper, config=self.config, path_to_results=path_to_results)
            print('----------------------------------------------------------------------')
            print('Creating the model: ', rs.name, '...')

            print('----------------------------------------------------------------------')
            print('Splitting the dataset for cross-validation...')
            # Split data-set into training, validation and test sets
            if self.config.zero_shot:
                self.train_valid_test_split_zero_shot(self.DataHelper.get_data(), non_linear=(isinstance(rs, models.NECTRModel)))
            else:
                self.train_valid_test_split(self.DataHelper.get_data(), as_mat=(isinstance(rs, models.NMFModel)), non_linear=(isinstance(rs, models.NECTRModel)))

            print('----------------------------------------------------------------------')
            print('Training the model using parameters: ', parameter, '...')
            self.config.epoch_range = range(0, self.config.n_epoch)
            if isinstance(rs, models.NECTRModel):
                loss = rs.train(self.train, self.train_solutions_masked, self.train_solutions)
            else:
                loss = rs.train(self.train)
            print('Loss after {} epochs = {}'.format(self.config.n_epoch, loss[-1]))
            best_model = {'loss_train': loss, 'parameter': json.dumps({(k):(v.name if isinstance(v,Enum) else v) for k, v in parameter.items()}), 'rs': rs}

            if self.config.cross_validation:
                scores, _ = rs.test(self.valid_masked)
                metrics = self.evaluate(self.valid, self.valid_mask, scores)
                print('\nTesting on validation set... MRR after {} epochs = {}'.format(self.config.n_epoch, metrics[16]))

                if self.config.early_stopping:
                    self.config.retrain_model = True
                    for i in range(self.config.n_epoch_max // self.config.n_epoch - 1):
                        start_epoch = (i+1) * self.config.n_epoch
                        end_epoch = start_epoch + self.config.n_epoch
                        self.config.epoch_range = range(start_epoch, end_epoch)
                        opt_metrics = metrics
                        rs.copy()
                        print('Training continued...')
                        if isinstance(rs, models.NECTRModel):
                            loss = rs.train(self.train, self.train_solutions_masked, self.train_solutions)
                        else:
                            loss = rs.train(self.train)
                        print('Loss after {} epochs = {}'.format(end_epoch, loss[-1]))
                        scores, _ = rs.test(self.valid_masked)
                        metrics = self.evaluate(self.valid, self.valid_mask, scores)
                        print('\nTesting on validation set... MRR after {} epochs = {}'.format(end_epoch, metrics[16]))
                        if metrics[16] < opt_metrics[16]:
                            print('Early stopping at epoch: ', start_epoch)
                            break

                    self.config.retrain_model = False
                    rs.save()

                results.append({'loss_train': loss, 'parameter': json.dumps({(k):(v.name if isinstance(v,Enum) else v) for k, v in parameter.items()}), 'metrics': metrics, 'rs': rs})
                with open(path_to_results + best_model['rs'].name + '_intermediate_results.pickle', 'ab') as fp:
                    pickle.dump({'loss_train': loss, 'parameter': json.dumps({(k):(v.name if isinstance(v,Enum) else v) for k, v in parameter.items()}), 'metrics': metrics}, fp)
                print('Training completed.')

        if self.config.cross_validation:
            print('----------------------------------------------------------------------')
            print('Choosing best model based on Mean reciprocal rank (in the filtered setting)...')
            metrics_all = np.array([i['metrics'] for i in results])
            best = np.argsort((-metrics_all[:, 16]))[0]
            best_model = results[best]
            # Store the results
            results = [{k: v for k, v in i.items() if k != 'rs'} for i in results]

        print('----------------------------------------------------------------------')
        print('Testing the best model on test set...')
        scores, model_result = best_model['rs'].test(self.test_masked)
        metrics = self.evaluate(self.test, self.test_mask, scores)

        cv_result = {'validation': results, 'test': metrics, 'best': best_model['parameter']}
        with open(path_to_results + best_model['rs'].name + '_results.pickle', 'wb') as fp:
            pickle.dump(cv_result, fp)
        model_result['entity2id'] = self.DataHelper.df_entity2id
        model_result['relation2id'] = self.DataHelper.df_relation2id
        with open(path_to_results + best_model['rs'].name + '_RecommenderSystem.pickle', 'wb') as fp:
            pickle.dump(model_result, fp)
        with open(path_to_results + best_model['rs'].name + '_ModelVariables.pickle', 'wb') as fp:
            pickle.dump(model_result.get('model_variables'), fp)

        np.savetxt(path_to_results + 'transform.csv', model_result.get('transform', []), delimiter=';')
        np.savetxt(path_to_results + 'item_embeddings.csv', model_result.get('item_embeddings', []), delimiter=';')
        np.savetxt(path_to_results + 'core_tensor.csv', model_result.get('core_tensor', []), delimiter=';')
        model_result.get('entity2id').to_csv(path_to_results + 'entity2id.csv', index=False, sep=';', header=None)

        all_metrics, best_metrics = extract_metrics(cv_result)

        print('----------------------------------------------------------------------')
        print("Evaluation results on the validation set:\n", all_metrics)
        print('----------------------------------------------------------------------')
        print("Evaluation results on the test set:\n", best_metrics)

        # Plot the results
        if plot:
            def get_category(row):
                categories = self.DataHelper.get_categories()
                if row['type'] == 'item' and self.DataHelper.has_category(row['id']):
                    return categories.iloc[self.DataHelper.get_category(int(row['id']))]['category']
                else:
                    return 'Unknown'

            df_entity2id, _, entity_embeddings, _ = model_result.get('entity2id'), model_result.get('core_tensor'), model_result.get('item_embeddings'), model_result.get('transform')
            df_entity2id['category'] = df_entity2id.apply(lambda row: get_category(row), axis=1)
            items = df_entity2id[df_entity2id['category'] != 'Unknown']
            # print('Items with known categories: ', len(items))
            item_embeddings = entity_embeddings[items.index.tolist()]

            visualize_item_embeddings(item_embeddings, categories=items['category'].tolist(), path_to_results=path_to_results + rs.name)

    def grid_search(self):
        """
        Perform a grid search over all parameters to create a list of all possible parameter combinations
        :param None
        :return: A list of all possible parameter combinations
        """
        if self.parameters is None:
            return [vars(self.config)]

        names = []
        parameters = []
        for p in self.parameters:
            names.append(p['name'])
            if p.get('value') is not None:
                parameters.append([p['value']])
            elif p.get('range') is not None:
                parameters.append(p['range'])
            else:
                parameters.append(generate_range(p['min'], p['max'], p.get('type', 'int'), p.get('step')))
        return [dict(zip(names, x)) for x in itertools.product(*parameters)]

    def mask_test_set(self, df_test_solutions, mask=None, as_mat=None):
        """

        :param df_test_solutions: Dataframe consisting of the solutions to be masked
        :param mask: Masking mechanism to use. Refer the "Mask" class
        :param as_mat: Parameter that indicates that the mask should be applied on the solutions matrix
        :return: Tuple consisting of the masked solutions and a list of all the masked items
        """
        global all_masked_items
        all_masked_items = []

        def mask_random_mat(df_solutions, sol_mat, mask):
            sol_mat, sol2index = sol_mat
            solutions_masked = sol_mat.copy()

            for row in df_solutions.itertuples():
                h, t = row.h, row.t

                items = t.split('\t')
                if mask is Mask.RANDOM:
                    masked_items = random.sample(items, len(items) // 2)
                else:
                    masked_items = [i for i in items if int(i) in mask]

                all_masked_items.append(masked_items)
                for t in masked_items:
                    solutions_masked[sol2index[h], int(t)] = 0

            return solutions_masked

        def mask_random(df_solutions):
            df_solutions_masked = df_solutions.copy()

            def mask_items(items):
                global all_masked_items
                items = items.split('\t')
                masked_items = random.sample(items, len(items) // 2)
                non_masked_items = [i for i in items if i not in masked_items]
                all_masked_items.append(masked_items)
                return '\t'.join(non_masked_items)

            df_solutions_masked['t'] = df_solutions_masked['t'].apply(lambda x: mask_items(x))
            return df_solutions_masked

        def mask_odd(df_solutions):
            df_solutions_masked = df_solutions.copy()

            def mask_items(items):
                global all_masked_items
                items = items.split('\t')
                masked_items = items[1::2]  # Mask items at odd indices
                non_masked_items = items[0::2]
                all_masked_items.append(masked_items)
                return '\t'.join(non_masked_items)

            df_solutions_masked['t'] = df_solutions_masked['t'].apply(lambda x: mask_items(x))
            return df_solutions_masked

        def mask_odd_in_category(df_solutions):
            df_solutions_masked = df_solutions.copy()

            def mask_items(items):
                global all_masked_items
                items = items.split('\t')
                masked_items = [i for i in items[1::2] if self.DataHelper.has_category(i)]  # Mask items at odd indices if they have a known category
                non_masked_items = items[0::2] + [i for i in items[1::2] if not self.DataHelper.has_category(i)]
                all_masked_items.append(masked_items)
                return '\t'.join(non_masked_items)

            df_solutions_masked['t'] = df_solutions_masked['t'].apply(lambda x: mask_items(x))
            return df_solutions_masked

        def mask_custom(df_solutions, mask):
            df_solutions_masked = df_solutions.copy()

            def mask_items(items):
                global all_masked_items
                items = items.split('\t')
                masked_items = [i for i in items if int(i) in mask]
                non_masked_items = [i for i in items if int(i) not in mask]
                all_masked_items.append(masked_items)
                return '\t'.join(non_masked_items)

            df_solutions_masked['t'] = df_solutions_masked['t'].apply(lambda x: mask_items(x))
            return df_solutions_masked

        if as_mat is not None:
            df_test_solutions_masked = mask_random_mat(df_test_solutions, as_mat, mask)
            df_test_solutions['index'] = df_test_solutions.index
        else:
            df_test_solutions['index'] = df_test_solutions.index

            if mask == Mask.RANDOM:
                df_test_solutions_masked = mask_random(df_test_solutions)
            elif mask == Mask.ODD:
                df_test_solutions_masked = mask_odd(df_test_solutions)
            elif mask == Mask.ODD_IN_CATEGORY:
                df_test_solutions_masked = mask_odd_in_category(df_test_solutions)
            else:
                df_test_solutions_masked = mask_custom(df_test_solutions, mask)

            df_test_solutions_masked.drop('index', axis=1, inplace=True)

        return df_test_solutions_masked, all_masked_items

    def evaluate(self, df_test, test_mask, result):
        """
        Performs a ranking of the items based on the scores generated by the model and computes a variety of qualitative
        metrics
        :param df_test: A dataframe corresponding to the unmasked test set
        :param test_mask: A list of all the masked items
        :param result: Scores obtained after running the model on the test set
        :return: An array of all the metrics computed on the cross-validation results
        """

        n_items_in_largest_category = self.DataHelper.get_largest_category()['n_item']

        def get_masked(index):
            masked = list(map(int, test_mask[index]))
            return masked

        # Function to get the ranks of items in the specified solution
        def get_rank(index, filtered=False, in_category=False, h=None):

            scores = np.array(result[index])
            # Evaluate only with respect to the masked items
            masked = list(map(int, test_mask[index]))

            if in_category:
                return [get_category_rank(i, scores, filtered) for i in masked if self.DataHelper.has_category(i)]
            elif filtered:
                ranks = []
                full_data = self.DataHelper.get_data()
                for i in masked:
                    truth = list(map(int, set(full_data[(full_data['h'] == h) & (full_data['r'] == 0)]['t'].astype('int')).difference({i})))
                    temp = scores.copy()
                    temp[truth] = -float('inf')
                    ranks.append(get_rank_from_scores([i], temp))
                return ranks
            else:
                return get_rank_from_scores(masked, scores)

        def get_rank_from_scores(masked, scores):
            scores_sorted = np.argsort(-scores)
            ranks = np.empty_like(scores_sorted)
            ranks[scores_sorted] = np.arange(len(scores))

            return (ranks[masked] + 1).tolist()

        def get_category_rank(id, scores, filtered=False):
            category_scores = np.full(scores.shape, -float('inf'))
            items_in_category = self.DataHelper.get_items_in_category(self.DataHelper.get_category(id))

            if filtered:
                truth = set(np.where(scores == -float('inf'))[0])
                items_in_category = list(set(items_in_category).difference(truth))

            category_scores[items_in_category] = scores[items_in_category]
            scores_sorted = np.argsort(-category_scores)
            ranks = np.empty_like(scores_sorted)
            ranks[scores_sorted] = np.arange(len(category_scores))
            rank = ranks[id]

            # Normalize category rank
            new_rank = (rank * n_items_in_largest_category) / len(items_in_category)
            return new_rank + 1

        bins = list(range(0, 11)) + list(range(11, 100, 10)) + list(range(101, 1000, 500)) + list(range(1001, self.DataHelper.get_item_count(), 1000)) + [self.DataHelper.get_item_count()]
        perc_bins = [i * 0.01 * self.DataHelper.get_item_count() for i in range(10)] + [i * 0.1 * self.DataHelper.get_item_count() for i in range(1, 11)]

        category_bins = list(range(0, 11)) + list(range(11, n_items_in_largest_category, 5)) + [n_items_in_largest_category]
        category_perc_bins = [i * 0.01 * n_items_in_largest_category for i in range(10)] + [i * 0.1 * n_items_in_largest_category for i in range(1, 11)]

        recall_bins = list(range(0, 350, 50))

        df_test['rank_raw'] = [get_rank(row['index']) for _, row in df_test.iterrows()]
        df_test['rank_filtered'] = [get_rank(row['index'], filtered=True, h=row['h']) for _, row in df_test.iterrows()]
        df_test['rank_in_category_raw'] = [get_rank(row['index'], in_category=True) for _, row in df_test.iterrows()]
        df_test['rank_in_category_filtered'] = [get_rank(row['index'], filtered=True, in_category=True, h=row['h']) for _, row in df_test.iterrows()]
        df_test['n_masked'] = [len(get_masked(row['index'])) for _, row in df_test.iterrows()]
        df_test['recall_raw'] = [np.cumsum(np.histogram(row['rank_raw'], recall_bins)[0])/row['n_masked'] if row['n_masked'] > 0 else [1]*(len(recall_bins)-1) for _, row in df_test.iterrows()]
        df_test['recall_filtered'] = [np.cumsum(np.histogram(row['rank_filtered'], recall_bins)[0])/row['n_masked'] if row['n_masked'] > 0 else [1]*(len(recall_bins)-1) for _, row in df_test.iterrows()]

        # Compute the evaluation metrics based in the raw setting
        mean_rank_raw = np.mean(df_test['rank_raw'].sum())
        median_rank_raw = np.median(df_test['rank_raw'].sum())
        mrr_raw = np.mean(np.reciprocal(np.array(df_test['rank_raw'].sum(), dtype=np.float32)))
        histogram_perc_rank_raw = np.array(np.histogram(df_test['rank_raw'].sum(), perc_bins))
        histogram_num_rank_raw = np.array(np.histogram(df_test['rank_raw'].sum(), bins))
        hits_top10_raw = histogram_num_rank_raw[0][:11].sum()/histogram_num_rank_raw[0].sum()
        hits_top10_perc_raw = histogram_perc_rank_raw[0][:10].sum()/histogram_perc_rank_raw[0].sum()
        recall_raw = np.mean(np.array(df_test['recall_raw'].tolist()), axis=0)

        # Compute the evaluation metrics based in the filtered setting
        mean_rank_filtered = np.mean(df_test['rank_filtered'].sum())
        median_rank_filtered = np.median(df_test['rank_filtered'].sum())
        mrr_filtered = np.mean(np.reciprocal(np.array(df_test['rank_filtered'].sum(), dtype=np.float32)))
        histogram_perc_rank_filtered = np.array(np.histogram(df_test['rank_filtered'].sum(), perc_bins))
        histogram_num_rank_filtered = np.array(np.histogram(df_test['rank_filtered'].sum(), bins))
        hits_top10_filtered = histogram_num_rank_filtered[0][:11].sum()/histogram_num_rank_filtered[0].sum()
        hits_top10_perc_filtered = histogram_perc_rank_filtered[0][:10].sum()/histogram_perc_rank_filtered[0].sum()
        recall_filtered = np.mean(np.array(df_test['recall_filtered'].tolist()), axis=0)

        # Compute the within-category evaluation metrics in the raw setting
        mean_rank_in_category_raw = np.mean(df_test['rank_in_category_raw'].sum())
        median_rank_in_category_raw = np.median(df_test['rank_in_category_raw'].sum())
        mrr_in_category_raw = np.mean(np.reciprocal(np.array(df_test['rank_in_category_raw'].sum(), dtype=np.float32)))
        histogram_perc_rank_in_category_raw = np.array(np.histogram(df_test['rank_in_category_raw'].sum(), category_perc_bins))
        histogram_num_rank_in_category_raw = np.array(np.histogram(df_test['rank_in_category_raw'].sum(), category_bins))
        hits_top10_in_category_raw = histogram_num_rank_in_category_raw[0][:11].sum()/histogram_num_rank_in_category_raw[0].sum()
        hits_top10_perc_in_category_raw = histogram_perc_rank_in_category_raw[0][:10].sum()/histogram_perc_rank_in_category_raw[0].sum()

        # Compute the within-category evaluation metrics in the filtered setting
        mean_rank_in_category_filtered = np.mean(df_test['rank_in_category_filtered'].sum())
        median_rank_in_category_filtered = np.median(df_test['rank_in_category_filtered'].sum())
        mrr_in_category_filtered = np.mean(np.reciprocal(np.array(df_test['rank_in_category_filtered'].sum(), dtype=np.float32)))
        histogram_perc_rank_in_category_filtered = np.array(np.histogram(df_test['rank_in_category_filtered'].sum(), category_perc_bins))
        histogram_num_rank_in_category_filtered = np.array(np.histogram(df_test['rank_in_category_filtered'].sum(), category_bins))
        hits_top10_in_category_filtered = histogram_num_rank_in_category_filtered[0][:11].sum()/histogram_num_rank_in_category_filtered[0].sum()
        hits_top10_perc_in_category_filtered = histogram_perc_rank_in_category_filtered[0][:10].sum()/histogram_perc_rank_in_category_filtered[0].sum()

        metrics = np.array(
            [mean_rank_raw, median_rank_raw, mrr_raw, hits_top10_raw, hits_top10_perc_raw, histogram_perc_rank_raw, histogram_num_rank_raw, mean_rank_in_category_raw,
             median_rank_in_category_raw, mrr_in_category_raw, hits_top10_in_category_raw, hits_top10_perc_in_category_raw, histogram_perc_rank_in_category_raw, histogram_num_rank_in_category_raw,
             mean_rank_filtered, median_rank_filtered, mrr_filtered, hits_top10_filtered, hits_top10_perc_filtered, histogram_perc_rank_filtered, histogram_num_rank_filtered,
             mean_rank_in_category_filtered, median_rank_in_category_filtered, mrr_in_category_filtered, hits_top10_in_category_filtered, hits_top10_perc_in_category_filtered,
             histogram_perc_rank_in_category_filtered, histogram_num_rank_in_category_filtered, recall_raw, recall_filtered])

        return metrics

    def rank(self, model, partial_solution, path_to_results=''):
        """
        Computes the scores after running the specified model on the data
        :param model: Specific model to use
        :param partial_solution: Binary partial solution with 1s indicating that the corresponding items were configured
        :param path_to_results: Path to store the resultant files (e.g., TensorFlow model, cross-validation results)
        :return: Scores obtained after running the model on the test set
        """
        for parameter in self.grid_search():
            for p_name, p_value in parameter.items():
                setattr(self.config, p_name, p_value)

        rs = model(datahelper=self.DataHelper, config=self.config, path_to_results=path_to_results)
        print('----------------------------------------------------------------------')
        print('Creating the model: ', rs.name, '...')

        scores = rs.rank(partial_solution)
        return scores
