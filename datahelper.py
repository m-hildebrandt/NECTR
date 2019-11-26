import pandas as pd
import numpy as np
import random
from enum import Enum
from scipy.sparse import csr_matrix


class Sampling(Enum):
    """
    Class used to represent the various mechanisms for sampling negative examples

    CONSTRAINED_RANDOM:    Rather than getting negative examples by randomly corrupting true triples,
                    we impose the additional constraint that the new tail appears as object in a known triple
                    with respect to the same relation as the true triple.

    DISTANCE:   Distance-based sampling where the distances or similarities between items are used to guide the
            sampling, with the intuition being that the larger the distance between two items, the more dissimilar they
            are and thereby less likely to appear in place of one another.

    DISTANCE_IN_CATEGORY:   Distance-based sampling within category is an extension of DISTANCE sampling. Instead of
            considering all items, we only consider those items that belong to the same category as the item in question
            and use the distances to them to guide the sampling.
    """

    CONSTRAINED_RANDOM = 1
    DISTANCE = 2
    DISTANCE_IN_CATEGORY = 3


class DataHelper:
    """
    Class responsible for all data-related functionality, such as loading data from files, transforming data into
    suitable data structures for optimal access, and sampling negative examples.
    """

    categories = None
    item2category = None
    category2item = None
    df_triple2id = None
    df_entity2id = None
    item_ids = None
    solution_ids = None
    df_relation2id = None
    item_distance = None
    data = None
    adjacency = None
    sampling = Sampling.CONSTRAINED_RANDOM

    def __init__(self):
        pass

    @staticmethod
    def setup(data_path, parameters=None):
        """
        Loads data from the file system and sets the relevant properties on the DataHelper object.
        :param data_path: Path to the folder containing all the data files
        :param parameters: Dictionary of parameter keys and values to be set on the DataHelper object
        :return: None
        """
        DataHelper.categories = pd.read_csv(data_path + 'Categories.csv', sep=';')
        temp = DataHelper.categories.apply(lambda x: {int(i): x.name for i in x['entity_id'].split(',')},
                                    axis=1).values.tolist()
        DataHelper.item2category = {k: v for i in temp for k, v in i.items()}
        temp = DataHelper.categories.apply(lambda x: {x.name: list(map(int, x['entity_id'].split(',')))},
                                    axis=1).values.tolist()
        DataHelper.category2item = {k: v for i in temp for k, v in i.items()}
        temp = None

        DataHelper.df_entity2id = pd.read_table(data_path + 'entity2id.txt', index_col=None, header=0,
                                     names=['entity', 'id', 'type'], encoding='latin1')
        DataHelper.item_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'item']['id'].astype(int).tolist()
        DataHelper.solution_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'solution']['id'].astype(int).tolist()
        DataHelper.df_relation2id = pd.read_table(data_path + 'relation2id.txt', index_col=None, header=0,
                                       names=['relation', 'id'])

        try:
            DataHelper.df_triple2id = pd.read_table(data_path + 'triple2id.txt', index_col=None, header=0,
                                                    names=['h', 't', 'r', 'n'])
        except pd.errors.ParserError:
            print('No count data found for the triples')
            print('Continuing...')
            DataHelper.df_triple2id = pd.read_table(data_path + 'triple2id.txt', index_col=None, header=0,
                                                    names=['h', 't', 'r'])

        DataHelper.set_parameters(parameters)

        if DataHelper.sampling == Sampling.DISTANCE or DataHelper.sampling == Sampling.DISTANCE_IN_CATEGORY:
            DataHelper.item_distance = np.load(data_path + 'item2item_distance.pickle')

        DataHelper.set_data(DataHelper.df_triple2id)

    @staticmethod
    def setup_prod_data(df_entity2id, df_relation2id):
        """
        Sets the specified entity and relation mappings on the DataHelper object.
        This method should be used when the data needs to be setup using the passed parameter values
        and not loaded from the file system.
        :param df_entity2id: DataFrame with the entity to id mapping
        :param df_relation2id: DataFrame with the relation to id mapping
        :return: None
        """
        DataHelper.df_entity2id = df_entity2id
        DataHelper.item_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'item']['id'].astype(int).tolist()
        DataHelper.solution_ids = DataHelper.df_entity2id[DataHelper.df_entity2id['type'] == 'solution']['id'].astype(int).tolist()
        DataHelper.df_relation2id = df_relation2id

    @staticmethod
    def set_parameters(parameters):
        """
        Sets the specified parameters on the DataHelper object
        :param parameters: Dictionary of parameter keys and values
        :return: None
        """
        if parameters:
            for p_name, p_value in parameters.items():
                setattr(DataHelper, p_name, p_value)

    @staticmethod
    def has_category(item):
        """
        Tells whether or not the specified item has a category
        :param item: Id of the item
        :return: True if the item has a known category and False otherwise
        """
        return item in DataHelper.item2category

    @staticmethod
    def get_category(item):
        """
        Fetches the category of the specified item
        :param item: Id of the item
        :return: Category of the specified item
        """
        return DataHelper.item2category[item]

    @staticmethod
    def get_category_encoding(item):
        """
        Fetches the one-hot encoding of an item based on the categories it belongs to
        :param item: Id of the item
        :return: One-hot encoded list of categories for the specified item
        """
        encoding = [0] * DataHelper.get_category_count()
        if DataHelper.has_category(item):
            encoding[DataHelper.get_category(item)] = 1
        return encoding

    @staticmethod
    def get_categories():
        """
        Fetches all known categories
        :param None
        :return: A DataFrame of all known categories with category name, item names, and item ids
        """
        return DataHelper.categories

    @staticmethod
    def get_items_in_category(category):
        """
        Fetches all items in the specified category
        :param category: Id of the category
        :return: List of ids of all items in the specified category
        """
        return DataHelper.category2item[category]

    @staticmethod
    def get_items_with_category_count():
        """
        Fetches the number of items with a known category
        :param None
        :return: Integer denoting the number of items with a known category
        """
        if DataHelper.item2category is None:
            return 0
        return len(DataHelper.item2category)

    @staticmethod
    def get_largest_category():
        """
        Fetches the category with the maximum number of items
        :param None
        :return: Dictionary with the id of the largest category and the number of items in that category
        """
        category = max(DataHelper.category2item, key=lambda p: len(DataHelper.category2item[p]))
        return {'category': category, 'n_item': len(DataHelper.category2item[category])}

    @staticmethod
    def get_item_count():
        """
        Fetches the number of items
        :param None
        :return: Integer denoting the number of items
        """
        if DataHelper.item_ids is None:
            return 0
        return len(DataHelper.get_items())

    @staticmethod
    def get_solution_count():
        """
        Fetches the number of solutions
        :param None
        :return: Integer denoting the number of solutions
        """
        return len(DataHelper.solution_ids)

    @staticmethod
    def get_category_count():
        """
        Fetches the number of categories
        :param None
        :return: Integer denoting the number of categories
        """
        if DataHelper.category2item is None:
            return 0
        return len(DataHelper.category2item)

    @staticmethod
    def get_entity_id(entity):
        """
        Fetches the id of the specified entity
        :param entity: Name of the entity
        :return: Id of the specified entity
        """
        return DataHelper.df_entity2id[DataHelper.df_entity2id['entity'].isin(entity)]['id'].astype(int).tolist()

    @staticmethod
    def get_items(copy=False):
        """
        Fetches all the items
        :param copy: Parameter that determines whether the original items or a copy will be returned
        :return: If copy=True, a copy of the items is returned. This is useful in cases where the result will be further modified.
        Else, the items are directly returned. This should be used for read-only purposes.
        """
        if copy:
            return DataHelper.item_ids[:]
        return DataHelper.item_ids

    @staticmethod
    def get_entity_count():
        """
        Fetches the number of entities
        :param None
        :return: Integer denoting the number of entities
        """
        if DataHelper.df_entity2id is None:
            return 0
        return len(DataHelper.df_entity2id)

    @staticmethod
    def get_relation_count():
        """
        Fetches the number of relations
        :param None
        :return: Integer denoting the number of relations
        """
        if DataHelper.df_relation2id is None:
            return 0
        return len(DataHelper.df_relation2id)

    @staticmethod
    def get_relation_id(relation):
        """
        Fetches the id of the specified relation
        :param relation: Name of the relation
        :return: Id of the specified relation
        """
        return DataHelper.df_relation2id[DataHelper.df_relation2id['relation'] == relation]['id'].iloc[0]

    @staticmethod
    def set_data(data):
        """
        Sets the data property on the DataHelper object and also creates an adjacency map
        to allow fast access and traversal of linked entities and relations.
        :param data: DataFrame containing data in the form of triples
        :return: None
        """
        DataHelper.data = data

        adjacency = {'h': set(), 't': set(), 'r': {}}
        for r in DataHelper.df_relation2id['id']:
            adjacency['r'][r] = {'h': set(), 't': set(), 'h_map': {}, 't_map': {}}
            r_triples = DataHelper.data.loc[DataHelper.data['r'] == r, ['h', 't']]
            for triple in r_triples.itertuples():
                _, h, t = triple
                adjacency['h'].add(h)
                adjacency['t'].add(t)
                adjacency['r'][r]['h'].add(h)
                adjacency['r'][r]['t'].add(t)
                adjacency['r'][r]['h_map'].setdefault(h, set()).add(t)
                adjacency['r'][r]['t_map'].setdefault(t, set()).add(h)
        DataHelper.adjacency = adjacency

    @staticmethod
    def get_data():
        """
        Fetches the data in the form of triples
        :param None
        :return: DataFrame containing all the triples
        """
        return DataHelper.data

    @staticmethod
    def get_solutions():
        """
        Fetches all the solutions
        :param None
        :return: DataFrame containing all the solutions
        """
        return DataHelper.data[DataHelper.data['r'] == 0]

    @staticmethod
    def get_solution_matrix(solution_ids):
        """
        Fetches the matrix of Solutions X Items with the entries corresponding to counts
        :param solution_ids: List of solution ids
        :return: Tuple consisting of a sparse matrix (Solutions X Items) and a mapping from solution ids to indices.
        The entries of the matrix indicate the number of times an item has been configured in the corresponding solution.
        """
        temp = DataHelper.data[DataHelper.data['h'].isin(solution_ids)]
        solution_id2index = {j: i for i, j in enumerate(solution_ids)}

        solution_matrix = csr_matrix((temp['n'], ([solution_id2index[i] for i in temp['h']], temp['t'])),
                                     (len(solution_ids), DataHelper.get_item_count()))

        return solution_matrix, solution_id2index

    @staticmethod
    def sol2triple(solutions):
        """
        Converts the solutions to triples
        :param solutions: DataFrame containing solutions in the form of a solution id and a tab-separated string of item ids
        :return: DataFrame containing the solutions in the form of triples, i.e., solution id, item id, contains relation id
        """
        temp = []
        for row in solutions.itertuples():
            _, h, t = row
            temp += [[h, int(i), 0] for i in t.split('\t')]
        triples = pd.DataFrame(temp, columns=['h', 't', 'r'])
        return triples

    @staticmethod
    def sol2mat(solutions, sparse=False):
        """
        Converts the solutions to a matrix
        :param solutions: DataFrame containing solutions in the form of a solution id and a tab-separated string of item ids
        :param sparse: Parameter that determines whether a dense or a sparse matrix will be returned
        :return: Binary matrix of Solutions X Items where 1 indicates that an item has been configured in the corresponding solution
        """
        rows, cols = [], []
        for row in solutions.itertuples():
            index, h, t = row
            if t:
                temp = [int(i) for i in t.split('\t')]
                rows += [index] * len(temp)
                cols += temp

        if sparse:
            mat = csr_matrix(([1] * len(rows), (rows, cols)), (len(solutions), DataHelper.get_item_count()))
        else:
            mat = np.zeros((len(solutions), DataHelper.get_item_count()))
            mat[rows, cols] = 1

        return mat

    @staticmethod
    def get_item_distance(item_id):
        """
        Fetches the distances to all items from the specified item
        :param item_id: DataFrame with the entity to id mapping
        :return: Numpy array of distances to all items from the specified item
        """
        return DataHelper.item_distance[item_id]

    @staticmethod
    def corrupt_tail(triple, n=1):
        """
        Generates a list of corrupt tail entities for the specified triple
        :param triple: DataFrame row denoting a triple of the form h, t, r
        :param n: Integer denoting the number of corrupt tail entities to generate
        :return: List of corrupted tail entities
        """
        h, t, r = triple.h, triple.t, triple.r
        corrupt_tails = DataHelper.adjacency['r'][r]['t'].difference(DataHelper.adjacency['r'][r]['h_map'][h])
        t_corrupt = []

        if r == 0:
            # Sampling items by distance
            if DataHelper.sampling == Sampling.DISTANCE or DataHelper.sampling == Sampling.DISTANCE_IN_CATEGORY:
                distances = DataHelper.get_item_distance(t)

                if DataHelper.sampling == Sampling.DISTANCE_IN_CATEGORY:
                    # For items with known category, sample items within the same category
                    if DataHelper.has_category(t):
                        corrupt_tails = corrupt_tails.intersection(set(DataHelper.get_items_in_category(DataHelper.get_category(t))))

                distances = distances[list(corrupt_tails)]
                distances /= distances.sum()

                t_corrupt = np.random.choice(range(len(distances)), n, p=distances).tolist()

            # Sampling items randomly from the constrained set of corrupted tails
            elif DataHelper.sampling == Sampling.CONSTRAINED_RANDOM:
                t_corrupt = random.sample(corrupt_tails, min(n, len(corrupt_tails)))
        else:
            h_primes = DataHelper.adjacency['r'][r]['h'].difference({h})

            if not len(h_primes) or not len(corrupt_tails):
                t_corrupt = random.sample(DataHelper.adjacency['t'].difference(DataHelper.adjacency['r'][r]['h_map'][h]), n)
            else:
                t_corrupt = random.sample(corrupt_tails, min(n, len(corrupt_tails)))

        return t_corrupt

    @staticmethod
    def get_batch(data, batch_size, negative2positive_ratio=1, category=False, partial_data=None, complete_data=None):
        """
        Fetches a batch of the specified batch size from the data
        :param data: DataFrame containing data in the form of triples
        :param batch_size: Size of the batch
        :param negative2positive_ratio: Ratio of positive to negative samples
        :param category: Boolean parameter that determines whether or not category data is included in the response
        :param partial_data: Sparse matrix with data corresponding to partial solutions
        :param complete_data: Sparse matrix with data corresponding to complete solutions
        :return: List of lists or a dictionary consisting of batches of the specified data
        """
        batch_index = 0
        if category:
            columns = ['h', 't', 'r', 'nh', 'nt', 'nr', 't_category', 'nt_category']
        else:
            columns = ['h', 't', 'r', 'nh', 'nt', 'nr']

        while batch_index < len(data.index):
            sample = data.iloc[batch_index: batch_index+batch_size]
            triples = []
            for x in sample.itertuples():
                temp = DataHelper.corrupt_tail(x, negative2positive_ratio)
                h, t, r = x.h, x.t, x.r
                if category:
                    t_category = DataHelper.get_category_encoding(t)
                    triples += [[h, t, r, h, i, r, t_category, DataHelper.get_category_encoding(i)] for i in temp]
                else:
                    triples += [[h, t, r, h, i, r] for i in temp]
            batch = pd.DataFrame(triples, columns=columns)

            if partial_data is not None:
                partial_sample = partial_data[batch_index: batch_index + batch_size]
                complete_sample = complete_data[batch_index: batch_index + batch_size]
                yield {'triple': [list(i) for i in zip(*batch.values)], 'solution': [partial_sample] + [complete_sample]}
            else:
                yield [list(i) for i in zip(*batch.values)]
            batch_index += batch_size

    @staticmethod
    def get_batch_solution(partial_data, complete_data, batch_size):
        """
        Fetches a batch of the specified batch size from the data
        :param partial_data: Sparse matrix with data corresponding to partial solutions
        :param complete_data: Sparse matrix with data corresponding to complete solutions
        :param batch_size: Size of the batch
        :return: Dictionary consisting of batches of the specified data
        """
        batch_index = 0
        while batch_index < partial_data.shape[0]:
            partial_sample = partial_data[batch_index: batch_index + batch_size]
            complete_sample = complete_data[batch_index: batch_index + batch_size]
            yield {'partial_solution': partial_sample, 'complete_solution': complete_sample}
            batch_index += batch_size

    @staticmethod
    def get_test_batch(data_h, batch_size):
        """
        Fetches a batch of the specified batch size from the data
        :param data_h: List of h entities to extract the batch from
        :param batch_size: Size of the batch
        :return: List of triples of the form h, t, r where each of h, t, and r is a list of entity/relation ids.
        For each h in data_h, combinations with all possible t values, i.e., all items,
        and the contains relation (id=0) are returned.
        """
        batch_index = 0
        while batch_index < len(data_h):
            yield [data_h[batch_index: batch_index + batch_size] * DataHelper.get_item_count(),
                   [i for i in range(DataHelper.get_item_count()) for j in range(batch_size)],
                   [0] * DataHelper.get_item_count() * batch_size]
            batch_index += batch_size
