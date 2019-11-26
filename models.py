import tensorflow as tf
import time
from sklearn.utils import shuffle
import numpy as np
from sklearn.decomposition import NMF
from abc import ABC, abstractmethod
import pickle
from enum import Enum


class TrainingMode(Enum):
    """
    Class used to represent the various modes of training the non-linear models

    ALTERNATING: The completion and recommendation loss  are minimized in an alternating fashion
    SIMULTANEOUS: The completion and recommendation loss are minimized simultaneously
    """

    ALTERNATING = 1
    SIMULTANEOUS = 2


class Model(ABC):
    """
    Base class for all factorization models
    """

    def __init__(self, datahelper=None, config=None, path_to_results=''):
        self.DataHelper = datahelper
        self.config = config
        self.path_to_results = path_to_results

    @abstractmethod
    def model(self, config):
        pass

    @property
    @abstractmethod
    def name(self):
        return ''

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def test(self, partial_solution):
        pass

    @abstractmethod
    def rank(self, partial_solution):
        pass


class MatrixModel(Model):
    """
    Base class for all matrix factorization models
    """

    def train(self, data):
        train_model = self.model(self.config)
        loss = train_model.fit(data)
        train_model.save(self.path_to_results + self.name + '_model')
        return loss

    def test(self, partial_solution):
        partial_solution = self.DataHelper.sol2mat(partial_solution)
        train_model = self.model(self.config)
        train_model.restore(self.path_to_results + self.name + '_model')
        return train_model.predict(partial_solution)

    def rank(self, partial_solution):
        pass


class TensorModel(Model):
    """
    Base class for all tensor factorization models
    """

    @abstractmethod
    def get_parameters(self, mode='train'):
        return []

    def init_vars(self):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.train_model = self.model(self.config)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.train_model.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=None)

    class Manager:
        def __enter__(self):
            self.device = tf.device('/gpu:0')
            self.device.__enter__()
            g = tf.Graph()
            self.g_cm = g.as_default()
            self.g_cm.__enter__()
            self.sess = tf.Session()
            self.sess_cm = self.sess.as_default()
            self.sess_cm.__enter__()

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.sess_cm.__exit__(exc_type, exc_val, exc_tb)
            self.sess.close()
            self.g_cm.__exit__(exc_type, exc_val, exc_tb)
            self.device.__exit__(exc_type, exc_val, exc_tb)

    def copy(self):
        with self.Manager() as mgr:
            self.init_vars()
            self.saver.restore(mgr.sess, self.path_to_results + self.name + '_model.vec')
            self.saver.save(mgr.sess, self.path_to_results + self.name + '_model_copy.vec')

    def save(self):
        with self.Manager() as mgr:
            self.init_vars()
            self.saver.restore(mgr.sess, self.path_to_results + self.name + '_model_copy.vec')
            self.saver.save(mgr.sess, self.path_to_results + self.name + '_model.vec')

    def train(self, data):
        tic = time.time()
        self.config.batch_size = len(data) // self.config.n_batch

        with self.Manager() as mgr:
            self.init_vars()
            mgr.sess.run(tf.initialize_all_variables())

            if self.config.retrain_model:
                self.saver.restore(mgr.sess, self.path_to_results + self.name + '_model.vec')

            def train_step(param_values):

                feed_dict = {self.learning_rate: self.config.learning_rate}
                for i, p in enumerate(self.get_parameters()):
                    feed_dict[getattr(self.train_model, p)] = param_values[i]

                _, step, loss, pos = mgr.sess.run(
                    [self.train_op, self.global_step, self.train_model.loss, self.train_model.pos], feed_dict)

                if self.config.clipE or self.config.clipR:
                    _ = mgr.sess.run([self.train_model.clip])
                return loss

            loss_train = []
            item_embs_history = {}

            isCategoryModel = isinstance(self, RESCOMCategoryModel)
            self.config.register('learning_rate')
            for times in self.config.epoch_range:
                if self.config.track_history:
                    item_embs_history[times] = mgr.sess.run(self.train_model.item_embeddings)

                loss = 0.0
                data = shuffle(data)
                for batch in self.DataHelper.get_batch(data, self.config.batch_size, self.config.negative2positive_ratio, category=isCategoryModel):
                    loss += train_step(batch)
                    _ = tf.train.global_step(mgr.sess, self.global_step)
                print('Epoch=', times, 'loss=', loss)
                loss_train.append(loss)
                if self.config.adaptive_learning_rate and times > 0:
                    if loss_train[-1] > loss_train[-2]:
                        self.config.learning_rate = self.config.learning_rate / 2
                        print("Half the learning rate: l_rate = ", self.config.learning_rate)
            self.config.recall('learning_rate')
            self.saver.save(mgr.sess, self.path_to_results + self.name + '_model.vec')
            if self.config.track_history:
                with open(self.path_to_results + 'item_embeddings_history.pickle', 'wb') as fp:
                    pickle.dump(item_embs_history, fp)

        toc = time.time()
        print('Time taken for training: {} (in hours)'.format((toc - tic) / 3600))
        return loss_train

    def test(self, partial_solution):
        with self.Manager() as mgr:
            self.init_vars()
            mgr.sess.run(tf.initialize_all_variables())
            self.saver.restore(mgr.sess, self.path_to_results + self.name + '_model.vec')

            def test_step(param_values):
                feed_dict = {self.learning_rate: self.config.learning_rate}
                for i, p in enumerate(self.get_parameters(mode='test')):
                    feed_dict[getattr(self.train_model, p)] = param_values[i]

                step, complete_solution, results = mgr.sess.run(
                    [self.global_step, self.train_model.predict, self.train_model.results], feed_dict)

                return complete_solution, results

            def test_by_projection(partial_solution):
                # print("partial_solution", np.unique(partial_solution))
                items = np.array(range(self.config.n_item), dtype=np.int32)
                feed_dict = {
                    self.train_model.partial_solution: partial_solution.toarray(),
                    self.train_model.pos_t: items,  # item embeddings
                    self.train_model.pos_r: np.array([0], dtype=np.int32),  # index of the 'contains' relation
                }
                if isinstance(self, RESCOMCategoryModel):
                    feed_dict[self.train_model.pos_t_category] = [self.DataHelper.get_category_encoding(i) for i in items]

                step, complete_solution, results = mgr.sess.run(
                    [self.global_step, self.train_model.predict, self.train_model.results], feed_dict)

                results['model_variables'] = self.get_model_variables(mgr)
                return complete_solution, results

            if isinstance(self, RESCOMModel) or isinstance(self, RESCOMCategoryModel) or isinstance(self, NECTRModel):
                if not self.config.nectr_item_counts:
                    partial_solution = self.DataHelper.sol2mat(partial_solution, sparse=True)
                complete_solution, results = test_by_projection(partial_solution)
            else:
                complete_solution = []
                results = {}
                for batch in self.DataHelper.get_test_batch(partial_solution['h'].unique().tolist(), batch_size=1):
                    p, results = test_step(batch)
                    complete_solution.append(p)
                complete_solution = np.array(complete_solution).squeeze(2)

        return complete_solution, results

    def rank(self, partial_solution):
        pass

    def get_model_variables(self, mgr):
        model_variables = {}
        if mgr is None:
            mgr = self.Manager()
        trainable_variables = tf.trainable_variables()
        trainable_variables_values = mgr.sess.run(trainable_variables)
        # Iterate over all the trainable variables and store them by name and value
        for variable, value in zip(trainable_variables, trainable_variables_values):
            model_variables[variable.name] = value
        return model_variables


class NMFModel(MatrixModel):
    """
    Class implementing non-negative matrix factorization (NMF)
    """

    @property
    def name(self):
        return 'NMF'

    class NMFWrapper:
        def __init__(self, config):
            self.model = NMF(alpha=config.regularization, init='random', l1_ratio=0.0, max_iter=config.n_epoch, n_components=config.hidden_size,
                             random_state=0, shuffle=False, solver='cd', tol=config.margin, verbose=0)

        def fit(self, data):
            self.model.fit(data)
            return [self.model.reconstruction_err_]

        def predict(self, partial_solution):
            partial_solution_transform = self.model.transform(partial_solution)
            complete_solution = self.model.inverse_transform(partial_solution_transform)
            result = {'item_embeddings': self.model.components_.T}
            return complete_solution, result

        def save(self, path):
            with open(path + '.pickle', 'wb') as fp:
                pickle.dump(self.model, fp)

        def restore(self, path):
            with open(path + '.pickle', 'rb') as fp:
                self.model = pickle.load(fp)

    def model(self, config):
        return self.NMFWrapper(config)


class TransDModel(TensorModel):
    """
    Class implementing TransD
    """

    @property
    def name(self):
        return 'TransD'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class TransD(object):

        def calc(self, e, t, r):
            return tf.nn.l2_normalize(e + tf.reduce_sum(e * t, 1, keep_dims=True) * r, 1)

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            with tf.name_scope("embedding"):

                self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.ent_transfer = tf.get_variable(name="ent_transfer", shape=[config.n_entity, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.rel_transfer = tf.get_variable(name="rel_transfer", shape=[config.n_relation, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
                pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
                pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)

                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
                neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
                neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
                neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)

                pos_h_e = self.calc(pos_h_e, pos_h_t, pos_r_t)
                pos_t_e = self.calc(pos_t_e, pos_t_t, pos_r_t)
                neg_h_e = self.calc(neg_h_e, neg_h_t, neg_r_t)
                neg_t_e = self.calc(neg_t_e, neg_t_t, neg_r_t)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)

            self.clip = tf.group(*clip_ops)

            self.results = {
                'relation_embeddings': self.rel_embeddings,
                'item_embeddings': self.ent_embeddings
            }

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
                    self.neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = -self.pos
            else:
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
                    self.neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos

    def model(self, config):
        return self.TransD(config)


class TransEModel(TensorModel):
    """
    Class implementing TransE
    """

    @property
    def name(self):
        return 'TransE'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class TransE(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            with tf.name_scope("embedding"):

                self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                      shape=[config.n_entity, config.hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                                      shape=[config.n_relation, config.hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)

            self.clip = tf.group(*clip_ops)

            self.results = {
                'relation_embeddings': self.rel_embeddings,
                'item_embeddings': self.ent_embeddings
            }

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
                    self.neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = -self.pos
            else:
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
                    self.neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos

    def model(self, config):
        return self.TransE(config)


class TransHModel(TensorModel):
    """
    Class implementing TransH
    """

    @property
    def name(self):
        return 'TransH'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class TransH(object):

        def calc(self, e, n):
            norm = tf.nn.l2_normalize(n, 1)
            return e - tf.reduce_sum(e * norm, 1, keep_dims=True) * norm

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            with tf.name_scope("embedding"):

                self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, config.hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.normal_vector = tf.get_variable(name="normal_vector", shape=[config.n_relation, config.hidden_size],
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

                pos_norm = tf.nn.embedding_lookup(self.normal_vector, self.pos_r)
                neg_norm = tf.nn.embedding_lookup(self.normal_vector, self.neg_r)

                pos_h_e = self.calc(pos_h_e, pos_norm)
                pos_t_e = self.calc(pos_t_e, pos_norm)
                neg_h_e = self.calc(neg_h_e, neg_norm)
                neg_t_e = self.calc(neg_t_e, neg_norm)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)

            self.clip = tf.group(*clip_ops)

            self.results = {
                'relation_embeddings': self.rel_embeddings,
                'item_embeddings': self.ent_embeddings
            }

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
                    self.neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = -self.pos
            else:
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
                    self.neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos

    def model(self, config):
        return self.TransH(config)


class TransRModel(TensorModel):
    """
    Class implementing TransR
    """

    @property
    def name(self):
        return 'TransR'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class TransR(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [config.batch_size])
            self.pos_t = tf.placeholder(tf.int32, [config.batch_size])
            self.pos_r = tf.placeholder(tf.int32, [config.batch_size])

            self.neg_h = tf.placeholder(tf.int32, [config.batch_size])
            self.neg_t = tf.placeholder(tf.int32, [config.batch_size])
            self.neg_r = tf.placeholder(tf.int32, [config.batch_size])

            with tf.name_scope("embedding"):
                if config.initE is None:
                    self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, config.hidden_sizeE],
                                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                else:
                    self.ent_embeddings = tf.Variable(np.loadtxt(config.initE), name="ent_embedding", dtype=np.float32)

                if config.initR is None:
                    self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_sizeR],
                                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                else:
                    self.rel_embeddings = tf.Variable(np.loadtxt(config.initR), name="rel_embedding", dtype=np.float32)

                rel_matrix = np.zeros([config.n_relation, config.hidden_sizeR * config.hidden_sizeE], dtype=np.float32)
                for i in range(config.n_relation):
                    for j in range(config.hidden_sizeR):
                        for k in range(config.hidden_sizeE):
                            if j == k:
                                rel_matrix[i][j * config.hidden_sizeE + k] = 1.0
                self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix")

            with tf.name_scope("lookup_embedding"):
                pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h), [-1, config.hidden_sizeE, 1])
                pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t), [-1, config.hidden_sizeE, 1])
                pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r), [-1, config.hidden_sizeR])
                neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h), [-1, config.hidden_sizeE, 1])
                neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t), [-1, config.hidden_sizeE, 1])
                neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r), [-1, config.hidden_sizeR])
                pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.pos_r), [-1, config.hidden_sizeR, config.hidden_sizeE])
                neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_r), [-1, config.hidden_sizeR, config.hidden_sizeE])

                pos_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_h_e), [-1, config.hidden_sizeR]), 1)
                pos_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_t_e), [-1, config.hidden_sizeR]), 1)
                neg_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_h_e), [-1, config.hidden_sizeR]), 1)
                neg_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_t_e), [-1, config.hidden_sizeR]), 1)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)

            self.clip = tf.group(*clip_ops)

            self.results = {
                'relation_embeddings': self.rel_embeddings,
                'item_embeddings': self.ent_embeddings
            }

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims=True)
                    self.neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = -self.pos
            else:
                with tf.name_scope("output"):
                    self.pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims=True)
                    self.neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims=True)
                    self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + config.margin, 0))
                self.predict = -self.pos

    def model(self, config):
        return self.TransR(config)


class RESCOMModel(TensorModel):
    """
    Class implementing RESCOM
    """

    @property
    def name(self):
        return 'RESCOM'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class RESCOM(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            self.partial_solution = tf.placeholder(tf.float32, (None, config.n_item))

            with tf.name_scope("embedding"):
                if config.path_pretrained_ent_embs:
                    pretrained_ent_embs = np.expand_dims(np.loadtxt(config.path_pretrained_ent_embs, delimiter=';', dtype=np.float32), axis=1)

                    self.ent_embeddings_no_train = tf.get_variable(name="ent_embedding_no_train", initializer=pretrained_ent_embs, trainable=False)
                    self.ent_embeddings_train = tf.get_variable(name="ent_embedding_train", initializer=np.random.randn(config.n_entity - len(pretrained_ent_embs), 1, config.hidden_size).astype(np.float32))
                    self.ent_embeddings = tf.concat([self.ent_embeddings_no_train, self.ent_embeddings_train], 0)
                else:
                    self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, 1, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            temp = np.zeros(self.rel_embeddings.shape, dtype=bool)
            diag_r, diag_c = np.where(np.eye(config.hidden_size, dtype=bool))
            temp[:, diag_r, diag_c] = 1
            diag_rel = tf.assign(self.rel_embeddings, tf.where(temp, self.rel_embeddings, np.zeros(self.rel_embeddings.shape)))
            ident_rel = tf.assign(self.rel_embeddings, tf.where(temp, np.ones(self.rel_embeddings.shape, dtype='float32'),
                                                                np.zeros(self.rel_embeddings.shape, dtype='float32')))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)
            if config.diagonalR:
                clip_ops.append(diag_rel)
            if config.identityR:
                clip_ops.append(ident_rel)

            self.clip = tf.group(*clip_ops)

            self.pos = tf.reshape(tf.matmul(tf.matmul(pos_h_e, pos_r_e), tf.transpose(pos_t_e, perm=[0, 2, 1])), [-1])
            self.neg = tf.reshape(tf.matmul(tf.matmul(neg_h_e, neg_r_e), tf.transpose(neg_t_e, perm=[0, 2, 1])), [-1])

            transform = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.matmul(tf.matmul(pos_r_e, tf.transpose(pos_t_e, perm=[1, 2, 0])), tf.transpose(pos_t_e, perm=[1, 0, 2])), tf.transpose(pos_r_e, perm=[0, 2, 1]))), pos_r_e), tf.transpose(pos_t_e, perm=[1, 2, 0]))
            projection = tf.matmul(transform, tf.transpose(tf.expand_dims(self.partial_solution, 0), perm=[0, 2, 1]))
            self.complete = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(projection, perm=[0, 2, 1]), pos_r_e), tf.transpose(pos_t_e, perm=[1, 2, 0])), axis=0)

            self.results = {
                'core_tensor': tf.squeeze(pos_r_e, axis=0),
                'item_embeddings': tf.squeeze(pos_t_e, axis=1),
                'entity_embeddings': tf.squeeze(self.ent_embeddings, axis=1),
                'relation_embeddings': self.rel_embeddings,
                'transform': tf.squeeze(transform, axis=0)
            }

            self.item_embeddings = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, np.array(range(config.n_item), dtype=np.int32)), axis=1)

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.loss = tf.reduce_sum(tf.abs(1 - self.pos) + tf.abs(self.neg)) + config.regularization * tf.reduce_sum(
                        pos_h_e + pos_t_e + neg_h_e + neg_t_e)
                self.predict = self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = self.pos
            else:
                with tf.name_scope("output"):
                    self.loss = tf.reduce_sum((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_sum(
                                    tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))
                self.predict = self.pos

            self.predict = self.complete

    def model(self, config):
        return self.RESCOM(config)


class RESCOMCategoryModel(TensorModel):
    """
    Class implementing RESCOM_Category
    """

    @property
    def name(self):
        return 'RESCOMCategory'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r', 'pos_t_category']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r', 'pos_t_category', 'neg_t_category']

    class RESCOMCategory(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            self.pos_t_category = tf.placeholder(tf.float32, [None, None])
            self.neg_t_category = tf.placeholder(tf.float32, [None, None])

            self.partial_solution = tf.placeholder(tf.float32, (None, None))

            with tf.name_scope("embedding"):
                self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, 1, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.category_embeddings = tf.get_variable(name="category_embedding", shape=[config.n_category, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            pos_t_e_category = tf.matmul(self.pos_t_category, self.category_embeddings)
            neg_t_e_category = tf.matmul(self.neg_t_category, self.category_embeddings)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))
            clip_type = tf.assign(self.category_embeddings, tf.maximum(0., self.category_embeddings))

            temp = np.zeros(self.rel_embeddings.shape, dtype=bool)
            diag_r, diag_c = np.where(np.eye(config.hidden_size, dtype=bool))
            temp[:, diag_r, diag_c] = 1
            diag_rel = tf.assign(self.rel_embeddings, tf.where(temp, self.rel_embeddings, np.zeros(self.rel_embeddings.shape)))
            ident_rel = tf.assign(self.rel_embeddings, tf.where(temp, np.ones(self.rel_embeddings.shape, dtype='float32'),
                                                                np.zeros(self.rel_embeddings.shape, dtype='float32')))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)
            if config.clipCategory:
                clip_ops.append(clip_type)
            if config.diagonalR:
                clip_ops.append(diag_rel)
            if config.identityR:
                clip_ops.append(ident_rel)

            self.clip = tf.group(*clip_ops)

            pos_t_e_full = tf.expand_dims(pos_t_e_category, 1) + pos_t_e
            neg_t_e_full = tf.expand_dims(neg_t_e_category, 1) + neg_t_e

            self.pos = tf.reshape(tf.matmul(tf.matmul(pos_h_e, pos_r_e), tf.transpose(pos_t_e_full, perm=[0, 2, 1])), [-1])
            self.neg = tf.reshape(tf.matmul(tf.matmul(neg_h_e, neg_r_e), tf.transpose(neg_t_e_full, perm=[0, 2, 1])), [-1])

            temp = tf.matmul(pos_r_e, tf.transpose(pos_t_e_full, perm=[1, 2, 0]))
            transform = tf.matmul(tf.matmul(tf.transpose(temp, perm=[0, 2, 1]),
                                            tf.matrix_inverse(tf.matmul(temp, tf.transpose(temp, perm=[0, 2, 1])))), temp)
            self.project = tf.squeeze(tf.matmul(tf.expand_dims(self.partial_solution, 0), transform), axis=0)

            self.results = {
                'core_tensor': tf.squeeze(pos_r_e, axis=0),
                'item_embeddings': tf.squeeze(pos_t_e_full, axis=1),
                'entity_embeddings': tf.squeeze(self.ent_embeddings, axis=1),
                'relation_embeddings': self.rel_embeddings,
                'transform': tf.squeeze(transform, axis=0),
                'category_embeddings': self.category_embeddings
            }

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.loss = tf.reduce_sum(tf.abs(1 - self.pos) + tf.abs(self.neg)) + config.regularization * tf.reduce_sum(
                        pos_h_e + pos_t_e + neg_h_e + neg_t_e)
                self.predict = self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = self.pos
            else:
                with tf.name_scope("output"):
                    self.loss = tf.reduce_sum((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_sum(
                                    tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))
                self.predict = self.pos

            self.predict = self.project

    def model(self, config):
        return self.RESCOMCategory(config)


class NECTRModel(TensorModel):
    """
    Class implementing NECTR
    """

    @property
    def name(self):
        return 'NECTR'

    def init_vars(self):
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        super().init_vars()

        optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars_completion = optimizer1.compute_gradients(self.train_model.loss_completion)
        self.train_op_completion = optimizer1.apply_gradients(grads_and_vars_completion, global_step=self.global_step)

        optimizer2 = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars_recommendation = optimizer2.compute_gradients(self.train_model.loss_recommendation)
        self.train_op_recommendation = optimizer2.apply_gradients(grads_and_vars_recommendation, global_step=self.global_step)

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class NECTR(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            with tf.name_scope("embedding"):
                if config.path_pretrained_ent_embs:
                    pretrained_ent_embs = np.expand_dims(np.loadtxt(config.path_pretrained_ent_embs, delimiter=';', dtype=np.float32), axis=1)

                    self.ent_embeddings_no_train = tf.get_variable(name="ent_embedding_no_train", initializer=pretrained_ent_embs, trainable=False)
                    self.ent_embeddings_train = tf.get_variable(name="ent_embedding_train", initializer=np.random.randn(config.n_entity - len(pretrained_ent_embs), 1, config.hidden_size).astype(np.float32))
                    self.ent_embeddings = tf.concat([self.ent_embeddings_no_train, self.ent_embeddings_train], 0)
                else:
                    self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, 1, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            temp = np.zeros(self.rel_embeddings.shape, dtype=bool)
            diag_r, diag_c = np.where(np.eye(config.hidden_size, dtype=bool))
            temp[:, diag_r, diag_c] = 1
            diag_rel = tf.assign(self.rel_embeddings, tf.where(temp, self.rel_embeddings, np.zeros(self.rel_embeddings.shape)))
            ident_rel = tf.assign(self.rel_embeddings, tf.where(temp, np.ones(self.rel_embeddings.shape, dtype='float32'),
                                                                np.zeros(self.rel_embeddings.shape, dtype='float32')))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)
            if config.diagonalR:
                clip_ops.append(diag_rel)
            if config.identityR:
                clip_ops.append(ident_rel)

            self.clip = tf.group(*clip_ops)

            self.pos = tf.reshape(tf.matmul(tf.matmul(pos_h_e, pos_r_e), tf.transpose(pos_t_e, perm=[0, 2, 1])), [-1])
            self.neg = tf.reshape(tf.matmul(tf.matmul(neg_h_e, neg_r_e), tf.transpose(neg_t_e, perm=[0, 2, 1])), [-1])

            self.item_embeddings = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, np.array(range(config.n_item), dtype=np.int32)), axis=1)

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.loss_completion = tf.reduce_sum(tf.abs(1 - self.pos) + tf.abs(self.neg)) + config.regularization * tf.reduce_sum(
                        pos_h_e + pos_t_e + neg_h_e + neg_t_e)
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss_completion = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
            else:
                with tf.name_scope("output"):
                    if config.mean_loss:
                        self.loss_completion = tf.reduce_mean((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_mean(
                            tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))
                    else:
                        self.loss_completion = tf.reduce_sum((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_sum(
                            tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))

            self.loss_completion = config.nectr_lambda_completion * self.loss_completion

            self.results = {
                'item_embeddings': tf.squeeze(pos_t_e, axis=1),
                'entity_embeddings': tf.squeeze(self.ent_embeddings, axis=1),
                'relation_embeddings': self.rel_embeddings
            }

            ###########################################################################################################

            # Non-linearity
            # TODO Change to sparse placeholders and also use int type instead of float
            self.partial_solution = tf.placeholder(tf.float32, [None, config.n_item])
            self.complete_solution = tf.placeholder(tf.float32, [None, config.n_item])
            self.item_counts = tf.placeholder(tf.float32, [None, config.n_item])

            hidden_layer = self.partial_solution

            for i in range(config.nectr_n_hidden_layers):
                hidden_layer = tf.layers.dense(hidden_layer, config.nectr_n_neurons, activation=tf.nn.relu, name="hidden_layer_"+str(i))
            output_layer = tf.layers.dense(hidden_layer, config.hidden_size, activation=None, name="output_layer")

            if config.mean_loss:
                if config.nectr_poisson:
                    self.logits = tf.exp(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_mean(self.complete_solution * (-self.item_counts * tf.log(self.logits + 1e-9) + tf.reduce_sum(self.logits))), - tf.reduce_mean((1 - self.complete_solution) * (-self.item_counts * tf.log(1 - self.logits + 1e-9) + tf.reduce_sum(1 - self.logits))), name="loss_recommendation")
                else:
                    self.logits = tf.sigmoid(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_mean(self.complete_solution * tf.log(self.logits + 1e-9)), - tf.reduce_mean((1 - self.complete_solution) * tf.log(1 - self.logits + 1e-9)), name="loss_recommendation")
            else:
                if config.nectr_poisson:
                    self.logits = tf.exp(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_sum(self.complete_solution * (-self.item_counts * tf.log(self.logits + 1e-9) + tf.reduce_sum(self.logits))), - tf.reduce_sum((1 - self.complete_solution) * (-self.item_counts * tf.log(1 - self.logits + 1e-9) + tf.reduce_sum(1 - self.logits))), name="loss_recommendation")
                else:
                    self.logits = tf.sigmoid(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_sum(self.complete_solution * tf.log(self.logits + 1e-9)), - tf.reduce_sum((1 - self.complete_solution) * tf.log(1 - self.logits + 1e-9)), name="loss_recommendation")

            all_variables = tf.trainable_variables()
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=config.nectr_nn_regularization, scope=None)
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=config.nectr_nn_regularization, scope=None)
            if config.nectr_nn_regularization_type == 'l1':
                regularizer = l1_regularizer
            else:
                regularizer = l2_regularizer
            reg = tf.contrib.layers.apply_regularization(regularizer, all_variables)

            self.loss = self.loss_completion + self.loss_recommendation + reg

            self.predict = self.logits

    def model(self, config):
        return self.NECTR(config)

    def rank(self, partial_solution):
        with self.Manager() as mgr:
            self.init_vars()
            mgr.sess.run(tf.initialize_all_variables())
            self.saver.restore(mgr.sess, self.path_to_results + self.name + '_model.vec')

            feed_dict = {self.train_model.partial_solution: partial_solution}

            # tvars = tf.trainable_variables()
            # tvars_vals = mgr.sess.run(tvars)
            # for var, val in zip(tvars, tvars_vals):
            #     print(var.name, val)  # Prints the name of the variable alongside its value.

            step, complete_solution = mgr.sess.run(
                [self.global_step, self.train_model.logits], feed_dict)

        return complete_solution

    def train(self, data, partial_data=None, complete_data=None):
        tic = time.time()
        self.config.batch_size = len(data) // self.config.n_batch

        with self.Manager() as mgr:
            self.init_vars()
            mgr.sess.run(tf.initialize_all_variables())

            if self.config.retrain_model:
                self.saver.restore(mgr.sess, self.path_to_results + self.name + '_model.vec')

            def train_step_completion(param_values):

                feed_dict = {self.learning_rate: self.config.learning_rate}
                for i, p in enumerate(self.get_parameters()):
                    feed_dict[getattr(self.train_model, p)] = param_values[i]

                _, step, loss, pos = mgr.sess.run(
                    [self.train_op_completion, self.global_step, self.train_model.loss_completion, self.train_model.pos], feed_dict)

                if self.config.clipE or self.config.clipR:
                    _ = mgr.sess.run([self.train_model.clip])
                return loss

            def train_step_recommendation(param_values):
                if self.config.nectr_item_counts:
                    temp = param_values['complete_solution'].toarray()
                    temp[temp > 1] = 1
                    feed_dict = {
                        self.learning_rate: self.config.nectr_learning_rate,
                        self.train_model.partial_solution: param_values['partial_solution'].toarray(),
                        self.train_model.item_counts: param_values['complete_solution'].toarray(),
                        self.train_model.complete_solution: temp
                    }
                else:
                    feed_dict = {
                        self.learning_rate: self.config.nectr_learning_rate,
                        self.train_model.partial_solution: param_values['partial_solution'].toarray(),
                        self.train_model.complete_solution: param_values['complete_solution'].toarray()
                    }

                _, step, loss = mgr.sess.run(
                    [self.train_op_recommendation, self.global_step, self.train_model.loss_recommendation], feed_dict)
                return loss

            def train_step(batch, batch_solution):
                feed_dict = {self.learning_rate: self.config.learning_rate}
                for i, p in enumerate(self.get_parameters()):
                    feed_dict[getattr(self.train_model, p)] = batch[i]

                if self.config.nectr_item_counts:
                    temp = batch_solution['complete_solution'].toarray()
                    temp[temp > 1] = 1
                    feed_dict[getattr(self.train_model, 'partial_solution')] = batch_solution['partial_solution'].toarray()
                    feed_dict[getattr(self.train_model, 'item_counts')] = batch_solution['complete_solution'].toarray()
                    feed_dict[getattr(self.train_model, 'complete_solution')] = temp

                else:
                    feed_dict[getattr(self.train_model, 'partial_solution')] = batch_solution['partial_solution'].toarray()
                    feed_dict[getattr(self.train_model, 'complete_solution')] = batch_solution['complete_solution'].toarray()

                _, step, loss, pos = mgr.sess.run(
                    [self.train_op, self.global_step, self.train_model.loss, self.train_model.pos], feed_dict)

                if self.config.clipE or self.config.clipR:
                    _ = mgr.sess.run([self.train_model.clip])
                return loss

            loss_train = []
            loss_train_completion = []
            loss_train_recommendation = []
            item_embs_history = {}

            self.config.register('learning_rate')
            # Train for the completion loss for 'config.nectr_n_epoch_completion' epochs and
            # then train on the overall loss (completion and recommendation loss)
            print('Training the tensor factorization component (for the completion loss)...')
            for times in range(self.config.nectr_n_epoch_completion):
                loss = 0.0
                data = shuffle(data)
                for batch in self.DataHelper.get_batch(data, self.config.batch_size, self.config.negative2positive_ratio):
                    loss += train_step_completion(batch)
                    _ = tf.train.global_step(mgr.sess, self.global_step)
                print('Epoch=', times, 'loss=', loss)
                loss_train.append(loss)
                if self.config.adaptive_learning_rate and times > 0:
                    if loss_train[-1] > loss_train[-2]:
                        self.config.learning_rate = self.config.learning_rate / 2
                        print("Half the learning rate: l_rate = ", self.config.learning_rate)
            self.config.recall('learning_rate')

            print('Training both tensor factorization and neural network components (overall completion and recommendation loss)...')

            self.config.register('learning_rate')
            self.config.register('nectr_learning_rate')
            for times in self.config.epoch_range:
                if self.config.track_history:
                    item_embs_history[times] = mgr.sess.run(self.train_model.item_embeddings)

                if self.config.nectr_training_mode == TrainingMode.ALTERNATING:
                    loss_completion = 0.0
                    data = shuffle(data)
                    for batch in self.DataHelper.get_batch(data, self.config.batch_size, self.config.negative2positive_ratio):
                        loss_completion += train_step_completion(batch)
                        _ = tf.train.global_step(mgr.sess, self.global_step)
                    print('Epoch=', times, 'completion loss=', loss_completion)
                    loss_train_completion.append(loss_completion)
                    if self.config.adaptive_learning_rate and times > 0:
                        if loss_train_completion[-1] > loss_train_completion[-2]:
                            self.config.learning_rate = self.config.learning_rate / 2
                            print("Half the learning rate: l_rate = ", self.config.learning_rate)

                    loss_recommendation = 0.0
                    # Different batch_size for the recommendation step
                    # E.g., n_batch for completion step = 100, then batch_size = ~2K if len(data) = 200000
                    # If same n_batch is used for recommendation, then batch_size = 20 if len(data) = 2000
                    batch_size = partial_data.shape[0] // self.config.n_batch
                    for batch in self.DataHelper.get_batch_solution(partial_data, complete_data, batch_size):
                        loss_recommendation += train_step_recommendation(batch)
                        _ = tf.train.global_step(mgr.sess, self.global_step)
                    print('Epoch=', times, 'recommendation loss=', loss_recommendation)
                    loss_train_recommendation.append(loss_recommendation)
                    if self.config.adaptive_learning_rate and times > 0:
                        if loss_train_recommendation[-1] > loss_train_recommendation[-2]:
                            self.config.nectr_learning_rate = self.config.nectr_learning_rate / 2
                            print("Half the learning rate: l_rate = ", self.config.nectr_learning_rate)

                    loss_train.append(loss_completion + loss_recommendation)

                elif self.config.nectr_training_mode == TrainingMode.SIMULTANEOUS:
                    loss = 0.0
                    batch_size = partial_data.shape[0] // self.config.n_batch
                    for batch, batch_solution in zip(self.DataHelper.get_batch(data, self.config.batch_size, self.config.negative2positive_ratio), self.DataHelper.get_batch_solution(partial_data, complete_data, batch_size)):
                        loss += train_step(batch, batch_solution)
                        _ = tf.train.global_step(mgr.sess, self.global_step)
                    print('Epoch=', times, 'loss=', loss)
                    loss_train.append(loss)
                    if self.config.adaptive_learning_rate and times > 0:
                        if loss_train[-1] > loss_train[-2]:
                            self.config.learning_rate = self.config.learning_rate / 2
                            print("Half the learning rate: l_rate = ", self.config.learning_rate)

            self.config.recall('learning_rate')
            self.config.recall('nectr_learning_rate')
            self.saver.save(mgr.sess, self.path_to_results + self.name + '_model.vec')
            if self.config.track_history:
                with open(self.path_to_results + 'item_embeddings_history.pickle', 'wb') as fp:
                    pickle.dump(item_embs_history, fp)

        toc = time.time()
        print('Time taken for training: {} (in hours)'.format((toc - tic) / 3600))
        return loss_train


class NECTRCategoryModel(NECTRModel):
    """
    Class implementing NECTR_Category
    """

    @property
    def name(self):
        return 'NECTRCategory'

    class NECTRCategory(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            with tf.name_scope("embedding"):
                if config.path_pretrained_ent_embs:
                    pretrained_ent_embs = np.expand_dims(np.loadtxt(config.path_pretrained_ent_embs, delimiter=';', dtype=np.float32), axis=1)

                    self.ent_embeddings_no_train = tf.get_variable(name="ent_embedding_no_train", initializer=pretrained_ent_embs, trainable=False)
                    self.ent_embeddings_train = tf.get_variable(name="ent_embedding_train", initializer=np.random.randn(config.n_entity - len(pretrained_ent_embs), 1, config.hidden_size).astype(np.float32))
                    self.ent_embeddings = tf.concat([self.ent_embeddings_no_train, self.ent_embeddings_train], 0)
                else:
                    self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, 1, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            temp = np.zeros(self.rel_embeddings.shape, dtype=bool)
            diag_r, diag_c = np.where(np.eye(config.hidden_size, dtype=bool))
            temp[:, diag_r, diag_c] = 1
            diag_rel = tf.assign(self.rel_embeddings, tf.where(temp, self.rel_embeddings, np.zeros(self.rel_embeddings.shape)))
            ident_rel = tf.assign(self.rel_embeddings, tf.where(temp, np.ones(self.rel_embeddings.shape, dtype='float32'),
                                                                np.zeros(self.rel_embeddings.shape, dtype='float32')))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)
            if config.diagonalR:
                clip_ops.append(diag_rel)
            if config.identityR:
                clip_ops.append(ident_rel)

            self.clip = tf.group(*clip_ops)

            self.pos = tf.reshape(tf.matmul(tf.matmul(pos_h_e, pos_r_e), tf.transpose(pos_t_e, perm=[0, 2, 1])), [-1])
            self.neg = tf.reshape(tf.matmul(tf.matmul(neg_h_e, neg_r_e), tf.transpose(neg_t_e, perm=[0, 2, 1])), [-1])

            self.item_embeddings = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, np.array(range(config.n_item), dtype=np.int32)), axis=1)

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.loss_completion = tf.reduce_sum(tf.abs(1 - self.pos) + tf.abs(self.neg)) + config.regularization * tf.reduce_sum(
                        pos_h_e + pos_t_e + neg_h_e + neg_t_e)
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss_completion = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
            else:
                with tf.name_scope("output"):
                    if config.mean_loss:
                        self.loss_completion = tf.reduce_mean((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_mean(
                            tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))
                    else:
                        self.loss_completion = tf.reduce_sum((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_sum(
                            tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))

            self.loss_completion = config.nectr_lambda_completion * self.loss_completion

            self.results = {
                'item_embeddings': tf.squeeze(pos_t_e, axis=1),
                'entity_embeddings': tf.squeeze(self.ent_embeddings, axis=1),
                'relation_embeddings': self.rel_embeddings
            }

            # Non-linearity
            self.partial_solution = tf.placeholder(tf.float32, [None, config.n_item])
            self.complete_solution = tf.placeholder(tf.float32, [None, config.n_item])
            self.item_counts = tf.placeholder(tf.float32, [None, config.n_item])

            hidden_layer = tf.log(self.partial_solution+1)

            # Set the number of neurons in the first hidden layer to be equal to the number of item categories
            n_neurons = [config.nectr_n_neurons] * config.nectr_n_hidden_layers
            n_neurons[0] = config.n_category + 1  # one additional for the unknown category

            def weights_by_category(weights):
                temp = np.ones(weights.shape, dtype=bool)
                categories = set(range(0, config.n_category+1))
                items_with_category = []
                # Each item belongs to a category, for every other category, set the weights to 0
                for c in range(config.n_category):
                    temp[np.ix_(config.category2item[c], list(categories.difference({c})))] = 0
                    items_with_category += config.category2item[c]
                # Do the same for items with an unknown category
                items_without_category = list(set(range(config.n_item)).difference(set(items_with_category)))
                temp[np.ix_(items_without_category, list(range(config.n_category)))] = 0

                weights = tf.assign(weights, tf.where(temp, weights, np.zeros(weights.shape)))
                return weights

            for i in range(config.nectr_n_hidden_layers):
                if i == 0:
                    # First layer summarizes the categories
                    # layer_constraint = weights_by_category
                    layer_constraint = None
                else:
                    layer_constraint = None
                hidden_layer = tf.layers.dense(hidden_layer, n_neurons[i], activation=tf.nn.relu, name="hidden_layer_"+str(i), kernel_constraint=layer_constraint)
            output_layer = tf.layers.dense(hidden_layer, config.hidden_size, activation=None, name="output_layer")

            for c in range(config.n_category):
                item_embeddings_cat = tf.squeeze(
                    tf.nn.embedding_lookup(self.ent_embeddings, config.category2item[c]),axis=1)
                weight = tf.get_variable(name="cat_weigt"+str(c), shape=[config.hidden_size,config.hidden_size], dtype=tf.float32)
                step = tf.matmul(output_layer, weight)
                temp = tf.sigmoid(tf.matmul(step, tf.transpose(item_embeddings_cat)))
                if c == 0:
                    self.logits = temp
                else:
                    self.logits = tf.concat([self.logits,temp], axis=1)

            idx = np.arange(config.n_items_with_category, config.n_item)
            item_embeddings_cat = tf.squeeze(tf.nn.embedding_lookup(self.ent_embeddings, idx), axis=1)
            weight = tf.get_variable(name="cat_weigt" + str(config.n_category), shape=[config.hidden_size, config.hidden_size], dtype=tf.float32)
            step = tf.matmul(output_layer, weight)
            temp = tf.sigmoid(tf.matmul(step, tf.transpose(item_embeddings_cat)))
            self.logits = tf.concat([self.logits, temp], axis=1)

            if config.mean_loss:
                if config.nectr_poisson:
                    self.logits = tf.exp(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_mean(self.complete_solution * (-self.item_counts * tf.log(self.logits + 1e-9) + tf.reduce_sum(self.logits))), - tf.reduce_mean((1 - self.complete_solution) * (-self.item_counts * tf.log(1 - self.logits + 1e-9) + tf.reduce_sum(1 - self.logits))), name="loss_recommendation")
                else:
                    # self.logits = tf.sigmoid(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_mean(self.complete_solution * tf.log(self.logits + 1e-9)), - tf.reduce_mean((1 - self.complete_solution) * tf.log(1 - self.logits + 1e-9)), name="loss_recommendation")
            else:
                if config.nectr_poisson:
                    self.logits = tf.exp(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_sum(self.complete_solution * (-self.item_counts * tf.log(self.logits + 1e-9) + tf.reduce_sum(self.logits))), - tf.reduce_sum((1 - self.complete_solution) * (-self.item_counts * tf.log(1 - self.logits + 1e-9) + tf.reduce_sum(1 - self.logits))), name="loss_recommendation")
                else:
                    # self.logits = tf.sigmoid(tf.matmul(output_layer, tf.transpose(self.item_embeddings)))
                    self.loss_recommendation = tf.add(-tf.reduce_sum(self.complete_solution * tf.log(self.logits + 1e-9)), - tf.reduce_sum((1 - self.complete_solution) * tf.log(1 - self.logits + 1e-9)), name="loss_recommendation")

            self.all_variables = tf.trainable_variables()
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=config.nectr_nn_regularization, scope=None)
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=config.nectr_nn_regularization, scope=None)
            if config.nectr_nn_regularization_type == 'l1':
                regularizer = l1_regularizer
            else:
                regularizer = l2_regularizer
            reg = tf.contrib.layers.apply_regularization(regularizer, self.all_variables)

            self.loss = self.loss_completion + self.loss_recommendation + reg

            self.predict = self.logits

    def model(self, config):
        return self.NECTRCategory(config)


class HolEModel(TensorModel):
    """
    Class implementing HolE
    """

    @property
    def name(self):
        return 'HolE'

    def get_parameters(self, mode='train'):
        if mode == 'test':
            return ['pos_h', 'pos_t', 'pos_r']
        return ['pos_h', 'pos_t', 'pos_r', 'neg_h', 'neg_t', 'neg_r']

    class HolE(object):

        def __init__(self, config):

            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])

            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])

            with tf.name_scope("embedding"):
                self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[config.n_entity, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[config.n_relation, config.hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

                pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
                pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
                pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
                neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
                neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
                neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            clip_ent = tf.assign(self.ent_embeddings, tf.maximum(0., self.ent_embeddings))
            clip_rel = tf.assign(self.rel_embeddings, tf.maximum(0., self.rel_embeddings))

            clip_ops = []
            if config.clipE:
                clip_ops.append(clip_ent)
            if config.clipR:
                clip_ops.append(clip_rel)

            self.clip = tf.group(*clip_ops)

            self.pos = tf.reduce_sum(tf.matmul(pos_r_e, tf.transpose(tf.real(tf.ifft(tf.fft(tf.complex(pos_h_e, tf.zeros(tf.shape(pos_h_e)))) * tf.fft(tf.complex(pos_t_e, tf.zeros(tf.shape(pos_t_e)))))))), 1, keep_dims=True)
            self.neg = tf.reduce_sum(tf.matmul(neg_r_e, tf.transpose(tf.real(tf.ifft(tf.fft(tf.complex(neg_h_e, tf.zeros(tf.shape(neg_h_e)))) * tf.fft(tf.complex(neg_t_e, tf.zeros(tf.shape(neg_t_e)))))))), 1, keep_dims=True)

            self.results = {
                'relation_embeddings': self.rel_embeddings,
                'item_embeddings': self.ent_embeddings
            }

            if config.loss_type == 'l1':
                with tf.name_scope("output"):
                    self.loss = tf.reduce_sum(tf.abs(1 - self.pos) + tf.abs(self.neg)) + config.regularization * tf.reduce_sum(pos_h_e + pos_t_e + neg_h_e + neg_t_e)
                self.predict = self.pos
            elif config.loss_type == 'cross_entropy':
                with tf.name_scope("output"):
                    self.loss = - tf.reduce_sum(tf.log(tf.sigmoid(self.pos) + np.finfo(np.float32).eps) + tf.log(
                        1 - tf.sigmoid(self.neg) + np.finfo(np.float32).eps))
                self.predict = self.pos
            else:
                with tf.name_scope("output"):
                    self.loss = tf.reduce_sum((1 - self.pos) ** 2 + self.neg ** 2) + config.regularization * tf.reduce_sum(tf.square(pos_h_e) + tf.square(pos_t_e) + tf.square(neg_h_e) + tf.square(neg_t_e))
                self.predict = self.pos

    def model(self, config):
        return self.HolE(config)
