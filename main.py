from datahelper import DataHelper
from cross_validation import CrossValidation, Mask
import models

# Define paths to data
PATH_TO_DATA = './data/'
PATH_TO_RESULTS = PATH_TO_DATA + 'results/'

# Define model parameters
model = models.NECTRModel# [NMF, TransE, TransD, TransR, TransH, RESCOM]
parameters = [{'name': 'hidden_size', 'min': 15, 'max': 15, 'step': 15},
              {'name': 'n_epoch', 'min': 2, 'max': 2, 'step': 2},
              {'name': 'mask', 'value': Mask.RANDOM},
              # {'name': 'learning_rate', 'value': 0.01},
              # (parameters for NECTR)
              # {'name': 'nectr_n_hidden_layers', 'min': 1, 'max': 2, 'step': 1},
              # {'name': 'nectr_n_neurons', 'min': 5, 'max': 45, 'step': 10}]
              # {'name': 'nectr_poisson', 'value': True},
              # {'name': 'nectr_item_counts', 'value': True},
              # {'name': 'nectr_train_tf_on_solutions', 'value': False},
              # {'name': 'nectr_learning_rate', 'value': 0.1},
              # {'name': 'nectr_nn_regularization_type', 'value': 'l1'},
              # {'name': 'nectr_nn_regularization', 'type': 'exp', 'min': 1e-2, 'max': 1e-2},
              # {'name': 'nectr_lambda_completion', 'type': 'exp', 'min': 2e-2, 'max': 2e-0},
              {'name': 'nectr_n_epoch_completion', 'min': 2, 'max': 2, 'step': 2}]


# Setup DataHelper utils
DataHelper.setup(PATH_TO_DATA)

# Setup cross validation
cv = CrossValidation(datahelper=DataHelper, parameters=parameters)

# Run the cross validation pipeline
cv.run_pipeline(model, PATH_TO_RESULTS, plot=False)
