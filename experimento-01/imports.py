# import numpy as np
import sklearn as sk
import os
from sklearn.model_selection import train_test_split
import random
import glob
# from sklearn.metrics import accuracy_score # deu problema no ultimo treino e agora adicionei isso para que a variavel acc funcione 

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras import layers
from keras.optimizers import SGD, Adam
from sklearn import metrics
from keras import metrics
from keras.models import load_model, Model
from tensorflow.keras.utils import plot_model

from hyperopt import hp
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt import STATUS_OK, STATUS_FAIL

import json
from bson import json_util

import keras.backend as K
from sklearn.utils import class_weight
from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score

import traceback
import pickle
import uuid

# from datetime import datetime

import visualkeras
import wandb
from wandb.integration.keras import WandbCallback  # Esta é a importação correta agora

from sklearn.metrics import confusion_matrix

# import seaborn as sns
# # import time
# import matplotlib.pyplot as plt

