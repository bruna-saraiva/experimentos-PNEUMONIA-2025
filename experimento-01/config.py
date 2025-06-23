from hyperopt import hp
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from hyperopt import STATUS_OK, STATUS_FAIL
import os
import tensorflow as tf

img_width = 180
img_height =  180
batch_size = 8 #batch_size para o treino

batch_size_val = 1
attention_module = 'Squeeze and Excitation'
epochs = 20

RESULTS_DIR = "results/" #pasta para salvar os resultados dos treinamentos

train_data_dir = "database/split1/train"
validation_data_dir = "database/split1/val"
test_data_dir = "database/split1/test"

num_classes_exp = 3

eps = 1.1e-5


space = {
    'num_blocks': hp.choice('num_blocks', [2,3]), #3
    'num_layers_per_block' : hp.choice('num_layers_per_block', [2,3]), #2
    'growth_rate': hp.choice('growth_rate', [16,32]), #32
    'dropout_rate' : hp.uniform('dropout_rate', 0.2, 0.35),
    'compress_factor' : hp.choice('compress_factor', [0.5, 1]), #0.5
    'num_filters' : hp.choice('num_filters', [32,64]), #64
    'se_config': hp.choice('se_config', [
        'nenhum',
        'apenas_topo',
        'apenas_transicao',
        'apenas_H',
        'transicao_e_H',
        'transicao_e_topo',
        'H_e_topo',
        'todas']),
    # para os modelos transfer learning
    'base_model': hp.choice('base_model', ['efficientnet', 'resnet', 'vgg']),
    'freeze_layers': hp.choice('freeze_layers', [True, False]),  # Fine-tuning (False) ou Feature Extraction (True)
}

# configurando memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limita otimizações XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'  