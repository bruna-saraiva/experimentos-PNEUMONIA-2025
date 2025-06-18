
from data import *
from visualization import *
# import numpy as np
# import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras import layers
from keras.optimizers import SGD, Adam
from sklearn import metrics
from keras import metrics
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        
        # Handle InputLayer specially
        if isinstance(layer, tf.keras.layers.InputLayer):
            # For InputLayer, use the model's input shape
            out_shape = (batch_size,) + model.input_shape[1:]
        else:
            # For other layers, try to get output_shape
            try:
                out_shape = layer.output_shape
                if isinstance(out_shape, list):
                    out_shape = out_shape[0]
            except AttributeError:
                continue  # Skip layers without output_shape
        
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory



def se_block(input_tensor, ratio=8, name=None):
    """
    Squeeze-and-Excitation block melhorado para TensorFlow 2.x/Keras.
    
    Args:
        input_tensor: Tensor de entrada (feature map).
        ratio: Fator de redução de canais (default 8 como no paper original).
        name: Prefixo para nomes das camadas (opcional).
    
    Returns:
        Tensor com atenção recalibrada.
    """
    # Obter número de canais/filtros
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D(name=f'{name}_gap' if name else None)(input_tensor)
    se = layers.Reshape(se_shape, name=f'{name}_reshape' if name else None)(se)
    
    # Excitation: Two FC layers with ReLU and Sigmoid
    se = layers.Dense(filters // ratio, 
                     activation='relu',
                     kernel_initializer='he_normal',
                     use_bias=False,
                     name=f'{name}_fc1' if name else None)(se)
    se = layers.Dense(filters, 
                     activation='sigmoid',
                     kernel_initializer='he_normal',
                     use_bias=False,
                     name=f'{name}_fc2' if name else None)(se)
    
    # Scale: Multiply input with excitation weights
    x = layers.Multiply(name=f'{name}_scale' if name else None)([input_tensor, se])
    
    return x


def H( inputs, num_filters , dropout_rate,use_se): #adicionando use_se por causa do opt
    x = layers.BatchNormalization( epsilon=eps )( inputs )
    x = layers.Activation('relu')(x)

    out_conv = []
    for i in [(1,1),(3,3),(5,5),(0,0)]:
        p = x
        if i == (1,1):
                p = layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p)
                out_conv.append(layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p))
        elif i == (0,0):
                p = layers.MaxPool2D(pool_size=(2, 2), padding="same",strides=(1,1))(p)
                out_conv.append(layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p))
        else:
                p = layers.Conv2D(num_filters, (1,1), padding="same",activation="relu")(p)
                p = layers.SeparableConv2D(num_filters, i, padding="same",activation="relu")(p)
                out_conv.append(layers.SeparableConv2D(num_filters, i, padding="same",activation="relu")(p))
                
    
    x = layers.concatenate(out_conv, axis = -1)
    # Adicionando o SE condicionalmente
    if use_se:
        x = se_block(x, ratio=8, name=None)

    x = layers.Dropout(rate=dropout_rate )(x)
    return x

def transition(inputs, num_filters , compression_factor , dropout_rate, use_se):
    # compression_factor is the 'θ'
    x = layers.BatchNormalization( epsilon=eps )(inputs)
    x = layers.Activation('relu')(x)
    num_feature_maps = inputs.shape[1] # The value of 'm'

    x = layers.Conv2D(int(np.floor(num_feature_maps * compression_factor)) ,
                        kernel_size=(1, 1), use_bias=False, padding='same' ,
                        kernel_initializer='he_normal')(x)
    x = layers.Dropout(rate=dropout_rate)(x)

    # adicionando atencao SE condicionalmente
    if use_se:
        x = se_block(x, ratio=8, name=None)

    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x

def dense_block( inputs, num_layers, num_filters, growth_rate , dropout_rate,block_idx,use_se_in_H ):
    for i in range(num_layers): # num_layers is the value of 'l'
        conv_outputs = H(inputs, num_filters , dropout_rate,use_se=use_se_in_H ) # por causa do use_s em H
        inputs = layers.Concatenate()([conv_outputs, inputs])
        num_filters += growth_rate # To increase the number of filters for each layer.
    return inputs, num_filters

def get_model(input_shape,
           num_blocks,
           num_layers_per_block,
           growth_rate,
           dropout_rate,
           compress_factor,
           num_filters,
           num_classes,
           se_config): # passamos se_config para definir ele
    
    # Determinar onde colocar os Se_blocks baseado na combinação do opt
    use_se_in_H = se_config in ['apenas_H','transicao_e_H','H_e_topo', 'todas']
    use_se_in_transition = se_config in ['apenas_transicao', 'transicao_e_H', 'transicao_e_topo', 'todas']
    use_se_in_final = se_config in ['apenas_topo','transicao_e_topo','H_e_topo','todas']

    inputs = layers.Input( shape=input_shape )
    x = layers.Conv2D( num_filters , kernel_size=( 3 , 3 ) , padding="same", use_bias=False, kernel_initializer='he_normal')( inputs )
    for i in range( num_blocks ):
        x, num_filters = dense_block(x, num_layers_per_block , num_filters, growth_rate , dropout_rate,block_idx=i,use_se_in_H=use_se_in_H)
        x = transition(x, num_filters , compress_factor , dropout_rate,use_se=use_se_in_transition)
        
    # x = cbam_block(x, ratio=8, name="cbam_final")
    # x = se_block(x,ratio=8, name="se_final")
    if use_se_in_final:
        x = se_block(x, ratio=8,name="se_final")
    x = layers.GlobalAveragePooling2D()( x )
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense( num_classes )( x )
    outputs = layers.Activation( 'softmax' )( x )

    model = Model( inputs , outputs )
    
    model.compile( loss='categorical_crossentropy' ,optimizer=Adam(learning_rate=0.001),
                    metrics=[ 'accuracy',
                              metrics.Recall(thresholds=0.5, class_id=0,name='r_normal'),
                              metrics.Recall(thresholds=0.5, class_id=1,name='r_covid'),
                              metrics.Recall(thresholds=0.5, class_id=2,name='r_viral')])
    return model



def build_and_train(hype_space):
    print (hype_space)

    model_final = get_model(input_shape=(img_width, img_height, 1), # rgb: 3 ; grayscale: 1
            num_blocks = int(hype_space['num_blocks']),
            num_layers_per_block = int(hype_space['num_layers_per_block']),
            growth_rate = int(hype_space['growth_rate']),
            dropout_rate = hype_space['dropout_rate'],
            compress_factor = hype_space['compress_factor'],
            num_filters = hype_space['num_filters'],
            num_classes = num_classes_exp,
            se_config=hype_space['se_config'])
# ----------------------------------------------------------------------------
    model_size = keras_model_memory_usage_in_bytes(model = model_final,
                       batch_size = batch_size)
    model_size = model_size/1000000000

    #print("Model size: " + str(model_size) )
    if (model_size > 12.5):
        model_name = "model_" + str(uuid.uuid4())[:5]
        result = {
            'space': hype_space,
            'status': STATUS_FAIL
        }
        # wandb.finish()
        return model_final, model_name, result
    #-------------------------------------------------------------------------
    # Para carregar pesos pre-existentes
# Codigo mais recente que verifica se existe 
    # weights_file = 'weights_best_etapa1.keras'
    # if os.path.exists(weights_file):
    #     print("Carregando pesos pré-existentes...")
    #     model_final = load_model(weights_file)
    # else:
    #     print("Nenhum peso encontrado. Treinando do zero...")

# Codigo mais antigo que nao verifica a existencia de modelo
    # model_final = load_model('weights_best_etapa1.keras')

# ----------------------------------------------------------------------------
    #inicio da fase de treino
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=7,verbose=1, mode='auto')
    checkpoint = ModelCheckpoint('weights_best_etapa1.keras', monitor='val_loss',verbose=1,
                                 save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    
    # Troquei wandbcallback() pelo metricsLogger e modelcheckpoint
    # Treino
    start_time = time.time()
    history = model_final.fit(train_gen,
                    epochs=epochs,
                    # steps_per_epoch=int(train_samples/batch_size),
                    validation_data=val_gen,
                    # validation_steps=batch_size_val,
                    class_weight = class_weight,
                    # adicionando wandb aos callbacks
                    verbose=1, callbacks=[early_stopping,checkpoint,reduce_lr,
                                        # WandbMetricsLogger(), # Loga métricas automaticamente
                                        # WandbModelCheckpoint("wandb_model.keras")  # Salva o modelo no wandb .keras
    ])
    # Avaliação
    preds = model_final.predict(test_gen, test_samples) #realiza o teste de classificação das imagens na rede
    y_pred = np.argmax(preds, axis=1)
    #print(classification_report(test_gen.classes, y_pred))#, target_names=target_names))
    acc = accuracy_score(test_gen.classes, y_pred) #calcula o acurácia era metrics.accuracy_score....
    class_report = classification_report(test_gen.classes, y_pred, output_dict=True)#, target_names=target_names)

    training_time = time.time() - start_time
    
    # gerando matriz de confusao
    cm = confusion_matrix(test_gen.classes,y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues',
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys()
                )
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')


    model_name = "model_{}_{}".format(str(acc), str(uuid.uuid4())[:5])
    # plot_model(model_final, to_file= RESULTS_DIR + model_name + '_plot.png', show_shapes=True, show_layer_names=True)

    # salvando matriz de confusao
    cm_filename = RESULTS_DIR + model_name + '_confusion_matrix.png'
    plt.savefig(cm_filename)
    plt.close()

    plot_training_history(history, hype_space, model_name)

    history_data = {
    'accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
}
    time_data = {
        'total_time_seconds': training_time,
        'time_per_epoch_seconds': training_time/epochs
    }

    result = {
        'history': history_data,
        'epoch': epochs,
        'batch_treino' : batch_size,
        'batch_teste' : batch_size_val,
        'loss': 1-acc,
        'acurracy': acc,
        'report': class_report,
        'attention_module': hype_space['se_config'],
        'confusion_matrix': cm.tolist(), #salvando a cm no json
        # 'acurracy_p_1': acc_p_1,
        # 'report_p_1': class_report_p_1,
        'model_name': model_name,
        'space': hype_space,
        'status': STATUS_OK,
        'data_execucao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Formato: Ano-Mês-Dia Hora:Minuto:Segundo
        'time_data': time_data
    }


    return model_final, model_name, result