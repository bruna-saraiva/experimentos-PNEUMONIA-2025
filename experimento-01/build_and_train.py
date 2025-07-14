from customized_model import *
from transfer_learning_model import *

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def build_and_train(hype_space):
    print (hype_space)

# Para usar o customizado é só comentar o transfer learning e descomentar o customizado
# Modelo customizado 
    model_final = get_model(input_shape=(img_width, img_height, 1), # rgb: 3 ; grayscale: 1
            num_blocks = int(hype_space['num_blocks']),
            num_layers_per_block = int(hype_space['num_layers_per_block']),
            growth_rate = int(hype_space['growth_rate']),
            dropout_rate = hype_space['dropout_rate'],
            compress_factor = hype_space['compress_factor'],
            num_filters = hype_space['num_filters'],
            num_classes = num_classes_exp,
            se_config=hype_space['se_config'])
    
# # Modelo Transfer Learning 
#     model_final = get_transfer_learning_model(
#                 base_model_name=hype_space['base_model'],
#                 input_shape=(img_width, img_height, 1),
#                 num_classes=num_classes_exp,
#                 dropout_rate=hype_space['dropout_rate'],
#                 freeze_layers=hype_space['freeze_layers']
#             )

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