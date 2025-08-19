import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import save_model, load_model
import time

# Configurações do experimento
RESULTS_DIR = "holdout_results/"
NUM_SPLITS = 1
EPOCHS = 20
IMG_WIDTH = 180
IMG_HEIGHT = 180
BATCH_SIZE = 8
BATCH_SIZE_VAL = 1
EPS = 1.1e-5

# Cria a pasta de resultados automaticamente se não existir
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Espaço de hiperparâmetros predefinido
HYPERSPACE = {
    'num_blocks': 3,
    'num_layers_per_block': 2,
    'growth_rate': 16,
    'dropout_rate': 0.25,
    'compress_factor': 0.5,
    'num_filters': 64,
    'se_config': 'apenas_H'
}

def get_num_classes_from_directory(directory):
    """Detecta automaticamente o número de classes baseado nas subpastas"""
    if not os.path.exists(directory):
        return 0
    
    # Lista todas as subpastas (cada uma representa uma classe)
    classes = [d for d in os.listdir(directory) 
               if os.path.isdir(os.path.join(directory, d)) and not d.startswith('.')]
    
    return len(classes)

def validate_split_consistency(split_num, train_dir, val_dir, test_dir):
    """Valida se todos os diretórios do split têm o mesmo número de classes"""
    print(f"Validando split {split_num}...")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    print(f"  Test: {test_dir}")
    
    train_classes = get_num_classes_from_directory(train_dir)
    val_classes = get_num_classes_from_directory(val_dir)
    test_classes = get_num_classes_from_directory(test_dir)
    
    print(f"Split {split_num} - Classes detectadas: Train={train_classes}, Val={val_classes}, Test={test_classes}")
    
    if train_classes == 0 or val_classes == 0 or test_classes == 0:
        raise ValueError(f"Algum diretório do split {split_num} está vazio ou não existe")
    
    if train_classes == val_classes == test_classes:
        return train_classes
    else:
        raise ValueError(f"Inconsistência no Split {split_num}: Train={train_classes}, Val={val_classes}, Test={test_classes}")

def run_holdout_experiment():
    """Executa o experimento holdout com 5 splits"""
    
    all_results = {}
    
    for split_num in range(1, NUM_SPLITS + 1):
        print(f"\n{'='*50}")
        print(f"PROCESSANDO SPLIT {split_num}/{NUM_SPLITS}")
        print(f"{'='*50}")
        
        # Configura os paths para o split atual - SOMENTE ESTES PATHS SERÃO USADOS
        base_path = f"database/split{split_num}/"
        train_data_dir = base_path + "train"
        validation_data_dir = base_path + "val"
        test_data_dir = base_path + "test"
        
        print(f"Procurando diretórios:")
        print(f"  Train: {train_data_dir} - Existe: {os.path.exists(train_data_dir)}")
        print(f"  Val: {validation_data_dir} - Existe: {os.path.exists(validation_data_dir)}")
        print(f"  Test: {test_data_dir} - Existe: {os.path.exists(test_data_dir)}")
        
        # Verifica se os diretórios existem
        if not all(os.path.exists(path) for path in [train_data_dir, validation_data_dir, test_data_dir]):
            print(f"⚠️  Diretórios do split {split_num} não encontrados. Pulando...")
            continue
        
        try:
            # Valida consistência do split
            num_classes = validate_split_consistency(
                split_num, 
                train_data_dir, 
                validation_data_dir, 
                test_data_dir
            )
            
            print(f"✅ Split {split_num} validado - {num_classes} classes")
            
            # Executa o treinamento e avaliação
            result = train_and_evaluate_split(
                split_num, 
                train_data_dir, 
                validation_data_dir, 
                test_data_dir,
                num_classes
            )
            
            all_results[f"split_{split_num}"] = result
            
        except ValueError as e:
            print(f"❌ Erro no Split {split_num}: {e}")
            continue
        except Exception as e:
            print(f"❌ Erro inesperado no Split {split_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Limpa a sessão do TensorFlow para liberar memória
            tf.keras.backend.clear_session()
            time.sleep(2)  # Pequena pausa para garantir limpeza completa
    
    # Salva todos os resultados
    if all_results:
        save_final_results(all_results)
    else:
        print("❌ Nenhum split foi processado com sucesso!")
    
    return all_results

def train_and_evaluate_split(split_num, train_dir, val_dir, test_dir, num_classes):
    """Treina e avalia o modelo para um split específico"""
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.utils.class_weight import compute_class_weight
    
    print(f"📊 Configurando generators para split {split_num}...")
    
    # DataGenerator para treino com augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=.05,
        height_shift_range=.05,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="constant"
    )
    
    # DataGenerator para validação e teste
    validgen = ImageDataGenerator(rescale=1./255)
    
    # Generators - MOSTRA INFORMAÇÕES DETALHADAS
    print(f"  Carregando treino...")
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    
    print(f"  Carregando validação...")
    val_gen = validgen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE_VAL,
        class_mode="categorical",
        shuffle=True
    )
    
    print(f"  Carregando teste...")
    test_gen = validgen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=BATCH_SIZE_VAL,
        class_mode="categorical",
        shuffle=False
    )
    
    print(f"✅ Generators configurados:")
    print(f"  Train: {len(train_gen.filenames)} imagens, {len(train_gen.class_indices)} classes")
    print(f"  Val: {len(val_gen.filenames)} imagens, {len(val_gen.class_indices)} classes")
    print(f"  Test: {len(test_gen.filenames)} imagens, {len(test_gen.class_indices)} classes")
    print(f"  Nomes das classes: {list(train_gen.class_indices.keys())}")
    
    # Calcula class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_gen.classes), 
        y=train_gen.classes
    )
    class_weight = dict(enumerate(class_weights))
    
    # Constrói o modelo
    from customized_model import get_model
    
    print("🏗️  Construindo modelo...")
    model = get_model(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
        num_blocks=int(HYPERSPACE['num_blocks']),
        num_layers_per_block=int(HYPERSPACE['num_layers_per_block']),
        growth_rate=int(HYPERSPACE['growth_rate']),
        dropout_rate=HYPERSPACE['dropout_rate'],
        compress_factor=HYPERSPACE['compress_factor'],
        num_filters=HYPERSPACE['num_filters'],
        num_classes=num_classes,
        se_config=HYPERSPACE['se_config']
    )
    
    # Callbacks
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    
    split_results_dir = os.path.join(RESULTS_DIR, f"split_{split_num}")
    if not os.path.exists(split_results_dir):
        os.makedirs(split_results_dir)
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(split_results_dir, 'best_model.keras'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=1,
        mode='auto'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1
    )
    
    # Treinamento
    print("🚀 Iniciando treinamento...")
    start_time = time.time()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    training_time = time.time() - start_time
    
    # Avaliação
    print("📈 Avaliando modelo...")
    preds = model.predict(test_gen, steps=len(test_gen), verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    
    # Métricas
    acc = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(test_gen.class_indices.keys()),
        yticklabels=list(test_gen.class_indices.keys())
    )
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão - Split {split_num}')
    
    # Salva a matriz de confusão
    cm_filename = os.path.join(split_results_dir, 'confusion_matrix.png')
    plt.savefig(cm_filename)
    plt.close()
    
    # Salva o modelo final
    final_model_path = os.path.join(split_results_dir, 'final_model.keras')
    save_model(model, final_model_path)
    
    # Prepara resultados
    result = {
        'split_number': split_num,
        'num_classes': num_classes,
        'class_names': list(test_gen.class_indices.keys()),
        'hyperparameters': HYPERSPACE,
        'accuracy': float(acc),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'training_time_seconds': float(training_time),
        'epochs_trained': len(history.history['loss']),
        'data_execucao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': final_model_path,
        'confusion_matrix_path': cm_filename
    }
    
    # Salva resultados do split
    split_result_path = os.path.join(split_results_dir, 'results.json')
    with open(split_result_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"✅ Split {split_num} concluído - {num_classes} classes - Acurácia: {acc:.4f}")
    
    return result

def save_final_results(all_results):
    """Salva todos os resultados em um arquivo JSON final"""
    
    # Calcula métricas agregadas
    accuracies = [result['accuracy'] for result in all_results.values()]
    
    final_results = {
        'experiment_type': 'holdout',
        'num_splits_completed': len(all_results),
        'hyperparameters_used': HYPERSPACE,
        'split_results': all_results,
        'aggregated_metrics': {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
        },
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Salva resultados finais
    final_path = os.path.join(RESULTS_DIR, 'holdout_final_results.json')
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\n🎯 EXPERIMENTO HOLDOUT CONCLUÍDO!")
    print(f"📊 Acurácia média: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"📈 Melhor split: {np.max(accuracies):.4f}")
    print(f"📉 Pior split: {np.min(accuracies):.4f}")
    print(f"📁 Resultados salvos em: {final_path}")

if __name__ == "__main__":
    
    # Configura memory growth para GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Executa o experimento holdout
    start_time = time.time()
    print("🎯 INICIANDO EXPERIMENTO HOLDOUT")
    print("📁 Usando SOMENTE os paths dos splits (database/splitX/)")
    print("🚫 Ignorando quaisquer paths hardcoded antigos")
    
    results = run_holdout_experiment()
    total_time = time.time() - start_time
    
    print(f"\n⏰ Tempo total do experimento: {total_time:.2f} segundos")