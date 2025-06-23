from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from keras.optimizers import SGD, Adam

from sklearn import metrics
from keras import metrics

from tensorflow.keras.layers import Conv2D

def get_transfer_learning_model(base_model_name, input_shape, num_classes, dropout_rate=0.5, freeze_layers=True):
    """Adaptado para aceitar imagens em grayscale (1 canal)."""
    
    # Verifica se a imagem é grayscale (1 canal)
    if input_shape[-1] == 1:
        # Cria uma camada de entrada com 1 canal
        inputs = Input(shape=input_shape)
        # Converte grayscale para "pseudo-RGB" replicando o canal 3 vezes
        x = Conv2D(3, (1, 1), padding='same', activation='linear', name='grayscale_to_rgb')(inputs)
    else:
        inputs = Input(shape=input_shape)
        x = inputs

    # Seleciona a arquitetura base (agora com input_shape=(height, width, 3))
    if base_model_name == 'efficientnet':
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3),  # Força 3 canais
            pooling=None
        )
    elif base_model_name == 'resnet':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling=None
        )
    elif base_model_name == 'vgg':
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling=None
        )
    elif base_model_name == 'mobilenet':
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(input_shape[0], input_shape[1], 3),
            pooling=None
        )
    else:
        raise ValueError(f"Arquitetura não suportada: {base_model_name}")

    # Congela as camadas da base se necessário
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False

    # Conecta ao modelo base
    x = base_model(x, training=not freeze_layers)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            'accuracy',
            metrics.Recall(thresholds=0.5, class_id=0, name='r_normal'),
            metrics.Recall(thresholds=0.5, class_id=1, name='r_covid'),
            metrics.Recall(thresholds=0.5, class_id=2, name='r_viral')
        ]
    )
    
    return model