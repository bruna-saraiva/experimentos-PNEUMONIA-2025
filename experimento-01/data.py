from config import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight



#DataGenerator utilizado para fazer o augmentation on the batch
datagen = ImageDataGenerator(rescale=1./255, # era assim antes de adicionar o clahe
    # featurewise_center=True,
    rotation_range=5,
    width_shift_range=.05,
    height_shift_range=.05,
    # shear_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="constant"
    # preprocessing_function = pre_process,
)
 #Generators

validgen = ImageDataGenerator(rescale=1./255, 
                            #   featurewise_center=True,
                            # preprocessing_function = pre_process,
                            ) 

train_gen = datagen.flow_from_directory( 
    train_data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale', 
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True)

val_gen = validgen.flow_from_directory( 
    validation_data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size_val,
    class_mode="categorical",
    shuffle=True)

test_gen = validgen.flow_from_directory( 
    test_data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size_val,
    class_mode="categorical",
    shuffle=False)


# Calcule os pesos automaticamente a partir dos dados

class_weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weight = dict(enumerate(class_weights))

#pega a quantidade de amostras de cada generator
train_samples = len(train_gen.filenames)
validation_samples = len(val_gen.filenames)
test_samples = len(test_gen.filenames)



# tf.keras.backend.clear_session() # ?

