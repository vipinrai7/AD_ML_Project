from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def predictors(name):
    p_model = Sequential();
    jsonfile = open('model_mac2_neural1.json','r')
    model_json = jsonfile.read()
    p_model = model_from_json(model_json)
    p_model.load_weights('first_try_mac3_neural1.h5')
    p_model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    img = image.load_img(name, target_size=(150,150))
    x=image.img_to_array(img)
    x=x.reshape((1,)+x.shape)
    test_datagen = ImageDataGenerator(rescale=1. /255)
    m=test_datagen.flow(x,batch_size=1)


    preds = p_model.predict_generator(m,1,verbose=1)

    normal=preds[0] * 100
    alzhiemers = 100 - normal

    return normal[0],alzhiemers[0]

