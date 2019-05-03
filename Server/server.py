import keras
from keras.preprocessing.image import img_to_array
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib
from scipy import misc
import os

import io
import flask
from flask import send_file,  abort

from Unet import unet_sigmoid
from helpers import load_model,prepare_image_for_model, prediction_to_image, prepare_image_for_model,resize_img,prediction_to_mask


# initialise Flask application and Keras model 
app = flask.Flask(__name__)
predictedImageName = 'predicted.png'


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/lastPredicted')
def lastPredictedGet():
    img = misc.imread(predictedImageName)
    return send_file(predictedImageName,mimetype='image/jpeg')
	
	
	
@app.route('/predict', methods=["POST"])
def predict():

    if os.path.exists("name.png"):
        os.remove("name.png")
    # initialise the data dictionary that will be returned 
    # from the view
    print("New request")
    data = {'success': False}

    if flask.request.method == "POST":
        if flask.request.files.get('image'):
            # read the image in PIL format 
            imageFormRequest = flask.request.files['image'].read()
            row_img = Image.open(io.BytesIO(imageFormRequest))

            smallImage = resize_img(np.array(row_img))
            dataForModel =  prepare_image_for_model(smallImage)

            with graph.as_default():
                preds = model.predict(dataForModel)
                image_with_mask = prediction_to_image(preds,smallImage,0.20 )
                matplotlib.image.imsave(predictedImageName, image_with_mask)

                return send_file(predictedImageName,mimetype='image/jpeg')

    return abort(401)
      


      
if __name__=="__main__":
    print(('* loading Keras model and Flask starting server'))
    global model
    model = load_model()
    global graph
    graph = tf.get_default_graph()

    app.run(host='0.0.0.0')