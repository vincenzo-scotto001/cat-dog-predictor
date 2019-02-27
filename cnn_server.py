import os
from flask import Flask, request, render_template,redirect,url_for
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
model = load_model('Dog_Cat_Model')
graph = tf.get_default_graph()

APP_ROOT = os.path.dirname(os.path.abspath('__file__'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/handledata",methods = ['POST'])
def handledata():
    target = os.path.join(APP_ROOT, 'images/')
    
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist('file'):
        filename ='test.jpg'
        destination = '/'.join([target,filename])
        file.save(destination)
    
    test_image = image.load_img('./images/test.jpg', target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    
    with graph.as_default():
        result = model.predict(test_image)
        print(result)
        if result[0][0] ==1:
            prediction = 'dog'
        else:
            prediction = 'cat'
    return render_template('index.html',prediction = prediction.upper())
    

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port = port)



# app.run()