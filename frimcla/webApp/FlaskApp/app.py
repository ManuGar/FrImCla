from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import json
import cPickle
import frimcla.extractor.extractor as e
import frimcla.utils.dataset as dataset
import numpy as np
import os


app = Flask(__name__)
app.config['SERVER_NAME'] = 'localhost:5000'

@app.route("/")
def main():
    prediction=""
    # Para enviar variables al html lo unico que hay que hacer es llamar a las variables con el nombre con el que iran en la pagina
    return render_template('index.html', **locals())

@app.route("/predict", methods = ["POST"])
def prediction():
    prediction =""
    f = request.files['file']
    f.save(secure_filename(f.filename))
    with open('FlaskApp/ConfModel.json') as data:
        datos = json.load(data)

    #Tener cuidado con esto, cuando suba la nueva version de frimcla a pip hay que quitar el literal eval
    featureExtractor = [str(datos["featureExtractor"]["model"]), datos["featureExtractor"]["params"]]
    classificationModel = datos["classificationModel"]
    cPickleFile = "FlaskApp/classifier_" + featureExtractor[0] + "_" + classificationModel + ".cpickle"
    labelEncoderPath = "FlaskApp/le-" + featureExtractor[0] + ".cpickle"
    le = cPickle.loads(open(labelEncoderPath).read())
    model = cPickle.loads(open(cPickleFile).read())
    oe = e.Extractor(featureExtractor)
    (labels, images) = dataset.build_batch([f.filename], featureExtractor[0])
    features = oe.describe(images)
    for (label, vector) in zip(labels, features):
        prediction = model.predict(np.atleast_2d(vector))[0]
        prediction = le.inverse_transform(prediction)
        # print("[INFO] predicted: {}".format(prediction))
    os.remove(f.filename)
    return render_template('index.html', **locals())

if __name__ == "__main__":
    app.run()