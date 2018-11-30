from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
from scipy import stats
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

    extractors =datos["featureExtractors"]
    # classificators = [datos["classificationModel"]]
    labelEncoderPath = "FlaskApp/le.cpickle"
    le = cPickle.loads(open(labelEncoderPath).read())
    predictions = []

    for ext in extractors:
        #Tener cuidado con esto, cuando suba la nueva version de frimcla a pip hay que quitar el literal eval
        featureExtractor = [str(ext["model"]), ext["params"]]
        # classificationModel = datos["classificationModel"]
        for classi in ext["classificationModels"]:
            cPickleFile = "FlaskApp/classifiers/classifier_" + ext["model"] + "_" + classi + ".cpickle"
            model = cPickle.loads(open(cPickleFile).read())
            oe = e.Extractor(featureExtractor)
            (labels, images) = dataset.build_batch([f.filename], featureExtractor[0])
            features = oe.describe(images)
            for (label, vector) in zip(labels, features):
                prediction = model.predict(np.atleast_2d(vector))[0]
                predictions.append(prediction)
                # prediction = le.inverse_transform(prediction)
                # print("[INFO] predicted: {}".format(prediction))

    os.remove(f.filename)
    aux = np.array(predictions)
    mode = stats.mode(aux[0])
    prediction = le.inverse_transform(mode[0])[0]

    return render_template('index.html', **locals())

if __name__ == "__main__":
    app.run()