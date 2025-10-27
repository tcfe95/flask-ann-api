from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Charger le modèle
model = tf.keras.models.load_model('iris_ann_model.h5')

# Initialiser l'application Flask
app = Flask(__name__)

# Route d'accueil
@app.route('/')
def home():
    return "API Flask pour modèle ANN - opérationnelle."


# Route de test de disponibilité
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"})


# Route pour prédiction sur une seule entrée
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})


# Route pour prédiction sur plusieurs entrées (batch)
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        features_list = data['features_list']
        features_array = np.array(features_list)
        predictions = model.predict(features_array)
        predicted_classes = np.argmax(predictions, axis=1)
        return jsonify({'predicted_classes': predicted_classes.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


# Route pour récupérer des informations sur le modèle
@app.route('/model_info', methods=['GET'])
def model_info():
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    return jsonify({"model_summary": model_summary})


if __name__ == '__main__':
    app.run(debug=True)

	
