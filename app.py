from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

# Cargar modelo y scaler
modelo = joblib.load('./model/GasRF_model.pkl')
scaler = joblib.load('./model/scaler_gasB.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        at = float(request.form['at'])
        ah = float(request.form['ah'])
        tit = float(request.form['tit'])
        tey = float(request.form['tey'])
        cdp = float(request.form['cdp'])
        co = float(request.form['co'])
        tat = float(request.form['tat'])

        # Escalar
        datos = np.array([[at, ah, tit, tey, cdp, co, tat]])
        datos_escalados = scaler.transform(datos)

        # Predecir
        prediccion = modelo.predict(datos_escalados)[0]

        return jsonify({'nox': round(prediccion, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
