import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib

app = Flask(__name__, static_url_path ='/static')
model = joblib.load('./model.pkl')

@app.route('/')
def main_route():
    return render_template('template.html')
@app.route('/verificar', methods= ['POST'])
def verificar():
        tipo_caso_conf = request.form.get('Tipo do caso conf')
        tipo_caso_desc = request.form.get('Tipo do caso desc')
        tipo_caso_susp = request.form.get('Tipo do caso susp')
        idade = request.form.get('idade')
        eup = request.form.get('eup')
        iot = request.form.get('iot')
        leito_solicitado = request.form.get('leito solicitado')
        leito_entrada = request.form.get('leito entrada')
        leito_saida = request.form.get('leito saida')
        tempo_internacao = request.form.get('tempo internacao')
        teste = np.array([[tipo_caso_conf, tipo_caso_desc, tipo_caso_susp, idade, eup, iot, leito_solicitado, leito_entrada,
                           leito_saida, tempo_internacao]])
        
        classe = model.predict(teste)[0]
        return render_template('template.html', classe=str(classe))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    app.run(debug = True, host='0.0.0.0', port = port)

