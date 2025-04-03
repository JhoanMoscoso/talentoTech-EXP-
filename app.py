from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "edad": request.form['edad'],
            "nivel_educativo": request.form["nivel_educativo"],
            "sexo": request.form['sexo']
        }
        df = pd.DataFrame(data, index=[0])
        with open(os.getcwd()+"/modeloSalariosEdu.pkl", 'rb') as file:
            modeloPrediccion = pickle.load(file)

        if isinstance(df, pd.Series):
            X_new_array =  df.values
        elif isinstance(df, pd.DataFrame):
            X_new_array =  df.values
        elif isinstance(df, np.ndarray):
            X_new_array =  df
        else:
            X_new_array =  np.array(data)
        
        
        # Predict scaled salaries
        y_pred = modeloPrediccion.predict(X_new_array)
        # Convertir en DataFrame
        
        return render_template('resultado.html', data=data, y_pred = y_pred)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
