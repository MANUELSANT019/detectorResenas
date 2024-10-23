import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from flask import render_template, request
from bp_main import bp_main

# Carga del dataset seleccionando solo las columnas necesarias (review_body, stars, language)
df = pd.read_csv('test.csv', usecols=['review_body', 'stars', 'language'])

# Filtramos solo las reseñas en español
df = df[df['language'] == 'es']

# Clasificación de reseñas: Positivas (>3 estrellas), Negativas (<=2 estrellas)
df = df[df['stars'] != 3]  # Eliminar reseñas neutrales
df['sentiment'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)

# División en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['review_body'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorización del texto
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- MODELO MULTINOMIAL NAIVE BAYES ----------
modelo_nb = MultinomialNB()
modelo_nb.fit(X_train_vec, y_train)

# Predicción con Naive Bayes
y_pred_nb = modelo_nb.predict(X_test_vec)

# Cálculo de las métricas para Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)

@bp_main.route('/')
def index():
    # Mostrar las métricas calculadas
    return render_template('index.html', 
                            accuracy_nb=accuracy_nb, 
                            precision_nb=precision_nb, 
                            recall_nb=recall_nb)

@bp_main.route('/clasificacion', methods=['POST'])
def clasificacion():
    if request.method == 'POST':
        texto = request.form['texto'] 
        
        # Predicción con Naive Bayes
        prediccion_nb = modelo_nb.predict(vectorizer.transform([texto]))[0]
        resultado_nb = 'Positiva' if prediccion_nb == 1 else 'Negativa'

    # Mostrar la predicción y las métricas en la misma vista
    return render_template('index.html', 
                            p_texto=texto, 
                            p_prediccion_nb=resultado_nb,
                            accuracy_nb=accuracy_nb, 
                            precision_nb=precision_nb, 
                            recall_nb=recall_nb)


