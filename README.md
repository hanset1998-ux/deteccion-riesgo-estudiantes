# deteccion-riesgo-estudiantes
 🌌 Machine Learning Hub - Estilo Realidad Aumentada (RA)
# 🎓 Alertas Tempranas - Detección de Estudiantes en Riesgo Académico
¡Bienvenido al portal interactivo y futurista diseñado para explorar y comprender a fondo el universo de **Machine Learning (ML)**! Esta aplicación web utiliza un diseño moderno hiperrealista inspirado en interfaces de Realidad Aumentada (RA/HUD) y Glassmorphism para llevarte a través de los conceptos y modelos de IA de manera visual e interactiva.
Este proyecto implementa un modelo de Machine Learning supervisado bajo la metodología estructurada **CRISP-ML** para predecir si un estudiante se encuentra en riesgo de fracaso académico o deserción. Incluye un cuaderno de análisis `.ipynb`, una interfaz interactiva de predicción en **Streamlit** y una Landing Page premium con estética de **Realidad Aumentada (RA)**.
---
## 🚀 Badges del Proyecto
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Visuals](https://img.shields.io/badge/Visuals-AI_Generated-00f2fe?style=for-the-badge)
![Visuals](https://img.shields.io/badge/Visuals-AI_Generated_AR-38bdf8?style=for-the-badge)
---
## 🧠 ¿Qué es Machine Learning?
## 📂 Archivos del Proyecto
El **Aprendizaje Automático (Machine Learning)** es una disciplina de la Inteligencia Artificial que permite a los sistemas informáticos aprender directamente a partir de datos históricos, identificando patrones complejos para tomar decisiones o realizar predicciones con una intervención humana mínima.
- `index.html`: Landing Page premium interactiva (glassmorphism con tema oscuro y gráficos RA).
- `app.py`: Servidor Streamlit interactivo para inferencias de riesgo estudiantil.
- `train.py`: Generador de datos sintéticos y pipeline de entrenamiento/serialización del modelo.
- `student_analysis.ipynb`: Cuaderno Jupyter con las 6 fases de la metodología CRISP-ML documentadas paso a paso.
- `requirements.txt`: Dependencias de librerías del proyecto.
- `assets/`: Directorio que contiene las imágenes de Realidad Aumentada generadas por IA.
Este proyecto detalla y desglosa de manera exhaustiva sus tres paradigmas fundamentales:
---
### 1. Aprendizaje Supervisado (Supervised Learning)
Los modelos se entrenan utilizando **datos etiquetados** (entradas con sus respectivas respuestas correctas conocidas). Su objetivo es aprender una función de mapeo para predecir salidas de nuevos datos de entrada.
## 🛠️ Ejecución Local
*   **Modelos Detallados:**
    *   **Regresión Lineal:** Predice un valor numérico continuo modelando la relación lineal entre variables.
    *   **Regresión Logística:** Clasifica datos en categorías binarias estimando la probabilidad de ocurrencia.
    *   **Máquinas de Vectores de Soporte (SVM):** Encuentra el hiperplano óptimo en un espacio multidimensional que maximiza la separación entre clases.
    *   **Árboles de Decisión (Decision Trees):** Estructura jerárquica de nodos y reglas booleanas para tomar decisiones secuenciales.
    *   **Bosques Aleatorios (Random Forest):** Combinación de múltiples árboles de decisión (Ensemble) que reduce el sobreajuste y mejora la precisión general.
Sigue los siguientes pasos en la terminal de tu sistema para configurar y lanzar el proyecto:
### 2. Aprendizaje No Supervisado (Unsupervised Learning)
Los modelos analizan **datos no etiquetados** para descubrir patrones ocultos, agrupaciones o estructuras intrínsecas por sí mismos.
### 1. Clonar o navegar a la carpeta del proyecto
```bash
cd C:/Users/hanse/Desktop/maestria
```
*   **Modelos Detallados:**
    *   **K-Means:** Agrupa datos similares en $K$ clústeres basándose en la distancia media hacia centroides móviles.
    *   **DBSCAN:** Agrupación espacial de datos basada en densidades de puntos, ideal para clústeres de formas irregulares y detección de anomalías.
    *   **PCA (Análisis de Componentes Principales):** Reduce la dimensionalidad del conjunto de datos proyectándolo en direcciones de máxima varianza.
    *   **t-SNE:** Técnica no lineal de reducción de dimensiones, ideal para la visualización en 2D/3D de datos altamente complejos.
    *   **Apriori:** Algoritmo de minería de reglas de asociación para descubrir relaciones de dependencia frecuentes en transacciones.
### 2. Crear y activar el entorno virtual de Python
```bash
# Crear entorno virtual
python -m venv venv
### 3. Aprendizaje por Refuerzo (Reinforcement Learning)
Un **agente autónomo** aprende a tomar decisiones óptimas interactuando de manera dinámica con su entorno para **maximizar una recompensa acumulada** a lo largo del tiempo, guiado por ensayo y error.
# Activar el entorno (Windows PowerShell)
.\venv\Scripts\activate
*   **Modelos Detallados:**
    *   **Q-Learning:** Algoritmo clásico libre de modelo que aprende los valores de calidad de las acciones ($Q$-values) para cada estado.
    *   **DQN (Deep Q-Networks):** Combina redes neuronales profundas con Q-learning para manejar estados de dimensiones masivas (como videojuegos o robótica).
    *   **SARSA:** Algoritmo en-política (on-policy) que actualiza los valores basándose en la acción real elegida por la política actual del agente.
    *   **Gradiente de Políticas (Policy Gradients):** Optimiza de manera directa la distribución de probabilidad de las acciones de la política sin requerir una función de valor intermedia.
# Activar el entorno (CMD clásico de Windows)
.\venv\Scripts\activate.bat
```
---
### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```
## 🐍 Ejemplo Básico de Código Python
### 4. Generar datos y entrenar el modelo
Este comando creará el archivo `student_data.csv` y guardará los modelos serializados (`.pkl`):
```bash
python train.py
```
En la landing page y a continuación se presenta un código práctico básico escrito en Python usando `scikit-learn` para demostrar el **Aprendizaje Supervisado** entrenando y clasificando flores en el famoso dataset de Iris:
### 5. Iniciar la aplicación Streamlit
```bash
streamlit run app.py
```
```python
# Importar librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
### 6. Abrir la Landing Page
Haz doble clic sobre el archivo [index.html](index.html) para visualizar la Landing Page en tu navegador.
# 1. Cargar dataset de prueba (flores Iris)
iris = load_iris()
X, y = iris.data, iris.target
---
# 2. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
## 📈 Guía Paso a Paso para Desplegar el Proyecto en GitHub
# 3. Inicializar el clasificador RandomForest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
Sigue estas instrucciones al pie de la letra para publicar este proyecto en tu perfil personal de GitHub:
# 4. Entrenar el modelo (Aprendizaje Supervisado)
modelo.fit(X_train, y_train)
### Paso 1: Configurar Git en tu Computadora (Si no está configurado)
Si es la primera vez que usas Git, debes definir tu nombre y correo en la consola (puedes abrir Git Bash, CMD o PowerShell):
```bash
git config --global user.name "Tu Nombre Completo"
git config --global user.email "tu-correo-de-github@example.com"
```
# 5. Evaluar predicciones
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)
### Paso 2: Inicializar el Repositorio Local
Navega a la carpeta del proyecto (`C:/Users/hanse/Desktop/maestria`) y ejecuta:
```bash
# Inicializar el repositorio Git
git init
```
print(f"Exactitud del Modelo: {precision * 100:.2f}%")
### Paso 3: Agregar los Archivos al Escenario (Staging Area)
El archivo `.gitignore` ya está configurado para no rastrear la pesada carpeta `venv/`. Para añadir el resto de los entregables corre:
```bash
git add .
```
---
### Paso 4: Guardar los Cambios (Primer Commit)
Guarda una instantánea de los archivos en tu historial de versiones local con un mensaje descriptivo:
```bash
git commit -m "Commit Inicial: Alerta Temprana de Riesgo Académico con CRISP-ML y Streamlit"
```
## 🎨 Características de Diseño (Estilo RA)
### Paso 5: Crear el Repositorio Remoto en GitHub
1. Ingresa a tu cuenta en [GitHub.com](https://github.com).
2. Haz clic en el botón verde **"New"** (Nuevo) en la sección superior izquierda o entra a [github.com/new](https://github.com/new).
3. Nombra tu repositorio (por ejemplo: `deteccion-riesgo-estudiantes`).
4. Déjalo como **Público** y **NO** selecciones añadir README, .gitignore o licencias (puesto que ya los tenemos creados localmente).
5. Haz clic en **"Create repository"** (Crear Repositorio).
*   **Glassmorphic Cards:** Paneles con desenfoque de fondo profundo (`backdrop-filter`) y sutiles bordes translúcidos con gradientes de neón.
*   **Holographic Hover Effects:** Iluminación holográfica 3D interactiva que sigue la posición del cursor del mouse sobre las tarjetas principales.
*   **HUD Laser Scanner:** Animación de barrido de haz láser que simula un escaneo de datos de Realidad Aumentada.
*   **Interactive Terminal:** Un simulador de código interactivo que permite copiar al portapapeles el script de Python y ver las explicaciones en tiempo real.
### Paso 6: Vincular tu Repositorio Local con GitHub
Copia la URL de tu repositorio de GitHub (que se verá similar a `https://github.com/tu-usuario/deteccion-riesgo-estudiantes.git`) y ejecútala en la terminal:
```bash
# Enlazar tu máquina local con GitHub
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
---
# Renombrar la rama principal a 'main'
git branch -M main
```
## 🛠️ Ejecución Local
### Paso 7: Subir tus Archivos (Git Push)
Sube tu rama local `main` al servidor remoto de GitHub:
```bash
git push -u origin main
```
*Si es la primera vez, el sistema te pedirá iniciar sesión o ingresar tu Token de Acceso Personal (PAT) de GitHub para autorizar la subida.*
Para visualizar la landing page interactiva en tu computadora:
¡Listo! Tu proyecto ahora estará desplegado en GitHub y visible para cualquier persona.
1.  Asegúrate de que todos los archivos (`index.html`, `style.css`, `script.js` y el directorio `assets/`) se encuentren en la misma carpeta.
2.  Haz doble clic en el archivo `index.html` para abrirlo directamente en tu navegador web de preferencia, o bien inicia un servidor local rápido ejecutando en la consola:
    ```bash
    python -m http.server 8000
    ```
    Y visita en tu navegador la dirección: [http://localhost:8000](http://localhost:8000).
