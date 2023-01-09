################################################################################################
from flask import Flask, render_template, request
from flask import g
import pandas as pd                    # Para la manipulación y análisis de datos
import numpy as np                     # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt        # Para la generación de gráficas a partir de los datos
import seaborn as sns                  # Para la visualización de datos basado en matplotlib
from joblib import load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import uuid
################################################################################################

app=Flask(__name__)

class LinkData():
    link = None

class LinkData():
    link = None

LinkDeData = LinkData()

class ModeloY():
    Y = None
    X = None

ValorY = ModeloY()
################################################################################################

@app.route('/') #Funcion que permite ver index
def index():
    return render_template('index.html')



@app.route('/menu', methods=["GET", "POST"]) #Funcion en la que se muestran los algoritmos
def menu():
    LinkDeData.link = request.form['linkdata']+"?raw=true"
    return render_template('menu.html')
################################################################################################

@app.route('/predict', methods=["GET", "POST"]) #EDA
def predict():
    img_path="app/static/"
    #linkdata = request.form['linkdata']+"?raw=true"        #marca error
    datosData=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    #datosData.hist(figsize=(14,14), xrot=45) #histogramas
    #plt.savefig(img_path+'Histograma.jpg')
    
    CorrData = datosData.corr()
    plt.figure(figsize=(20,7))
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True) #matriz de calor CORR
    plt.savefig(img_path+'Correlaciones.jpg')

    plt.figure(figsize=(20,7))
    MatrizInf = np.triu(CorrData)
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True, mask=MatrizInf) #correlaciones media
    plt.savefig(img_path+'prueba.jpg')

    plt.clf()
    for col in datosData.select_dtypes(include='object'):
        if datosData[col].nunique()<10:sns.countplot(y=col, data=datosData) #valores atipicos unicos
        plt.savefig(img_path+'categoricas1.jpg')
    return render_template('result.html')

@app.route('/pca', methods=["GET", "POST"]) #PCA
def pca():

    img_path="app/static/"
    datosData1=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0) #matriz de correlaciones
    CorrData1 = datosData1.corr(method='pearson')
    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(CorrData1)
    sns.heatmap(CorrData1, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig(img_path+'CorrPCA.jpg')
    plt.clf()
    
    Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    NuevaMatriz = datosData1.select_dtypes(exclude=['object'])
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)
    pd.DataFrame(MEstandarizada, columns=NuevaMatriz.columns)
    #pca = PCA(n_components=9)     #Se instancia el objeto PCA    #pca=PCA(n_components=None), pca=PCA(.85)
    pca = PCA(n_components='mle',svd_solver='full')
    pca.fit(MEstandarizada)
    Varianza = pca.explained_variance_ratio_
    
    plt.plot(np.cumsum(Varianza)) #grafica de varianza acumulada
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza acumulada')
    plt.grid()
    plt.savefig(img_path+'VarianzaAcumPCA1.jpg')
    
    plt.clf()
    sns.pairplot(datosData1, hue = None ) #dispersion
    plt.savefig(img_path+'Visual.jpg')

    return render_template('pca.html')


@app.route('/ADecision', methods=["GET", "POST"])
def ADecision():
    img_path="app/static/"
    datosData2=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(datosData2.corr())
    sns.heatmap(datosData2.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig(img_path+'ADCorr.jpg')
    plt.clf()
    X=datosData2.columns.values
    return render_template('ADecision.html')
################################################################################################

@app.route('/ADArbol', methods=["GET", "POST"])
def ADArbol():
    img_path="app/static/"
    ValorY.Y = request.form['valory']
    ValorY.X = request.form['valorx']
    img_path="app/static/"
    datosData2=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    
    X = np.array(datosData2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
    Y = np.array(datosData2[['Outcome']])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)
    ClasificacionAD = DecisionTreeClassifier(max_depth=14,min_samples_split=4,min_samples_leaf=2,random_state=0)
    ClasificacionAD.fit(X_train, Y_train)
    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
    accuracy_score(Y_validation, Y_ClasificacionAD)
    
    plt.figure(figsize=(16,16))  
    plot_tree(ClasificacionAD, feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age'])
    plt.savefig(img_path+'Arbol.jpg')
    plt.clf()
    ClasificacionBA = RandomForestClassifier(n_estimators=105,max_depth=7, min_samples_split=4, min_samples_leaf=2, random_state=1234)
    RocCurveDisplay.from_estimator(ClasificacionAD,X_validation,Y_validation)
    plt.savefig(img_path+'GrafArbol1.jpg')
    plt.clf()
    metrics.RocCurveDisplay.from_estimator(ClasificacionBA,X_validation,Y_validation)
    plt.savefig(img_path+'GrafArbol2.jpg')

    return render_template('ADecision.html')

@app.route('/VerLaData', methods=["GET", "POST"]) #VER TODOS LOS DATOS
def VerLaData():
    VerLaData = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    return VerLaData.to_html()

@app.route('/VerTail', methods=["GET", "POST"]) #VER LOS ULTIMOS 20 DATOS
def VerTail():
    VerTail = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    VerTailResult = VerTail.tail(20)
    return VerTailResult.to_html()

@app.route('/TiposDeDatos', methods=["GET", "POST"]) #VER LOS TIPOS DE DATOS
def TiposDeDatos():
    TiposDeDatos = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Tipos = TiposDeDatos.dtypes.to_frame()
    
    return Tipos.to_html()

@app.route('/Nulos', methods=["GET", "POST"]) #VER LOS NULOS
def Nulos():
    Nulos = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    NulosResult = Nulos.isnull().sum().to_frame() 
    return NulosResult.to_html()

@app.route('/ResumenEstadistico', methods=["GET", "POST"]) #DESCRIBE
def ResumenEstadistico():
    ResumenEstadistico = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    ResumenEstadisticoResult = ResumenEstadistico.describe() 
    return ResumenEstadisticoResult.to_html()

@app.route('/DistribucionCat', methods=["GET", "POST"])
def DistribucionCat():
    DistribucionCat = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    DistribucionCatResult = DistribucionCat.describe(include='object') 
    return DistribucionCatResult.to_html()

@app.route('/MatrizCorr1', methods=["GET", "POST"]) #MATRIZ DE CORRELACIONES
def MatrizCorr1():
    MatrizCorr1 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    MatrizCorr1Result = MatrizCorr1.corr()
    return MatrizCorr1Result.to_html()

@app.route('/MatrizCorr2', methods=["GET", "POST"]) #MATRIZ DE CORRELACIONES
def MatrizCorr2():
    MatrizCorr2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    MatrizCorr2Result = MatrizCorr2.corr(method='pearson')
    return MatrizCorr2Result.to_html()

@app.route('/Estandarizar', methods=["GET", "POST"]) #matriz estandarizada
def Estandarizar():
    Data2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Estandarizar = StandardScaler()
    NuevaMatriz = Data2.select_dtypes(exclude=['object'])
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)
    Matriz = pd.DataFrame(MEstandarizada, columns=NuevaMatriz.columns) 
    return Matriz.to_html()

@app.route('/Normalizar', methods=["GET", "POST"]) #matriz estandarizada
def Normalizar():
    Data2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Normalizar = MinMaxScaler()
    NuevaMatriz = Data2.select_dtypes(exclude=['object'])
    MEstandarizada = Normalizar.fit_transform(NuevaMatriz)
    Matriz = pd.DataFrame(MEstandarizada, columns=NuevaMatriz.columns) 
    return Matriz.to_html()

@app.route('/Cargas', methods=["GET", "POST"]) #CARGAS
def Cargas():
    Data2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Estandarizar = StandardScaler()
    NuevaMatriz = Data2.select_dtypes(exclude=['object'])
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)
    pca = PCA(n_components='mle',svd_solver='full')
    pca.fit(MEstandarizada)
    CargasComponentes = pd.DataFrame(abs(pca.components_), columns=NuevaMatriz.columns)
    return CargasComponentes.to_html()

@app.route('/pca/GrafDisper', methods=["GET", "POST"]) #GRAFICA DE DISPERSION
def GrafDisper():
    img_path="app/static/"
    Data3 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    clase1 = request.form['x']
    clase2 = request.form['y']
    clase3 = request.form['hist']
    sns.scatterplot(x=clase1, y =clase2, data=Data3, hue=clase3)
    plt.title('Gráfico de dispersión')
    plt.xlabel(clase1)
    plt.ylabel(clase2)
    plt.savefig(img_path+'GraficoDisper.jpg')
    return render_template('GrafDisper.html')

@app.route('/MatrizClasif', methods=["GET", "POST"])
def MatrizClasif():
    Data4 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    
    X = np.array(Data4[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
    Y = np.array(Data4[['Outcome']])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)
    ClasificacionAD = DecisionTreeClassifier(max_depth=14,min_samples_split=4,min_samples_leaf=2,random_state=0)
    ClasificacionAD.fit(X_train, Y_train)
    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
    accuracy_score(Y_validation, Y_ClasificacionAD)

    ClasificacionBA = RandomForestClassifier(n_estimators=105,max_depth=7, min_samples_split=4, min_samples_leaf=2, random_state=1234)
    ClasificacionBA.fit(X_train, Y_train)
    Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
    ModeloClasificacion2 = ClasificacionBA.predict(X_validation)
    Matriz_Clasificacion2 = pd.crosstab(Y_validation.ravel(), ModeloClasificacion2,rownames=['Reales'], colnames=['Clasificación'])
    return Matriz_Clasificacion2.to_html()

@app.route('/EfiYC', methods=["GET", "POST"])
def EfiYC():
    Data4 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    
    X = np.array(Data4[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])
    Y = np.array(Data4[['Outcome']])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)
    ClasificacionAD = DecisionTreeClassifier(max_depth=14,min_samples_split=4,min_samples_leaf=2,random_state=0)
    ClasificacionAD.fit(X_train, Y_train)
    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
    accuracy_score(Y_validation, Y_ClasificacionAD)

    ClasificacionBA = RandomForestClassifier(n_estimators=105,max_depth=7, min_samples_split=4, min_samples_leaf=2, random_state=1234)
    ClasificacionBA.fit(X_train, Y_train)
    ModeloClasificacion2 = ClasificacionBA.predict(X_validation)
    Importancia2 = pd.DataFrame({'Variable': list(Data4[['Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction','Age']]), 'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
    
    return Importancia2.to_html()
################################################################################################

if __name__=='__main__':
    app.run(debug=True, port=5000)
################################################################################################

@app.route('/hello', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = load('model.joblib')
        np_arr = floats_string_to_np_arr(text)
        make_picture2('AgesAndHeights.pkl', model, np_arr, path)

        return render_template('index.html', href=path)

def make_picture2(training_data_filename, model, new_inp_np_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data['Age']
  data = data[ages > 0]
  ages = data['Age']
  heights = data['Height']
  x_new = np.array(list(range(19))).reshape(19, 1)
  preds = model.predict(x_new)

  fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)',
                                                                                'y': 'Height (inches)'})

  fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

  new_preds = model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
  
  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()

def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)