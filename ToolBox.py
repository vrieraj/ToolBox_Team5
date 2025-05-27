import numpy as np
import pandas as pd

from scipy.stats import f_oneway, mannwhitneyu, pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

## DESCRIPCIÓN DATAFRAME - VARIABLES ##

def describe_df(df:pd.DataFrame):
    '''
    Resume las principales características de cada variable de un dataframe dado, tales como tipo, valores nulos y cardinalidad. 

    Argumentos:
    df: (DataFrame): DataFrame que se pretende analizar

    Retorna:
    DataFrame: Retorna un DataFrame cuyo índice son las variables del DataFrame a analizar, y las columnas los parámetros analizados.
    
    Por cada variable, devuelve:
    DATA_TYPE (str): Tipo de dato.
    MISSINGS(%) (float): Porcentaje de valores nulos.
    UNIQUE_VALUES (int): Número de valores únicos.
    CARDIN(%) (float): Porcentaje de cardinalidad.
    '''
    
    df_var = pd.DataFrame(index=pd.Index([], name='COL_N'))
    for variable in df.columns:
        df_var.loc['DATA_TYPE', variable]    = df[variable].dtype
        df_var.loc['MISSINGS (%)', variable] = round(df[variable].isnull().sum()/len(df[variable]), 2)
        
        cardinalidad = df[variable].nunique()
        porc_card = round(df[variable].nunique()/len(df)*100, 2)
                
        df_var.loc['UNIQUE_VALUES', variable] = cardinalidad
        df_var.loc['CARDIN (%)', variable]    = porc_card
    
    df_var = df_var.reset_index()
    df_var.index = pd.Index(np.full((1,len(df_var)),'')[0]) 

    return df_var

def tipifica_variables(df:pd.DataFrame, umbral_categoria:int, umbral_continua:float):
    '''
    Sugiere el tipo categórico de cada variable de un dataframe en función del número máximo de categorías a considerar y 
    del porcentaje de cardinalidad dado como umbral para considerar una variable numérica como continua.

    Argumentos:
    df (DataFrame): DataFrame que se pretende analizar
    umbral_categoria (int): Número máximo de categorías a considerar por variable
    umbral_continua (float): Porcentaje de cardinalidad a partir del cuál se considerará una variable como continua

    Retorna:
    DataFrame: Retorna un DataFrame cuyas columnas son las variables del DataFrame a analizar, y el índice los parámetros analizados.

    NOMBRE_VARIABLE (str): Variable del dataframe dado.
    TIPO_SUGERIDO (str): Sugerencia sobre el tipo de variable a analizar: 'Binaria', 'Categórica', 'Numérica discreta', 'Numérica continua'. 
    '''

    df = describe_df(df).set_index('COL_N').T
    
    df.loc[(df.UNIQUE_VALUES >= umbral_categoria) & (df['CARDIN (%)'] > umbral_continua), 'TIPO_SUGERIDO'] = 'Numérica continua'
    df.loc[(df.UNIQUE_VALUES >= umbral_categoria) & (df['CARDIN (%)'] < umbral_continua), 'TIPO_SUGERIDO'] = 'Numérica discreta'
    df.loc[df.UNIQUE_VALUES < umbral_categoria, 'TIPO_SUGERIDO'] = 'Categórica'
    df.loc[df.UNIQUE_VALUES == 2, 'TIPO_SUGERIDO'] = 'Binaria'

    return pd.DataFrame([df.index, df.TIPO_SUGERIDO]).T.rename(columns = {0: "nombre_variable", 1: "tipo_sugerido"})

def categoriza_variables(df:pd.DataFrame, umbral_categoria:int, umbral_continua:float):
    '''
    Resume las principales características de cada variable de un dataframe dado, tales como tipo, valores nulos, cardinalidad...
    Además, sgiere el tipo categórico de cada variable de un dataframe en función del número máximo de categorías a considerar y 
    del porcentaje de cardinalidad dado como umbral para considerar una variable numérica como continua.

    Argumentos:
    df (DataFrame): DataFrame que se pretende analizar
    umbral_categoria (int): Número máximo de categorías a considerar por variable
    umbral_continua (float): Porcentaje de cardinalidad a partir del cuál se considerará una variable como continua

    Retorna:
    DataFrame: Retorna un DataFrame cuyas columnas son las variables del DataFrame a analizar, y el índice los parámetros analizados.

    NOMBRE_VARIABLE (str): Variable del dataframe dado.
    TIPO_SUGERIDO (str): Sugerencia sobre el tipo de variable a analizar: 'Binaria', 'Categórica', 'Numérica discreta', 'Numérica continua'.    
    '''

    df_var = pd.DataFrame(df.columns, columns=['Features'])
    for variable in df.columns:
        df_var.loc[df_var.Features == variable, 'Data_type'] = df[variable].dtype
        df_var.loc[df_var.Features == variable, '%_Missings'] = round(df[variable].isnull().sum()/len(df[variable]), 2)
        df_var.loc[df_var.Features == variable, 'Unique_values'] = df[variable].nunique()
        df_var.loc[df_var.Features == variable, '%-Cardinalidad'] = round(df[variable].nunique()/len(df)*100, 2)

        df_var.loc[(df_var.Unique_values >= umbral_categoria) & (df_var['%-Cardinalidad'] > umbral_continua), 'Tipo_sugerido'] = 'Numérica continua'
        df_var.loc[(df_var.Unique_values >= umbral_categoria) & (df_var['%-Cardinalidad'] < umbral_continua), 'Tipo_sugerido'] = 'Numérica discreta'
        df_var.loc[df_var.Unique_values < umbral_categoria, 'Tipo_sugerido'] = 'Categórica'
        df_var.loc[df_var.Unique_values == 2, 'Tipo_sugerido'] = 'Binaria'
    
    return df_var 

## FUNCIONES AUXILIARES ##

def check_parametros(df:pd.DataFrame, target_col:str, umbral_corr = 0.5, umbral_categoria = 0, umbral_continua = 0.5, pvalue = None):
    """
    DESCRIPCIÓN:

    Comprobación de argumentos de las funciones de selección de features numéricas y categóricas: get_features_num_regression y get_features_cat_regression.

    ARGUMENTOS:

    df (DataFrame): Dataset de train del conjunto de datos de un hipotético modelo de regresión lineal que queremos entrenar. La función comprobará que el argumento es un 
    DataFrame o nos devolverá un mensaje de error con el motivo.

    target_col (float): Columna del dataset que constiuye el 'target' de nuestro hipotético modelo de regresión. Variable numérica continua o discreta con alta cardinalidad.
    La función comprobará que el target es una columna del dataset del tipo numérica con alta cardinalidad o nos devolverá un mensaje de error con el motivo.

    umbral_corr (float): Argumento por defecto con valor 0. Establece el corte de la correlación entre la variable target y las variables numéricas. La función comprobará que 
    el valor del argumento esté entre 0 y 1 o nos devolverá un mensaje de error con el motivo.

    umbral_categorica (int): Argumento por defecto con valor 0. Establece el corte de las variables que se consideran categóricas (todas aquellas cuyo número total de valores únicos
    quede por debajo de este umbral). La función comprobará que el argumento sea un número entero mayor que 0 o nos devolverá un mensaje de error con el motivo.

    umbral_continua (float): Argumento por defecto con valor 0. Establece el corte de las variables que se consideran numéricas continuas, (todas aquellas cuyo número total 
    de valores únicos quede por encima de este umbral). La función comprobará que el argumento sea un número decimal mayor que 0 o nos devolverá un mensaje de error con el motivo.

    pvalue (float): Valor del pvalue (default = None). Si el pvalue no es None, la función comprobará si el valor del argumento se encuentra entre 1 y 0 o nos devolverá un mensaje 
    de error con el motivo. 

    RETURN:

    (string): 'OK' en caso de superar todas las comprobaciones descritas en los parámetros.

    """

    if not isinstance(df, pd.DataFrame):
        print("Error: el primer argumento debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: la columna '{target_col}' no es numérica.")
        return None

    if df[target_col].nunique() < 10:
        print(f"Error: '{target_col}' no parece ser una variable continua (baja cardinalidad).")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if type(umbral_categoria) != int or umbral_categoria < 0:
        print("Error: 'umbral_categoria' debe ser un número entero.")
        return None

    if type(umbral_continua) != float or umbral_continua < 0:
        print("Error: 'umbral_continua' debe ser un número decimal.")
        return None

    if pvalue is not None:
        if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1 o None.")
            return None
    return 'OK'

## VARIABLES NUMÉRICAS ##

def columnas_correlacionadas(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve una lista de columnas numéricas cuya correlación con target supera el umbral de correlación.
    Si especifica 'pvalue', tambien verifica que la correlación sea significativa.
    
    Argumentoss:
    df (pd.DataFrame): DataFrame a introducir.
    target_col (str): Nombre de la columna objetivo (debe ser numérica y con alta cardinalidad).
    umbral_corr (float): Umbral de correlación (entre 0 y 1).
    pvalue (float, optional): Nivel de significancia deseado (por ejemplo 0.05). Por defecto None.
    
    Retorna:
    list or None: Lista de nombres de columnas que cumplen los criterios. Imprime errores si no es válido.
    """
    
    if check_parametros(df=df, target_col=target_col, umbral_corr = umbral_corr, pvalue = pvalue) != 'OK':
        return None

    columnas_validas = []

    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)

    for col in num_cols:
        series = df[[target_col, col]].dropna()
        corr, pval = pearsonr(series[target_col], series[col])

        if abs(corr) >= umbral_corr:
            if pvalue is None or pval <= (1 - pvalue):
                columnas_validas.append(col)

    return columnas_validas

def plot_features_num_regression(df, target_col='', columns=[], umbral_corr=0, pvalue=None):
    '''
    Genera graficos pairplot entre la variable target y otras variables numéricas
    del DataFrame que presenten una correlación significativa.

    Argumentos:
    df (DataFrame): DataFrame que contiene los datos a analizar.
    target_col (str): Nombre de la variable objetivo con la que se evaluará la correlación.
    columns (list): Lista de nombres de columnas numéricas a considerar. Si se deja vacía,
                    se tomarán todas las variables numéricas.
    umbral_corr (float): Valor mínimo absoluto de correlación para que una variable sea incluida.
    pvalue (float o None): Umbral de significancia estadística (p-valor). Si es distinto de None,
                           se filtran también las variables cuya correlación con la variable objetivo
                           no sea estadísticamente significativa.

    Retorna:
    None. Muestra en pantalla uno o varios pairplots con las variables seleccionadas.
    '''
    if check_parametros(df=df, target_col=target_col, umbral_corr = umbral_corr, pvalue = pvalue) != 'OK':
        return None

    if len(columns) == 0:
        columns = [variable for variable in df.columns if np.issubdtype(df[variable].dtype, np.number)]

    columnas = columnas_correlacionadas(df[columns], target_col=target_col, umbral_corr=umbral_corr, pvalue=pvalue)
    sns.pairplot(df, x_vars=columnas, y_vars=target_col)

## VARIABLES CATEGÓRICAS ##

def get_features_cat_regression(dataframe:pd.DataFrame, target_col:float, pvalue = 0.05, umbral_categoria = 6, umbral_continua = 25.0): 
    
    """
    DESCRIPCIÓN:

    Análisis bivariante de la variable target contra las variables categóricas de un dataset para su posterior selección de features categóricas ante la construcción de un 
    hipotético modelo de regresión lineal.

    REQUISITOS:
    
    Es necesario importar mannwhitneyu y stats de scipy, ejecuta:
    
    from scipy.stats import mannwhitneyu 
    
    from scipy import stats

    ARGUMENTOS:

    dataframe (DataFrame): Dataset de train del conjunto de datos de un hipotético modelo de regresión lineal que queremos entrenar

    target_col (float): Columna del dataset que constiuye el 'target' de nuestro hipotético modelo de regresión. Variable numérica continua o discreta con alta cardinalidad

    pvalue (float): Valor del pvalue (default = 0.05) 

    umbral_categoria (int): Argumento por defecto con valor 6 (lo que se considera estándard). Establece el corte de las variables que se consideran categóricas 
    (todas aquellas cuyo número total de valores únicos quede por debajo de este umbral). Este argumento es necesario para las funciones tipifica_variables y check_parametros
     que se invocan dentro de la descrita. 

    umbral_continua (float): Argumento por defecto con valor 25.00. Establece el corte de las variables que se consideran numéricas continuas, (todas aquellas cuyo número total 
    de valores únicos quede por debajo de este umbral). Este argumento es necesario para la función tipifica_variables y check_parametros que se invocan dentro de la descrita. 

    RETURN:

    (list): Variables categóricas que superen en confianza estadística el test de relación pertinente tras un análisis bivariante.

    """
    if check_parametros(dataframe, target_col, umbral_categoria = umbral_categoria, umbral_continua = umbral_continua, pvalue=pvalue) != 'OK':
        return None

    df_tipo = tipifica_variables(dataframe, umbral_categoria, umbral_continua)

    # me quedo con las categóricas y las vuelco en una lista el nombre de la variable, que está en el índice del dataset
    
    es_catego = df_tipo.tipo_sugerido == "Categórica"
    es_binaria = df_tipo.tipo_sugerido == "Binaria"

    lista_categoricas = df_tipo.loc[es_catego | es_binaria].index.to_list()

    features_categoricas = []
    for categoria in lista_categoricas:
        # si mi variable es binaria, aplicamos U de Mann-Whitney

        if len(dataframe[categoria].unique()) == 2:      
            
            # from scipy.stats import mannwhitneyu --> preguntar si esto debería ir aquí o añadir en el stringdoc que es necesario importarlo para usar la función
            
            es_a = dataframe[categoria].unique()[0]   # obtengo las dos agrupaciones
            es_b = dataframe[categoria].unique()[1]
            
            grupo_a = dataframe.loc[dataframe[categoria] == es_a][target_col]   # y separo mi dataset en función de ellas
            grupo_b = dataframe.loc[dataframe[categoria] == es_b][target_col]
            
            u_stat, p_valor = mannwhitneyu(grupo_a, grupo_b)

            if p_valor <= pvalue:
                features_categoricas.append(categoria)
            

        # si no es binaria, aplicamos ANOVA

        else:   
            # from scipy import stats 
            grupos = dataframe[categoria].unique()  # obtengo todos valores de la variable

            # obtenemos los valores del target por cada valor de las diferentes categorias con un list comprehension 
            argumento_stats = [dataframe[dataframe[categoria] == grupo][target_col] for grupo in grupos] 
                 
            f_val, p_valor = stats.f_oneway(*argumento_stats) # El método * separa todos los elementos de la lista y los pasa como argumento a la función                                                   

            if p_valor <= pvalue:
                features_categoricas.append(categoria)

    return features_categoricas  

def plot_features_cat_regression(df, target_col = '', columns=[], pvalue=0.05, with_individual_plot=False, umbral_categoria = 6, umbral_continua = 25.0, escala_log=False):

    """
    DESCRIPCIÓN:

    Visualización de los histogramas agrupados de la variable target para cada uno de los valores de las variables categóricas pre-seleccionadas de un dataset.
    Esta función llama a get_features_cat_regression para llevar a cabo dicha selección (ver documentación de la función).
   
    REQUISITOS:
    
    Es necesario importar seaborn, ejecuta:
    
    import seaborn as sns

    ARGUMENTOS:

    dataframe (DataFrame): Dataset de train del conjunto de datos de un hipotético modelo de regresión lineal que queremos entrenar

    target_col (float): Columna del dataset que constiuye el 'target' de nuestro hipotético modelo de regresión. Variable numérica continua o discreta con alta cardinalidad

    columns (list) = Lista de strings vacía por defecto. Si la lista está vacía, la función la igualará a todas las variables categóricas del dataset. Si no lo está, sólo empleará
    las que aparezcan en la lista para su análisis, selección y posterior visualización. 

    pvalue (float): Valor del pvalue (default = 0.05) 

    with_individual_plot (bool) = False por defecto. La función muestra una figura con tantos axes como variables se comparan contra el target.

    umbral_categoria (int): Argumento por defecto con valor 6 (lo que se considera estándard). Establece el corte de las variables que se consideran categóricas 
    (todas aquellas cuyo número total de valores únicos quede por debajo de este umbral). Este argumento es necesario para la función get_features_cat_regression que se invoca
    dentro de ésta. 

    umbral_continua (float): Argumento por defecto con valor 25.00. Establece el corte de las variables que se consideran numéricas continuas, (todas aquellas cuyo número total 
    de valores únicos quede por encima de este umbral). Este argumento es necesario para la función get_features_cat_regression que se invoca dentro de ésta. 

    RETURN:

    (list): Variables categóricas que superen en confianza estadística el test de relación pertinente tras un análisis bivariante.
    if check_parametros(df, target_col, umbral_categoria = umbral_categoria, umbral_continua = umbral_continua, pvalue=pvalue) != 'OK':
        return None

    """
    if check_parametros(df, target_col, umbral_categoria = umbral_categoria, umbral_continua = umbral_continua, pvalue=pvalue) != 'OK':
        return None

    if len(columns) == 0:
        
        df_tipo = tipifica_variables(dataframe=df, umbral_categoria= umbral_categoria, umbral_continua= umbral_continua)
        es_catego = df_tipo.Tipo == "Categórica"
        es_binaria = df_tipo.Tipo == "Binaria"

        columns = df_tipo.loc[es_catego | es_binaria].index.to_list()
    
    columns.append(target_col)
    columnas = get_features_cat_regression(df[columns], target_col=target_col, pvalue=pvalue, umbral_categoria = 6, umbral_continua = 25.0)
    
    fig, ax = plt.subplots(len(columnas), figsize=(10,10))
    for index, columna in enumerate(columnas):
        sns.histplot(df, x=target_col, hue=columna, ax=ax[index], log_scale=escala_log)