# -*- coding: utf-8 -*-
"""
Created on Thu May 30 01:00:36 2019

@author: Mauricio

propublica.py:
    
El presente programa es una reproduccion del estudio elaborado por ProPublica,
del cual los autores concluyen que la herramienta COMPAS tiene un prejuicio en
contra de individuos de raza negra. 
"""

"""
0.-Importacion de modulos
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tf.enable_eager_execution()
"""
I.- CARGADO, PREPROCESADO Y VISUALIZACION DE LOS DATOS
"""

#importar datos
raw_data = pd.read_csv(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Arts\Bias\ProPublica COMPAS\compas-analysis-master\compas-analysis-master\compas-scores-two-years.csv")
cats = raw_data.columns

"""
Preprocesado de datos:
    
1.-Remover registros donde la fecha del cargo no sea dentro de 
30 dias de la fecha de arresto, por si no se esta considerando
la ofensa adecuada (30 >= days_b_screening_arrest >= -30)

2.-Datos sin caso tratado por COMPAS (is_recid == -1)

3.-Ofensas que no resulten en encarcelamiento (c_charge_degree == 'O')

4.-En los datos solo hay registros de gente que reincido en dos anios o paso 
al menos dos anios en libertad
"""
#Paso a paso para mayor claridad:
data_valid_date = raw_data.loc[(raw_data['days_b_screening_arrest'] >= -30) & (raw_data['days_b_screening_arrest'] <= 30)]
data_valid_COMPAS = data_valid_date.loc[data_valid_date['is_recid'] != -1]
data_valid = data_valid_COMPAS.loc[data_valid_COMPAS['c_charge_degree'] != 0]

#Observar correlacion entre puntuacion COMPAS y permanencia
jail_out = pd.to_datetime(data_valid['c_jail_out'])
jail_in = pd.to_datetime(data_valid['c_jail_in'])
length_of_stay = jail_out.sub(jail_in)
length_of_stay = length_of_stay.dt.total_seconds()

#correlacion entre tiempo de encarcelamiento y puntaje COMPAS
length_of_stay.corr(data_valid['decile_score'])

#Observar la distribucion por edad, raza, puntaje, sexo y reincidencia
#Edad:
data_valid.groupby(['age_cat']).size()

#Raza:
data_valid.groupby(['race']).size()
(data_valid.groupby(['race']).size())*100/data_valid['race'].size

#Puntaje categorico (score_text)
data_valid.groupby(['score_text']).size()

#Tabla: sexos y razas
#nota: la columna 'event' solo se usa aqui para poner los valores de los conteos
pd.pivot_table(data_valid[['sex','race','event']],index=['sex','race'], aggfunc = 'count')

#Sexo
data_valid.groupby(['sex']).size()
(data_valid.groupby(['sex']).size())*100/data_valid['race'].size

#Reincidencia:
two_year_rec_count = (data_valid.loc[(data_valid['two_year_recid'] == 1)]).shape[0]
two_year_rec_count*100/data_valid.shape[0]

#Puntaje por raza
xtable = pd.pivot_table(data_valid[['decile_score','race','event']],index=['decile_score','race'], aggfunc = 'count', dropna = False, fill_value = 0)
deciles_data = data_valid[['race','decile_score']]
black_deciles = deciles_data.loc[deciles_data['race'] == "African-American"]
white_deciles = deciles_data.loc[deciles_data['race'] == "Caucasian"]


#Grafica de puntajes para razas caucasica y negra:
black_deciles['decile_score'].plot.hist(grid = True, bins = 10, rwidth = 1.0)
plt.title('Puntajes COMPAS para raza negra')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)

white_deciles['decile_score'].plot.hist(grid = True, bins = 10, rwidth = 1.0)
plt.title('Puntajes COMPAS para raza blanca')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)

"""
II. ESTUDIO DEL PREJUICIO RACIAL EN COMPAS
"""

"""
Modelos de regresion logistica

Mediante un modelo de regresion logistica que predice la probabilidad
de obtener puntaje de riesgo medio o alto (expresado en una variable binaria)
se comparan las probabilidades para agresores de ciertas cualidades (masculinos, caucasicos, de edad
entre 25 y 40 años, sin ofensas previas que haya cometido crimen grave (felony)
y que no hayan reincidido en al menos dos años) contra todos aquellos que se 
separen de esta norma.

Una de las criticas principales al trabajo de ProPublica fue juntar riesgo medio
y alto, sin comparar el resultado de combinar riesgo bajo y medio.
"""

#Conversion de algunos datos en factores categoricos
data_valid_logreg = data_valid.copy()
data_valid_logreg['crime_factor'] = data_valid_logreg['c_charge_degree'].astype('category')
crime_factor_logreg = pd.get_dummies(data_valid_logreg['crime_factor'], prefix = ['crime_factor'], drop_first = True)

data_valid_logreg['age_factor'] = data_valid_logreg['age_cat'].astype('category')
age_factor_logreg = pd.get_dummies(data_valid_logreg['age_factor'], prefix = ['age_factor'], drop_first = True)



data_valid_logreg['race_factor'] = data_valid_logreg['race'].astype('category')
data_valid_logreg['race_factor'] = data_valid_logreg['race_factor'].cat.reorder_categories(['Caucasian', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other'])
race_factor_logreg = pd.get_dummies(data_valid_logreg['race_factor'], prefix = ['race_factor'], drop_first = True)


data_valid_logreg['gender_factor'] = data_valid_logreg['sex'].astype('category')
data_valid_logreg['gender_factor'] = data_valid_logreg['gender_factor'].cat.reorder_categories(['Male', 'Female'])
gender_factor_logreg = pd.get_dummies(data_valid_logreg['gender_factor'], prefix = ['gender_factor'], drop_first = True)

data_valid_logreg['score_text_1'] = data_valid_logreg['score_text'].replace('Medium', 'High')
data_valid_logreg['score_factor_MH'] = data_valid_logreg['score_text_1'].astype('category')
data_valid_logreg['score_factor_MH'] = data_valid_logreg['score_factor_MH'].cat.reorder_categories(['Low', 'High'])
score_factor_MH_logreg = pd.get_dummies(data_valid_logreg['score_factor_MH'], prefix = ['score_factor_MH'], drop_first = True)

logreg_bias = np.ones((np.size(score_factor_MH_logreg), 1))
lgim = np.array(pd.concat(
        [gender_factor_logreg, age_factor_logreg, race_factor_logreg, data_valid_logreg['priors_count'], crime_factor_logreg, data_valid_logreg['two_year_recid']], axis = 1
        ))

lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)

trgt = np.array(score_factor_MH_logreg)
logreg_target = tf.reshape(tf.cast(trgt, tf.float32), [-1])

 
coeffs_mh, predicted_mh, isconv_mh, iters_mh = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)
coeffs_mh_alt, predicted_mh_alt, isconv_mh_alt, iters_mh_alt = tfp.glm.fit(model_matrix = logreg_input_matrix_alt, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)

#Unir Low y Medium en vez de Medium y High
data_valid_logreg['score_text_2'] = data_valid_logreg['score_text'].replace('Medium', 'Low')
data_valid_logreg['score_factor_LM'] = data_valid_logreg['score_text_2'].astype('category')
data_valid_logreg['score_factor_LM'] = data_valid_logreg['score_factor_LM'].cat.reorder_categories(['Low', 'High'])
score_factor_LM_logreg = pd.get_dummies(data_valid_logreg['score_factor_LM'], prefix = ['score_factor_LM'], drop_first = True)

trgt = np.array(score_factor_LM_logreg)
logreg_target = tf.reshape(tf.cast(trgt, tf.float32), [-1])

coeffs_lm, predicted_lm, isconv_lm, iters_lm = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)
coeffs_lm_alt, predicted_lm_alt, isconv_lm_alt, iters_lm_alt = tfp.glm.fit(model_matrix = logreg_input_matrix_alt, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)

def GetOddsRatio(reference_coeff, case_coeff):
    "Obtiene el 'Odds Ratio (OR) o Razon de Momios dados los coeficientes del termino de bias y de la caracteristica a explorar"
    ref_p = 1/(1 + np.exp(-reference_coeff))
    case_p = 1/(1 + np.exp(-(reference_coeff + case_coeff)))
    OR = case_p/ref_p
    return OR

OR_black = GetOddsRatio(coeffs_mh[0], coeffs_mh[4])
OR_woman = GetOddsRatio(coeffs_mh[0], coeffs_mh[1])
OR_u25 = GetOddsRatio(coeffs_mh[0], coeffs_mh[3])

OR_black_lm = GetOddsRatio(coeffs_lm[0], coeffs_lm[4])
OR_woman_lm = GetOddsRatio(coeffs_lm[0], coeffs_lm[1])
OR_u25_lm = GetOddsRatio(coeffs_lm[0], coeffs_lm[3])

"""
III. ESTUDIO DEL RIESGO DE REINCIDENCIA VIOLENTA
"""

"""
El análisis es similar al elaborado para agresores en general, pero ahora
el estudio se concentra en predicciones y reincidencias violentas. El preproce-
sado de los datos es igual al realizado en el punto I.
"""

#importar datos
raw_data_violent = pd.read_csv(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Arts\Bias\ProPublica COMPAS\compas-analysis-master\compas-analysis-master\compas-scores-two-years-violent.csv")
cats_violent = np.array(raw_data.columns)

#Preprocesamiento paso a paso:
data_violent_date = raw_data_violent.loc[(raw_data_violent['days_b_screening_arrest'] >= -30) & (raw_data_violent['days_b_screening_arrest'] <= 30)]
data_violent_COMPAS = data_violent_date.loc[data_violent_date['is_recid'] != -1]
data_violent = data_violent_COMPAS.loc[data_violent_COMPAS['c_charge_degree'] != 0]

#Observar la distribucion por edad, raza, puntaje (violento) y reincidencia
#Edad:
data_violent.groupby(['age_cat']).size()

#Raza:
data_violent.groupby(['race']).size()

#Puntajevategorico (casos violentos) (v_score_text)
data_violent.groupby(['v_score_text']).size()

#Reincidencia:
two_year_rec_count_violent = (data_violent.loc[(data_violent['two_year_recid'] == 1)]).shape[0]
two_year_rec_count_violent*100/data_violent.shape[0]

#Puntaje (agresores violentos) por raza
xtable_v = pd.pivot_table(data_violent[['v_decile_score','race','event']],index=['v_decile_score','race'], aggfunc = 'count', dropna = False, fill_value = 0)
deciles_data_v = data_violent[['race','v_decile_score']]
black_deciles_v = deciles_data_v.loc[deciles_data_v['race'] == "African-American"]
white_deciles_v = deciles_data_v.loc[deciles_data_v['race'] == "Caucasian"]


#Grafica de puntajes para razas caucasica y negra:
black_deciles_v['v_decile_score'].plot.hist(grid = True, bins = 10, rwidth = 1.0)
plt.title('Puntajes COMPAS para raza negra (casos violentos)')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)

white_deciles_v['v_decile_score'].plot.hist(grid = True, bins = 10, rwidth = 1.0)
plt.title('Puntajes COMPAS para raza blanca (casos volentos)')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)

"""
Regresion logistica para  reincidencia violenta
"""

#Conversion de algunos datos en factores categoricos
data_violent_logreg = data_violent.copy()
data_violent_logreg['crime_factor'] = data_violent_logreg['c_charge_degree'].astype('category')
crime_factor_logreg_v = pd.get_dummies(data_violent_logreg['crime_factor'], prefix = ['crime_factor'], drop_first = True)

data_violent_logreg['age_factor'] = data_violent_logreg['age_cat'].astype('category')
age_factor_logreg_v = pd.get_dummies(data_violent_logreg['age_factor'], prefix = ['age_factor'], drop_first = True)



data_violent_logreg['race_factor'] = data_violent_logreg['race'].astype('category')
data_violent_logreg['race_factor'] = data_violent_logreg['race_factor'].cat.reorder_categories(['Caucasian', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other'])
race_factor_logreg_v = pd.get_dummies(data_violent_logreg['race_factor'], prefix = ['race_factor'], drop_first = True)


data_violent_logreg['gender_factor'] = data_violent_logreg['sex'].astype('category')
data_violent_logreg['gender_factor'] = data_violent_logreg['gender_factor'].cat.reorder_categories(['Male', 'Female'])
gender_factor_logreg_v = pd.get_dummies(data_violent_logreg['gender_factor'], prefix = ['gender_factor'], drop_first = True)

data_violent_logreg['score_text_1'] = data_violent_logreg['v_score_text'].replace('Medium', 'High')
data_violent_logreg['score_factor_MH'] = data_violent_logreg['score_text_1'].astype('category')
data_violent_logreg['score_factor_MH'] = data_violent_logreg['score_factor_MH'].cat.reorder_categories(['Low', 'High'])
score_factor_MH_logreg_v = pd.get_dummies(data_violent_logreg['score_factor_MH'], prefix = ['score_factor_MH'], drop_first = True)

logreg_bias_v = np.ones((np.size(score_factor_MH_logreg_v), 1))
lgim = np.array(pd.concat(
        [gender_factor_logreg_v, age_factor_logreg_v, race_factor_logreg_v, data_violent_logreg['priors_count'], crime_factor_logreg_v, data_violent_logreg['two_year_recid']], axis = 1
        ))

lgim_bias = np.hstack((logreg_bias_v, lgim))
logreg_input_matrix_v = tf.cast(lgim_bias, tf.float32)


trgt = np.array(score_factor_MH_logreg_v)
logreg_target_v = tf.reshape(tf.cast(trgt, tf.float32), [-1])
 
coeffs_mh_v, predicted_mh_v, isconv_mh_v, iters_mh_v = tfp.glm.fit(model_matrix = logreg_input_matrix_v, response = logreg_target_v, model = tfp.glm.Bernoulli(), maximum_iterations = 5)


trgt = np.array(score_factor_LM_logreg)
logreg_target = tf.reshape(tf.cast(trgt, tf.float32), [-1])

coeffs_lm, predicted_lm, isconv_lm, iters_lm = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)
coeffs_lm_alt, predicted_lm_alt, isconv_lm_alt, iters_lm_alt = tfp.glm.fit(model_matrix = logreg_input_matrix_alt, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)

OR_black_v = GetOddsRatio(coeffs_mh_v[0], coeffs_mh_v[4])
OR_u25_v = GetOddsRatio(coeffs_mh_v[0], coeffs_mh_v[3])


"""
IV. PRECISION PREDICTIVA DE COMPAS
"""

"""
ProPublica estudia la precision predictiva de Compas mediante una
regresion de Cox.

En el preprocesado de datos resultantes, se omiten agresores cuya fecha de termino
sea anterior a la de inicio.
"""

"""
wataru
#importar datos
raw_data_cox = pd.read_csv(r"C:\zUzzzsers\Mauricio\Documents\Tesis\Algo Bias\Arts\Bias\ProPublica COMPAS\compas-analysis-master\compas-analysis-master\cox-parsed.csv")
cats_cox = raw_data_cox.columns

#Preprocesado
data_valid_date_cox = raw_data_cox.loc[(raw_data_cox['end'] > raw_data_cox['start'])]
data_valid_score_cox = data_valid_date_cox.dropna(subset = ['score_text'])
data_valid_cox = data_valid_score_cox.drop_duplicates(subset = 'id') 

#Conversion de algunos datos en factores categoricos
data_cox = data_valid_cox.copy()

data_cox['race_factor'] = data_cox['race'].astype('category')
data_cox['race_factor'] = data_cox['race_factor'].cat.reorder_categories(['Caucasian', 'African-American', 'Asian', 'Hispanic', 'Native American', 'Other'])
race_factor_cox = pd.get_dummies(data_cox['race_factor'], prefix = ['race_factor'], drop_first = True)

data_cox['score_factor'] = data_cox['score_text'].astype('category')
data_cox['score_factor'] = data_cox['score_factor'].cat.reorder_categories(['Low', 'High', 'Medium'])
score_factor_cox = pd.get_dummies(data_cox['score_factor'], prefix = ['score_factor'], drop_first = True)


#Observar la distribucion por factores de puntaje y raza
#Puntaje:
data_cox.groupby(['score_factor']).size()

#Raza:
data_cox.groupby(['race_factor']).size()
"""
























