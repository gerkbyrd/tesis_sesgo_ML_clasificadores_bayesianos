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
import helper

import os



tf.enable_eager_execution()
"""
I.- CARGADO, PREPROCESADO Y VISUALIZACION DE LOS DATOS
"""

#importar datos
raw_data = pd.read_csv(r"SRC/compas-scores-two-years.csv")
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

bns = np.arange(0.75,10.75,0.5)
#Grafica de puntajes para razas caucasica y negra:
black_deciles['decile_score'].plot.hist(grid = True, bins = bns, rwidth = None, color='red')
plt.title('Puntajes COMPAS para raza negra')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)
plt.xticks(np.arange(11))

white_deciles['decile_score'].plot.hist(grid = True, bins = bns, rwidth = None, color='blue')
plt.title('Puntajes COMPAS para raza blanca')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)
plt.xticks(np.arange(11))

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
"""
BerMod = tfp.glm.CustomExponentialFamily(
        distribution_fn = lambda mu: tfp.distributions.Bernoulli(probs=mu),
        linear_model_to_mean_fn = tf.nn.sigmoid,
        is_canonical = True,
        name = 'Bernoulli_Fake'
        )

SmxMod = tfp.glm.CustomExponentialFamily(
        distribution_fn = lambda mu: tfp.distributions.OneHotCategorical(probs=mu),
        linear_model_to_mean_fn = tf.nn.softmax,
        is_canonical = True,
        name = 'OneHot'
        )
CatMod = tfp.glm.CustomExponentialFamily(
        distribution_fn = lambda mu: tfp.distributions.OneHotCategorical(probs=mu,event_shape = [10]),
        linear_model_to_mean_fn = tf.nn.softmax,
        is_canonical = False,
        name = 'Cat'
        )
"""
coeffs_mh, predicted_mh, isconv_mh, iters_mh = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)
"""
coefs1 = tf.reshape(coeffs_mh, (12,1))
coef_mtx = coefs1
for i in range (0,9):
    coef_mtx = tf.concat([coef_mtx, coefs1], 1)
    
#Usar 10 cats
scores_softmax = pd.get_dummies(data_valid_logreg['decile_score'], prefix = ['decile_score'], drop_first = False)
scores_cat = data_valid_logreg['decile_score']
smx_trgt = np.array(scores_softmax)
smx1 = np.reshape(smx_trgt, (10,6172))
cat_trgt = np.array(scores_cat)
smx_target = tf.cast(smx_trgt, tf.float32)
smx_target1 = tf.reshape(smx_target, (10,6172))
coeffs_mh, predicted_mh, isconv_mh, iters_mh = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = BerMod, maximum_iterations = 5)
coeffs_mh_smx, predicted_mh_smx, isconv_mh_smx, iters_mh_smx = tfp.glm.fit(model_matrix = logreg_input_matrix, response = smx_target1, model = SmxMod, maximum_iterations = 5)

coeffs_mh_smx, predicted_mh_smx, isconv_mh_smx, iters_mh_smx = tfp.glm.fit(model_matrix = tf.reshape(logreg_input_matrix,(6172,1,12)), response = tf.reshape(smx_target, (6172,10,)), model = SmxMod, maximum_iterations = 5, fast_unsafe_numerics = False)


preds = np.matmul(logreg_input_matrix, coeffs_mh)
preds = np.matmul(logreg_input_matrix, coefs1)
preds = np.matmul(logreg_input_matrix, coef_mtx)
probs0 = tf.nn.sigmoid(predicted_mh)
probs1 = tf.nn.sigmoid(preds)
probs1 = tf.nn.softmax(preds, axis=0)

coeffs_mh1, predicted_mh1, isconv_mh1, iters_mh1 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = BerMod, maximum_iterations = 5)
"""

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
raw_data_violent = pd.read_csv(r"SRC/compas-scores-two-years-violent.csv")
cats_violent = np.array(raw_data_violent.columns)

#Preprocesamiento paso a paso:
data_violent_date = raw_data_violent.loc[(raw_data_violent['days_b_screening_arrest'] >= -30) & (raw_data_violent['days_b_screening_arrest'] <= 30)]
data_violent_COMPAS = data_violent_date.loc[data_violent_date['is_recid'] != -1]
data_violent = data_violent_COMPAS.loc[data_violent_COMPAS['c_charge_degree'] != 0]



#Observar correlacion entre puntuacion COMPAS y permanencia (crimenes violentos)
jail_out = pd.to_datetime(data_violent['c_jail_out'])
jail_in = pd.to_datetime(data_violent['c_jail_in'])
length_of_stay = jail_out.sub(jail_in)
length_of_stay = length_of_stay.dt.total_seconds()

#correlacion entre tiempo de encarcelamiento y puntaje COMPAS (crimenes violentos)
length_of_stay.corr(data_violent['v_decile_score'])

#Observar la distribucion por edad, raza, puntaje (violento) y reincidencia
#Edad:
data_violent.groupby(['age_cat']).size()

#Raza:
data_violent.groupby(['race']).size()
(data_violent.groupby(['race']).size())*100/data_violent['race'].size

#Puntajevategorico (casos violentos) (v_score_text)
data_violent.groupby(['v_score_text']).size()


#Tabla: sexos y razas
#nota: la columna 'event' solo se usa aqui para poner los valores de los conteos
pd.pivot_table(data_violent[['sex','race','event']],index=['sex','race'], aggfunc = 'count')

#Sexo
data_violent.groupby(['sex']).size()
(data_violent.groupby(['sex']).size())*100/data_violent['race'].size

#Reincidencia:
two_year_rec_count_violent = (data_violent.loc[(data_violent['two_year_recid'] == 1)]).shape[0]
two_year_rec_count_violent*100/data_violent.shape[0]

#Puntaje (agresores violentos) por raza
xtable_v = pd.pivot_table(data_violent[['v_decile_score','race','event']],index=['v_decile_score','race'], aggfunc = 'count', dropna = False, fill_value = 0)
deciles_data_v = data_violent[['race','v_decile_score']]
black_deciles_v = deciles_data_v.loc[deciles_data_v['race'] == "African-American"]
white_deciles_v = deciles_data_v.loc[deciles_data_v['race'] == "Caucasian"]


bns = np.arange(0.75,10.75,0.5)
#Grafica de puntajes para razas caucasica y negra:
black_deciles_v['v_decile_score'].plot.hist(grid = True, bins = bns, rwidth = None, color='red')
plt.title('Puntajes COMPAS para raza negra (reincidencia violenta)')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)
plt.xticks(np.arange(11))


white_deciles_v['v_decile_score'].plot.hist(grid = True, bins = bns, rwidth = 1.0)
plt.title('Puntajes COMPAS para raza blanca (reincidencia violenta)')
plt.xlabel('Puntaje obtenido')
plt.ylabel('Conteo total en la poblacion')
plt.grid(axis='y', alpha=0.75)
plt.xticks(np.arange(11))

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
 
coeffs_mh_v, predicted_mh_v, isconv_mh_v, iters_mh_v = tfp.glm.fit(model_matrix = logreg_input_matrix_v, response = logreg_target_v, model = tfp.glm.Bernoulli(), maximum_iterations = 6)

data_violent_logreg['score_text_2'] = data_violent_logreg['score_text'].replace('Medium', 'Low')
data_violent_logreg['score_factor_LM'] = data_violent_logreg['score_text_2'].astype('category')
data_violent_logreg['score_factor_LM'] = data_violent_logreg['score_factor_LM'].cat.reorder_categories(['Low', 'High'])
score_factor_LM_logreg_v = pd.get_dummies(data_violent_logreg['score_factor_LM'], prefix = ['score_factor_LM'], drop_first = True)

trgt = np.array(score_factor_LM_logreg_v)
logreg_target_v = tf.reshape(tf.cast(trgt, tf.float32), [-1])


coeffs_lm, predicted_lm, isconv_lm, iters_lm = tfp.glm.fit(model_matrix = logreg_input_matrix_v, response = logreg_target_v, model = tfp.glm.Bernoulli(), maximum_iterations = 6)
coeffs_lm_alt, predicted_lm_alt, isconv_lm_alt, iters_lm_alt = tfp.glm.fit(model_matrix = logreg_input_matrix_alt, response = logreg_target, model = tfp.glm.Bernoulli(), maximum_iterations = 5)

OR_black_v = GetOddsRatio(coeffs_mh_v[0], coeffs_mh_v[4])
OR_woman_v = GetOddsRatio(coeffs_mh_v[0], coeffs_mh_v[1])
OR_u25_v = GetOddsRatio(coeffs_mh_v[0], coeffs_mh_v[3])


OR_black_lm_v = GetOddsRatio(coeffs_lm[0], coeffs_lm[4])
OR_woman_lm_v = GetOddsRatio(coeffs_lm[0], coeffs_lm[1])
OR_u25_lm_v = GetOddsRatio(coeffs_lm[0], coeffs_lm[3])


"""
Regresión Softmax
"""
"""
#Usar 10 cats
scores_softmax = pd.get_dummies(data_valid_logreg['decile_score'], prefix = ['decile_score'], drop_first = False)

smx_trgt = np.array(scores_softmax)
smx_target = tf.cast(trgt, tf.float32)

coeffs_lm, predicted_lm, isconv_lm, iters_lm = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.CustomExponentialFamily(), maximum_iterations = 5)
"""

"""
II.A ESTUDIO DE PREJUICIO RACIAL (FLORES, LOWENKAMP)
"""
data_valid_BW = data_valid_COMPAS.loc[(data_valid_COMPAS['c_charge_degree'] != 0)&((data_valid_COMPAS['race'] == 'African-American')|(data_valid_COMPAS['race'] == 'Caucasian'))]

data_valid_logreg = data_valid_BW.copy()

"""
Análisis iniciales
"""
#Reincidencia por raza
data_valid_BW.loc[data_valid_BW['two_year_recid']==1].shape[0]/data_valid_BW.shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'African-American')].shape[0]/data_valid_BW.loc[data_valid_COMPAS['race'] == 'African-American'].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'Caucasian')].shape[0]/data_valid_BW.loc[data_valid_COMPAS['race'] == 'Caucasian'].shape[0]

#Reincidencia por raza y COMPAS categórico (BAJO)
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['score_text'] == 'Low')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['score_text'] == 'Low')].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'African-American')&(data_valid_COMPAS['score_text'] == 'Low')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['race'] == 'African-American')&(data_valid_COMPAS['score_text'] == 'Low')].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'Caucasian')&(data_valid_COMPAS['score_text'] == 'Low')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['race'] == 'Caucasian')&(data_valid_COMPAS['score_text'] == 'Low')].shape[0]

#Reincidencia por raza y COMPAS categórico (MEDIO)
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['score_text'] == 'Medium')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['score_text'] == 'Medium')].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'African-American')&(data_valid_COMPAS['score_text'] == 'Medium')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['race'] == 'African-American')&(data_valid_COMPAS['score_text'] == 'Medium')].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'Caucasian')&(data_valid_COMPAS['score_text'] == 'Medium')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['race'] == 'Caucasian')&(data_valid_COMPAS['score_text'] == 'Medium')].shape[0]

#Reincidencia por raza y COMPAS categórico (ALTO)
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['score_text'] == 'High')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['score_text'] == 'High')].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'African-American')&(data_valid_COMPAS['score_text'] == 'High')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['race'] == 'African-American')&(data_valid_COMPAS['score_text'] == 'High')].shape[0]
data_valid_BW.loc[(data_valid_BW['two_year_recid']==1)&(data_valid_COMPAS['race'] == 'Caucasian')&(data_valid_COMPAS['score_text'] == 'High')].shape[0]/data_valid_BW.loc[(data_valid_COMPAS['race'] == 'Caucasian')&(data_valid_COMPAS['score_text'] == 'High')].shape[0]


data_valid_logreg['age_factor'] = data_valid_logreg['age'].astype('int64')
age_factor_logreg = data_valid_logreg['age_factor']


data_valid_logreg['race_factor'] = data_valid_logreg['race'].astype('category')
data_valid_logreg['race_factor'] = data_valid_logreg['race_factor'].cat.reorder_categories(['Caucasian', 'African-American'])
race_factor_logreg = pd.get_dummies(data_valid_logreg['race_factor'], prefix = ['race_factor'], drop_first = True)


data_valid_logreg['gender_factor'] = data_valid_logreg['sex'].astype('category')
data_valid_logreg['gender_factor'] = data_valid_logreg['gender_factor'].cat.reorder_categories(['Male', 'Female'])
gender_factor_logreg = pd.get_dummies(data_valid_logreg['gender_factor'], prefix = ['gender_factor'], drop_first = True)


raceXdec_factor_logreg = np.multiply(np.reshape(np.array(data_valid_logreg['decile_score']), (5278,1)), race_factor_logreg)
logreg_bias = np.ones((np.size(data_valid_logreg['two_year_recid']), 1))

"""
PRIMER MODELO
"""
lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg,  race_factor_logreg], axis = 1
        ))


#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)

trgt = np.array(data_valid_logreg['two_year_recid'])
logreg_target = tf.reshape(tf.cast(trgt, tf.float32), [-1])

coeffs_bw1, predicted_bw1, isconv_bw1, iters_bw1 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())
OR_black_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[3])
OR_female_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[2])
OR_age_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[1])
OR_self_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[0])
"""
SEGUNDO MODELO
"""

lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg, data_valid_logreg['decile_score']], axis = 1
        ))
#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)


coeffs_bw2, predicted_bw2, isconv_bw2, iters_bw2 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())

OR_female_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[2])
OR_age_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[1])
OR_DEC_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[3])
OR_self_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[0])
"""
TERCER MODELO
"""

lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg,  race_factor_logreg, data_valid_logreg['decile_score']], axis = 1
        ))
#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)


coeffs_bw3, predicted_bw3, isconv_bw3, iters_bw3 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())
OR_black_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[3])
OR_female_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[2])
OR_age_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[1])
OR_DEC_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[4])
OR_self_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[0])

"""
CUARTO MODELO
"""

lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg, race_factor_logreg, data_valid_logreg['decile_score'], raceXdec_factor_logreg], axis = 1
        ))
#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)


coeffs_bw4, predicted_bw4, isconv_bw4, iters_bw4 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())
OR_black_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[3])
OR_female_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[2])
OR_age_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[1])
OR_DEC_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[4])
OR_blackXdec_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[5])
OR_self_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[0])

preds = np.array(tf.sigmoid(predicted_bw4))
preds = preds.reshape((preds.size, 1))
deciles = np.array(data_valid_logreg['decile_score'])
deciles = deciles.reshape((deciles.size, 1))
race = np.array(race_factor_logreg)
plot_data = np.array(np.concatenate(
        [race, deciles, preds], axis = 1
        ))
plot_data_w = plot_data[np.where(plot_data[:, 0] == 0)][:,1:3]
plot_data_b = plot_data[np.where(plot_data[:, 0] == 1)][:,1:3]

res_w, res_b  = np.zeros(10), np.zeros(10)
for x in range(10):
    res_w[x] = np.mean(plot_data_w[np.where(plot_data_w[:,0] == x + 1)][:,1])


res_b = np.zeros(10)
for x in range(10):
    res_b[x] = np.mean(plot_data_b[np.where(plot_data_b[:,0] == x + 1)][:,1])

from matplotlib import figure
from matplotlib.backends import backend_agg
points = np.arange(11)[1:11]
labels = np.array(['Elementos de raza negra', 'Elementos de raza blanca'])
fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)
ax.set_prop_cycle(color = ['red', 'blue'])

for i, y in enumerate([res_b, res_w]):
    ax.plot(points, y, lw=2, label = labels[i])
    
ax.set_xlim([0, 10.1])
ax.set_ylim([0.18, 0.81])
ax.set_xticks(np.arange(11))

"""
ax.vlines(means, -0.1, 12., linestyles='dashed', lw = 1.5)#np.max(pdfs)])
ax.hlines(0., -3., 4., linestyles='solid', lw = 1.)
"""
ax.set_title('Probabilidad promedio estimada de reincidencia general por raza')
ax.set_xlabel('Puntaje COMPAS', fontsize = 15)
ax.set_ylabel('Probabilidad promedio', fontsize = 12)

ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

ax.legend()
fname=os.path.join(r"SRC/Flores et Al/Figuras/",
                                         "FloresFig1.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))


"""
III.A ESTUDIO DE PREJUICIO RACIAL (FLORES, LOWENKAMP) para reincidencia violenta
"""
data_violent_BW = data_violent_COMPAS.loc[(data_violent['c_charge_degree'] != 0)&((data_violent_COMPAS['race'] == 'African-American')|(data_violent_COMPAS['race'] == 'Caucasian'))]

#data_violent_BW = raw_data_violent.loc[((raw_data_violent['race'] == 'African-American')|(raw_data_violent['race'] == 'Caucasian'))]

#Reincidencia por raza
data_violent_BW.loc[data_violent_BW['two_year_recid']==1].shape[0]/data_violent_BW.shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'African-American')].shape[0]/data_violent_BW.loc[data_violent_BW['race'] == 'African-American'].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'Caucasian')].shape[0]/data_violent_BW.loc[data_violent_BW['race'] == 'Caucasian'].shape[0]

#Reincidencia por raza y COMPAS categórico (BAJO)
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['score_text'] == 'Low')].shape[0]/data_violent_BW.loc[(data_violent_BW['score_text'] == 'Low')].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'African-American')&(data_violent_BW['score_text'] == 'Low')].shape[0]/data_violent_BW.loc[(data_violent_BW['race'] == 'African-American')&(data_violent_BW['score_text'] == 'Low')].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'Caucasian')&(data_violent_BW['score_text'] == 'Low')].shape[0]/data_violent_BW.loc[(data_violent_BW['race'] == 'Caucasian')&(data_violent_BW['score_text'] == 'Low')].shape[0]

#Reincidencia por raza y COMPAS categórico (MEDIO)
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['score_text'] == 'Medium')].shape[0]/data_violent_BW.loc[(data_violent_BW['score_text'] == 'Medium')].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'African-American')&(data_violent_BW['score_text'] == 'Medium')].shape[0]/data_violent_BW.loc[(data_violent_BW['race'] == 'African-American')&(data_violent_BW['score_text'] == 'Medium')].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'Caucasian')&(data_violent_BW['score_text'] == 'Medium')].shape[0]/data_violent_BW.loc[(data_violent_BW['race'] == 'Caucasian')&(data_violent_BW['score_text'] == 'Medium')].shape[0]

#Reincidencia por raza y COMPAS categórico (ALTO)
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['score_text'] == 'High')].shape[0]/data_violent_BW.loc[(data_violent_BW['score_text'] == 'High')].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'African-American')&(data_violent_BW['score_text'] == 'High')].shape[0]/data_violent_BW.loc[(data_violent_BW['race'] == 'African-American')&(data_violent_BW['score_text'] == 'High')].shape[0]
data_violent_BW.loc[(data_violent_BW['two_year_recid']==1)&(data_violent_BW['race'] == 'Caucasian')&(data_violent_BW['score_text'] == 'High')].shape[0]/data_violent_BW.loc[(data_violent_BW['race'] == 'Caucasian')&(data_violent_BW['score_text'] == 'High')].shape[0]

data_violent_logreg = data_violent_BW.copy()

data_violent_logreg['age_factor'] = data_violent_logreg['age'].astype('int64')
age_factor_logreg = data_violent_logreg['age_factor']


data_violent_logreg['race_factor'] = data_violent_logreg['race'].astype('category')
data_violent_logreg['race_factor'] = data_violent_logreg['race_factor'].cat.reorder_categories(['Caucasian', 'African-American'])
race_factor_logreg = pd.get_dummies(data_violent_logreg['race_factor'], prefix = ['race_factor'], drop_first = True)


data_violent_logreg['gender_factor'] = data_violent_logreg['sex'].astype('category')
data_violent_logreg['gender_factor'] = data_violent_logreg['gender_factor'].cat.reorder_categories(['Male', 'Female'])
gender_factor_logreg = pd.get_dummies(data_violent_logreg['gender_factor'], prefix = ['gender_factor'], drop_first = True)

raceXdec_factor_logreg = np.multiply(np.reshape(np.array(data_violent_logreg['decile_score']), (data_violent_logreg.shape[0],1)), race_factor_logreg)
logreg_bias = np.ones((np.size(data_violent_logreg['two_year_recid']), 1))

"""
PRIMER MODELO
"""
lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg,  race_factor_logreg], axis = 1
        ))


#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)

trgt = np.array(data_violent_logreg['two_year_recid'])
logreg_target = tf.reshape(tf.cast(trgt, tf.float32), [-1])

coeffs_bw1, predicted_bw1, isconv_bw1, iters_bw1 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())
OR_black_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[3])
OR_female_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[2])
OR_age_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[1])
OR_self_bw1 = GetOddsRatio(coeffs_bw1[0], coeffs_bw1[0])
"""
SEGUNDO MODELO
"""

lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg, data_violent_logreg['decile_score']], axis = 1
        ))
#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)


coeffs_bw2, predicted_bw2, isconv_bw2, iters_bw2 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())

OR_female_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[2])
OR_age_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[1])
OR_DEC_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[3])
OR_self_bw2 = GetOddsRatio(coeffs_bw2[0], coeffs_bw2[0])
"""
TERCER MODELO
"""

lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg,  race_factor_logreg, data_violent_logreg['decile_score']], axis = 1
        ))
#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)


coeffs_bw3, predicted_bw3, isconv_bw3, iters_bw3 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())
OR_black_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[3])
OR_female_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[2])
OR_age_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[1])
OR_DEC_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[4])
OR_self_bw3 = GetOddsRatio(coeffs_bw3[0], coeffs_bw3[0])

"""
CUARTO MODELO
"""

lgim = np.array(pd.concat(
        [age_factor_logreg, gender_factor_logreg, race_factor_logreg, data_violent_logreg['decile_score'], raceXdec_factor_logreg], axis = 1
        ))
#lgim_alt = (lgim > 0) #explorar ofensas previas en terminos binarios (no importa cuantas veces se haya ofendido antes)
lgim_bias = np.hstack((logreg_bias, lgim))
#lgim_bias_alt = np.hstack((logreg_bias, lgim_alt))

logreg_input_matrix = tf.cast(lgim_bias, tf.float32)
lgrg_in_mtx_base = tf.cast(lgim, tf.float32)
lgrg_in_bias = np.array(logreg_input_matrix)
#logreg_input_matrix_alt = tf.cast(lgim_bias_alt, tf.float32)


coeffs_bw4, predicted_bw4, isconv_bw4, iters_bw4 = tfp.glm.fit(model_matrix = logreg_input_matrix, response = logreg_target, model = tfp.glm.Bernoulli())
OR_black_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[3])
OR_female_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[2])
OR_age_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[1])
OR_DEC_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[4])
OR_blackXdec_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[5])
OR_self_bw4 = GetOddsRatio(coeffs_bw4[0], coeffs_bw4[0])

preds = np.array(tf.sigmoid(predicted_bw4))
preds = preds.reshape((preds.size, 1))
deciles = np.array(data_violent_logreg['decile_score'])
deciles = deciles.reshape((deciles.size, 1))
race = np.array(race_factor_logreg)
plot_data = np.array(np.concatenate(
        [race, deciles, preds], axis = 1
        ))
plot_data_w = plot_data[np.where(plot_data[:, 0] == 0)][:,1:3]
plot_data_b = plot_data[np.where(plot_data[:, 0] == 1)][:,1:3]

res_w, res_b  = np.zeros(10), np.zeros(10)
for x in range(10):
    res_w[x] = np.mean(plot_data_w[np.where(plot_data_w[:,0] == x + 1)][:,1])


res_b = np.zeros(10)
for x in range(10):
    res_b[x] = np.mean(plot_data_b[np.where(plot_data_b[:,0] == x + 1)][:,1])

from matplotlib import figure
from matplotlib.backends import backend_agg
points = np.arange(11)[1:11]
labels = np.array(['Elementos de raza negra', 'Elementos de raza blanca'])
fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)
ax.set_prop_cycle(color = ['red', 'blue'])

for i, y in enumerate([res_b, res_w]):
    ax.plot(points, y, lw=2, label = labels[i])
    
ax.set_xlim([0, 10.1])
ax.set_ylim([0.05, 0.6])
ax.set_xticks(np.arange(11))

"""
ax.vlines(means, -0.1, 12., linestyles='dashed', lw = 1.5)#np.max(pdfs)])
ax.hlines(0., -3., 4., linestyles='solid', lw = 1.)
"""
ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos filtrados, N = 3377)')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos sin filtrar, N = 3967)')
ax.set_xlabel('Puntaje COMPAS', fontsize = 15)
ax.set_ylabel('Probabilidad promedio', fontsize = 12)

ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

ax.legend()
fname=os.path.join(r"SRC/Flores et Al/Figuras/",
                                         "FloresFig2.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))

fname=os.path.join(r"SRC/Flores et Al/Figuras/",
                                         "FloresFig2_nf.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))







































































