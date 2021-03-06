# tesis_sesgo_ML_clasificadores_bayesianos
ESTUDIOS DE SESGOS O PREJUICIOS EN EL APRENDIZAJE COMPUTACIONAL: ENFOQUE EN EL DISEÑO DE MODELOS CON CLASIFICADORES BAYESIANOS
Este repositorio contiene el código desarrollado en la elaboración de la tesis con el mismo título.

Para facilitar la reproducción del trabajo, se ponen disponibles varios arreglos que es necesario obtener de los distintos programas implementados haciendo las modificaciones pertinentes y guardando los resultados de varias ejecuciones.

SRC: además de contener las carpetas que separan el código en distintas secciones, contiene los datos del caso de estudio en formato .csv (que también se pueden obtener en https://github.com/propublica/compas-analysis/) y dos programas importantes. Propublica-Flores (frequentist).py contiene el código utilizado para obtener todos los resultados relacionados con implementaciones frecuentistas, desde las secciones de "REPRODUCCIÓN" para los capítulos 11 y 12 de la tesis, hasta las secciones del anexo: "DISTRIBUCIONES DE PUNTAJE COMPAS POR RAZA", "VARIABLE BINARIA DE PUNTAJE COMPAS ALTO", y "DIFERENCIAS POR CAMBIOS EN LA OPTIMIZACIÓN"; sin embargo, para la mayoría de estas secciones los arreglos con los datos necesarios ya están almacenados en otras carpetas, por lo que no es necesario ejecutar este archivo para obtener esos resultados. El archivo helper.py, contiene varias funciones que se definieron para simplificar la mayoría de las implementaciones en otros archivos, y se importa en casi todos los programas.

SRC/Anexo: Contiene el código y arreglos necesarios para la sección del anexo "RENDIMIENTO DE MODELOS CLASIFICADORES". 

SRC/Flores et Al.: Contiene el código usado en la reproducción y en la implementación bayesiana del trabajo de Flores et. Al. (https://www.researchgate.net/publication/306032039_False_Positives_False_Negatives_and_False_Analyses_A_Rejoinder_to_Machine_Bias_There's_Software_Used_Across_the_Country_to_Predict_Future_Criminals_And_it's_Biased_Against_Blacks). "Lowenkamp bayesian.py" contiene el código necesario para obtener los resultados bayesianos y los resultados frecuentistas que usan optimizador Adam para reincidencia general, y "Lowenkamp violent bayesian.py" contiene el código para este mismo propósito con el caso de reincidencia violenta. En "performance_lowenkamp.py" está el código para obtener las medidas de rendimiento, tanto bayesianas como frecuentistas, dentro del análisis de este caso de estudio en el capítulo 11. Se incluyen los arreglos necesarios en la carpeta correspondiente. También se incluyen algunas figuras generadas con el código en esta carpeta, correspondientes también al capítulo 11 del escrito.

SRC/Otros: simplemente contiene el código usado para la figura de la función sigmoide presentada en el capítulo 5 "CLASIFICADORES", en sigmplot.py también se incluye la figura generada.

SRC/ProPublica: contiene el código y arreglos necesarios para las implementaciones bayesianas y frecuentistas (con optimizador Adam) correspondientes al caso de estudio de ProPublica (https://github.com/propublica/compas-analysis), en el capítulo 10 del escrito. "Propublica bayesian.py", y "Propublica bayesian violent.py" contienen el código para los casos de reincidencia general y violenta, respectivamente. Ambos códigos también se usaron para el desarrollo de las secciones "MODIFICACIONES DE LA VARIANZA A PRIORI" y "DIFERENCIAS POR CAMBIOS EN LA OPTIMIZACIÓN" en el anexo del escrito. "performance_propublica.py" contiene el código necesario para la primera parte de la sección "RENDIMIENTO DE MODELOS CLASIFICADORES" de dicho anexo. También se incluyen algunas de las figuras generadas para el capítulo 10 en la subcarpeta pertinente. 

SRC/Sinteticos: contiene el código necesario para generar los resultados del capítulo 7 del escrito, y también las figuras generadas para el mismo.



