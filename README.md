# Perceptron-and-Decision-Tree

il dataset usato per l'esercizio può essere scaricato qui: https://github.com/zalandoresearch/fashion-mnist
(il link viene fornito in quando github non mi permette di caricare una parte del dataset in quanto superiora la dimensione dei 25mb) 

Il codice consiste di 2 file .py:

il primo, Functions.py, contiene tutte le funzioni necessarie per ottenere i risultati del confronto fra gli alberi di decisione e il perceptron.
Il file contiene:
Una funzione Plot, presa dal materiale fornito nella lezione del 14/12/2017: http://ai.dinfo.unifi.it/teaching/ai_2017.html, necessaria per mostrare il grafico che rappresenta la learning curve di un classificatore, riceve come parametri i valori restituiti dalla funzione learning_curve presente nella libreria sklearn e un titolo con una descrizione opzionale.
Una funzione Parameters che serve a definire i parametri della cross validation e il numero di punti che saranno presenti nel grafico della learning curve. In questo esercizio viene usata la ShuffleSplit della libreria sklearn per effettuare la cross validation.
Due funzioni per plottare le learning curve del perceptron e del decision tree che prendono in ingresso la cross validation selezionata e il train sizes, il set di train e il set di test divisi entrambi in input (X) e output (y) e i parametri necessari per la funzione sklearn che implementa il classificatore. Le funzioni fornisco lo score del training, la accuratezza della predizione su set di test, la confusion_matrix e la classification report di sklearn. Queste informazioni consentono di valutare le prestazioni del classificatore.
Altre due funzioni che effettuano la grid search usando la funzione sklearn gridsearchcv, questa funzione prende come parametro la funzione per la cross validation e restituisce il miglior set di parametri, il miglior score fra tutte le combinazione dei parametri scelti all'interno della funzione e lo score sul training set di ogni combinazione di parametri con relativa imprecisione. 
Queste funzioni sono state prese da: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
L'ultima funzione, BestEstimatorCurve, riceve come parametri il tipo di classificatore, i set di train e test, il train sizes e la funzione di cross validation. Essa effettua la ricerca dei parametri migliori per il classificatore scelto e li usa per plottare la learning curve corrispettiva. Restituisce anche la accuratezza del training, la accuratezza della predizione, la confusion matrix e la classification report per il classificatore usando i parametri che restituiscono il miglior score nel training.
