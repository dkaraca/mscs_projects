import plotly.express as px
import numpy as np
import random
from scipy import optimize

import scipy
import mlrose_hiive

import time

time.time()


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, QuantileTransformer

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import (RocCurveDisplay, precision_recall_curve, roc_curve, det_curve, roc_auc_score, brier_score_loss,
                            confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,
                            make_scorer)

import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

def main():

    df = pd.read_csv("train.csv")
    df['Label'] = df['Label'].map({'subjective': 1, 'objective': 0})
    frequency_columns = ['CC', 'CD', 'DT', 'EX', 'FW', 'INs', 'JJ',
           'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
           'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TOs', 'UH', 'VB',
           'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB',
           'baseform', 'Quotes', 'questionmarks', 'exclamationmarks',
           'fullstops', 'commas', 'semicolon', 'colon', 'ellipsis',
           'pronouns1st', 'pronouns2nd', 'pronouns3rd', 'compsupadjadv',
           'past', 'imperative', 'present3rd', 'present1st2nd','semanticobjscore','semanticsubjscore']

    sss = StratifiedShuffleSplit(test_size=0.2, random_state=42, n_splits=2)
    for train_idx, test_idx in sss.split(df, df['Label']):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        hold_out_df = df.iloc[test_idx].reset_index(drop=True)

    frequency_norm_cols = [col + '_norm' for col in frequency_columns]
    train_cols = frequency_norm_cols + ['totalWordsCount_norm','sentence1st','sentencelast']

    for col in frequency_columns:
        train_df[col + '_norm'] = train_df[col] / train_df['totalWordsCount']
        hold_out_df[col + '_norm'] = hold_out_df[col] / hold_out_df['totalWordsCount']
        
    ss_word_count = StandardScaler()

    train_df['totalWordsCount_norm'] = ss_word_count.fit_transform(train_df['totalWordsCount'].values.reshape(-1, 1))
    hold_out_df['totalWordsCount_norm'] = ss_word_count.transform(hold_out_df['totalWordsCount'].values.reshape(-1, 1))

    exclude = [e + '_norm' for e in ['JJS','NNP','PRP','TOs','VBD','VBG','WP','WRB','exclamationmarks','fullstops','commas','semicolon','colon','ellipsis','EX','FW','INs']]
    train_cols = [e for e in train_cols if e not in exclude]

    train_df


    # In[285]:


    gds = ['gradient_descent']
    gd_curves = []
    for gd in gds:
        start_time = time.time()
        nn = mlrose_hiive.neural.NeuralNetwork(hidden_nodes=[5, 5],
                         activation='tanh',
                         algorithm=gd,
                         max_iters=1000,
                         bias=True,
                         is_classifier=True,
                         learning_rate=0.001,
                         early_stopping=False,
                         clip_max=3,
                         restarts=res,
                         schedule=mlrose_hiive.algorithms.decay.GeomDecay(),
                         pop_size=200,
                         mutation_prob=0.1,
                         max_attempts=10,
                         random_state=None,
                         curve=True)
        nn.fit(train_df[train_cols], train_df['Label'])
        end_time = time.time()
        gd_curves.append((gd, nn.fitness_curve, accuracy_score(hold_out_df['Label'], nn.predict(hold_out_df[train_cols])), end_time-start_time))


    # In[277]:


    restarts = [0, 10, 100, 1000]
    restart_curves = []
    for res in restarts:
        print(res)
        start_time = time.time()
        nn = mlrose_hiive.neural.NeuralNetwork(hidden_nodes=[5, 5],
                         activation='tanh',
                         algorithm='random_hill_climb',
                         max_iters=1000,
                         bias=True,
                         is_classifier=True,
                         learning_rate=0.001,
                         early_stopping=False,
                         clip_max=3,
                         restarts=res,
                         schedule=mlrose_hiive.algorithms.decay.GeomDecay(),
                         pop_size=200,
                         mutation_prob=0.1,
                         max_attempts=10,
                         random_state=None,
                         curve=True)
        nn.fit(train_df[train_cols], train_df['Label'])
        end_time = time.time()
        restart_curves.append((res, nn.fitness_curve, accuracy_score(hold_out_df['Label'], nn.predict(hold_out_df[train_cols])), end_time-start_time))


    # In[317]:


    schedules = [('exponential', mlrose_hiive.algorithms.decay.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001))]
    #('geometric', mlrose_hiive.algorithms.decay.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)),
    #          ('arithmetic', mlrose_hiive.algorithms.decay.ArithDecay(init_temp=1.0, decay=0.99, min_temp=0.001)),
    #          ('exponential', mlrose_hiive.algorithms.decay.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001))
    schedule_curves = []
    for sched in schedules:
        print(res)
        start_time = time.time()
        nn = mlrose_hiive.neural.NeuralNetwork(hidden_nodes=[5, 5],
                         activation='tanh',
                         algorithm='simulated_annealing',
                         max_iters=1000,
                         bias=True,
                         is_classifier=True,
                         learning_rate=0.001,
                         early_stopping=False,
                         clip_max=3,
                         restarts=res,
                         schedule=sched[1],
                         pop_size=200,
                         mutation_prob=0.1,
                         max_attempts=10,
                         random_state=None,
                         curve=True)
        nn.fit(train_df[train_cols], train_df['Label'])
        end_time = time.time()
        schedule_curves.append((sched[0], nn.fitness_curve, accuracy_score(hold_out_df['Label'], nn.predict(hold_out_df[train_cols])), end_time-start_time))


    # In[281]:


    mut_probs = [0.1]#, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mut_curves = []
    for prob in mut_probs:
        print(res)
        start_time = time.time()
        nn = mlrose_hiive.neural.NeuralNetwork(hidden_nodes=[5, 5],
                         activation='tanh',
                         algorithm='genetic_alg',
                         max_iters=1000,
                         bias=True,
                         is_classifier=True,
                         learning_rate=0.001,
                         early_stopping=False,
                         clip_max=3,
                         pop_size=train_df.shape[0],
                         mutation_prob=prob,
                         max_attempts=10,
                         random_state=None,
                         curve=True)
        nn.fit(train_df[train_cols], train_df['Label'])
        end_time = time.time()
        mut_curves.append((prob, nn.fitness_curve, accuracy_score(hold_out_df['Label'], nn.predict(hold_out_df[train_cols])), end_time-start_time))


    # In[316]:


    loss_df = pd.concat([
        pd.DataFrame(
            [(i, 'RHC', val) for i, val in enumerate(restart_curves[0][1][:, 0])],
            columns=['Iterations', 'Algorithm', 'Loss']
        ),
        pd.DataFrame(
            [(i, 'SA', val) for i, val in enumerate(schedule_curves[0][1][:, 0])],
            columns=['Iterations', 'Algorithm', 'Loss']
        ),
        pd.DataFrame(
            [(i, 'GA', val) for i, val in enumerate(mut_curves[0][1][:, 0])],
            columns=['Iterations', 'Algorithm', 'Loss']
        ),
        pd.DataFrame(
            [(i, 'GD', val) for i, val in enumerate(gd_curves[0][1]*-1)],
            columns=['Iterations', 'Algorithm', 'Loss']
        )
    ], axis=0).reset_index(drop=True)

    import plotly.express as px

    px.line(loss_df, x='Iterations', y='Loss', color='Algorithm', title='Loss per Iteration using different optimizers')



    import plotly.express as px
    import numpy as np
    import random
    from scipy import optimize

    import scipy


    # In[3]:


    n_nodes = 100

    import networkx as nx
    import matplotlib.pyplot as plt

    kcolors = ['red', 'green', 'blue', 'yellow']
    color_to_bin = {'red': 1, 'green': 0, 'blue': 2, 'yellow': 3}

    random_edges = {}
    degrees = {}
    for i in range(n_nodes):
        random_edges[i] = []
        if i not in degrees:
            degrees[i] = 0
        for _ in range(2):
            conn_node = random.sample(range(10), 1)[0]
            if conn_node not in degrees:
                degrees[conn_node] = 0
            if conn_node != i and conn_node not in random_edges[i]:
                random_edges[i].append(conn_node)
                degrees[i] += 1
                degrees[conn_node] += 1
                
    G = nx.Graph(random_edges)
    colors = []
    init_colors = []
    for n in G.nodes():
        curr_color = random.sample(kcolors, 1)[0]
        colors.append(curr_color)
        init_colors.append(color_to_bin[curr_color])

    nx.draw(G, node_color=colors)
    plt.show()


    # In[4]:


    def get_fitness_kcolors(color_state):
        fitness = 0
        for edge in G.edges():
            if color_state[edge[0]] != color_state[edge[1]]:
                fitness+= 1
        return fitness
    get_fitness_kcolors(init_colors)


    # In[11]:


    import mlrose_hiive
    import time


    # In[30]:


    restarts = [0, 10, 100, 1000]
    restart_curves = []
    for res in restarts:
        kcolor_fitness = mlrose_hiive.fitness.MaxKColor(G.edges())
        opt = mlrose_hiive.MaxKColorOpt(edges=G.edges(), length=n_nodes, fitness_fn=kcolor_fitness, maximize=True, max_colors=4)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.random_hill_climb(
              opt, max_attempts=10, max_iters=100, restarts=res,
              init_state=None, curve=True, random_state=None,
              state_fitness_callback=None, callback_user_info=None
        )
        end = time.time()
        restart_curves.append((res, opt_sol, opt_val, curr_fitness, end-start))


    # In[98]:


    decays = [('geometric', mlrose_hiive.algorithms.decay.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)),
              ('arithmetic', mlrose_hiive.algorithms.decay.ArithDecay(init_temp=1.0, decay=0.99, min_temp=0.001)),
              ('exponential', mlrose_hiive.algorithms.decay.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001))]
    decay_curves = []
    for dec in decays:
        kcolor_fitness = mlrose_hiive.fitness.MaxKColor(G.edges())
        opt = mlrose_hiive.MaxKColorOpt(edges=G.edges(), length=n_nodes, fitness_fn=kcolor_fitness, maximize=True, max_colors=4)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.simulated_annealing(
            opt, schedule=dec[1],
            max_attempts=10, max_iters=1000, init_state=None, curve=True,
            fevals=False, random_state=None,
            state_fitness_callback=None, callback_user_info=None
        )
        end = time.time()
        decay_curves.append((dec[0], opt_sol, opt_val, curr_fitness, end-start))


    # In[22]:


    pop_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    percent_curves = []

    for perc in pop_percent:
        kcolor_fitness = mlrose_hiive.fitness.MaxKColor(G.edges())
        opt = mlrose_hiive.MaxKColorOpt(edges=G.edges(), length=n_nodes, fitness_fn=kcolor_fitness, maximize=True, max_colors=4)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.genetic_alg(
            opt, pop_size=len(G.nodes()), pop_breed_percent=perc, elite_dreg_ratio=0.95,
            minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
            max_attempts=10, max_iters=100, curve=True, random_state=None
        )
        end = time.time()
        percent_curves.append((perc, opt_sol, opt_val, curr_fitness, end-start))


    # In[108]:


    keep_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    keep_curves = []
    for perc in keep_percent:
        print(perc)
        kcolor_fitness = mlrose_hiive.fitness.MaxKColor(G.edges())
        opt = mlrose_hiive.MaxKColorOpt(edges=G.edges(), length=n_nodes, fitness_fn=kcolor_fitness, maximize=True, max_colors=4)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.mimic(
            opt, pop_size=len(G.nodes()), keep_pct=perc, max_attempts=10,
            max_iters=100, curve=True, random_state=None,
            state_fitness_callback=None, callback_user_info=None, noise=0.0
        )
        end = time.time()
        keep_curves.append((perc, opt_sol, opt_val, curr_fitness, end-start))


    # In[38]:


    import pandas as pd
    rhc_df = pd.DataFrame(
        [('RHC', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in restart_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Restarts', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    sa_df = pd.DataFrame(
        [('SA', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in decay_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Decay Type', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    ga_df = pd.DataFrame(
        [('GA', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in percent_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Breeding Percentage', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    mimic_df = pd.DataFrame(
        [('MIMIC', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in keep_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Keep Percentage', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    fig = px.line(rhc_df, x='Iteration', y='Fitness Value', color='Restarts', title='RHC Performance in 4-Color with Different Restart Counts')
    fig.update_yaxes(range=[120, 190])
    fig.update_xaxes(range=[0, 1000])
    fig.show()

    fig = px.line(sa_df, x='Iteration', y='Fitness Value', color='Decay Type', title='SA Performance in 4-Color with Different Decay Functions')
    fig.update_yaxes(range=[120, 190])
    fig.update_xaxes(range=[0, 1000])
    fig.show()

    fig = px.line(ga_df, x='Iteration', y='Fitness Value', color='Breeding Percentage', title='GA Performance in 4-Color with Different Breeding Percentages')
    fig.update_yaxes(range=[120, 190])
    fig.update_xaxes(range=[0, 1000])
    fig.show()

    fig = px.line(mimic_df, x='Iteration', y='Fitness Value', color='Keep Percentage', title='MIMIC Performance in 4-Color with Different Keep Percentages')
    fig.update_yaxes(range=[120, 190])
    fig.update_xaxes(range=[0, 1000])
    fig.show()



    import plotly.express as px
    import numpy as np
    import random
    from scipy import optimize

    import scipy


    # In[2]:


    import mlrose_hiive
    import time


    # In[ ]:


    restarts = [0, 10, 100, 1000]
    restart_curves = []
    for res in restarts:
        fit_func = mlrose_hiive.fitness.SixPeaks(t_pct=0.1)
        opt = mlrose_hiive.DiscreteOpt(length=1000, fitness_fn=fit_func, max_val=2)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.random_hill_climb(
              opt, max_attempts=10, max_iters=100, restarts=1000,
              init_state=None, curve=True, random_state=None,
              state_fitness_callback=None, callback_user_info=None
        )
        end = time.time()
        restart_curves.append((res, opt_sol, opt_val, curr_fitness, end-start))


    # In[ ]:


    decays = [('geometric', mlrose_hiive.algorithms.decay.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)),
              ('arithmetic', mlrose_hiive.algorithms.decay.ArithDecay(init_temp=1.0, decay=0.99, min_temp=0.001)),
              ('exponential', mlrose_hiive.algorithms.decay.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001))]
    decay_curves = []
    for dec in decays:
        fit_func = mlrose_hiive.fitness.SixPeaks(t_pct=0.1)
        opt = mlrose_hiive.DiscreteOpt(length=1000, fitness_fn=fit_func, max_val=2)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.simulated_annealing(
            opt, schedule=dec[1],
            max_attempts=10, max_iters=100, init_state=None, curve=True,
            fevals=False, random_state=None,
            state_fitness_callback=None, callback_user_info=None
        )
        end = time.time()
        decay_curves.append((dec[0], opt_sol, opt_val, curr_fitness, end-start))


    # In[41]:


    pop_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    percent_curves = []

    for perc in pop_percent:
        fit_func = mlrose_hiive.fitness.SixPeaks(t_pct=0.1)
        opt = mlrose_hiive.DiscreteOpt(length=1000, fitness_fn=fit_func, max_val=2)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.genetic_alg(
            opt, pop_size=100, pop_breed_percent=perc, elite_dreg_ratio=0.95,
            minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
            max_attempts=10, max_iters=100, curve=True, random_state=None
        )
        end = time.time()
        percent_curves.append((perc, opt_sol, opt_val, curr_fitness, end-start))


    # In[35]:


    keep_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    keep_curves = []
    for perc in keep_percent:
        print(perc)
        fit_func = mlrose_hiive.fitness.SixPeaks(t_pct=0.1)
        opt = mlrose_hiive.DiscreteOpt(length=200, fitness_fn=fit_func, max_val=2)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.mimic(
            opt, pop_size=200, keep_pct=perc, max_attempts=10,
            max_iters=100, curve=True, random_state=None,
            state_fitness_callback=None, callback_user_info=None, noise=0.0
        )
        end = time.time()
        keep_curves.append((perc, opt_sol, opt_val, curr_fitness, end-start))


    # In[43]:


    import pandas as pd
    rhc_df = pd.DataFrame(
        [('RHC', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in restart_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Restarts', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    sa_df = pd.DataFrame(
        [('SA', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in decay_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Decay Type', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    ga_df = pd.DataFrame(
        [('GA', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in percent_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Breeding Percentage', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    mimic_df = pd.DataFrame(
        [('MIMIC', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in keep_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Keep Percentage', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    fig = px.line(rhc_df, x='Iteration', y='Fitness Value', color='Restarts', title='RHC Performance in Six Peaks with Different Restart Counts')
    #fig.update_yaxes(range=[120, 190])
    #fig.update_xaxes(range=[0, 1000])
    fig.show()

    fig = px.line(sa_df, x='Iteration', y='Fitness Value', color='Decay Type', title='SA Performance in Six Peaks with Different Decay Functions')
    #fig.update_yaxes(range=[120, 190])
    #fig.update_xaxes(range=[0, 1000])
    fig.show()

    fig = px.line(ga_df, x='Iteration', y='Fitness Value', color='Breeding Percentage', title='GA Performance in Six Peaks with Different Breeding Percentages')
    #fig.update_yaxes(range=[120, 190])
    #fig.update_xaxes(range=[0, 1000])
    fig.show()

    fig = px.line(mimic_df, x='Iteration', y='Fitness Value', color='Keep Percentage', title='MIMIC Performance in Six Peaks with Different Keep Percentages')
    #fig.update_yaxes(range=[120, 190])
    #fig.update_xaxes(range=[0, 1000])
    fig.show()


    import plotly.express as px
    import numpy as np
    import random
    from scipy import optimize

    import scipy


    # In[2]:


    import mlrose_hiive
    import time


    # In[8]:


    restarts = [0, 10, 100, 1000]
    restart_curves = []
    for res in restarts:
        fit_func = mlrose_hiive.fitness.FlipFlop()
        opt = mlrose_hiive.FlipFlopOpt(length=20, fitness_fn=fit_func)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.random_hill_climb(
              opt, max_attempts=10, max_iters=100, restarts=1000,
              init_state=None, curve=True, random_state=None,
              state_fitness_callback=None, callback_user_info=None
        )
        end = time.time()
        restart_curves.append((res, opt_sol, opt_val, curr_fitness, end-start))


    # In[33]:


    decays = [('geometric', mlrose_hiive.algorithms.decay.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.01)),
              ('arithmetic', mlrose_hiive.algorithms.decay.ArithDecay(init_temp=1.0, decay=0.99, min_temp=0.01)),
              ('exponential', mlrose_hiive.algorithms.decay.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001))]
    decay_curves = []
    for dec in decays:
        fit_func = mlrose_hiive.fitness.FlipFlop()
        opt = mlrose_hiive.FlipFlopOpt(length=20, fitness_fn=fit_func)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.simulated_annealing(
            opt, schedule=dec[1],
            max_attempts=10, max_iters=1000, init_state=None, curve=True,
            fevals=False, random_state=None,
            state_fitness_callback=None, callback_user_info=None
        )
        end = time.time()
        print(opt_val)
        decay_curves.append((dec[0], opt_sol, opt_val, curr_fitness, end-start))


    # In[10]:


    pop_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    percent_curves = []

    for perc in pop_percent:
        fit_func = mlrose_hiive.fitness.FlipFlop()
        opt = mlrose_hiive.FlipFlopOpt(length=20, fitness_fn=fit_func)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.genetic_alg(
            opt, pop_size=500, pop_breed_percent=perc, elite_dreg_ratio=0.95,
            minimum_elites=0, minimum_dregs=0, mutation_prob=0.1,
            max_attempts=10, max_iters=100, curve=True, random_state=None
        )
        end = time.time()
        percent_curves.append((perc, opt_sol, opt_val, curr_fitness, end-start))


    # In[27]:


    keep_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    keep_curves = []
    for perc in keep_percent:
        print(perc)
        fit_func = mlrose_hiive.fitness.FlipFlop()
        opt = mlrose_hiive.FlipFlopOpt(length=20, fitness_fn=fit_func)
        start = time.time()
        opt_sol, opt_val, curr_fitness = mlrose_hiive.algorithms.mimic(
            opt, pop_size=500, keep_pct=perc, max_attempts=10,
            max_iters=100, curve=True, random_state=None,
            state_fitness_callback=None, callback_user_info=None, noise=0.0
        )
        end = time.time()
        keep_curves.append((perc, opt_sol, opt_val, curr_fitness, end-start))


    # In[43]:


    import pandas as pd
    rhc_df = pd.DataFrame(
        [('RHC', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in restart_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Restarts', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    sa_df = pd.DataFrame(
        [('SA', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in decay_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Decay Type', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    ga_df = pd.DataFrame(
        [('GA', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in percent_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Breeding Percentage', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    mimic_df = pd.DataFrame(
        [('MIMIC', curve_tuples[-1], curve_tuples[0], elem[0], elem[1], idx) for curve_tuples in keep_curves for idx, elem in enumerate(curve_tuples[3])],
        columns=['Algorithm', 'Time', 'Keep Percentage', 'Fitness Value', 'Fitness Evaluations', 'Iteration']
    )
    fig = px.line(rhc_df, x='Iteration', y='Fitness Value', color='Restarts', title='RHC Performance in Flip-Flop with Different Restart Counts')
    fig.update_yaxes(range=[5, 20])
    fig.update_xaxes(range=[0, 150])
    fig.show()

    fig = px.line(sa_df, x='Iteration', y='Fitness Value', color='Decay Type', title='SA Performance in Flip-Flop with Different Decay Functions')
    fig.update_yaxes(range=[5, 20])
    fig.update_xaxes(range=[0, 150])
    fig.show()

    fig = px.line(ga_df, x='Iteration', y='Fitness Value', color='Breeding Percentage', title='GA Performance in Flip-Flop with Different Breeding Percentages')
    fig.update_yaxes(range=[5, 20])
    fig.update_xaxes(range=[0, 150])
    fig.show()

    fig = px.line(mimic_df, x='Iteration', y='Fitness Value', color='Keep Percentage', title='MIMIC Performance in Flip-Flop with Different Keep Percentages')
    fig.update_yaxes(range=[5, 20])
    fig.update_xaxes(range=[0, 150])
    fig.show()


if __name__ == '__main__':
    main()