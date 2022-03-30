from cmath import pi
import sys
import pickle
from multiprocessing.pool import ThreadPool
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from pandas.api.types import is_object_dtype, is_bool_dtype
from pyod.models.copod import COPOD

from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.exceptions import InconsistentConstraintsException

from anomatools.anomaly_detection import INNE

from image_generation import generate_image

def convert_problematic_data(data):
    '''
    Takes a pandas dataframe
    Converts all data that is not numerical into numerical data. 
    Returns an updated dataframe with one hot encoding. 
    '''
    df = data.infer_objects() 
    category_columns = []
    # Convert int datatypes to float datatypes
    # This might be needed cause of the weird abod thingy
    # for column in df:
    #    if(df[column].dtype == "int64"):
    #        df[column] = df[column].astype(float)
    #        break

    # Determine which columns are categorical
    for column in df:
        if(df[column].dtype == "object"):
            df[column] = df[column].astype('category')
            df = df.drop(columns=[column]) #Drop em for now but gotta actually do something with em later
        elif(df[column].dtype == "bool"):
            df = df.drop(columns=[column]) #Drop these too for now
    #TODO: One hot encoding

    return df

def create_constraint(links):
    '''
    Takes a list of (index, index) lists. 
    Exports links to be symettric and linked based off of logic within links. 

    Input: [(40, 41), (42, 41)]
    Output: [(40, 41), (41, 40), (42, 41), (41, 42), (40, 42), (42, 40)]

    Input: [(40, 41), (42, 43)]
    Output: [(40, 41), (41, 40), (42, 43), (43, 42)]
    '''
    final_link = []
    for link in links:
        final_link.append((int(link[0]), int(link[1])))
        final_link.append((int(link[1]), int(link[0])))
    links_new = final_link.copy()
    for link in links_new:
        for link2 in links_new:
            if link != link2 and link[1] == link2[0] and link[0] != link2[1] and (link[0], link2[1]) not in final_link:
                final_link.append((int(link[0]), int(link2[1])))
                final_link.append((int(link2[1]), int(link[0])))
    return final_link

def constraint_already_exists(constraint_sets, v1, v2):
    '''
    Checks if a constraint between v1 and v2 exists in constraint_sets
    constraint_sets should be a 2d list of constraints 
    eg. if the must link constraints were (1,2) and (3,4), the cannot link constrains were (5,6) and (7,8) and the unknown constraints were (9,0) constraint_sets would look like:
    [[1,2,3,4],
    [5,6,7,8],
    [9,0]]
    '''
    for constraint_set in constraint_sets:
        for i in range(0, len(constraint_set)-1, 2):
            if int(constraint_set[i]) == int(v1) and int(constraint_set[i+1]) == int(v2) or int(constraint_set[i]) == int(v2) and int(constraint_set[i+1]) == int(v1):
                return True
    return False

def find_nearest_neighbor(neighbors, numpy_data, value, labels, same_cluster, constraints=[]):
    '''
    Finds and returns the nearest neighbor to the item at numpy_data[value] with either the same or different label depending on same_cluster 
    '''
    closest_neighbor_index = neighbors.kneighbors(numpy_data[value].reshape(1, -1), n_neighbors=len(numpy_data))[1][0]

    for i in range(1, len(closest_neighbor_index)):
        # If the labels are the same and same_cluster is true or the labels are different and same_cluster is false
        if (labels[closest_neighbor_index[i]] == labels[value[0]]) == same_cluster and not constraint_already_exists(constraints, closest_neighbor_index[i], value[0]):
            return closest_neighbor_index[i]
    
    print(3)
    raise IndexError("Unable to find another Sample to match "+ str(value[0]) +" with due to constraints.")

def evaluate_model(model, numpy_data, evaluation_algorithms, cluster_iter):
    '''
    Evaluates the clustering model with several metrics
    Returns a dictionary of the different scores as well as a list of normalized MAGIC scores for each data point
    The evaluation algorithms being used will only effect the MAGIC score and not the dictionary with scores
    '''
    # ================Evaluate clustering model================

    labels = model.labels_

    evaluation_scores = {}

    #iNNE
    iNNEVal = INNE().fit(numpy_data).predict(numpy_data)[0]
    norm_inne_scores = list(map(lambda x, r=float(np.max(iNNEVal) - np.min(iNNEVal)): (1 - (x - np.min(iNNEVal)) / r), iNNEVal))
    
    evaluation_scores['iNNE'] = sum(norm_inne_scores)/len(norm_inne_scores)

    #COPOD
    copod_scores = COPOD().fit(numpy_data).decision_scores_
    norm_copod_scores = list(map(lambda x, r=float(np.max(copod_scores) - np.min(copod_scores)): (1 - (x - np.min(copod_scores)) / r), copod_scores))

    evaluation_scores['COPOD'] = sum(norm_copod_scores)/len(copod_scores)

    #Isolation Forest Anomaly Score
    if_samp = IsolationForest(random_state=0).fit(numpy_data).score_samples(numpy_data)
    norm_if_scores = list(map(lambda x, r=float(np.max(if_samp) - np.min(if_samp)): ((x - np.min(if_samp)) / r), if_samp))
    
    evaluation_scores['IF'] = sum(norm_if_scores)/len(norm_if_scores)

    #Local Outlier Factor
    neg_out_arr = LocalOutlierFactor().fit(numpy_data).negative_outlier_factor_
    norm_nog = list(map(lambda x, r=float(np.max(neg_out_arr) - np.min(neg_out_arr)): ((x - np.min(neg_out_arr)) / r), neg_out_arr))
    
    evaluation_scores['LOF'] = sum(norm_nog)/len(norm_nog)

    #Sihlouette
    # Passing my data (data) and the certain cluster that each data point from X should be based on our model.
    sil_arr = metrics.silhouette_samples(numpy_data, labels)
    norm_sil = list(map(lambda x, r=float(np.max(sil_arr) - np.min(sil_arr)): ((x - np.min(sil_arr)) / r), sil_arr))
    
    evaluation_scores['SIL'] = sum(norm_sil)/len(norm_sil)

    #Davies-Bouldin
    #TODO this one is fucked because lower is better for this score but I don't know what the theoretical maximum score is to do (max - score/max) with
    # We gonna have to read through the wikipedia or something to figure that out
    davies_bouldin_index = 1 - davies_bouldin_score(numpy_data, labels)
    evaluation_scores['DB'] = davies_bouldin_index

    #Calinski-Harabasz
    calinski_harabasz_index = calinski_harabasz_score(numpy_data, labels)

    # If cluster_iter = 1 then we save the Calinski-Harabasz score to a file that we can read later to get the +- bounds
    # Else we map the value to between 0 and 1 where 0 and 1 are +- 10% of the original value
    if int(cluster_iter) == 1:
        pickle.dump(calinski_harabasz_index, open('interactive-constrained-clustering/src/model/ch.sav', 'wb'))
        evaluation_scores['Calinski-Harabasz'] = 0.5
    else:
        old_score = pickle.load(open('interactive-constrained-clustering/src/model/ch.sav', 'rb'))
        evaluation_scores['Calinski-Harabasz'] = (calinski_harabasz_index - (old_score - old_score * 0.1))/((old_score + old_score * 0.1) - (old_score - old_score * 0.1))

    #evaluation_scores['Calinski-Harabasz'] = calinski_harabasz_index

    #Take all the normalized metric arrays, determine the avg to provide for question determination
    normalized_magic = [(v*int(evaluation_algorithms[0]) + w*int(evaluation_algorithms[1]) + x*int(evaluation_algorithms[2]) + y*int(evaluation_algorithms[3]) + z*int(evaluation_algorithms[4]))/evaluation_algorithms.count('1') 
    for v, w, x, y, z in zip(norm_inne_scores, norm_copod_scores, norm_if_scores, norm_nog, norm_sil)]

    return evaluation_scores, normalized_magic

def fit_model(model, numpy_data, ml=[], cl=[]):
    '''
    Fits the model to the data and returns the model
    This is in its own method so that I can call it asynchronously
    '''
    model.fit(numpy_data, ml, cl)

    return model

def compute_questions(filename, cluster_iter, question_num, cluster_num, must_link_constraints, cant_link_constraints, unknown_constraints, reduction_algorithm, evaluation_algorithms):
    '''
    Args:
        filename: name of the csv file
        cluster_iter: what iteration we are currently on
        question_num: Questions per iteration rounded down to nearest even number
        must_link_constraints: 
        cant_link_constraints:
        unknown_constraints:
        evaluation_algorithms: 
    '''
    
    # ================Generate clustering model================

    df = convert_problematic_data(pd.read_csv('datasets/' + filename))
    numpy_data = df.to_numpy()
    # Will not be aware of ml or cl constraints until after user passes Iteration 1
    if int(cluster_iter) != 1:
        ml_converted = [i for i in zip(*[iter(must_link_constraints)]*2)]
        cl_converted = [i for i in zip(*[iter(cant_link_constraints)]*2)]
        # Generates the setup for constraints from input from the user.
        ml = create_constraint(ml_converted)
        cl = create_constraint(cl_converted)
        # Applying new constraints to the model
        model = PCKMeans(n_clusters=cluster_num)
        try:
            #Create each model in its own thread to save time
            pool = ThreadPool(processes=3)
            model_result = pool.apply_async(fit_model, (model, numpy_data, ml, cl))

            clusters_inc_model = PCKMeans(n_clusters=cluster_num+1)
            model_inc_result = pool.apply_async(fit_model, (clusters_inc_model, numpy_data, ml, cl))

            if cluster_num > 2:
                clusters_dec_model = PCKMeans(n_clusters=cluster_num-1)
                model_dec_result = pool.apply_async(fit_model, (clusters_dec_model, numpy_data, ml, cl))

            model = model_result.get()
            clusters_inc_model = model_inc_result.get()
            if cluster_num > 2:
                clusters_dec_model = model_dec_result.get()

        except InconsistentConstraintsException:
            # Error 2 sent to client to handle properly.
            print(2)
            raise InconsistentConstraintsException("Inconsistent constraints")
    else:
        model = PCKMeans(n_clusters=cluster_num)
        try:
            model.fit(numpy_data)
        except TypeError:
            # Error 1 sent to client to handle properly.
            print(1)
            raise TypeError("There exists a string values in the dataset that the tool was unable to handle properly.")

    # ================Save the pickle================

    #dump(obj, open(filename, mode))
    pickle.dump(model, open('interactive-constrained-clustering/src/model/finalized_model.sav', 'wb'))

    #Temporary pickles from the case study so that we can go back if we forget things
    pickle.dump(model, open('interactive-constrained-clustering/src/model/finalized_model_' + str(cluster_iter) + '.sav', 'wb'))
    pickle.dump(numpy_data, open('interactive-constrained-clustering/src/model/numpy_data_' + str(cluster_iter) + '.sav', 'wb'))
    if(int(cluster_iter) != 1):
        pickle.dump(ml, open('interactive-constrained-clustering/src/model/ml_' + str(cluster_iter) + '.sav', 'wb'))
        pickle.dump(cl, open('interactive-constrained-clustering/src/model/cl_' + str(cluster_iter) + '.sav', 'wb'))

    # ================Evaluate clustering model================

    evaluation_scores, normalized_magic = evaluate_model(model, numpy_data, evaluation_algorithms, cluster_iter)
    avg_magic = sum(normalized_magic)/len(normalized_magic)
    
    # Suggest +- 1 clusters if the magic score will improve

    #cluster+1 and cluster-1 portion for silhoutte. Determine if we must flag the notif in front-end app. 
    sil_change_value = 0
    if int(cluster_iter) != 1:
        _, normalized_magic_inc = evaluate_model(clusters_inc_model, numpy_data, evaluation_algorithms, cluster_iter)
        avg_inc = sum(normalized_magic_inc)/len(normalized_magic_inc)
        if int(cluster_num) > 2:
            _, normalized_magic_dec = evaluate_model (clusters_dec_model, numpy_data, evaluation_algorithms, cluster_iter)
            avg_dec = sum(normalized_magic_dec)/len(normalized_magic_dec)
            # Increase clusters
            if avg_magic < avg_inc and avg_dec < avg_inc:
                sil_change_value = 4
            # Decrease clusters
            elif avg_magic < avg_dec and avg_inc < avg_dec:
                sil_change_value = 5
        else:
            # Increase clusters
            if avg_magic < avg_inc:
                sil_change_value = 4

    sorted_norm_magic = sorted(normalized_magic)

    #Min
    print(sorted_norm_magic[0])
    print("SEPERATOR")
    #Avg
    print(str(sum(normalized_magic)/len(normalized_magic)))
    print("SEPERATOR")
    #Max
    print(sorted_norm_magic[-1])
    print("SEPERATOR")

    # ================Generate graph for website================
    
    labels = model.labels_
    generate_image(cluster_iter, numpy_data, labels, reduction_algorithm, cluster_num, evaluation_scores)

    # ================Decide what questions to ask clustering model================

    question_set_indices = []
    # Interested in question_num/2 unreliable data points as we will compare the nearest neighbour of same node and nearest neighbour of a diffrent node
    # Converting the lowest indecies into an array of list(index,index) based on nearest sets of clusters.
    for v in sorted_norm_magic[:int(question_num/2)]:
        question_set_indices += np.where(normalized_magic == v)

    #Creating neighbor to determine nearest nodes. 
    neighbor = NearestNeighbors()
    neighbor.fit(numpy_data)
    # Format for question_set: [valueQuestioned(VQ), VQSameNeighbor, VQ, VQDiffNeighbor, VQ2, VQ2SameNeighbor, VQ2, VQ2DiffNeighbor,...]
    # This format is used to support the transfer into javascript.
    question_set = []

    for value in question_set_indices:
        # Creates an "odd" question with value and the nearest neighbour to value that is in the same cluster and has no existing constraints with value
        question_set.append(value[0])
        question_set.append(find_nearest_neighbor(neighbor, numpy_data, value, labels, True, constraints=[must_link_constraints, cant_link_constraints, unknown_constraints]))
        # Creates and "even" question with value and the nearest neighbour to value that is in a different cluster and has no existing constraints with value
        question_set.append(value[0])
        question_set.append(find_nearest_neighbor(neighbor, numpy_data, value, labels, False, constraints=[must_link_constraints, cant_link_constraints, unknown_constraints]))

    print(question_set)
    print("SEPERATOR")
    print(sil_change_value)


'''
filename - filename within datasets folder to search for. 
clustering_iter - to support the naming of the clustering in images.
question_num - the input from the landing page will set the num of samples that will be collected.
cluster_num - the number of clusters for the PCKmeans algorithm.
ml - The must link constraints from the oracle.
cl - The can't link constraints from the oracle.
unknown - The unknown constraints from the oracle. 
'''
# Handle incoming values from program call.
filename = str(sys.argv[1])
cluster_iter = str(sys.argv[2])
question_num = int(sys.argv[3])
cluster_num = int(sys.argv[4])
ml = sys.argv[5].split(",")
cl = sys.argv[6].split(",")
unknown = sys.argv[7].split(",")
reduction_algorithm = sys.argv[8]
evaluation_algorithms = sys.argv[9].split(',')

# filename = 'Pokemon_no_string.csv'
# cluster_iter = '1'
# question_num = 4
# cluster_num = 5
# ml = []
# cl = []
# unknown = []
# reduction_algorithm = "TSNE"
# evaluation_algorithms = ['1','1','1','1','1']

compute_questions(filename, cluster_iter, question_num, cluster_num, ml, cl, unknown, reduction_algorithm, evaluation_algorithms)