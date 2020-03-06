"""

Nick Smith
CS 445 - Machine Learning
HW #5 - K-means Clustering
Due Tuesday March 5th

"""
import itertools
import numpy as np
from PIL import Image

"""
Just hit run and this program will do the rest. 
It will create .png files with the 8x8 greyscale 
representation of the clusters, as well as .csv 
files for the confusion matrix and the labels for 
both K = 10 and K = 30
"""

# Global Variables
global_k = 10
classes = 10  # digits 0 to 9
num_trials = 5


# ---- DATA LOADER ---- #
# Load Data from file
def data_loader(filename="optdigits.train"):
    file = open('./optdigits/' + filename, 'r')
    unprocessed_data = file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []
        split_line = line.split(',')
        for element in split_line[:-1]:
            feature_vector.append(float(element))
        features.append(feature_vector)
        labels.append(int(split_line[-1]))

    return features, labels


# ---- POINT DISTANCE ---- #
# Function to determine the distance between the points
# and all the centers
def distance(point, center):
    square_sums = 0.0
    for point_i, center_i in zip(point, center):
        square_sums += (point_i - center_i) ** 2
    return np.sqrt(square_sums)


# ---- NEAREST CENTROID ---- #
# Using the Distance function, find the 
# nearest centroid
def nearest_centroid(point, centers):
    distances = list()
    for center in centers:
        distances.append(distance(point, center))
    dist_array = np.array(distances)

    first_min_distance = dist_array.argmin()

    min_distances = list()
    for i in range(len(distances)):
        if distances[i] - distances[first_min_distance] < 10 ** -10:
            min_distances.append(i)
    return np.random.choice(min_distances)


# ---- RANDOM CENTER ---- #
# Generate random centers at the beginning of each trial
def random_center():
    return np.random.randint(0, 16, 64).tolist()


# ---- ERROR ---- #
# Mean squared error for all clusters
def mean_squared_error(clustering, centers, data):
    error = 0
    for i in range(global_k):
        cluster = clustering[i]
        center = centers[i]
        for data_point_index in cluster:
            datapoint = data[data_point_index]
            error += distance(datapoint, center) ** 2
    error /= global_k
    return error


# ---- SEPARATION ---- #
# Mean square separation for all clusters
def mean_squared_separation(clustering, centers):
    pairs = itertools.combinations([i for i in range(global_k)], 2)
    separation = 0
    for pair in pairs:
        separation += distance(centers[pair[0]], centers[pair[1]]) ** 2
    separation /= global_k
    return separation


# ---- ENTROPY FUNCTION ---- #
# Using the formula we learned in class, calculate the
# entropy on all clusters
def entropy(cluster, labels):
    entropy_sum = 0

    class_representation_in_cluster = [0 for i in range(classes)]
    total_instances = len(cluster)

    if total_instances == 0:  # Special case: 0*log_2(0) is just 0, okay?
        return 0

    for point in cluster: 
        class_representation_in_cluster[labels[point]] += 1

    class_ratios = [float(class_representation_in_cluster[i]) / total_instances
                    for i in range(classes)]
    for i in range(classes):
        if class_representation_in_cluster[i] < 1:  # Let Log_2(0) = 0
            product = 0.0
        else:
            product = class_ratios[i] * np.log2(class_ratios[i])
        entropy_sum += product

    return -1 * entropy_sum


# ---- MEAN ENTROPY ---- #
# Find the mean entropy using the previous function
def mean_entropy(clustering, labels):
    instances_per_cluster = [len(cluster) for cluster in clustering]
    total_number_of_instances = sum(instances_per_cluster)
    ratios = [float(instances_per_cluster[i]) / total_number_of_instances \
              for i in range(global_k)]

    weighted_entropies = [ratios[i] * entropy(clustering[i], labels) \
                          for i in range(global_k)]

    mean = float(sum(weighted_entropies)) / len(weighted_entropies)
    return mean


# ---- K FUNCTION ---- #
# This will be used so we can do both
# Experiments in one run.
def set_global_k(number):
    global global_k
    global_k = number


# ---- CENTER CHECKER ---- #
# This function will be used to check if the
# centers are moving after each update.
def check_centers(old_centers, centers):
    difference = 0
    for old, new in zip(old_centers, centers):
        difference += np.sum(np.abs(np.array(old) - np.array(new)))
    if difference < 10 ** -1:  # if the difference is arbitrarily close to 0
        return True
    else:
        return False


# ---- MOST POPULAR CLASS ---- #
# Find the class most likely to be the correct digit
def most_popular_class(cluster, labels):
    class_representation_in_cluster = [0 for i in range(classes)]
    total_instances = len(cluster)

    if total_instances == 0:  # Special case: 0*log_2(0) is just 0, okay?
        return None

    for point in cluster:
        class_representation_in_cluster[labels[point]] += 1

    most_popular_count = max(class_representation_in_cluster)
    first_most_popular_index = class_representation_in_cluster.index(
        most_popular_count)

    if class_representation_in_cluster.count(most_popular_count) is 1:
        return first_most_popular_index
    else:  # There might be a tie between classes
        indices_of_tied_classes = []
        for c in class_representation_in_cluster:
            if c == most_popular_count:
                indices_of_tied_classes.append(
                    class_representation_in_cluster.index(c))

        return np.random.choice(indices_of_tied_classes)


# ---- CLUSTER CLASSIFY ---- #
# Use the nearest center cluster to classify
# what digit is being represented
def classify(centers, cluster_to_class, test):
    closest = nearest_centroid(test, centers)
    return cluster_to_class[closest]


# ---- K-MEANS FUNCTION ---- #
# This will combine our previous functions to
# classify the data.
def k_means(testing_features, testing_labels, training_features,
            training_labels):

    k_means_trials = dict()
    for trial in range(num_trials):
        print("Classification starting:")
        print('Initializing random centers for %d clusters..' % global_k)
        centers = [random_center() for i in range(global_k)]

        print('Checking for Center Oscillation')
        print('Please wait...')
        change = False

        while change is False:
            # Calculate closest centers for each data point
            nearest_centroids = []
            for datapoint in training_features:
                nearest_centroids.append(nearest_centroid(datapoint, centers))
            clustering = [[] for i in range(global_k)]
            for i in range(len(nearest_centroids)):
                clustering[nearest_centroids[i]].append(i)

            # Calculate the centroid of each center's set of points
            centroids = []
            for cluster in clustering:
                mean_vector = np.array([0.0 for i in range(64)])  # sum feature
                # values
                for i in range(len(cluster)): 
                    mean_vector += np.array((training_features[cluster[i]]))
                if len(cluster) > 0:
                    mean_vector /= float(len(cluster))  # average the sums
                centroids.append(mean_vector)

            # Reassign each center
            old_centers = centers
            centers = centroids
            change = check_centers(old_centers, centers)
        print('\nFinal cluster centers for trial %d set.' % trial)
        mse = mean_squared_error(clustering, centers, training_features)
        mss = mean_squared_separation(clustering, centers)
        avg_entropy = mean_entropy(clustering, training_labels)

        k_means_trials[trial] = [centers, nearest_centroids, clustering, mse,
                                 mss, avg_entropy]

    print('%d trials are complete.\n' % num_trials)
    for i in range(num_trials):
        print('\nTrial #', str(i), ':')
        print('Mean-square Error: ', k_means_trials[i][3])
        print('Mean-square Separation: ', k_means_trials[i][4])
        print('Mean Entropy: ', k_means_trials[i][5])
    smallest_mse_index = 0
    for trial in range(1, len(k_means_trials)):
        if k_means_trials[trial][3] < k_means_trials[smallest_mse_index][3]:
            smallest_mse_index = trial
    print ('\nThe best trial was number %d' % smallest_mse_index)
    best_trial = k_means_trials[smallest_mse_index]
    best_centers = best_trial[0]
    best_clustering = best_trial[2]

    print('Assigning classes to each cluster.')
    cluster_labels = [most_popular_class(cluster, training_labels) for
                      cluster in best_clustering]

    print('Assigning classifications to each test instance.')
    classifications = [classify(best_centers, cluster_labels, test) for
                       test in testing_features]

    confusion_matrix = create_confusion_matrix(classifications, testing_labels)
    save_confusion_matrix(confusion_matrix)

    return [best_trial, cluster_labels, classifications, confusion_matrix]


# ---- CONFUSION MATRIX ---- #
# This will be displayed by the following function
def create_confusion_matrix(classifications, testing_labels):
    confusion_matrix = [[0 for i in range(classes)] for i in
                        range(classes)]
    for label, classification in zip(testing_labels, classifications):
        confusion_matrix[label][classification] += 1
    return confusion_matrix


# ---- C.M. WRITER ---- #
def save_confusion_matrix(confusion_matrix):
    filename = 'confusion_matrix_%d_clusters.csv' % global_k
    output = open(filename, 'w')
    for row in confusion_matrix:
        for col in row:
            output.write(str(col) + ',')
        output.write('\n')
    output.close()


# ---- ACCURACY PER TRIAL --- #
def accuracy(confusion_matrix):
    m = np.array(confusion_matrix)
    return float(np.sum(np.diagonal(m))) / np.sum(m)


# ---- PIXEL VALUE ---- #
# This will give the clusters a numeric
# value for display purposes
def pixel_value(value):
    value = int(np.floor(value))
    return value * 16


# ---- CREATE CLUSTER BITMAPS ---- #
# This will create individual files
def draw_center_as_bitmap(name_prefix, center_number, center):
    img = Image.new('L', (8, 8), "black")
    center_2d = np.array(center).reshape(8, 8)
    for i in range(img.size[0]):
        for j in range(img.size[0]):
            img.putpixel((j, i), pixel_value(int(center_2d[i][j])))
    name = name_prefix + str(center_number) + '.png'
    img.save(name)


# ---- CLUSTER LABELER ---- #
# Prints the labels assigned to each cluster
def cluster_labeler(cluster_labels):
    print('Labels for each cluster:\n')
    for i in range(len(cluster_labels)):
        if cluster_labels[i] is not None:
            print('Cluster %d\'s label is %d' % (i, cluster_labels[i]))
        elif cluster_labels[i] is None:
            print('Cluster %d\'s label is None' % i)


def main():

    #LOAD DATA
    training_features, training_labels = data_loader()
    testing_features, testing_labels = data_loader('optdigits.test')

    # ---- EXPERIMENT #1 ---- #
    # K = 10 WITH RANDOM SEEDS
    # ----------------------- #

    set_global_k(10)
    ten_clusters_results = k_means(testing_features, testing_labels,
                                   training_features, training_labels)
    accuracy_10_clusters = accuracy(ten_clusters_results[-1])
    print('Accuracy for 10 clusters: ', accuracy_10_clusters)
    ten_centers = ten_clusters_results[0][0]
    for i in range(global_k):
        draw_center_as_bitmap('exp1_center_', i, ten_centers[i])

    ten_cluster_labels = ten_clusters_results[1]
    cluster_labeler(ten_cluster_labels)

    # ---- EXPERIMENT #3 ---- #
    # K = 30 WITH RANDOM SEEDS
    # ----------------------- #

    set_global_k(30)
    thirty_clusters_results = k_means(testing_features, testing_labels,
                                      training_features, training_labels)
    accuracy_30_clusters = accuracy(thirty_clusters_results[-1])
    print('Accuracy for 30 clusters: ', accuracy_30_clusters)
    thirty_centers = thirty_clusters_results[0][0]
    for i in range(global_k):
        draw_center_as_bitmap('exp2_center_', i, thirty_centers[i])
    thirty_cluster_labels = thirty_clusters_results[1]
    cluster_labeler(thirty_cluster_labels)


if __name__ == "__main__":
    main()

