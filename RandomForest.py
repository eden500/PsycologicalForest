from concurrent.futures import ProcessPoolExecutor
import logging
import numpy as np
from DecisionTree import DecisionTreeClassifier
from multiprocessing import cpu_count
import pickle
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm


class RandomForestRegressor:
    def __init__(self, nb_trees, nb_samples=-1, max_depth=None, max_workers=cpu_count() - 1, criterion='mse',
                 min_samples_split=2, min_samples_leaf=1, max_features=None, bootstrap=True):
        """
        :param  nb_trees:           Number of decision trees to use
        :param  nb_samples:         Number of samples to give to each tree
        :param  max_depth:          Maximum depth of the trees. If None, then nodes are expanded until all leaves
                                    are pure or until all leaves contain less than min_samples_split samples.
        :param  max_workers:        Maximum number of processes to use for training
        :param  criterion:          The function used to split data at each node of the tree.
        :param  min_samples_split:  The minimum number of samples required to split an internal node.
        :param  min_samples_leaf:   The minimum number of samples required to be at a leaf node.
                                    A split point at any depth will only be considered if it leaves at least
                                    min_samples_leaf training samples in each of the left and right branches.
        :param  max_features:       The number of features to consider when looking for the best split:
                                    if int, number of features used in each tree will be max_features
                                    if float, number of features used in each tree round(max_features * n_features).
                                    if None, each tree will use all features.
        :param  bootstrap:          Whether bootstrap samples are used when building trees. If False, the
                                    whole dataset is used to build each tree.
        """
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.max_workers = max_workers
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.nb_features = 0
        self.max_features = max_features
        self.use_bootstrap = bootstrap
        self.feature_names = None
        self.forced = None

    def fit(self, X, y, force_split=None, splits=1, remove_from_split=0, show_histogram=False):
        """
        Trains self.nb_trees number of decision trees.
        :param X: The data used to train the decision tree.
        :param y: The true label matching to the data in X.
        :param force_split: feature to split by the first splits times
        :param splits: number of times to split by the given features
        :param remove_from_split: number of times not to use the feature after only using it.
                                  value of -1 removes the feature from all future use.
        :param show_histogram: show histogram of the values of the split features.
        """
        # input validation
        if len(X) < 1:
            raise ValueError("Not enough samples in the given dataset")
        if len(X) != len(y):
            raise ValueError("Found input variables with inconsistent numbers of "
                             "samples: %r" % [int(l) for l in [len(X), len(y)]])
        self.nb_features = len(X[0])
        if self.nb_features < 1:
            raise ValueError("data points are meaningless")

        if not isinstance(y[0], list):
            y = np.reshape(y, (-1, 1)).tolist()
        data = np.hstack((X, y))

        # setup
        if self.nb_samples == -1:
            self.nb_samples = len(X)

        if self.max_features is None:
            self.max_features = self.nb_features
        elif isinstance(self.max_features, int):
            self.max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                self.max_features = max(1, int(self.max_features * self.nb_features))
            else:
                self.max_features = 0

        # forced split
        if force_split is not None:
            if self.feature_names is not None and force_split in self.feature_names:
                force_split = self.feature_names.index(force_split)
            if force_split not in range(self.nb_features):
                if self.feature_names:
                    raise ValueError("invalid feature value {} in force_split\ntry using features from"
                                     " {}".format(force_split, self.feature_names))
                else:
                    raise ValueError("invalid feature value {} in force_split and no feature name given.\ntry using "
                                     "features from {}".format(force_split, range(self.nb_features)))

            self.forced = (force_split, splits, show_histogram, remove_from_split)

        # training the trees

        # sampling data points for each tree:
        if self.use_bootstrap:
            tree_samples = map(lambda tree_index: [tree_index, data[np.random.choice(len(data), self.nb_samples), :]],
                               range(self.nb_trees))
        else:
            tree_samples = map(
                lambda tree_index: [tree_index, data[np.random.choice(len(data), self.nb_samples, replace=False), :]],
                range(self.nb_trees))

        # parallel training trees:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self.trees = list(tqdm(executor.map(self.train_tree, tree_samples), total=self.nb_trees))

        if self.forced is not None and self.forced[2]:
            self.histogram_forced(self.forced[1])

    def train_tree(self, data):
        """
        Trains a single tree and returns it.
        :param  data:   A List containing the index of the tree being trained
                        and the data to train it
        """
        logging.info('Training tree {}'.format(data[0] + 1))
        tree = DecisionTreeClassifier(self.forced)
        tree.fit(data[1], max_depth=self.max_depth, criterion=self.criterion, min_samples_split=self.min_samples_split,
                 min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        return tree

    def predict(self, features):
        """
        Returns a prediction for the given feature. The result is the value that
        gets the most votes.
        :param  features:    The features used to predict
        """

        if not self.trees:
            raise NotFittedError("This RandomForestRegressor instance is not fitted yet."
                                 " Call 'fit' with appropriate arguments before using this estimator")

        if len(features) != self.nb_features:
            raise ValueError("given features to predict do not match training data features")

        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(features))

        return np.mean(predictions)

    def predict_multiple(self, data):
        prediction = []

        for features in data:
            prediction.append(self.predict(features))

        return prediction

    def save(self, file_name):
        """
        Saves model as a pickle file with the given file name
        :param file_name: name of the saved file (+.p)
        """
        pickle.dump(self, open(str(file_name) + '.p', "wb"))

    def print_tree_horizontal(self, tree_index, depth=-1, spacious=5):
        """
        Print the tree of the given index horizontally
        :param tree_index: index of the tree to print
        :param spacious: space between level of the tree in the printout.
        """
        self.trees[tree_index].print_horizontal(self.feature_names, depth, spacious)

    def print_tree_vertical(self, tree_index, depth=-1):
        """
        Print the tree of the given index vertiaclly
        :param tree_index: index of the tree to print
        """
        self.trees[tree_index].print_vertical(self.feature_names, depth)

    def print_tree_topdown(self, tree_index, depth=-1):
        """
        Print the tree of the given index topdown
        :param tree_index: index of the tree to print
        """
        self.trees[tree_index].print_topdown(self.feature_names, depth)

    def set_features_names(self, names: list):
        """
        Set feature names for printout functions
        :param names: list of names in same order features were given to the model
        """
        if len(names) != self.nb_features:
            raise ValueError(
                "Number of features mismatch. model was trained on {} features, {} names were given.".format(
                    self.nb_features, len(names)))
        self.feature_names = names

    def pre_set_features_names(self, names: list):
        """
        Set feature names before fitting the model (used in forced splitting until I find a better solution)
        :param names: list of names of features in same order features will be given to the model
        """
        self.feature_names = names

    def get_values_in_level(self, level, son=None):
        """Returns list of values of the splits in a given level"""

        def get_values_in_level_rec(node, lev):
            if node.feature == -1:  # leaves does not split
                return []
            if lev == 1:
                if self.feature_names:
                    return [(self.feature_names[node.feature], round(node.value))]
                else:
                    return [(node.feature, round(node.value))]
            return get_values_in_level_rec(node.fb, lev - 1) + get_values_in_level_rec(node.tb, lev - 1)

        values = []
        for tree in self.trees:
            if son:
                node = tree.root_node
                for direction in son:
                    if direction == 'left':
                        node = node.tb
                    else:
                        node = node.fb
                    level -= 1
                values.extend(get_values_in_level_rec(node, 1))
            else:
                values.extend(get_values_in_level_rec(tree.root_node, level))

        return values

    def get_features_in_level(self, levels):
        """Returns dictionary of count of each feature in given level"""

        def get_features_in_level_rec(node, lev):
            if node.feature == -1:  # leaves does not split
                return []
            if lev == 1:
                return [node.feature]
            return get_features_in_level_rec(node.fb, lev - 1) + get_features_in_level_rec(node.tb, lev - 1)

        features = {key: 0 for key in range(self.nb_features)}

        for tree in self.trees:
            tree_features = []
            for level in levels:
                tree_features.extend(get_features_in_level_rec(tree.root_node, level))
            for feature in tree_features:
                features[feature] += 1

        if self.feature_names:
            keys = list(features.keys())
            features = {self.feature_names[key]: features[key] for key in keys}
        else:
            print("no feature names, please use 'set_features_names' method")

        return features

    def get_statistics(self, levels=-1):
        """
        Returns models feature-values dictionary in all trees
        :param levels: If int or list of ints: returns features and values in given level
                       If -1: returns features and values from all levels
        :return: Dictionary of {features: [value1, value2, ...]}
        """

        def get_statistics_rec(node, lev):
            if node.feature == -1:  # leaves does not split
                return []
            if lev == 1:
                return [(node.feature, round(node.value, 3))]
            return get_statistics_rec(node.fb, lev - 1) + get_statistics_rec(node.tb, lev - 1)

        features = {key: [] for key in range(self.nb_features)}

        if levels != -1:
            levels = [levels] if isinstance(levels, int) else levels
            for level in levels:
                for tree in self.trees:
                    tree_features = get_statistics_rec(tree.root_node, level)
                    for feature, value in tree_features:
                        features[feature].append(value)
        else:
            done = False
            level = 1
            while not done:
                done = True
                for tree in self.trees:
                    tree_features = get_statistics_rec(tree.root_node, level)
                    if tree_features:
                        done = False
                    for feature, value in tree_features:
                        features[feature].append(value)
                level += 1

        if self.feature_names:
            keys = list(features.keys())
            features = {self.feature_names[key]: features[key] for key in keys}
        return features

    def histogram_forced(self, splits):
        feature_name = self.forced[0]
        if self.feature_names:
            feature_name = self.feature_names[feature_name]
        for i in range(splits):
            forced_splits = self.get_values_in_level(i)
            plt.hist(forced_splits)
            plt.title('plotting feature {} split value histogram on split {}'.format(feature_name, i + 1))
            plt.rcParams["figure.figsize"] = [16, 9]
            plt.show()
        for length in range(1, splits):
            permutations = product(['left', 'right'], repeat=length)
            for permute in permutations:
                forced_splits = self.get_values_in_level(length, son=permute)
                plt.hist(forced_splits)
                plt.title('plotting feature {} split value histogram on split {}'.format(feature_name, permute))
                plt.rcParams["figure.figsize"] = [16, 9]
                plt.show()

    def feature_histogram(self, tree_levels, hide_zeros=False, two_sided=False, resize=1, font_size=14, rotation=45):
        """
        Histogram features from all trees of the given level in the trees
        :param tree_levels: levels of the trees to histogram features from
        :param hide_zeros: remove feature that did not used in the given level form the histogram
        :param two_sided: Soon...
        :param resize: parameter to adjust size, the greater it is the bigger the figure
        :param font_size: adjustable parameter to change font size
        :param rotation: adjustable parameter to change font rotation
        """
        if not self.trees:
            raise NotFittedError("This RandomForestRegressor instance is not fitted yet."
                                 " Call 'fit' with appropriate arguments before using this estimator")
        if two_sided:
            print('todo')
            pass
        else:
            tree_levels = [tree_levels] if isinstance(tree_levels, int) else tree_levels
            features = self.get_features_in_level(tree_levels)
            if hide_zeros:
                features = {x: y for x, y in features.items() if y != 0}

            keys = list(features.keys())
            plt.bar(keys, features.values(), width=0.5, color='g')
            plt.title('plotting count of each feature in level {} in all trees'.format(tree_levels))
            plt.xticks(fontsize=font_size, rotation=rotation)
            # plt.figure(figsize=(25, 25))
            plt.rcParams["figure.figsize"] = [16 * resize, 9 * resize]
            plt.tight_layout()
            plt.show()

    def level_analyze(self):
        """
        by level model analyze of the third level.
        IMPORTANT - use force split on a feature for the first two slits two get valid results.
        """
        def info_dict(node, first_split):
            if node.value < first_split:
                return (
                    {'range': (0, node.value), 'feature': self.feature_names[node.fb.feature], 'value': node.fb.value},
                    {'range': (node.value, first_split), 'feature': self.feature_names[node.tb.feature],
                     'value': node.tb.value})
            return (
                {'range': (first_split, node.value), 'feature': self.feature_names[node.fb.feature],
                 'value': node.fb.value},
                {'range': (node.value, 1), 'feature': self.feature_names[node.tb.feature],
                 'value': node.tb.value})

        left_left = []  # 0<x<0.25
        left_right = []  # 0.25<x<0.5
        right_left = []  # 0.5<x<0.75
        right_right = []  # 0.75<x<1

        for tree in self.trees:
            node = tree.root_node
            first_split = node.value
            # left split:
            left_left.append(info_dict(node.fb, first_split)[0])
            left_right.append(info_dict(node.fb, first_split)[1])
            # right split:
            right_left.append(info_dict(node.tb, first_split)[0])
            right_right.append(info_dict(node.tb, first_split)[1])

        ranges = [left_left, left_right, right_left, right_right]
        # analyzing:
        plotting = []
        for r in ranges:
            print('-------------------------------------')
            # beast value:
            beasts = [x['range'] for x in r]
            average_beast = [round(sum(y) / len(y), 3) for y in zip(*beasts)]
            print(f'Average BEASTpred range of {average_beast}\t'
                  f'lowest range {[round(min(y), 3) for y in zip(*beasts)]}\t'
                  f'highest range {[round(max(y), 3) for y in zip(*beasts)]}')
            print()
            #  splits information
            feature_values = {feature: [] for feature in self.feature_names}
            for dict in r:
                range, feature, value = dict.values()
                feature_values[feature].append(value)

            top_features = []
            splits = []
            splits_values = []
            counter = 0
            for feature, values in feature_values.items():
                if len(values) > 0.05 * len(self.trees):
                    splits.append(f"{feature} - {len(values)}%")
                    splits_values.append(len(values))
                    counter += len(values)
                    top_features.append(
                        {'feature': feature, 'number of splits': len(values),
                         'average value': round(np.mean(values), 3),
                         'values range': (round(min(values), 3), round(max(values), 3))})
            if counter < len(self.trees):
                splits.append(f"others - {len(self.trees) - counter}%")
                splits_values.append(len(self.trees) - counter)
            for i, feature in enumerate(sorted(top_features, key=lambda k: k['number of splits'], reverse=True)):
                print(f"The {i + 1}'th best feature is: ")
                print(feature)

            plotting.append((f"BEASTpred range of {average_beast}", splits, splits_values))

        # plot some visualization
        figure, axis = plt.subplots(2, 2, figsize=(10, 7))
        axis[0, 0].pie(plotting[0][2], labels=plotting[0][1])
        axis[0, 0].set_title(plotting[0][0])
        axis[0, 1].pie(plotting[1][2], labels=plotting[1][1])
        axis[0, 1].set_title(plotting[1][0])
        axis[1, 0].pie(plotting[2][2], labels=plotting[2][1])
        axis[1, 0].set_title(plotting[2][0])
        axis[1, 1].pie(plotting[3][2], labels=plotting[3][1])
        axis[1, 1].set_title(plotting[3][0])

        # Combine all the operations and display
        plt.tight_layout()
        plt.show()


def get_tree_data(self, tree_index):
    """
    :param tree_index:
    :return: returns the data used to train the tree in the given index
    """
    # return self.trees_data[tree_index]
    pass


class NotFittedError(Exception):
    """
    Creating my own exceptions. lol
    """
    pass


def load_model(file_name) -> RandomForestRegressor:
    """
    load model form given file name
    :param file_name: file to load model from
    :return: loaded model
    """
    file_name = str(file_name)
    if len(file_name) < 3 or file_name[:-2] != '.p':
        file_name += '.p'
    return pickle.load(open(file_name, "rb"))
