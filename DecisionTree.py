import random
import numpy as np


class DecisionTreeClassifier:
    class DecisionNode:
        def __init__(self, feature=-1, value=None, results=None, tb=None, fb=None):
            self.feature = feature  # feature split
            self.value = value  # value split
            self.results = results  # result of prediction reaching this node
            self.tb = tb  # true branch - rows with matching feature and value
            self.fb = fb  # false branch - rows where feature and value are not matching

    def __init__(self, forced=None):
        self.root_node = None
        self.max_depth = None
        self.criterion = self.SSE
        self.min_samples_leaf = 1
        self.min_samples_split = 2
        self.max_features = None
        self.nb_features = None
        self.force_split = forced

    def fit(self, X, max_depth, max_features, criterion, min_samples_split, min_samples_leaf):
        """
        :param X: The data used to train the decision tree.
        :param  max_depth: Maximum number of splits during training. If None, then nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.
        :param max_features: The number of features to consider when looking for the best split.
        :param criterion: The function used to split data at each node of the tree.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param  min_samples_leaf:   The minimum number of samples required to be at a leaf node.
        """
        if not criterion or criterion == 'mse':
            self.criterion = self.SSE

        if max_depth is not None and (not (isinstance(max_depth, int) or max_depth < 0)):
            raise ValueError("invalid max_depth value {}".format(max_depth))
        self.max_depth = max_depth

        if min_samples_leaf < 1 or not isinstance(min_samples_leaf, int):
            raise ValueError(
                "invalid min_samples_leaf value, leaf can't have {} elements".format(min_samples_split))
        if min_samples_split < 2 or not isinstance(min_samples_split, int):
            raise ValueError(
                "invalid min_samples_split value, cant split node with {} elements".format(min_samples_split))
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        self.max_features = max_features

        depth = len(X) + 1 if self.max_depth is None else self.max_depth
        self.nb_features = X.shape[1] - 1

        self.root_node = self.build_tree(X, depth, self.force_split)

    def predict(self, features):
        """
        Returns a prediction for the given features.
        :param  features:   A list of values
        """
        return self.classify(features, self.root_node)

    def SSE(self, data):
        """
        Returns the SSE (sum squared error) on data.
        """
        predicted = np.mean(data[:, -1])
        error = np.sum((data[:, -1] - predicted) ** 2)
        return error

    def split_data(self, X, column, value):
        """
        Splits the given dataset depending on the value at the given column index.
        :param  X: The dataset
        :param  column: The index of the column used to split data
        :param  value:  The value used for the split
        """
        if isinstance(value, int) or isinstance(value, float):
            split = X[:, column] >= value
        else:
            split = X[:, column] == value
        return X[split], X[~split]

    def label_mean(self, data: np.ndarray):
        """
        Returns the average of the labels of all rows in data.
        labels are located in the end of each row
        """
        return np.mean(data[:, -1])

    def find_best_split(self, X, features):
        """
        Find best split in data on given features
        :param X: Data to split
        :param features: Features to split on
        :return: Best feature and split value.
        """
        best_error = self.criterion(X)
        best_criteria = None

        for feature in features:
            X = X[X[:, feature].argsort()]
            feature_values = X[:, feature]
            last_value = feature_values[self.min_samples_leaf - 1]
            end = -self.min_samples_leaf + 1 if self.min_samples_leaf > 1 else None
            for index, value in enumerate(feature_values[self.min_samples_leaf:end]):
                if value == last_value:
                    continue
                split_on = value
                if isinstance(last_value, int) or isinstance(last_value, float):
                    # upon continuous features take the average of two values
                    split_on = (last_value + value) / 2
                    last_value = value
                curr_index = index + self.min_samples_leaf
                split_error = self.criterion(X[:curr_index]) + self.criterion(X[curr_index:])
                if split_error < best_error:
                    best_error = split_error
                    best_criteria = (feature, split_on)

        return best_criteria

    def build_tree(self, X, depth, force_split):
        """
        Recursively creates the decision tree by splitting the dataset until no
        gain of information is added, or until the max depth is reached.
        :param  X: The dataset
        :param  depth: The current depth in the tree
        :param  force_split: Tuple of (feature, splits). Force the first #splits to be splitted on feature.
                If None, split normally.
        """
        # can't split any more:
        if len(X) < self.min_samples_split or depth == 0:
            return self.DecisionNode(results=self.label_mean(X))

        if self.criterion == self.SSE:
            # SSE criterion implementation

            features = random.sample(range(self.nb_features), self.max_features)

            if force_split:
                if force_split[1] > 0:
                    features = force_split[0] if isinstance(force_split[0], list) else [force_split[0]]
                    force_split = (force_split[0], force_split[1] - 1, force_split[2], force_split[3])
                else:
                    if force_split[3]:
                        features = [x for x in list(range(self.nb_features)) if x != force_split[0]]
                        features = random.sample(features, self.max_features)
                        force_split = (force_split[0], force_split[1], force_split[2], force_split[3] - 1)

            best_criteria = self.find_best_split(X, features)

            if best_criteria is not None:  # improving split founded
                best_sets = self.split_data(X, best_criteria[0], best_criteria[1])
                trueBranch = self.build_tree(best_sets[0], depth - 1, force_split)
                falseBranch = self.build_tree(best_sets[1], depth - 1, force_split)
                return self.DecisionNode(feature=best_criteria[0],
                                         value=best_criteria[1],
                                         tb=trueBranch, fb=falseBranch)
            else:
                return self.DecisionNode(results=self.label_mean(X))

    def classify(self, observation, tree):
        """
        Recursively going down the tree to predict the features value
        :param  observation: The features to use to predict
        :param  tree: The current node
        """
        if tree is None:
            raise RuntimeError("model is not fitted")
        if tree.results is not None:
            return tree.results
        else:
            v = observation[tree.feature]
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)

    def print_vertical(self, feature_names, depth):
        print('Left branch is the true branch')
        print('That is all observations with value x>=SplitValue')

        if feature_names is None:
            feature_names = range(self.nb_features)

        # right = fb
        def recursive_print(root, dep):
            """Returns list of strings, width, height, and horizontal coordinate of the root."""
            # No child.
            if dep == 0 or (root.fb is None and root.tb is None):
                if root.results:
                    line = '%s' % round(root.results, 3)
                else:
                    line = '%s' % feature_names[root.feature] + ': ' + str(round(root.value, 3))
                width = len(line)
                height = 1
                middle = width // 2
                return [line], width, height, middle

            # Only left child.
            if root.fb is None:
                lines, n, p, x = recursive_print(root.tb, dep - 1)
                s = '%s' % feature_names[root.feature] + ': ' + str(round(root.value, 3))
                u = len(s)
                first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
                second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
                shifted_lines = [line + u * ' ' for line in lines]
                return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

            # Only right child.
            if root.tb is None:
                lines, n, p, x = recursive_print(root.fb, dep - 1)
                s = '%s' % feature_names[root.feature] + ': ' + str(round(root.value, 3))
                u = len(s)
                first_line = s + x * '_' + (n - x) * ' '
                second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
                shifted_lines = [u * ' ' + line for line in lines]
                return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

            # Two children.
            left, n, p, x = recursive_print(root.tb, dep - 1)
            right, m, q, y = recursive_print(root.fb, dep - 1)
            s = '%s' % feature_names[root.feature] + ': ' + str(round(root.value, 3))
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
            second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
            if p < q:
                left += [n * ' '] * (q - p)
            elif q < p:
                right += [m * ' '] * (p - q)
            zipped_lines = zip(left, right)
            lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
            return lines, n + m + u, max(p, q) + 2, n + u // 2

        lines, *_ = recursive_print(self.root_node, depth-1)
        for line in lines:
            print(line)

    def print_topdown(self, feature_names, dep):
        """
        recursively print the tree topdowm.
        """
        print('Left branch is the true branch represented by the ' + chr(881) + ' sign')
        print('That is all observations with value x>=SplitValue')
        print('Right branch (false branch) is represented by ' + chr(8735) + ' sign')

        if feature_names is None:
            feature_names = list(range(self.nb_features))
        feature_names.append('leaf')
        rec = np.zeros(1000)

        def recursive_print(root, depth, stop):
            if root is None or not stop:
                return

            print('\t', end='')
            for i in range(depth):
                if i == depth - 1:
                    print(chr(881), end='') if rec[depth - 1] else print(chr(8735), end='')
                    print(chr(8212) * 3, end='')
                else:
                    print('|', end='') if rec[i] else print('  ', end='')
                    print('  ', end='')
            if root.feature == -1:
                print('%s' % str(round(root.results, 3)))
            else:
                print('%s' % str(feature_names[root.feature]) + ': ' + str(round(root.value, 3)))
            rec[depth] = 1
            recursive_print(root.tb, depth + 1, stop-1)
            rec[depth] = 0
            recursive_print(root.fb, depth + 1, stop-1)

        recursive_print(self.root_node, 0, dep)

    def print_horizontal(self, feature_names, depth, spacious):
        """
        Recursively print the tree horizontally. (root in leftmost side and leaves are at rightmost side)
        :param spacious: space between level of the tree in the printout.
        """
        print('The lower branch is the true branch')
        print('That is all observations with value x>=SplitValue')
        if feature_names is None:
            feature_names = range(self.nb_features)

        def recursive_print(root, dep, space):
            if root is None or dep == 0:
                return
            space += spacious
            recursive_print(root.fb, dep - 1, space)
            for i in range(spacious, space):
                print(end=" ")

            if root.feature == -1:
                print('%s' % round(root.results, 3))
            else:
                print('%s' % str(feature_names[root.feature]) + ': ' + str(round(root.value, 3)))
            recursive_print(root.tb, dep - 1, space)

        recursive_print(self.root_node, depth, 0)
