#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rich import print
import pandas as pd
from functools import partial
from typing import *
import matplotlib.pyplot as plt


# ## Disclaimer
# Here i use `rich` library for pretty-printing. Here is the link: https://github.com/willmcgugan/rich
# 
# It does produce a lot of html for visualization, so if you do not want to use it - simply remove first line of imports

# ## Task 1


class Tree:
    """Create a binary tree; keyword-only arguments `data`, `left`, `right`.

    Examples:
      l1 = Tree.leaf("leaf1")
      l2 = Tree.leaf("leaf2")
      tree = Tree(data="root", left=l1, right=Tree(right=l2))
    """

    def leaf(data):
        """Create a leaf tree"""
        return Tree(data=data)

    # pretty-print trees
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (
                self.data,
                self.left,
                self.right,
            )

    # all arguments after `*` are *keyword-only*!
    def __init__(self, *, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        """Check if this tree is a leaf tree"""
        return self.left == None and self.right == None

    def children(self):
        """List of child subtrees"""
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        """Compute the depth of a tree
        A leaf is depth-1, and a child is one deeper than the parent.
        """
        return max([x.depth() for x in self.children()], default=0) + 1


morning = Tree(
    data="morning?",
    left=Tree.leaf("like"),
    right=Tree.leaf("nah")
)
liked_other_sys = Tree(
    data="likedOtherSys?",
    left=Tree.leaf("nah"),
    right=Tree.leaf("like")
)
taken_other_sys = Tree(
    data="takenOtherSys?",
    left=morning,
    right=liked_other_sys
)
is_systems = Tree(data="isSystems?", left=Tree.leaf("like"), right=taken_other_sys)


print(is_systems)


# ## Task 2
# data.csv file contains dataset from the task

data = pd.read_csv("./data.csv")

data['ok'] = data['rating'] >= 0
print(data)


# ## Task 3

def get_most_common_value(series: pd.Series) -> bool:
    return series.value_counts()[:1].index.tolist()[0]

def single_feature_score(data, goal, feature):

    ## learning
    true_part = data[data[feature] == True][goal]
    false_part = data[data[feature] == False][goal]
    
    if len(false_part) != 0:
        most_common_for_false = get_most_common_value(false_part)
        n_matches_for_false = false_part.value_counts()[most_common_for_false]
    else:
        n_matches_for_false = 0
        
    if len(true_part) != 0:
        most_common_for_true = get_most_common_value(true_part)
        n_matches_for_true = true_part.value_counts()[most_common_for_true]
    else:
        n_matches_for_true = 0
    
    return (n_matches_for_true + n_matches_for_false) / len(data)


def best_feature(data, goal, features):
    # optional: avoid the lambda using `functools.partial`
    scorer = partial(single_feature_score, data, goal)
    return max(features, key=scorer)


feature_names = {'easy', 'ai', 'systems', 'theory', 'morning'}


best_feature(data, 'ok', feature_names)


# ## Task 4, 5
# Here i use `get_most_common_value` to select guess. Then i return leaf with guess if one of following conditions are met:
# - labels(`goal`) are unambiguous 
# - feature set exhausted
# - max depth reached
# 
# If none of these conditions are met, i select best feature using `best_feature` and separate the dataset by it.
# Then, i do recursive call to build trees for false and true subsets with `remaining_features`. If such subsets are empty i just return the leaf with `guess` value.

def decision_tree_train(data: pd.DataFrame, goal: str, features: Set[str], max_depth: int = 999) -> Tree:
    guess = get_most_common_value(data[goal])
    if len(set(data[goal])) == 1 or len(features) == 0 or max_depth == 0:
        return Tree.leaf(guess)
    else:
        current_best_feature = best_feature(data, goal, features)
        remaining_features = features.copy()
        remaining_features.remove(current_best_feature)
        
        true_subset = data[data[current_best_feature] == True]
        false_subset = data[data[current_best_feature] == False]
        
        if len(false_subset) != 0:
            left = decision_tree_train(false_subset, goal, remaining_features, max_depth-1)
        else:
            left = Tree.leaf(guess)
            
        if len(true_subset) != 0:
            right = decision_tree_train(true_subset, goal, remaining_features, max_depth-1)
        else:
            right = Tree.leaf(guess)
        
        return Tree(data=current_best_feature, left=left, right=right)

def decision_tree_test(tree: Tree, data_point: Dict[str, bool]) -> bool:
    if tree.is_leaf():
        return tree.data
    else:
        feature_name = tree.data
        if not data_point[feature_name]:
            return decision_tree_test(tree.left, data_point)
        else:
            return decision_tree_test(tree.right, data_point)


# This is the helper function to compute accuracy score for tree

def compute_score(tree: Tree, goal: str, data: pd.DataFrame) -> float:
    data_points = data.to_dict(orient='records')
    y_pred = [decision_tree_test(tree, point) for point in data_points]
    y_true = [point[goal] for point in data_points]
    n_matches = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    return n_matches / len(data)


# Here i show dependency between max_depth and accuracy.

max_depths = list(range(8))
scores = []
for i in max_depths:
    tree = decision_tree_train(data, 'ok', feature_names, max_depth=i)
    score = compute_score(tree, 'ok', data)
    scores.append(score)

plt.grid(True)
plt.ylabel("Accuracy")
plt.xlabel("Max Depth")
plt.plot(max_depths, scores, "-go")
plt.savefig("accuracy-vs-max_depth.png")

