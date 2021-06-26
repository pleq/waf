import pickle

with open(r"C:/Users/emirm/Desktop/project/Docker/waf/ml-model/saved-models/decision_tree_model.pickle", "rb") as fd:
     model = pickle.load(fd)
     print(model)