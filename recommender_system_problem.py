import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.datasets import fetch_stackexchange
from lightfm.evaluation import auc_score
import csv

# Fetch data that have rating equal or greater than 4.0
data = fetch_movielens(min_rating=4.0)

# Printing Training and Testing Data
print(repr(data['train']))
print(repr(data['test']))

# Create dictionary of different mmodels using different loss function
models = {
    'warp': LightFM(loss='warp', item_alpha=1e-6, no_components=3),
    'logistic': LightFM(loss='logistic', item_alpha=1e-6, no_components=3),
    'bpr': LightFM(loss='bpr', item_alpha=1e-6, no_components=3),
    'warp-kos': LightFM(loss='warp-kos', item_alpha=1e-6, no_components=3),
}


def sample_recommendation(model, data, user_ids):

    # Number of users and movies in Training Data
    n_users, n_items = data['train'].shape

    # Generates output for each input
    for user_id in user_ids:

        # Movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Movies Recommended by the model
        scores = model.predict(user_id, np.arange(n_items))

        # Rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # Printing the results
        print("----User %s---" % user_id)
        print("--Known positives--:")

        for x in known_positives[:3]:
            print("%s" % x)

        print()
        print("--Recommended--:")
        recommended = []
        for x in top_items[:3]:
            recommended.append("%s" % x)
            print("%s" % x)
    return(recommended)

# Iterating over each model to find recommended movies for each
for model_name, model in models.items():
    print("/-------%s-------/" % model_name)
    model.fit(data['train'], item_features=data['item_features'], epochs=30, num_threads=2)

    test_auc = auc_score(model, data['test'], train_interactions=data['train'], item_features=data['item_features'], num_threads=2).mean() 
    print("Test AUC: %.3f%%" % (test_auc * 100))
    print()
    
    # Using models dictionary to store accuracy and recommended movie
    models[model_name] = {
        "auc": test_auc,
        "recommended": sample_recommendation(model, data, [69, 789, 900]),
    }

max_auc = max([value["auc"] for value in models.values()])
model_name = [model_name for model_name, value in models.items() if value["auc"] == max_auc][0]

print()
print("Highest AUC of %.3f%% attained by %s." % (max_auc, model_name))
print()
print("Result is")
print(models[model_name]["recommended"])