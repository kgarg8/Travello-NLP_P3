import json
import os
filepath = 'experiments/base_model/val_best_weights.json'
if os.path.exists(filepath):
    f = open(filepath)
    data = json.load(f)
    best_val_acc = data['accuracy']
    print(best_val_acc)
