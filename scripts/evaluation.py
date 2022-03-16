import pandas as pd
import os
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


df = pd.read_csv(os.path.join(os.path.curdir,"OCSVM_scoresnew3.csv"))
df.reset_index(drop=True, inplace=True)
#print(df)
preds = df.iloc[:,1].to_numpy()         # array of predictions
print(preds)
labels = pd.read_csv(os.path.join(os.path.curdir, "labels.csv"))
labels = labels.iloc[:,1].to_numpy()            # array of test labels
print(labels)
scores = np.zeros_like(preds)           #scores based on predictions on test data
for i in range(preds.shape[0]):
    if preds[i] > 0:
      scores[i] = 0
    else:
        scores[i] = 1
print(scores)

###ROC-AUC___________________________________________
fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=0)
area_under_curve = metrics.auc(fpr, tpr)
print(area_under_curve)
plt.plot(fpr,tpr,label= "AUC= "+str(area_under_curve))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
#_____________________________________________________________

#Confusion Matrix_____________________________________________
cf_matrix = confusion_matrix(labels, scores)
print(cf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,display_labels=np.array(['Valid Images','Invalid Images']))
disp.plot()
plt.show()