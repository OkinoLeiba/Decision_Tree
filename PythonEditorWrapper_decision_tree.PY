# Prolog - Auto Generated #
import os, uuid, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import pandas

import sys
sys.tracebacklimit = 0

os.chdir(u'C:/Users/Owner/PythonEditorWrapper_242e5631-2b72-49ed-a355-dc4d6ac3ee14')
dataset = pandas.read_csv('input_df_839b6fea-46ba-41e8-aec6-e6747cbf8b35.csv')

matplotlib.pyplot.figure(figsize=(5.55555555555556,4.16666666666667), dpi=72)
matplotlib.pyplot.show = lambda args=None,kw=None: matplotlib.pyplot.savefig(str(uuid.uuid1()))
# Original Script. Please update your script content here and once completed copy below section back to the original editing window #
# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(salary, Total Time at the Company, Quit the Company)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# file_path = "C:/Users/Owner/source/vsc_repo/machine_learn_cookbook/Logistic_Regression_Quit-Predict/hr_file.csv"
# hr_data = pd.read_csv(file_path, sep=",", engine="python", encoding="utf-8", encoding_errors="strict")

hr_data = dataset

hr_data.rename(columns={'Departments ': 'Departments', 'salary' : 'Salary' }, inplace=True)
# hr_data.rename(str.title, axis='columns', inplace=True)
# hr_data['Departments'] = [s.title() for s in hr_data.Departments]
# hr_data.drop('Departments ', axis=1, inplace=True)
hr_data['Salary'] = [s.title() for s in hr_data['Salary']]
# hr_data['Departments'] = hr_data['Departments '].replace(["Hr", "It","Mng", "Randd"], ["HR","IT","MNG","R&D"], regex=True, inplace=False)

hr_dummies = pd.get_dummies(hr_data, prefix_sep="_", dummy_na=False, dtype=int, drop_first=True).dropna(axis=0, how="any", inplace=False)

X = hr_dummies.drop("Quit The Company", axis=1)
y = hr_dummies["Quit The Company"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

dtc = DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features="auto", min_samples_split=2, random_state=0)
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)

classNames=["No Quit, Quit"]
tree.plot_tree(dtc, feature_names=X.columns, filled=True, label="all", fontsize=8, impurity=True)
plt.show()

# Epilog - Auto Generated #
os.chdir(u'C:/Users/Owner/PythonEditorWrapper_242e5631-2b72-49ed-a355-dc4d6ac3ee14')
