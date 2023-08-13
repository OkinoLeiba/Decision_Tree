# %% [markdown]
# #  <span style="text-shadow: 80px 10px #CD5C5C; color: black;">Decision Tree</span> 

# %%
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# %%
file_path = "C:/Users/Owner/source/vsc_repo/machine_learn_cookbook/Logistic_Regression_Quit-Predict/hr_file.csv"
hr_data = pd.read_csv(file_path, sep=",", engine="python", encoding="utf-8", encoding_errors="strict")
hr_data.rename(columns={'Departments ': 'Departments', 'salary' : 'Salary' }, inplace=True)
# hr_data.rename(str.title, axis='columns', inplace=True)
hr_data['Departments'] = [s.title() for s in hr_data['Departments']]
# hr_data.drop('Departments ', axis=1, inplace=True)
hr_data['Salary'] = [s.title() for s in hr_data['Salary']]
hr_data['Departments'] = hr_data['Departments'].replace(["Hr", "It","Mng", "Randd"], ["HR","IT","MNG","R&D"], regex=True, inplace=False)

# %% [markdown]
# ##  <span style="text-shadow: 80px 10px black; color: #CD5C5C;">Exploratory Data Analysis</span>

# %%
hr_data.head(5)

# %%
hr_data.tail(5)

# %%
hr_data.sample(n=5, random_state=3)

# %%
hr_data.info()

# %%
hr_data.dtypes

# %%
hr_data.index

# %%
hr_data.shape

# %%
hr_data.ndim

# %%
hr_data.columns

# %%
hr_data["Departments"].unique()

# %%
hr_data["Salary"].unique()

# %%
hr_data.describe()

# %%
hr_data["Quit the Company"].loc[(hr_data["Departments"] == "IT") & (hr_data["Salary"] == "Low")].value_counts()

# %%
hr_data["Quit the Company"].loc[(hr_data["Departments"] == "IT") & (hr_data["Salary"] == "Medium")].value_counts()

# %%
hr_data["Quit the Company"].loc[(hr_data["Departments"] == "IT") & (hr_data["Salary"] == "High")].value_counts()

# %% [markdown]
# ##  <span style="text-shadow: 80px 10px #CD5C5C; color: black;">Preprocessing</span>

# %%

he = OneHotEncoder(categories="auto", drop="first", handle_unknown="error")
hr_encode = he.fit_transform(hr_data)
hr_dummies = pd.get_dummies(hr_data, prefix_sep="_", dummy_na=False, dtype=int, drop_first=True).dropna(axis=0, how="any", inplace=False)
hr_dummies.head(3)



# %%
le = LabelEncoder()
hr_data["Departments_Encode"] = le.fit_transform(hr_data["Departments"])
hr_data["Salary_Encode"] = le.fit_transform(hr_data["Salary"])
hr_data.head(5)

# %%
X = hr_dummies.drop("Quit the Company",  axis=1)
y = hr_dummies["Quit the Company"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# %% [markdown]
# ##  <span style="text-shadow: 80px 10px black; color: #CD5C5C;">Decision Tree Classifier</span>

# %%
from sklearn.model_selection import GridSearchCV
param_grid = {"criterion" : ["gini", "entropy", "log_loss"], "max_depth" : [6, 8, 10, 12], "min_samples_split" : [2, 4, 6, 8], "max_features" : ["auto", "sqrt", "log2"], 
              "random_state" : [0,42], }
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring="accuracy", verbose=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


dtc = DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features="auto", min_samples_split=2, random_state=0)
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)

# %% [markdown]
# ##  <span style="text-shadow: 80px 10px #CD5C5C; color: black;">Visualization: Tree Plot</span>

# %%
classNames=["No Quit, Quit"]
plt.figure(figsize=(10,8), dpi=150)
tree.plot_tree(dtc, feature_names=X.columns, filled=True, label="all", fontsize=8, impurity=True)
plt.show()

# %% [markdown]
# #  <span style="text-shadow: 80px 10px black; color:  #CD5C5C;">Decision Tree for Power BI</span>

# %%
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

file_path = "C:/Users/Owner/source/vsc_repo/machine_learn_cookbook/Logistic_Regression_Quit-Predict/hr_file.csv"
hr_data = pd.read_csv(file_path, sep=",", engine="python", encoding="utf-8", encoding_errors="strict")
hr_data.rename(columns={'Departments ': 'Departments', 'salary' : 'Salary' }, inplace=True)
# hr_data.rename(str.title, axis='columns', inplace=True)
hr_data['Departments'] = [s.title() for s in hr_data['Departments']]
# hr_data.drop('Departments ', axis=1, inplace=True)
hr_data['Salary'] = [s.title() for s in hr_data['Salary']]
hr_data['Departments'] = hr_data['Departments'].replace(["Hr", "It","Mng", "Randd"], ["HR","IT","MNG","R&D"], regex=True, inplace=False)

hr_dummies = pd.get_dummies(hr_data, prefix_sep="_", dummy_na=False, dtype=int, drop_first=True).dropna(axis=0, how="any", inplace=False)

X = hr_dummies.drop("Quit the Company", axis=1)
y = hr_dummies["Quit the Company"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

dtc = DecisionTreeClassifier(criterion="entropy", max_depth=6, max_features="auto", min_samples_split=2, random_state=0)
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)

classNames=["No Quit, Quit"]
plt.figure(figsize=(10,8), dpi=150)
tree.plot_tree(dtc, feature_names=X.columns, filled=True, label="all", fontsize=8, impurity=True)
plt.show()


