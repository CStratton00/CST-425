# Import packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
warnings.filterwarnings('ignore')


##################################
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape  # shape gives us the number of rows and columns
##################################
train.info()
##################################
print(train.isnull().sum())
##################################
# Test comparisons
print(f"Amount of transported people = {train['Transported'].value_counts()}")
f, ax = plt.subplots(1, 2, figsize=(12, 4))
train['Transported'].value_counts().plot.pie(
    explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False)
ax[0].set_title('Survivors (1) and the dead (0)')
ax[0].set_ylabel('')
sns.countplot('Transported', data=train, ax=ax[1])
ax[1].set_ylabel('Quantity')
ax[1].set_title('Survivors (1) and the dead (0)')
plt.show()


##################################

