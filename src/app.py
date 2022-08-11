# Import libarys
import pandas as pd
import numpy as np 

from datetime import datetime

import seaborn as sns
import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt

from pylab import rcParams
import statsmodels.api as sm

from pmdarima.arima import auto_arima

import pickle

# Load datasets



url_test_a = pd.read_csv('../data/raw/cpu-test-a.csv')
url_test_b = pd.read_csv('../data/raw/cpu-test-b.csv')

# change 'datetime' type to 'datetime64'
test_a['datetime']=test_a['datetime'].astype('datetime64')
test_b['datetime']=test_b['datetime'].astype('datetime64')

# Set index
test_a.set_index('datetime',inplace=True)
test_b.set_index('datetime',inplace=True)


# Save the model as a pickle
filename = '../models/best_model_b.pkl'
pickle.dump(stepwise_model, open(filename,'wb'))

