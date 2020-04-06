'''
Data preprocessing
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
print("Created on: {}".format(timestr))

# change dir
# dirname = os.path.dirname(__file__)
# os.chdir('../' + dirname + '/data')

os.chdir('/home/olga/Documents/fionamalone/data/')

data = pd.read_excel('data_extra.xlsx', index_col=None, header=2)
data.reset_index(drop=True, inplace=True)

# rename columns
cols = {'Type': 'Arch', 'Rigid/Flex?': 'Rigid', 'Type.1': 'Flowrate'}
data.rename(columns=cols, inplace=True)

# convert to 1 for Rigid and 0 for Flex
binary = {'Rigid': 1, 'Flexible': 0}
data['Rigid'] = data['Rigid'].map(binary)

# remove spaces in values


def remove_whitespace(x):
    """
    Helper function to remove any blank space from a string
    x: a string
    """
    try:
        # Remove spaces inside of the string
        x = "".join(x.split())

    except:
        pass
    return x


data['Thrombin'] = data['Thrombin'].apply(remove_whitespace)
data['Branch'] = data['Branch'].apply(remove_whitespace)
data['Arch'] = data['Arch'].apply(remove_whitespace)

# branches are not unique
# mixed up lower and upper cases

data.Branch.unique()
data['Branch'] = data['Branch'].str.lower()

# replace branch name
data[data.Branch == 'varight']
data.replace({'varight': 'rsub'}, inplace=True)
data[data.Branch == 'varight']

data.columns

data['Thrombin'] = data['Thrombin'].str.lower()
data['Arch'] = data['Arch'].str.lower()
data['Entry'] = data['Entry'].str.lower()
data['Flowrate'] = data['Flowrate'].str.lower()

data


data.to_csv('data.csv')
