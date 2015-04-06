import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


def create_X_y_files(input_filename):
    with open(input_filename) as the_file:
        df = pd.read_csv(the_file)

        #print df.head(3)

        df['category'] = 0

        for i in range(1,10):
            df.loc[df['target'] == 'Class_{}'.format(i), 'category'] = i

        #print df['category']

        X = df.ix[:,1:94] # zero indexed
        y = df['category']

        #write to CSV with headers
        X.to_csv('X_with_headers.csv', sep=',', header=True, index=False)
        y.to_csv('y_with_headers.csv', sep=',', header=True, index=False)
        #write to CSV without headers
        X.to_csv('X.csv', sep=',', header=False, index=False)
        y.to_csv('y.csv', sep=',', header=False, index=False)


        #print X.head()

create_X_y_files('sample.csv')