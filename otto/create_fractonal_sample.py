__author__ = 'harry'

import random

def create_fractional_sample(input_filename, fractional_sample, has_header=True):
    with open(input_filename, 'r') as the_file:
        if has_header:
            header = the_file.readline()
            print header,
        for line in the_file:
            if random.random() < fractional_sample:
                print line,

if __name__ == '__main__':
    create_fractional_sample('train.csv', 0.01)