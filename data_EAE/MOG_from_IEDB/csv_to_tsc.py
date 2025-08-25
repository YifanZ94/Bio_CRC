# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:58:51 2025

@author: a4945
"""

import csv

# Input and output file paths
input_file = 'std_MOG_assays.csv'
output_file = 'train.tsv'

# Convert CSV to TSV
with open(input_file, 'r', newline='', encoding='utf-8') as csvfile, \
     open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
    
    reader = csv.reader(csvfile, delimiter=',')
    writer = csv.writer(tsvfile, delimiter='\t')
    
    for row in reader:
        writer.writerow(row)
