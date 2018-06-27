import os
import pandas as pd
from data_utils import compare_submmits
from data_utils import prods_op
from data_utils import counter_csv
from data_utils import vote_val

# prods_path = './prods/'
# prods_op(prods_path, sub_name='raw_mfcc_5_models.csv', rand_num=5)

# root_path = 'submissions/'
multi_prods_csv_path = 'submission_1.csv'
origin_csv_path = 'submission_cnn_raw_data_mfcc_simlified_dropout_0.7_reg_0.2_sap_all.csv'
multi_prods_5_csv_path = 'raw_mfcc_5_models.csv'
strictly_vote_csv = 'strictly_vote_submissions.csv'
demo_csv_path = '1d_2d_ensembled_submission.csv'

indices = compare_submmits(demo_csv_path, strictly_vote_csv)
print(len(indices))

# path = 'submissions/'
# counter_csv(path)
# vote(path, 'vote_submissions.csv')

# vote_val(prods_path, sub_path='strictly_vote_submissions.csv')





