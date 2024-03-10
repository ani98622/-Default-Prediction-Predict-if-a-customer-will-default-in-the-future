import os
import sys
from dataclasses import dataclass
from tqdm import tqdm
import numpy 
import pandas as pd
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.pipeline import Pipeline
import optbinning 
from sklearn.model_selection import train_test_split
from src.Credit_Defaultor_Prediction.exception import CustomException
from src.Credit_Defaultor_Prediction.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path,labels_path):

        try:
            train =pd.read_feather(train_path)
            test =pd.read_feather(test_path)
            labels = pd.read_csv(labels_path)

            logging.info("Reading the train, test file, labels file")

            # We are considering last transactions of each customer_ID
            train = train.groupby("customer_ID").tail(1).reset_index(drop = True)
            test = test.groupby("customer_ID").tail(1).reset_index(drop = True)

            # merging with the targets values
            train = train.merge(labels, on = 'customer_ID', how = 'left')

            drop_cols = ['customer_ID','S_2','target']
            train_cols = [_ for _ in train.columns if _ not in drop_cols]
            
            categorical = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
            
            logging.info(f"Categorical Columns:{categorical}")

            # IV score claculation for categorical and numerical type of data
            iv_dict = {}
            for _ in tqdm(train_cols): # what is this tqdm doing here?
                if _ in categorical :
                    ob = optbinning.OptimalBinning(dtype = 'categorical')
                    ob.fit(train[_],train['target'])
                else:
                    ob = ob = optbinning.OptimalBinning(dtype = 'numerical')
                    ob.fit(train[_],train['target'])
                binning_table = ob.binning_table
                binning_table.build()  
                iv_dict[_] = binning_table.iv

            iv_df = pd.Series(iv_dict)
            iv_df.sort_values(ascending  = False, inplace = True )

            # Now, we will select the features for IV values > 0.5
            s_f = iv_df[iv_df > 0.5].index.values  
            categorical = [_ for _ in categorical if _ in s_f]
            train_cols = [_ for _ in train.columns if _ in s_f]

            def drop_feature_selection(r , c , corr , r_id , c_id):
                if r_id >= c_id:
                    return c
                else :
                    return r
                
            cor_mat = train[train_cols].corr().abs()
            upper_tri = cor_mat.where(np.triu(np.ones(cor_mat.shape),k=1).astype(np.bool_))
            corr_df = upper_tri.stack().reset_index()
            corr_df.columns = ['row', 'col', 'corr']
            corr_df = corr_df.drop_duplicates()
            corr_df = corr_df.sort_values('corr', ascending=False)
            corr_df = corr_df.query("corr >= 0.8")  # query similar to SQL
            corr_df['row_iv'] = corr_df['row'].map(iv_dict)
            corr_df['col_iv'] = corr_df['col'].map(iv_dict)
            corr_df['drop_feature'] = corr_df.apply(lambda x: drop_feature_selection(x['row'], x['col'], x['corr'], x['row_iv'], x['col_iv']), axis=1)

            corr_drop_features = corr_df['drop_feature'].unique().tolist()

            train_data, cv_data = train_test_split(train, test_size=0.3, random_state=42, shuffle=True, stratify=train['target'])

            s_f = [_ for _ in s_f if _ not in corr_drop_features]
            categorical = [_ for _ in categorical if _ in s_f]
            train_cols = [_ for _ in train.columns if _ in s_f]

            X_train = train_data[train_cols].copy()
            y_train = train_data['target'].copy()

            X_cv = cv_data[train_cols].copy()
            y_cv = cv_data['target'].copy()

            X_test = test[train_cols].copy()

            logging.info("Data Transformation Done")

            return (categorical,train_cols,X_train, y_train, X_cv, y_cv , X_test )
        
        except Exception as e:
            raise CustomException(sys,e)




