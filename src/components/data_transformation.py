import os 
import sys 
from src.exception import CustomException 
from src.logger import logging 
from dataclasses import dataclass 
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['writing_score','reading_score'],
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read both train and test data")
            preprocessor_obj=self.get_data_transformer_object()
            target_column=['math_score']
            numerical_columns=['reading_score','writing_score']
            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            target_feature_train_df=train_df[target_column]
            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            target_feature_test_df=test_df[target_column]
            logging.info(f"applying column transformer on training and testing data")
            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.fit_transform(input_feature_test_df)
            train_array=np.c_(input_feature_train_array,np.array(target_feature_train_df))
            test_array=np.c_(input_feature_test_array,np.array(target_feature_test_df))
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)
            logging.info(f"preprocessor object saved successfully")
            return (train_array,test_array,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)
        

