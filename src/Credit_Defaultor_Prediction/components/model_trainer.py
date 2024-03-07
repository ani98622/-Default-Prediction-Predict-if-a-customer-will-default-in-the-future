import os
import sys
import pandas as pd

import optbinnig
from dataclasses import dataclass
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from src.Credit_Defaultor_Prediction.exception import CustomException
from src.Credit_Defaultor_Prediction.logger import logging
from src.Credit_Defaultor_Prediction.utils import save_object,evaluate_models
from sklearn.linear_model import LinearRegression
from sklearn import metrics


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        
        # return rmse, mae, r2

        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            
            binning_process = optbinning.BinningProcess(variable_names = train_cols,
                                            categorical_variables = categorical)
            
            model = LogisticRegression()

            scorecard = optbinning.Scorecard(
            binning_process = binning_process,
            estimator = model,
            scaling_method = 'min_max',
            scaling_method_params = {'min':300,'max':850}
            )
            # 'pdo_odds' :  scaling_method

            scorecard.fit(X_train,y_train) # model training

            scorecard_df = scorecard.table(style = "detailed")

            train_data['pred_prob'] = scorecard.predict_proba(X_train)[:,1]
            cv_data['pred_prob'] = scorecard.predict_proba(X_cv)[:,1]

            train_score = metric(train_data['target'],train_data['pred_prob'])
            cv_score = metric(cv_data['target'],cv_data['pred_prob'])

            print("Training Score = ",train_score)
            print("CV score = ", cv_score)


            fpr_t , tpr_t , thresholds = metrics.roc_curve(y_train,train_data['pred_prob'])
            # returns three values i.e false +ve rate , true +ve rate , thresholds
            opt_idx = np.argmax(tpr_t - fpr_t)
            opt_threshold_t = thresholds[opt_idx]   
            print("Threshold Value(Train): ", opt_threshold_t)

            fpr_cv , tpr_cv , thresholds = metrics.roc_curve(y_cv,cv_data['pred_prob'])
            opt_idx = np.argmax(tpr_cv - fpr_cv)
            # argmax returns the index of maximum value

            opt_threshold_cv = thresholds[opt_idx]  
            print("Threshold Value(Train): ", opt_threshold_cv)

            auc_score_t = metrics.auc(fpr_t , tpr_t)
            auc_score_cv = metrics.auc(fpr_cv , tpr_cv)
            plt.title("ROC curve")
            plt.plot(fpr_t,tpr_t,'b',
                    label = 'Binning+LR: Train AUC = {0:.4f}'.format(auc_score_t))
            plt.plot(fpr_cv,tpr_cv,'r',
                    label = 'Binning+LR : CV AUC = {0:.4f}'.format(auc_score_cv))

            plt.legend(loc = 'lower right')
            plt.plot([0,1],[0,1],"--")
            plt.xlabel('False Postive Rate ')
            plt.ylabel('True Positive Rate ')
            plt.show()

            auc_score_t,auc_score_cv

            train_data['predict'] = (train_data['pred_prob']>opt_threshold_t).astype(int)
            # astype(data_type): converts pandas object to specified data type
            cv_data['predict'] = (cv_data['pred_prob']>opt_threshold_t).astype(int)


            # Now we will make confusion Matrix
            conf_matrix = metrics.confusion_matrix(train_data['target'],train_data['predict'])
            figure,(ax_1,ax_2) = plt.subplots(1,2, figsize = (7,5))

            sns.heatmap(conf_matrix, square = True, annot = True,
                        cmap = 'Blues',fmt = 'd' ,cbar = False ,ax= ax_1)
            sns.heatmap(conf_matrix/np.sum(conf_matrix), annot = True,
                        cmap = 'Blues',fmt = '.2%' ,cbar = False ,ax= ax_2)
            plt.show()


            conf_matrix = metrics.confusion_matrix(cv_data['target'],cv_data['predict'])
            figure,(ax_1,ax_2) = plt.subplots(1,2, figsize = (7,5))

            sns.heatmap(conf_matrix, square = True, annot = True,
                        cmap = 'Blues',fmt = 'd' ,cbar = False ,ax= ax_1)
            sns.heatmap(conf_matrix/np.sum(conf_matrix), annot = True,
                        cmap = 'Blues',fmt = '.2%' ,cbar = False ,ax= ax_2)
            plt.show()

            print(metrics.classification_report(train_data['target'], train_data['predict'], labels=[0, 1]))

            print(metrics.classification_report(cv_data['target'],cv_data['predict'], labels=[0, 1]))

            train_data['score'] = scorecard.score(X_train)
            cv_data['score'] = scorecard.score(X_cv)

            y_test = cv_data['target']
            score = cv_data['score']

            mask = y_test == 0

            figure, ax = plt.subplots(figsize=(13,8))
            plt.hist(score[mask],label = "Non-Default",color = 'b',alpha = 0.35)
            plt.hist(score[~mask],label = "Default",color = 'g',alpha = 0.35)
            plt.xlabel("score")
            plt.legend()
            plt.show()

            # Distribution of scores
            plt.figure(figsize=(20,10))

            plt.hist(score,
                    bins = 90,
                    edgecolor = 'blue',
                    color = "skyblue",
                    linewidth = 1.1)

            plt.title('Scorecard Dist.',fontweight = "bold",fontsize = 10)
            plt.xlabel('Score')
            plt.ylabel('Count')


            plt.figure(figsize = (15,9))

            plt.scatter(x = score,
                        y = cv_data['pred_prob'],
                        color = '#e82e60'
            )
            plt.title('Scores by Probability',fontweight = 'bold',fontsize = 14)
            plt.xlabel('Score')
            plt.ylabel('Prob')


            test['Probability'] = scorecard.predict_proba(X_test)[:,1]
            submit_df = test[['customer_ID','Probability']].copy()
            submit_df.to_csv("Submission.csv")

            temp = submit_df[submit_df['Probability']>0.5]




























         # Hidden
        
    def metric(y_true, y_pred, return_components=False) -> float:
            """Amex metric for ndarrays"""
        def top_four_percent_captured(df) -> float:
            """Corresponds to the recall for a threshold of 4 %"""
            df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
            four_pct_cutoff = int(0.04 * df['weight'].sum())
            df['weight_cumsum'] = df['weight'].cumsum()
            df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
            return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()

        def weighted_gini(df) -> float:
            df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
            df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
            total_pos = (df['target'] * df['weight']).sum()
            df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
            df['lorentz'] = df['cum_pos_found'] / total_pos
            df['gini'] = (df['lorentz'] - df['random']) * df['weight']
            return df['gini'].sum()

        def normalized_weighted_gini(df) -> float:
            """Corresponds to 2 * AUC - 1"""
            df2 = pd.DataFrame({'target': df.target, 'prediction': df.target})
            df2.sort_values('prediction', ascending=False, inplace=True)
            return weighted_gini(df) / weighted_gini(df2)

        df = pd.DataFrame({'target': y_true.ravel(), 'prediction': y_pred.ravel()})
        df.sort_values('prediction', ascending=False, inplace=True)
        g = normalized_weighted_gini(df)
        d = top_four_percent_captured(df)

        if return_components: return g, d, 0.5 * (g + d)
        return 0.5 * (g + d)

























            
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)


            









            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)