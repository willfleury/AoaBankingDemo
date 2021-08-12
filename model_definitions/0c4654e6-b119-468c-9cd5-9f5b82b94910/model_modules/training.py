from teradataml import create_context, DataFrame, copy_to_sql, remove_context
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from teradataml.analytics.valib import *
from teradataml import configure

import os

configure.val_install_location = os.environ.get("AOA_VAL_DB", "VAL")


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    features_table = data_conf["features"]
    
    ads = DataFrame(features_table)

    print("Starting training...")
    

    feature_names = ['income_bins', 'age_bins', 'tot_cust_years', 
                'tot_children', 'female_ind', 'single_ind', 'married_ind','separated_ind',
                'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind', 'il_resident_ind', 
                'az_resident_ind', 'oh_resident_ind','sv_acct_ind',
                'ck_avg_bal','sv_avg_bal','ck_avg_tran_amt','cc_avg_tran_amt',
                'q1_trans_cnt','q2_trans_cnt','q3_trans_cnt','q4_trans_cnt']
    target_name = "cc_acct_ind"

    model = valib.LogReg(data=ads, 
                           columns=feature_names, 
                           response_column=target_name, 
                           response_value=1,
                           threshold_output='true',
                           near_dep_report='true', 
                           cond_ind_threshold=int(hyperparams["cond_ind_threshold"]),
                           variance_prop_threshold=float(hyperparams["variance_prop_threshold"]))

   
    model.model.to_sql(table_name=kwargs.get("model_table"), if_exists="replace")
    model.statistical_measures.to_sql(table_name = kwargs.get("model_table") + "_rpt", if_exists = 'replace')
    
    print("Finished training")

    print("Calculating dataset statistics")
    
    stats.record_training_stats(ads,
                       features=feature_names,
                       predictors=[target_name],
                        # bug in VAL frequency won't allow us to specify more categorical columns
                        # tracked in https://github.com/ThinkBigAnalytics/AoaPythonClient/issues/155
                       categorical=[target_name, 'female_ind', 'single_ind', 'married_ind',
                                    'separated_ind','ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind'],
                       category_labels={target_name: {0: "false", 1: "true"}})
    
    print("Finished calculating dataset statistics")
    
    remove_context()
    