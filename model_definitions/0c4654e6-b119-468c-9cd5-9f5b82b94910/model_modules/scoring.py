from teradataml import create_context, DataFrame, copy_to_sql, remove_context, INTEGER
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from teradataml.analytics.valib import *
from teradataml import configure

import os

configure.val_install_location = os.environ.get("AOA_VAL_DB", "VAL")


def score(data_conf, model_conf, **kwargs):
    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    features = DataFrame("bank_features")

    # this should be available via kwargs. 
    # Being tracked in https://github.com/ThinkBigAnalytics/AoaPythonClient/issues/153
    model_table = "AOA_MODELS_{}".format(kwargs.get("model_version").split("-", 1)[0])
    model = DataFrame(model_table)
    
    score = valib.LogRegPredict(data=features, 
                                model=model, 
                                index_columns="cust_id",
                                estimate_column="cc_acct_ind",
                                prob_column="Probability")
    
    df = score.result
    df = df.assign(cc_acct_ind=df.cc_acct_ind.cast(type_=INTEGER))
    
    df.to_sql(table_name="bank_predictions", if_exists='replace')
   
    print("Finished Scoring")
    
    print("Calculating dataset statistics")
    
    # the number of rows output from VAL is different to the number of input rows.. nulls?
    # temporary fix - join back to features and filter features without predictions
    predictions = DataFrame("bank_predictions")
    features = DataFrame.from_query("SELECT F.* FROM bank_features F JOIN bank_predictions P ON F.cust_id = P.cust_id")

    stats.record_scoring_stats(features, DataFrame("bank_predictions"))
    
    print("Finished calculating dataset statistics")
    
    remove_context()
    