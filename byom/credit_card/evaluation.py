from sklearn import metrics
from teradataml import create_context
from teradataml.context.context import get_connection
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats

import os
import json
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(data_conf, model_conf, **kwargs):
    model_version = kwargs["model_version"]
    model_id = kwargs["model_id"]

    engine = create_context(host=os.environ["AOA_CONN_HOST"],
                            username=os.environ["AOA_CONN_USERNAME"],
                            password=os.environ["AOA_CONN_PASSWORD"])

    cursor = engine.raw_connection().cursor()
    conn = get_connection()

    with open("artifacts/input/model.pmml", "rb") as f:
        model_bytes = f.read()

    # we don't want to insert this into the models that can be used yet so add to temporary table and use there
    cursor.execute("""
    CREATE VOLATILE TABLE ivsm_models_tmp(
        model_version VARCHAR(255),
        model_id VARCHAR(255),
        model BLOB(2097088000)
    ) ON COMMIT PRESERVE ROWS;
    """)
    cursor.execute(f"INSERT INTO ivsm_models_tmp(model_version, model_id, model) "
                   "values(?,?,?)",
                   (model_version, model_id, model_bytes))

    scores_df = pd.read_sql(f"""
    SELECT cust_id, y_test, CAST(y_pred AS INT) FROM (
        SELECT cust_id, cc_acct_ind as y_test, CAST(score_result AS JSON).JSONExtractValue('$.predicted_cc_acct_ind') as y_pred FROM IVSM.IVSM_SCORE(
                    ON (SELECT * FROM {data_conf["features"]}) AS DataTable
                    ON (SELECT model_id, model FROM ivsm_models_tmp WHERE model_version = '{model_version}') AS ModelTable DIMENSION
                    USING
                        ModelID('{model_id}')
                        ColumnsToPreserve('cust_id', 'cc_acct_ind')
                        ModelType('PMML')
                ) sc
        ) T WHERE T.y_pred=0 OR T.y_pred=1;
    """, conn)

    y_pred = scores_df[["y_pred"]]
    y_test = scores_df[["y_test"]]

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # create confusion matrix plot
    cf = metrics.confusion_matrix(y_test, y_pred)

    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['0','1'])
    plt.yticks([0, 1], ['0','1'])

    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')

    fig = plt.gcf()
    fig.savefig('artifacts/output/confusion_matrix', dpi=500)
    plt.clf()

    predictions_table = "bank_predictions_tmp"
    predictions_df = scores_df[["cust_id", "y_pred"]].rename({'y_pred': 'cc_acct_ind'}, axis=1)
    copy_to_sql(df=predictions_df, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    # the number of rows output from VAL is different to the number of input rows.. nulls?
    # temporary workaround - join back to features and filter features without predictions
    ads = DataFrame.from_query(f"SELECT F.* FROM {data_conf['features']} F JOIN bank_predictions_tmp P ON F.cust_id = P.cust_id")

    stats.record_evaluation_stats(ads, DataFrame(predictions_table))
