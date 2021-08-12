
def compute_features():
    tdCustomer = DataFrame("bank_customer")
    tdAccounts = DataFrame("bank_accounts")
    tdTransactions = DataFrame("bank_transactions")

    # First, grab customer demographic variables and create binned variables and one-hot encoded variables from the customer table.

    fn = FillNa(style = "median")

    income_t = Binning(style = "bins", value = 100, columns = "income", out_columns = "income_bins", fillna = fn)
    age_t = Binning(style = "bins", value = 10, columns = "age", out_columns = "age_bins", fillna = fn)
    gender_t = OneHotEncoder(values = {"M":"male_ind", "F":"female_ind"}, columns = "gender", fillna = fn)
    marital_status_t = OneHotEncoder(values = {1:"single_ind", 2:"married_ind", 3:"separated_ind", 4:"widower_ind"}, 
                                     columns = "marital_status", fillna = fn)
    state_code_t = OneHotEncoder(values = {"CA":"ca_resident_ind", "NY":"ny_resident_ind", 
                                           "TX":"tx_resident_ind", "IL":"il_resident_ind",
                                           "AZ":"az_resident_ind", "OH":"oh_resident_ind"}, 
                                 columns = "state_code", fillna = fn)
    fillna_t1 = FillNa(style = "median", columns = "years_with_bank", out_columns = "tot_cust_years", datatype = 'integer')
    fillna_t2 = FillNa(style = "median", columns = "nbr_children", out_columns = "tot_children", datatype = 'integer')
    labelencoder_t = LabelEncoder(values={"CA": "CA", "NY": "NY", "TX": "TX", "OH": "OH", "AZ": "AZ", "IL": "IL"}, 
                                  columns="state_code", default="OTHER", datatype = 'char,6')

    cust = valib.Transform(data = tdCustomer,
                           bins = [income_t, age_t],
                           one_hot_encode = [gender_t, marital_status_t, state_code_t],
                           fillna = [fillna_t1, fillna_t2],
                           label_encode = labelencoder_t,
                           key_columns = "cust_id")
    
    # Next, create account indicators and then calculate account balances
    fn = FillNa(style = "literal", value=0)

    account_type_t = OneHotEncoder(values = {"CC":"cc_acct_ind", "CK":"ck_acct_ind", 
                                             "SV":"sv_acct_ind"}, 
                                   columns = "acct_type", fillna = fn)
    fillna_t = FillNa(style = "median", columns = ["cust_id", "starting_balance", "ending_balance"])

    acct = valib.Transform(data = tdAccounts,
                           one_hot_encode = [account_type_t],
                           fillna = fillna_t,
                           key_columns = "cust_id")

    acct_bal = acct.result.starting_balance + acct.result.ending_balance

    acct.result = acct.result.assign(cc_bal = case_when( [(acct.result.cc_acct_ind.expression == 1, acct_bal.expression)
                                                         ], else_=0 )
                            ).assign(ck_bal = case_when( [(acct.result.ck_acct_ind.expression == 1, acct_bal.expression)
                                                         ], else_=0 )
                            ).assign(sv_bal = case_when( [(acct.result.sv_acct_ind.expression == 1, acct_bal.expression)
                                                         ], else_=0 )
                            )
    
    # Next get the transaction information required for the Quarterly aggregation by pulling out the quarter the transaction was made.

    acct_mon = extract('month', tdTransactions.tran_date.expression).expression

    trans = tdTransactions.assign(q1_trans = case( [(acct_mon ==  "1", 1), (acct_mon ==  "2", 1), (acct_mon ==  "3", 1)], else_ = 0 ),
                                  q2_trans = case( [(acct_mon ==  "4", 1), (acct_mon ==  "5", 1), (acct_mon ==  "6", 1)], else_ = 0 ),
                                  q3_trans = case( [(acct_mon ==  "7", 1), (acct_mon ==  "8", 1), (acct_mon ==  "9", 1)], else_ = 0 ),
                                  q4_trans = case( [(acct_mon == "10", 1), (acct_mon == "11", 1), (acct_mon == "12", 1)], else_ = 0 ),
                                 )
    
    # Join the transformed Customer table to the transformed Account table

    cust_acct = cust.result.join(other = acct.result, how = "left", on = ["cust_id"],
                                 lsuffix = "cust", rsuffix = "acct")


    # Next Join the transformed Transaction table to the transformed Account table

    acct_tran_amt = trans.principal_amt + trans.interest_amt

    cust_acct_tran = cust_acct.join(other = trans, how = "left", on = ["acct_nbr"], 
                                    lsuffix = "cu_ac", rsuffix = "trans"
                           ).assign(cc_tran_amt = 
                                    case_when( [(cust_acct.cc_acct_ind.expression == 1, acct_tran_amt.expression)
                                               ], else_=0 )
                           ).assign(ck_tran_amt = 
                                    case_when( [(cust_acct.ck_acct_ind.expression == 1, acct_tran_amt.expression)
                                               ], else_=0 )
                           ).assign(sv_tran_amt = 
                                    case_when( [(cust_acct.sv_acct_ind.expression == 1, acct_tran_amt.expression)
                                               ], else_=0 )
                           )
    
    
    
    # Finally, aggregate and roll up by 'cust_id' all variables in the above join operation.  This pulls everything together into the 
    # analytic data set.

    ADS_Py = cust_acct_tran.groupby("cust_cust_id").agg(
                       {
                        "income_bins"     : "max",
                        "age_bins"        : "max",
                        "tot_cust_years"  : "max",
                        "tot_children"    : "max",
                        "female_ind"      : "max",
                        "single_ind"      : "max",
                        "married_ind"     : "max",
                        "separated_ind"   : "max",
                        "ca_resident_ind" : "max",
                        "ny_resident_ind" : "max",
                        "tx_resident_ind" : "max",
                        "il_resident_ind" : "max",
                        "az_resident_ind" : "max",
                        "oh_resident_ind" : "max",
                        "state_code"      : "max",
                        "ck_acct_ind"     : "max",
                        "sv_acct_ind"     : "max",
                        "cc_acct_ind"     : "max",
                        "ck_bal"          : "mean",
                        "sv_bal"          : "mean",
                        "cc_bal"          : "mean",
                        "ck_tran_amt"     : "mean",
                        "sv_tran_amt"     : "mean",
                        "cc_tran_amt"     : "mean",
                        "q1_trans"        : "sum",
                        "q2_trans"        : "sum",
                        "q3_trans"        : "sum",
                        "q4_trans"        : "sum"
                       }
             )

    # Rename Columns because of VAL bug with MEAN parsing

    columns = ['cust_id','income_bins','age_bins','tot_cust_years','tot_children','female_ind',
               'single_ind', 'married_ind', 'separated_ind', 'state_code', 'ca_resident_ind', 'ny_resident_ind',
               'tx_resident_ind','il_resident_ind','az_resident_ind', 'oh_resident_ind',
               'ck_acct_ind','sv_acct_ind','cc_acct_ind', 'ck_avg_bal','sv_avg_bal','cc_avg_bal',
               'ck_avg_tran_amt','sv_avg_tran_amt','cc_avg_tran_amt','q1_trans_cnt',
               'q2_trans_cnt','q3_trans_cnt','q4_trans_cnt']

    ADS_Py = ADS_Py.assign(drop_columns = True,
                           cust_id         = ADS_Py.cust_cust_id,
                           income_bins     = ADS_Py.max_income_bins,
                           age_bins        = ADS_Py.max_age_bins,
                           tot_cust_years  = ADS_Py.max_tot_cust_years,
                           tot_children    = ADS_Py.max_tot_children,
                           female_ind      = ADS_Py.max_female_ind,
                           single_ind      = ADS_Py.max_single_ind,
                           married_ind     = ADS_Py.max_married_ind,
                           separated_ind   = ADS_Py.max_separated_ind,
                           state_code      = ADS_Py.max_state_code,
                           ca_resident_ind = ADS_Py.max_ca_resident_ind,
                           ny_resident_ind = ADS_Py.max_ny_resident_ind,
                           tx_resident_ind = ADS_Py.max_tx_resident_ind,
                           il_resident_ind = ADS_Py.max_il_resident_ind,
                           az_resident_ind = ADS_Py.max_az_resident_ind,
                           oh_resident_ind = ADS_Py.max_oh_resident_ind,
                           ck_acct_ind     = ADS_Py.max_ck_acct_ind,
                           sv_acct_ind     = ADS_Py.max_sv_acct_ind,
                           cc_acct_ind     = ADS_Py.max_cc_acct_ind,
                           ck_avg_bal      = ADS_Py.mean_ck_bal,
                           sv_avg_bal      = ADS_Py.mean_sv_bal,
                           cc_avg_bal      = ADS_Py.mean_cc_bal,
                           ck_avg_tran_amt = ADS_Py.mean_ck_tran_amt,
                           sv_avg_tran_amt = ADS_Py.mean_sv_tran_amt,
                           cc_avg_tran_amt = ADS_Py.mean_cc_tran_amt,
                           q1_trans_cnt    = ADS_Py.sum_q1_trans,
                           q2_trans_cnt    = ADS_Py.sum_q2_trans,
                           q3_trans_cnt    = ADS_Py.sum_q3_trans,
                           q4_trans_cnt    = ADS_Py.sum_q4_trans).select(columns)

    copy_to_sql(ADS_Py, table_name="ADS_Py", if_exists="replace")
    


