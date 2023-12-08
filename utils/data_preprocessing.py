import pandas as pd
from pathlib import Path


def get_data(data_path):
    base_path = Path(data_path)
    clients_df = pd.read_csv(base_path / 'D_clients.csv')
    close_loan_df = pd.read_csv(base_path / 'D_close_loan.csv')
    job_df = pd.read_csv(base_path / 'D_job.csv') #
    last_credit_df = pd.read_csv(base_path / 'D_last_credit.csv')
    loan_df = pd.read_csv(base_path / 'D_loan.csv')
    pens_df = pd.read_csv(base_path / 'D_pens.csv')
    salary_df = pd.read_csv(base_path / 'D_salary.csv')
    target_df = pd.read_csv(base_path / 'D_target.csv')
    work_df = pd.read_csv(base_path / 'D_work.csv') #
    total_df = clients_df.merge(job_df, how='left', left_on='ID', right_on='ID_CLIENT').drop(['ID_CLIENT'], axis=1)
    total_df = total_df.merge(work_df, how='left', left_on='SOCSTATUS_WORK_FL', right_on='FLAG')\
        .drop(['ID_y', 'FLAG'], axis=1).rename({'ID_x': 'ID_CLIENT', 'COMMENT': 'WORK_COMMENT'}, axis=1)
    total_df = total_df.merge(pens_df, how='left', left_on='SOCSTATUS_PENS_FL', right_on='FLAG').\
        drop(['ID', 'FLAG'], axis=1).rename({'COMMENT': 'PENS_COMMENT'}, axis=1)
    total_df = total_df.merge(salary_df, how='left', on='ID_CLIENT')
    total_df = total_df.merge(last_credit_df, how='left', on='ID_CLIENT')
    total_df = total_df.merge(target_df, how='left', on='ID_CLIENT')
    total_df.drop_duplicates(inplace=True)

    loans = loan_df.merge(close_loan_df, how='inner', on='ID_LOAN')
    loans = loans.groupby('ID_CLIENT').aggregate({'ID_LOAN': 'count', 'CLOSED_FL': 'sum'}).rename({'ID_LOAN': 'LOANS_OPEN', 'CLOSED_FL': 'LOANS_CLOSED'}, axis=1)
    total_df = total_df.merge(loans, how='left', on='ID_CLIENT')
    return total_df


def remove_outliers(df, column, threshold=1.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    df_outliers_removed = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_outliers_removed


