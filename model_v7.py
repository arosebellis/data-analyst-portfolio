import argparse
import math
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    fbeta_score, precision_recall_curve, roc_auc_score, precision_score, recall_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

### DATA LOADING AND PREPROCESSING
# Reproducibility function
def set_seed(seed):
    """Sets a variety of random number generator methods using a set
    default value of 42.
    Inputs:
    seed: int = 42 
    Output:
    None."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Labelled columns
id_col = "TransactionID"
label_col = "Class"
time_col = "Time"
amount_col = "Amount"

# UID candidate columns
card_col = "card1"
addr_col = "addr1"
d1_col = "D1"

account_col = "account_uid"

# Codes and encoders
UNKNOWN_CODE = 0.0 # Code to fill any unknown variables with.
FREQ_ENCODE_CANDIDATES = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceInfo", "id_30_device", "id_31_browser", "DeviceInfo_device"
] # Columns used to create uid codes.

def load_ieee_data(transaction_path, identity_path):
    """
    Loads IEEE-CIS training transaction data table and left-joins the
    training identity table onto it.
    Input: 
    transaction_path: str,
    identity_path: str = None
    Output:
    df: pd.DataFrame
    """
    df = pd.read_csv(transaction_path)

    if identity_path is not None and os.path.exists(identity_path):
        identity_df = pd.read_csv(identity_path)
        df = df.merge(identity_df, on=id_col, how="left")

    rename_map = {"TransactionDT": time_col, "TransactionAmt": amount_col}

    # if loop included to safeguard against loading of the test dataset
    # the test dataset is not currently used in this analysis.
    if "isFraud" in df.columns:
        rename_map["isFraud"] = label_col

    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected IEEE-CIS columns: {missing}")

    df = df.rename(columns=rename_map)
    return df

def sort_by_time(df):
    """
    Sorts the input dataframe by time, according to the given column.
    Input:
    df: pd.Dataframe
    Output:
    df: pd.DataFrame
    """
    return df.sort_values(by=time_col).reset_index(drop=True)

def time_based_split(df, train_frac, val_frac):
    """
    Splits a time-sorted dataframe into train/val/test sets based on time.
    Input:
    df: pd.DataFrame
    train_frac: float = 0.6,
    val_frac: float = 0.3
    Output:
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df

def split_val_for_calibration(val_df, clf_fit_frac):
    """
    Splits the validation dataframe in two: one part to fit the classifier
    and one part for threshold calibration.
    Input:
    val_df: pd.DataFrame,
    clf_fit_frac: float = 0.5
    Output:
    clf_fit_df: pd.DataFrame,
    val_calib_df: pd.DataFrame
    """
    n = len(val_df)
    split_point = int(n * clf_fit_frac)
    clf_fit_df = val_df.iloc[:split_point].reset_index(drop=True)
    val_calib_df = val_df.iloc[split_point:].reset_index(drop=True)

    return clf_fit_df, val_calib_df

def engineer_time_features(df):
    """
    Creates hour and day features from the sequential time feature.
    Input:
    df: pd.DataFrame
    Output:
    df: pd.DataFrame
    """
    df = df.copy()
    df["Transaction_hour"] = (df[time_col] // 3600) % 24
    df["Transaction_day"] = (df[time_col] // (24*3600)) % 7
    return df

def add_account_uid(df):
    """Adds a unique id for each individual account, based on days since
    opening, address of the account, and card type.
    Input:
    df: pd.DataFrame
    Output:
    df: pd.DataFrame
    """
    df = df.copy()
    absolute_day = df[time_col] // (24 * 3600)
    d1n = df[d1_col] - absolute_day
    df["D1n"] = d1n

    has_addr = df[addr_col].notna()
    has_d1 = d1n.notna()
    known_account = has_addr & has_d1

    card = df[card_col].fillna(-1).astype("int64").astype(str)
    addr = df[addr_col].fillna(-1).astype("int64").astype(str)
    d1n_str = d1n.fillna(-1).astype("int64").astype(str)

    grouped_uid = card + "_" + addr + "_" + d1n_str
    fallback_uid = "unk_" + df[id_col].astype(str)
    df[account_col] = np.where(known_account, grouped_uid, fallback_uid)
    return df

def add_features(df):
    """Adds features to the dataset based on a breakdown of original
    features (i.e. breakdowns of two part features.)
    Input:
    df: pd.DataFrame
    Output:
    df: pd.DataFrame
    """
    df = df.copy()
    ## Adds a feature for the decimal part of the transaction amount, scaled
    ## to an integer
    amt = df[amount_col].fillna(0).clip(lower=0)
    df["TransactionAmt_decimal"] = ((amt - amt.astype(int)) * 1000).round().astype(np.int32)

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        p = df["P_emaildomain"].astype(str)
        r = df["R_emaildomain"].astype(str)
        df["email_domain_match"] = (p == r).astype(np.float32)
        df["P_emaildomain_suffix"] = p.str.split(".").str[-1]
        df["R_emaildomain_suffix"] = r.str.split(".").str[-1]

    if "id_30" in df.columns:
        df["id_30_device"] = df["id_30"].astype(str).str.split(" ").str[0]
    if "id_31" in df.columns:
        df["id_31_browser"] = df["id_31"].astype(str).str.split(" ").str[0]
    if "DeviceInfo" in df.columns:
        df["DeviceInfo_device"] = df["DeviceInfo"].astype(str).str.split("/").str[0]

    return df

def fit_v_column_reduction(train_df, corr_threshold):
    """Reduces the number of V columns based on matching numbers of missing
    values, high correlation, and zero variance.
    Input:
    train_df: pd.DataFrame
    corr_threshold: float = 0.9
    Output:
    info: dict
    keep_cols: list
    """

    v_cols = [c for c in train_df.columns if re.fullmatch(r"V\d+", c)]
    if not v_cols:
        return v_cols, {"nan_count_groups": 0, "dropped_zero_variance": [], "dropped_correlated": []}

    nan_counts = train_df[v_cols].isna().sum()
    nan_groups = {}
    for c in v_cols:
        nan_groups.setdefault(int(nan_counts[c]), []).append(c)

    keep_cols, dropped_zero_variance, dropped_correlated = [], [], []
    for _, cols in nan_groups.items():
        variances = train_df[cols].var(skipna=True)
        nonzero_cols = [c for c in cols if pd.notna(variances[c]) and variances[c] > 0]
        dropped_zero_variance += [c for c in cols if c not in nonzero_cols]
        if not nonzero_cols:
            continue
        if len(nonzero_cols) == 1:
            keep_cols += nonzero_cols
            continue

        corr = train_df[nonzero_cols].corr().abs()
        visited = set()
        for c in nonzero_cols:
            if c in visited:
                continue
            cluster = [c] + [
                other for other in nonzero_cols
                if other not in visited and other != c and corr.loc[c, other] > corr_threshold
            ]
            visited.update(cluster)
            keep_cols.append(cluster[0])
            dropped_correlated += cluster[1:]

    print(f"V-column reduction: {len(v_cols)} raw V columns -> {len(keep_cols)} kept "
          f"({len(dropped_zero_variance)} zero-variance dropped, {len(dropped_correlated)} dropped"
          f" as >{corr_threshold:.0%} correlated with a kept column, across {len(nan_groups)} NaN-count groups).")

    info = {
        "corr_threshold": corr_threshold,
        "n_nan_count_groups": len(nan_groups),
        "dropped_zero_variance": dropped_zero_variance,
        "dropped_correlated": dropped_correlated,
    }
    return keep_cols, info

def apply_v_column_reduction(df, v_keep_cols, all_v_cols):
    """Applies the v-column reduction constructed above to the dataframe.
    Input:
    df: pd.DataFrame,
    v_keep_cols: list = keep_cols,
    all_v_cols: list
    Output:
    df: pd.DataFrame
    """
    drop_cols = [c for c in all_v_cols if c not in v_keep_cols and c in df.columns]
    return df.drop(columns=drop_cols) if drop_cols else df

def fit_missing_indicator_cols(train_df, protected_cols, threshold):
    """Fits a missing indicator to all columns with missing data amounting to more
    than the set threshold.
    Input:
    train_df: pd.DataFrame,
    protected_cols: list,
    threshold: float = 0.3
    Output:
    cols: list
    """
    missing_frac = train_df.isna().mean()
    cols = [c for c in train_df.columns if c not in protected_cols and missing_frac[c] > threshold]
    if cols:
        print(f"Adding missingness indicators for {len(cols)} / {train_df.shape[1]} columns with >{threshold:.0%} missing in train.")

    return cols

def add_missing_indicators(df, cols):
    """
    Applies the missing indicator to columns as per fit_missing_indicator_cols.
    Input:
    df: pd.DataFrame,
    cols: list
    Output:
    df: pd.DataFrame
    """
    df = df.copy()
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        return df.copy()
    indicators = pd.DataFrame(
        {f"{c}_isna": df[c].isna().astype(np.float32) for c in present_cols},
        index=df.index
    )
    return pd.concat([df, indicators], axis=1)

def fit_frequency_encoders(train_df, cols):
    """
    Fits frequency encoders to selected columns.
    Input:
    train_df: pd.DataFrame,
    cols: list
    Output:
    encoders: dict
    """
    encoders = {}
    for c in cols:
        encoders[c] = train_df[c].value_counts(dropna=False).to_dict()
    return encoders

def apply_frequency_encoders(df, encoders):
    """
    Applies fitted frequency encoders to the dataframe.
    Input:
    df: pd.DataFrame,
    encoders: dict
    Output:
    df: pd.DataFrame
    """
    present = {c: mapping for c, mapping in encoders.items() if c in df.columns}
    if not present:
        return df.copy()
    freq_cols = pd.DataFrame(
        {f"{c}_freq": df[c].map(mapping).fillna(0).astype(np.float32) for c, mapping in present.items()},
        index=df.index,
    )
    return pd.concat([df, freq_cols], axis=1)

def add_causal_uid_aggregrates(df, group_col, agg_cols, time_col):
    """
    Adds aggregate columns for each UID.
    Input:
    df: pd.DataFrame,
    group_col: str,
    agg_cols: list,
    time_col: str
    Output:
    df: pd.DataFrame,
    new_cols: list
    """
    grp_time = df.groupby(group_col, sort=False)[time_col]
    prior_count = grp_time.cumcount()
    safe_count = prior_count.where(prior_count > 0, np.nan)

    new_data = {
        "uid_txn_count_prior": prior_count.astype(np.float32),
        "uid_is_new": (prior_count == 0).astype(np.float32),
        "uid_time_since_prev": (df[time_col] - grp_time.shift(1)).astype(np.float32),
        "uid_time_since_first": (df[time_col] - grp_time.transform("first")).astype(np.float32),
    }

    new_cols = list(new_data.keys())

    for col in agg_cols:
        values = df[col].astype(np.float64)
        sq = values ** 2

        cum_sum = values.groupby(df[group_col], sort=False).cumsum() - values
        cum_sq = sq.groupby(df[group_col], sort=False).cumsum() - sq

        mean_prior = cum_sum / safe_count
        var_prior = (cum_sq / safe_count) - mean_prior ** 2
        std_prior = np.sqrt(var_prior.clip(lower=0.0))

        mean_col, std_col = f"{col}_uid_mean_prior", f"{col}_uid_std_prior"
        new_data[mean_col] = mean_prior.astype(np.float32)
        new_data[std_col] = std_prior.astype(np.float32)
        new_cols += [mean_col, std_col]

    df = pd.concat([df, pd.DataFrame(new_data, index=df.index)], axis=1)
    return df, new_cols

class UidAggregateStore:
    def __init__(self, agg_cols, fallback_values: dict):
        self.agg_cols = list(agg_cols)
        self.fallback_values = fallback_values
        self.counts, self.first_time, self.last_time = {}, {}, {}
        self.sums = {col: {} for col in self.agg_cols}
        self.sumsqs = {col: {} for col in self.agg_cols}
 
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, uid_col: str, time_col: str, agg_cols, fallback_values: dict):
        store = cls(agg_cols, fallback_values)
        grouped = df.groupby(uid_col, sort=False)
        store.counts = grouped.size().to_dict()
        store.first_time = grouped[time_col].min().to_dict()
        store.last_time = grouped[time_col].max().to_dict()
        for col in agg_cols:
            store.sums[col] = grouped[col].sum().to_dict()
            store.sumsqs[col] = grouped[col].apply(lambda s: float((s.astype(np.float64) ** 2).sum())).to_dict()
        return store
 
    def featurize_and_update(self, uid, time_value: float, values: dict) -> dict:
        prior_count = self.counts.get(uid, 0)
        feat = {
            "uid_txn_count_prior": float(prior_count),
            "uid_is_new": 1.0 if prior_count == 0 else 0.0,
        }
 
        if prior_count == 0:
            feat["uid_time_since_prev"] = self.fallback_values.get("uid_time_since_prev", 0.0)
            feat["uid_time_since_first"] = 0.0
        else:
            feat["uid_time_since_prev"] = float(time_value - self.last_time[uid])
            feat["uid_time_since_first"] = float(time_value - self.first_time[uid])
 
        for col in self.agg_cols:
            mean_col, std_col = f"{col}_uid_mean_prior", f"{col}_uid_std_prior"
            if prior_count == 0:
                feat[mean_col] = self.fallback_values.get(mean_col, 0.0)
                feat[std_col] = self.fallback_values.get(std_col, 0.0)
            else:
                s, ssq = self.sums[col][uid], self.sumsqs[col][uid]
                mean_prior = s / prior_count
                var_prior = max(ssq / prior_count - mean_prior ** 2, 0.0)
                feat[mean_col] = mean_prior
                feat[std_col] = math.sqrt(var_prior)
 
        self.counts[uid] = prior_count + 1
        if prior_count == 0:
            self.first_time[uid] = time_value
        self.last_time[uid] = time_value
        for col in self.agg_cols:
            v = float(values[col])
            self.sums[col][uid] = self.sums[col].get(uid, 0.0) + v
            self.sumsqs[col][uid] = self.sumsqs[col].get(uid, 0.0) + v ** 2
 
        return feat

def identity_column_types(df, exclude_cols):
    """
    Separates remaining columns into categorical vs. numeric.
    Input:
    df: pd.DataFrame,
    exclude_cols: list
    Output:
    categorical_cols: list
    numeric_cols: list
    """
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    categorical_cols = [c for c in candidate_cols if df[c].dtype == object]
    numeric_cols = [c for c in candidate_cols if c not in categorical_cols]
    return categorical_cols, numeric_cols

def fit_categorical_encoders(train_df, categorical_cols):
    """
    Fits a category -> code lookup using the training set.
    Input: 
    train_df: pd.DataFrame,
    categorical_cols: list
    Output:
    encoders: dict
    """
    encoders = {}
    for col in categorical_cols:
        values = train_df[col].fillna("missing").astype(str)
        uniques = sorted(values.unique())
        encoders[col] = {val: float(idx + 1) for idx, val in enumerate(uniques)}
    return encoders

def apply_categorical_encoders(df, encoders):
    """
    Applies the fitted categorical encoders to the dataframe.
    Input:
    df: pd.DataFrame,
    encoders: dict
    Output:
    df: pd.DataFrame
    """
    df = df.copy()
    for col, mapping in encoders.items():
        values = df[col].fillna("missing").astype(str)
        df[col] = values.map(mapping).fillna(UNKNOWN_CODE).astype(np.float32)
    return df

def fit_numeric_imputers(train_df, numeric_cols):
    """
    Fits median numeric imputation to all numeric columns.
    Input: 
    train_df: pd.DataFrame,
    numeric_cols: list
    Output:
    medians: dict
    """
    return {col: float(train_df[col].median()) for col in numeric_cols}

def apply_numeric_imputation(df, medians):
    """
    Applies fitted numeric imputation to the dataframe.
    Input:
    df: pd.DataFrame,
    medians: dict
    Output:
    df: pd.DataFrame
    """
    df = df.copy()
    for col, median in medians.items():
        df[col] = df[col].fillna(median)
    return df

def build_feature_pipeline(train_df, val_df, test_df, missing_indicator_threshold = 0.3, v_corr_threshold = 0.9):
    protected = {id_col, label_col, time_col, card_col, addr_col, d1_col}
    all_v_cols = [c for c in train_df.columns if re.fullmatch(r"V\d+", c)]
    v_keep_cols, v_reduction_info = fit_v_column_reduction(train_df, v_corr_threshold)
    train_df = apply_v_column_reduction(train_df, v_keep_cols, all_v_cols)
    val_df = apply_v_column_reduction(val_df, v_keep_cols, all_v_cols)
    test_df = apply_v_column_reduction(test_df, v_keep_cols, all_v_cols)

    train_df = engineer_time_features(train_df)
    val_df = engineer_time_features(val_df)
    test_df = engineer_time_features(test_df)
    train_df = add_account_uid(train_df)
    val_df = add_account_uid(val_df)
    test_df = add_account_uid(test_df)
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    missing_indicator_cols = fit_missing_indicator_cols(train_df, protected, missing_indicator_threshold)
    train_df = add_missing_indicators(train_df, missing_indicator_cols)
    val_df = add_missing_indicators(val_df, missing_indicator_cols)
    test_df = add_missing_indicators(test_df, missing_indicator_cols)

    freq_cols = [c for c in FREQ_ENCODE_CANDIDATES if c in train_df.columns]
    freq_encoders = fit_frequency_encoders(train_df, freq_cols)
    train_df = apply_frequency_encoders(train_df, freq_encoders)
    val_df = apply_frequency_encoders(val_df, freq_encoders)
    test_df = apply_frequency_encoders(test_df, freq_encoders)

    exclude = {id_col, label_col, card_col, account_col, "D1n", addr_col, d1_col}
    categorical_cols, numeric_cols = identity_column_types(train_df, exclude)
    numeric_cols = [c for c in numeric_cols if c not in (time_col, amount_col)]

    cat_encoders = fit_categorical_encoders(train_df, categorical_cols)
    train_df = apply_categorical_encoders(train_df, cat_encoders)
    val_df = apply_categorical_encoders(val_df, cat_encoders)
    test_df = apply_categorical_encoders(test_df, cat_encoders)

    num_medians = fit_numeric_imputers(train_df, numeric_cols)
    train_df = apply_numeric_imputation(train_df, num_medians)
    val_df = apply_numeric_imputation(val_df, num_medians)
    test_df = apply_numeric_imputation(test_df, num_medians)

    for d in (train_df, val_df, test_df):
        d[amount_col] = np.log1p(d[amount_col].fillna(0).clip(lower=0))

    split_sizes = [len(train_df), len(val_df), len(test_df)]
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    uid_agg_source_cols = numeric_cols + [amount_col]
    full_df, uid_agg_cols = add_causal_uid_aggregrates(full_df, amount_col, uid_agg_source_cols, time_col)

    train_df = full_df.iloc[:split_sizes[0]].reset_index(drop=True)
    val_df = full_df.iloc[split_sizes[0]:split_sizes[0]+split_sizes[1]].reset_index(drop=True)
    test_df = full_df.iloc[split_sizes[0] + split_sizes[1]:].reset_index(drop=True)

    uid_agg_fill_cols = [c for c in uid_agg_cols if c not in ("uid_txn_count_prior", "uid_is_new")]
    uid_agg_medians = fit_numeric_imputers(train_df, uid_agg_fill_cols)
    train_df = apply_numeric_imputation(train_df, uid_agg_medians)
    val_df = apply_numeric_imputation(val_df, uid_agg_medians)
    test_df = apply_numeric_imputation(test_df, uid_agg_medians)

    feature_cols = categorical_cols + numeric_cols + [amount_col] + uid_agg_cols

    preprocessors = {
        "v_keep_cols": v_keep_cols,
        "v_reduction_info": v_reduction_info,
        "missing_indicator_cols": missing_indicator_cols,
        "freq_encoders": freq_encoders,
        "cat_encoders": cat_encoders,
        "num_medians": num_medians,
        "uid_agg_medians": uid_agg_medians,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    return train_df, val_df, test_df, feature_cols, uid_agg_source_cols, preprocessors


#-------------------#
## MODELLING FUNCTIONS

class SlidingWindowFraudDataset(Dataset):
    """A PyTorch Dataset that creates sliding windows of features."""
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, feature_cols, seq_len: int = 16, stride: int = 1, fit_scaler: bool = False, clip_value = 10.0):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        raw = df[feature_cols].values.astype(np.float32)
 
        if fit_scaler:
            scaler.fit(raw)
        self.scaler = scaler
        scaled = scaler.transform(raw).astype(np.float32)
 
        self.scaled = np.clip(scaled, -clip_value, clip_value)
        labels = df[label_col].values.astype(np.float32)
 
        accounts = df[account_col].fillna(-1).values
        self.n_features = self.scaled.shape[1]
 
        order = np.argsort(accounts, kind="stable")
        sorted_accounts = accounts[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_accounts[1:] != sorted_accounts[:-1], True])
        self.order = order
 
        window_abs_start, window_abs_end, window_labels, window_accounts = [], [], [], []
        for i in range(len(boundaries) - 1):
            group_start, group_end = boundaries[i], boundaries[i+1]
            idx = order[group_start:group_end]
            acc_labels = labels[idx]
            n = len(idx)
            acc_id = sorted_accounts[group_start]
 
            for end in range(0, n, stride):
                start = end - seq_len + 1
                real_start = max(0, start)
 
                window_abs_start.append(group_start + real_start)
                window_abs_end.append(group_start + end)
                window_labels.append(1.0 if acc_labels[end].max() > 0 else 0.0)
                window_accounts.append(acc_id)
 
        self.window_abs_start = np.array(window_abs_start, dtype=np.int64)
        self.window_abs_end = np.array(window_abs_end, dtype=np.int64)
        self.y = np.array(window_labels, dtype=np.float32)
        self.window_accounts = np.array(window_accounts)
 
    def get_account_ids(self):
        return self.window_accounts
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, idx):
        abs_start = self.window_abs_start[idx]
        abs_end = self.window_abs_end[idx]
 
        row_positions = self.order[abs_start:abs_end + 1]
        real_part = self.scaled[row_positions]
 
        real_len = real_part.shape[0]
        pad_len = self.seq_len - real_len
 
        mask_col = np.ones((real_len, 1), dtype=np.float32)
        if pad_len > 0:
            pad_block = np.zeros((pad_len, self.n_features), dtype=np.float32)
            pad_mask = np.zeros((pad_len, 1), dtype=np.float32)
            real_part = np.concatenate([pad_block, real_part], axis=0)
            mask_col = np.concatenate([pad_mask, mask_col], axis=0)
 
        window = np.concatenate([real_part, mask_col], axis=1)
 
        x = torch.from_numpy(window).transpose(0, 1)
        y = torch.tensor(self.y[idx])
        return x, y


def compute_balanced_sample_weights(labels):
    """
    Computes sample weights.
    Input:
    labels: np.ndarray
    Output:
    sample_weights: np.ndarray
    """
    labels = np.asarray(labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    weight_pos = len(labels) / (2 * max(n_pos, 1))
    weight_neg = len(labels) / (2 * max(n_neg, 1))
    return np.where(labels == 1, weight_pos, weight_neg).astype(np.float32)

def make_weighted_sampler(labels):
    """
    Returns a weighted random sampler based on computed sample weights.
    Input:
    labels: np.ndarray
    Output:
    sampler: WeightedRandomSampler
    """
    sample_weights = compute_balanced_sample_weights(labels)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

# TCN Block and model definitions
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.GroupNorm(1, n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.GroupNorm(1, n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.norm1, self.relu1, self.dropout1, 
                                 self.conv2, self.chomp2, self.norm2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNStack(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        prev_ch = in_ch
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(prev_ch, ch, kernel_size, dilation, dropout))
            prev_ch = ch

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TCNAutoencoderClassifier(nn.Module):
    """A hybrid model combining a TCN autoencoder"""
    def __init__(self, n_features, seq_len, enc_channels=(32, 32, 64), latent_dim=32, kernel_size=3, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
 
        # Encoder
        self.encoder = TCNStack(n_features + 1, list(enc_channels), kernel_size, dropout)
        self.to_latent = nn.Linear(enc_channels[-1], latent_dim)
 
        # Decoder
        dec_channels = list(reversed(enc_channels))
        self.from_latent = nn.Linear(latent_dim, dec_channels[0])
        self.decoder = TCNStack(dec_channels[0], dec_channels[1:] + [dec_channels[-1]], kernel_size, dropout)
        self.reconstructor = nn.Conv1d(dec_channels[-1], n_features, kernel_size=1)
 
        # Padding
        self.pad_embedding = nn.Parameter(torch.randn(n_features) * 0.02)
 
        self.n_extra_stats = 3
        clf_in_dim = latent_dim + n_features + self.n_extra_stats
        self.register_buffer("feature_mask", torch.ones(clf_in_dim, dtype=torch.bool))
 
    def _fill_padding(self, x):
        real = x[:, :self.n_features, :]
        mask = x[:, self.n_features:self.n_features + 1, :]
        pad_emb = self.pad_embedding.view(1, -1, 1)
        real_filled = real * mask + pad_emb * (1 - mask)
        return torch.cat([real_filled, mask], dim=1)
 
    def encode(self, x):
        z = self.encoder(x)
        z = z[:, :, -1]
        z = self.to_latent(z)
        return z
 
    def decode(self, z):
        z = self.from_latent(z)
        z = z.unsqueeze(-1).repeat(1, 1, self.seq_len)
        x_hat = self.decoder(z)
        x_hat = self.reconstructor(x_hat)
        return x_hat
 
    @staticmethod
    def last_step_error(x, x_hat):
        return torch.abs(x[:, :, -1] - x_hat[:, :, -1])
 
    @staticmethod
    def window_error_stats(x, x_hat, mask=None):
        err_full = torch.abs(x - x_hat)
        err_per_step = err_full.mean(dim=1)
 
        if mask is None:
            mean_err = err_per_step.mean(dim=1)
            max_err = err_per_step.max(dim=1).values
            std_err = err_per_step.std(dim=1)
        else:
            m = mask.squeeze(1)
            valid_counts = m.sum(dim=1).clamp(min=1.0)
            mean_err = (err_per_step * m).sum(dim=1) / valid_counts
 
            masked_for_max = err_per_step.masked_fill(m == 0, float("-inf"))
            max_err = masked_for_max.max(dim=1).values
            max_err = torch.where(torch.isfinite(max_err), max_err, mean_err)
 
            var = ((err_per_step - mean_err.unsqueeze(1)) ** 2 * m).sum(dim=1) / valid_counts
            std_err = torch.sqrt(var.clamp(min=1e-8))
        return torch.stack([mean_err, max_err, std_err], dim=1)
 
    def classifier_input_features(self, x, z=None, x_hat=None):
        real = x[:, :self.n_features, :]
        mask = x[:, self.n_features:self.n_features + 1, :]
 
        if z is None:
            z = self.encode(x)
        if x_hat is None:
            x_hat = self.decode(z)
 
        recon_error = self.last_step_error(real, x_hat)
        window_stats = self.window_error_stats(real, x_hat, mask=mask)
        return torch.cat([z, recon_error, window_stats], dim=1)
 
    def set_feature_mask(self, mask: torch.Tensor):
        mask = mask.to(dtype=torch.bool, device=self.feature_mask.device)
        self.feature_mask = mask
        print(f"Feature mask set: {int(mask.sum().item())} / {len(mask)} features selected for the XGBoost head.")
 
    def forward(self, x, compute_logit=True):
        z = self.encode(x)
        x_hat = self.decode(z)
        real = x[:, :self.n_features, :]
        recon_error = self.last_step_error(real, x_hat)
 
        return x_hat, z, recon_error

# Loss functions
def focal_loss_obj(alpha: float = 0.55, gamma: float = 2.0, fn_cost = 1.2):
    def objective(preds, dtrain):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        pt = np.where(y==1, p, 1-p)
        alpha_t = np.where(y == 1, alpha, 1 - alpha)
        focal_weight = alpha_t * (1 - pt) ** gamma
        focal_weight = np.where(y == 1, focal_weight * fn_cost, focal_weight)

        instance_weight = dtrain.get_weight()
        if instance_weight is not None and len(instance_weight) == len(y):
            focal_weight = focal_weight * instance_weight

        grad = focal_weight * (p - y)
        hess = focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return objective

def fbeta_feval(beta: float = 2.0, threshold: float = 0.5):
    def feval(preds, dtrain):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        y_pred = (p >= threshold).astype(int)
        return f"f{beta:g}", fbeta_score(y, y_pred, beta=beta, zero_division=0)
    return feval
 
def xgb_predict_proba(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    dmat = xgb.DMatrix(X)
    best_it = getattr(booster, "best_iteration", None)
    if best_it is not None:
        margin = booster.predict(dmat, iteration_range=(0, best_it + 1))
    else:
        margin = booster.predict(dmat)
 
    return 1.0 / (1.0 + np.exp(-margin))

## TRAINING AND EVALUATION

def train_ae_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
 
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, z, _ = model(x)
 
        real = x[:, :model.n_features, :]
        mask = x[:, model.n_features:model.n_features + 1, :]
        sq_err = (x_hat - real) ** 2 * mask
        n_valid = mask.sum() * real.size(1)
        loss = sq_err.sum() / n_valid.clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / n

def apply_fraud_cascade(account_ids, y_pred):
    account_ids = np.asarray(account_ids)
    y_pred = np.asarray(y_pred).astype(int)
 
    prior_flags = (
        pd.Series(y_pred)
        .groupby(pd.Series(account_ids), sort=False)
        .cumsum()
        - y_pred
    ).to_numpy()
 
    is_cascaded = prior_flags > 0
    cascaded_pred = np.where(is_cascaded, 1, y_pred).astype(int)
    return cascaded_pred, is_cascaded

 
def _best_fbeta_threshold(y_true, scores, beta: float = 2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]

    fbeta_curve = (1 + beta**2) * precision * recall / (beta**2 * precision * recall + 1e-8)
    if len(thresholds) == 0:
        return float(scores.mean())

    best_idx = int(np.nanargmax(fbeta_curve))
    return float(thresholds[best_idx])

 
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = model.encode(x)
        x_hat = model.decode(z)
        feat = model.classifier_input_features(x, z=z, x_hat=x_hat)
        feats.append(feat.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)
 
def evaluate(model, xgb_model, loader, device, threshold=None, calibrator=None, account_ids=None, threshold_beta=2.0):
    X, y_true = extract_features(model, loader, device)
    mask = model.feature_mask.cpu().numpy()
    X_sel = X[:, mask]
    scores = xgb_predict_proba(xgb_model, X_sel)
 
    if calibrator is not None:
        scores = calibrator.predict(scores)
 
    if account_ids is not None:
        account_ids = np.asarray(account_ids)
        if len(account_ids) != len(y_true):
            raise ValueError("Length of account_ids must match number of samples.")
 
        prelim_threshold = threshold if threshold is not None else _best_fbeta_threshold(y_true, scores, beta=threshold_beta)
        prelim_pred = (scores >= prelim_threshold).astype(int)
        _, is_cascaded = apply_fraud_cascade(account_ids, prelim_pred)
    else:
        is_cascaded = np.zeros(len(y_true), dtype=bool)
 
    clean = ~is_cascaded
    clean_true, clean_scores = y_true[clean], scores[clean]
 
    metrics = {
        "roc_auc": roc_auc_score(clean_true, clean_scores),
        "pr_auc": average_precision_score(clean_true, clean_scores)
    }
 
    if threshold is None:
        threshold = _best_fbeta_threshold(y_true, scores, beta=threshold_beta)
 
    base_pred = (scores >= threshold).astype(int)
 
    if account_ids is not None:
        y_pred, is_cascaded = apply_fraud_cascade(account_ids, base_pred)
        clean = ~is_cascaded
    else:
        y_pred = base_pred
 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
 
    metrics["calibrated_threshold"] = float(threshold)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["f2"] = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["false_negatives"] = int(fn)
    metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, digits=4, zero_division=0)
    if account_ids is not None and clean.sum() > 0:
        metrics["clean_f1"] = f1_score(y_true[clean], y_pred[clean])
 
    return metrics, y_true, scores, is_cascaded
 
### RFECV for Feature Selection
def select_features_rfecv(model, loader, device, step=0.1, cv=3, max_samples=20000, random_state=42):
    X, y = extract_features(model, loader, device)
 
    if len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        fraud_idx = np.flatnonzero(y == 1)
        clean_idx = np.flatnonzero(y == 0)
        n_clean = max(max_samples - len(fraud_idx), 0)
        clean_sample = rng.choice(clean_idx, size=min(n_clean, len(clean_idx)), replace=False)
        keep = np.concatenate([fraud_idx, clean_sample])
        X, y = X[keep], y[keep]
 
    if y.sum() < 2:
        raise ValueError("Need at least 2 fraud examples to run RFE")
 
    base_estimator = RandomForestClassifier(
        n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=1
    )
 
    rfecv = RFECV(estimator=base_estimator, step=step, cv=cv, scoring="average_precision", n_jobs=-1)
    rfecv.fit(X, y)
 
    support_mask = torch.tensor(rfecv.support_, dtype=torch.bool, device=device)
    print(f"RFECV selected {support_mask.sum().item()} features out of {len(support_mask)}")
    return support_mask
 
def train_xgb_classifier(model, train_loader, val_loader, device, params=None, num_boost_round=500,
                         early_stopping_rounds=30, focal_alph=0.55, focal_gamma=2.0, fn_cost=1.2, eval_beta=1.5, use_scale_pos_weight=False, seed=42, verbose_eval=25):
    mask = model.feature_mask.cpu().numpy()
 
    X_train, y_train = extract_features(model, train_loader, device)
    X_train = X_train[:, mask]
    sample_weight = compute_balanced_sample_weights(y_train)
 
    X_val, y_val = extract_features(model, val_loader, device)
    X_val = X_val[:, mask]
 
    dtrain = xgb.DMatrix(X_train, label=y_train, weight = sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)
 
    xgb_params = {
        "max_depth" : 6,
        "eta" : 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "seed": seed,
    }
    if use_scale_pos_weight:
        n_pos = max(int(y_train.sum()), 1)
        n_neg = len(y_train) - n_pos
        xgb_params["scale_pos_weight"] = n_neg / n_pos
    if params:
        xgb_params.update(params)
 
    booster = xgb.train(xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        obj= focal_loss_obj(alpha=focal_alph, gamma=focal_gamma, fn_cost=fn_cost),
        evals = [(dtrain, "train"), (dval, "val")],
        custom_metric = fbeta_feval(beta=eval_beta, threshold=0.5),
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,
        verbose_eval=verbose_eval)
 
    print(f"XGBoost head trained: best iteration {booster.best_iteration}"
          f"(best val aucpr {booster.best_score:.4f})")
    return booster
 
 
def bootstrap_stable_threshold(y_true, scores, n_boot=50, random_state=42, threshold_beta=2.0):
    rng=np.random.default_rng(random_state)
    n = len(y_true)
    thresholds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, sc = y_true[idx], scores[idx]
        if yt.sum() < 2 or yt.sum() == len(yt):
            continue
        thresholds.append(_best_fbeta_threshold(yt, sc, beta=threshold_beta))
    if not thresholds:
        return None
    return float(np.median(thresholds))
 
def bootstrap_f1_ci(y_true, y_pred, n_boot=1000, random_state=42, ci=95):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if yt.sum() == 0:
            continue
        boot_f1s.append(f1_score(yt, yp, zero_division=0))
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    return (float(np.mean(boot_f1s)), float(np.percentile(boot_f1s, lower_pct)),
            float(np.percentile(boot_f1s, upper_pct)))
 
 
 
def main():
 
    parser = argparse.ArgumentParser(description="Train a TCN Autoencoder + XGBoost Classifier for IEEE-CIS Fraud Detection")
    parser.add_argument("--transaction_data", type=str, help="Path to train_transaction.csv")
    parser.add_argument("--identity_data", type=str, help="Path to train_identity.csv (optional -- left-joined on TransactionID if provided)")
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.3)
    parser.add_argument("--missing_indicator_threshold", type=float, default=0.3)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs_ae", type=int, default=12)
    parser.add_argument("--lr_ae", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--rfecv_folds", type=int, default=5)
    parser.add_argument("--xgb_rounds", type=int, default=500)
    parser.add_argument("--xgb_early_stopping", type=int, default=30)
    parser.add_argument("--focal_alpha", type=float, default=0.55,
                         help="Weight on the fraud class in the focal loss term, on top of "
                              "balanced instance weights; keep modest (0.5-0.6) to avoid FP flood.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--fn_cost", type=float, default=1.2,
                         help="Extra multiplier on gradients/hessians for missed fraud (false negatives).")
    parser.add_argument("--eval_beta", type=float, default=1.5,
                         help="F-beta used for xgb early stopping; >1 weights recall over precision.")
    parser.add_argument("--use_scale_pos_weight", action="store_true")
    parser.add_argument("--threshold_beta", type=float, default=2.0,
                         help="Target ceiling on the fraction of fraud missed (false negatives). "
                              "Decision threshold is chosen to hit recall >= 1 - max_fn_rate "
                              "while maximizing precision, so this is a direct recall/FP knob: "
                              "raise it if there are too many false positives, lower it to catch "
                              "more fraud at the cost of more false alarms.")
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args(["--transaction_data", "train_transaction.csv", "--identity_data", "train_identity.csv"])
 
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    df = load_ieee_data(args.transaction_data, args.identity_data)
    df = sort_by_time(df)
    n_total = len(df)
    n_fraud_total = len(df[df["Class"] == 1])
    rate_total = (n_fraud_total / n_total) * 100
    print(f"n_total = {n_total}, n_fraud_total = {n_fraud_total}, rate_total = {rate_total}")

    train_df_raw, val_df, test_df = time_based_split(df, train_frac=args.train_frac, val_frac=args.val_frac)
 
    train_df_raw, val_df, test_df, feature_cols, uid_agg_source_cols, preprocessors = build_feature_pipeline(
        train_df_raw, val_df, test_df, missing_indicator_threshold=args.missing_indicator_threshold
    )
    uid_agg_medians = preprocessors["uid_agg_medians"]
    print(f"Using {len(feature_cols)} feature columns after feature engineering")
 
    train_df = train_df_raw[train_df_raw[label_col] == 0].reset_index(drop = True)
 
    clf_fit_df, val_calib_df = split_val_for_calibration(val_df, clf_fit_frac=0.5)
    clf_train_df = pd.concat([train_df_raw, clf_fit_df], axis=0, ignore_index=True)

    n_train = len(train_df)
    n_fraud_train = len(train_df[train_df["Class"] == 1])
    rate_train = (n_fraud_train / n_train) * 100
    print(f"n_total = {n_train}, n_fraud_total = {n_fraud_train}, rate_total = {rate_train}")

    n_val = len(val_df)
    n_fraud_val = len(val_df[val_df["Class"] == 1])
    rate_val = (n_fraud_val / n_val) * 100
    print(f"n_total = {n_val}, n_fraud_total = {n_fraud_val}, rate_total = {rate_val}")

    n_clf_fit = len(clf_fit_df)
    n_fraud_clf_fit = len(clf_fit_df[clf_fit_df["Class"] == 1])
    rate_clf_fit = (n_fraud_clf_fit / n_clf_fit) * 100
    print(f"n_total = {n_clf_fit}, n_fraud_total = {n_fraud_clf_fit}, rate_total = {rate_clf_fit}")

    n_val_calib = len(val_calib_df)
    n_fraud_val_calib = len(val_calib_df[val_calib_df["Class"] == 1])
    rate_val_calib = (n_fraud_val_calib / n_val_calib) * 100
    print(f"n_total = {n_val_calib}, n_fraud_total = {n_fraud_val_calib}, rate_total = {rate_val_calib}")

    n_test = len(test_df)
    n_fraud_test = len(test_df[test_df["Class"] == 1])
    rate_test = (n_fraud_test / n_test * 100)
    print(f"n_total = {n_test}, n_fraud_total = {n_fraud_test}, rate_total = {rate_test}")
 
    scaler = StandardScaler()
    train_ds = SlidingWindowFraudDataset(train_df, scaler, feature_cols, seq_len=args.seq_len, fit_scaler=True)
    clf_fit_ds = SlidingWindowFraudDataset(clf_train_df, scaler, feature_cols, seq_len=args.seq_len, fit_scaler=False)
    val_calib_ds = SlidingWindowFraudDataset(val_calib_df, scaler, feature_cols, seq_len=args.seq_len, fit_scaler=False)
    test_ds = SlidingWindowFraudDataset(test_df, scaler, feature_cols, seq_len=args.seq_len, fit_scaler=False)
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    clf_fit_loader = DataLoader(clf_fit_ds, batch_size=args.batch_size, shuffle=False)
    val_calib_loader = DataLoader(val_calib_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
 
    val_calib_account_ids = val_calib_ds.get_account_ids()
    test_account_ids = test_ds.get_account_ids()
 
    n_features = len(feature_cols)
    model = TCNAutoencoderClassifier(n_features=n_features, seq_len=args.seq_len, latent_dim=args.latent_dim).to(device)
    ae_params = list(model.encoder.parameters()) + list(model.to_latent.parameters()) + list(model.from_latent.parameters()) + list(model.decoder.parameters()) + list(model.reconstructor.parameters()) + [model.pad_embedding]
    ae_optimizer = torch.optim.Adam(ae_params, lr=args.lr_ae, weight_decay=1e-5)
    ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ae_optimizer, mode="min", factor=0.5, patience=2)
 
    ae_best_path = os.path.join(args.out_dir, "best_ae.pth")
    best_recon_loss = math.inf
 
    ae_already_trained = os.path.exists(ae_best_path)
 
    if ae_already_trained:
        print(f"Found existing autoencoder checkpoint at {ae_best_path} -- loading it and skipping AE training. "
              f"(pass --force_retrain_ae to retrain from scratch instead)")
        try:
            model.load_state_dict(torch.load(ae_best_path, map_location=device))
            print("Model loaded.")
        except RuntimeError as e:
            # Most likely cause: the feature pipeline changed since this checkpoint was saved
            # (different n_features), so the saved weights no longer match the model shape.
            print(f"Could not load checkpoint ({e}); it looks stale for the current feature set -- retraining instead.")
            ae_already_trained = False
 
    if not ae_already_trained:
        ae_params = list(model.encoder.parameters()) + list(model.to_latent.parameters()) + list(model.from_latent.parameters()) + list(model.decoder.parameters()) + list(model.reconstructor.parameters()) + [model.pad_embedding]
        ae_optimizer = torch.optim.Adam(ae_params, lr=args.lr_ae, weight_decay=1e-5)
        ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ae_optimizer, mode='min', factor=0.5, patience=2)
 
        best_recon_loss = math.inf
        for epoch in range(1, args.epochs_ae + 1):
            recon_loss = train_ae_epoch(model, train_loader, ae_optimizer, device)
            ae_scheduler.step(recon_loss)
            print(f"Epoch {epoch}/{args.epochs_ae} - AE Recon Loss: {recon_loss:.6f}")
 
            if recon_loss < best_recon_loss:
                best_recon_loss = recon_loss
                torch.save(model.state_dict(), ae_best_path)
 
        model.load_state_dict(torch.load(ae_best_path, map_location=device))
 
    feature_mask = select_features_rfecv(model, clf_fit_loader, device, cv=args.rfecv_folds)
    model.set_feature_mask(feature_mask)
 
 
    xgb_model = train_xgb_classifier(
        model, clf_fit_loader, val_calib_loader, device,
        num_boost_round=args.xgb_rounds, early_stopping_rounds=args.xgb_early_stopping,
        focal_alph=args.focal_alpha, focal_gamma=args.focal_gamma, fn_cost=args.fn_cost, eval_beta=args.eval_beta, use_scale_pos_weight=args.use_scale_pos_weight, seed=args.seed,
    )
 
    _, val_labels_raw, val_scores_raw, val_is_cascaded_05 = evaluate(
        model, xgb_model, val_calib_loader, device, threshold=0.5, account_ids=val_calib_account_ids
    )
    clean_05 = ~val_is_cascaded_05
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_scores_raw[clean_05], val_labels_raw[clean_05])
 
    val_metrics, val_true, val_scores, val_is_cascaded = evaluate(
        model, xgb_model, val_calib_loader, device, calibrator=calibrator, account_ids=val_calib_account_ids, threshold_beta=2.0,
    )
    final_threshold = val_metrics["calibrated_threshold"]
    print(f"cascade-overriden val rows excluded from calibration: {int(val_is_cascaded.sum())}/{len(val_is_cascaded)}")
    print(f"val ROC-AUC {val_metrics['roc_auc']:.4f} | val PR-AUC {val_metrics['pr_auc']:.4f} | "
          f"val F1 {val_metrics['f1']:.4f} | val F2 {val_metrics['f2']:.4f} | "
          f"val recall {val_metrics['recall']:.4f} | val precision {val_metrics['precision']:.4f} | "
          f"val FN {val_metrics['false_negatives']}")
 
    val_clean = ~val_is_cascaded
    stable_threshold = bootstrap_stable_threshold(val_true[val_clean], val_scores[val_clean], n_boot=args.n_boot, threshold_beta=2.0)
    if stable_threshold is not None:
        print(f"\nSingle-split threshold: {final_threshold:.4f} | "
              f"Bootstrap-stabilised threshold: {stable_threshold:.4f}")
        final_threshold = stable_threshold
 
    # --- Held-out generalization test: the third time-ordered split of the training file ---
    test_metrics, y_true, y_scores, test_is_cascaded = evaluate(
        model, xgb_model, test_loader, device, threshold=final_threshold, calibrator=calibrator, account_ids=test_account_ids, threshold_beta=2.0
    )
    print(f"\nROC-AUC:              {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:               {test_metrics['pr_auc']:.4f}")
    print(f"Calibrated threshold: {test_metrics['calibrated_threshold']:.4f}")
    print(f"F1 score:             {test_metrics['f1']:.4f}")
    print(f"F2 score:             {test_metrics['f2']:.4f}")
    print(f"Recall (catch rate):  {test_metrics['recall']:.4f}")
    print(f"Precision:            {test_metrics['precision']:.4f}")
    print(f"False negatives:      {test_metrics['false_negatives']} "
          f"({test_metrics['false_negative_rate']:.2%} of all fraud missed)")
    print("Confusion matrix (rows=true, cols=pred) [ [TN FP] [FN TP] ]:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("\nClassification report:")
    print(test_metrics["classification_report"])
 
    base_test_pred = (y_scores >= final_threshold).astype(int)
    y_pred, _ = apply_fraud_cascade(test_account_ids, base_test_pred)
    mean_f1, ci_low, ci_high = bootstrap_f1_ci(y_true, y_pred, n_boot=args.n_boot)
    print(f"Bootstrap test F1: mean={mean_f1:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}] "
          f"(n_boot={args.n_boot}, {int(y_true.sum())} positives in test set)")
 
    full_seen_df = pd.concat([train_df_raw, val_df, test_df], axis=0, ignore_index=True)
    uid_store_final = UidAggregateStore.from_dataframe(
        full_seen_df, account_col, time_col, uid_agg_source_cols, uid_agg_medians
    )
 
    artifacts_path = os.path.join(args.out_dir, "inference_artifacts.pkl")
    xgb_model_path = os.path.join(args.out_dir, "xgb_model.json")
    xgb_model.save_model(xgb_model_path)
    with open(artifacts_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "feature_cols": feature_cols,
            "uid_agg_source_cols": uid_agg_source_cols,
            "preprocessors": preprocessors,
            "uid_store": uid_store_final,
            "calibrator": calibrator,
            "final_threshold": final_threshold,
            "seq_len": args.seq_len,
            "feature_mask": model.feature_mask.cpu().numpy(),
            "xgb_model_path": xgb_model_path,
        }, f)
    print(f"Saved inference artifacts (incl. warm UidAggregateStore for new-user scoring) to {artifacts_path}")
 
    return {
        "model": model, "xgb_model": xgb_model, "test_metrics": test_metrics,
        "val_calib_loader": val_calib_loader, "test_loader": test_loader,
        "device": device, "args": args, "uid_store": uid_store_final,
        "uid_agg_source_cols": uid_agg_source_cols, "uid_agg_medians": uid_agg_medians,
    }
 
if __name__ == "__main__":
    main()





    
