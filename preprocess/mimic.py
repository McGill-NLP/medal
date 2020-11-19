import argparse

import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

tqdm.pandas()

parser = argparse.ArgumentParser()
parse.add_argument("--mimic_dir", help="The directory contaning all the required MIMIC files (ADMISSIONS, PATIENTS, DIAGNOSES_ICD, PROCEDURES_ICD, NOTEEVENTS).")
parse.add_argument("--save_dir", help="The directory where you want to save the processed files.")
args = parser.parse_args()


# ## Load data and clean admission

raw_adm = pd.read_csv(args.mimic_dir + "ADMISSIONS.csv.gz")
raw_patients = pd.read_csv(args.mimic_dir + 'PATIENTS.csv.gz')
dicd = pd.read_csv(args.mimic_dir + 'DIAGNOSES_ICD.csv.gz')
picd = pd.read_csv(args.mimic_dir + 'PROCEDURES_ICD.csv.gz')
raw_notes = pd.read_csv(args.mimic_dir + "NOTEEVENTS.csv")

# First only keep the admission IDs that are in the notes as well
adm = raw_adm.copy()
adm_in_notes = notes.HADM_ID.unique()
adm = adm[adm.HADM_ID.isin(adm_in_notes)]


# ## Process notes dataframe
# 
# - Add information about death and discharge time
# - Ensures to only keep notes that were written at least 24h before discharge
# - Sample negative notes by randomly selecting at most 4 notes per patient.
# - Recombine everything to get a balanced dataset

notes = raw_notes.copy()
categories_keep = ['Nursing', 'Physician ', 'Nursing/other']
notes = notes[notes.CATEGORY.isin(categories_keep)]

notes = notes.merge(
    adm[['HADM_ID', 'DISCHTIME', 'HOSPITAL_EXPIRE_FLAG']],
    on='HADM_ID', how='left'
)

# Time manipulation
notes.DISCHTIME = pd.to_datetime(notes.DISCHTIME)
notes.CHARTTIME = pd.to_datetime(notes.CHARTTIME)
notes.CHARTDATE = pd.to_datetime(notes.CHARTDATE) + pd.DateOffset(hours=23)

notes.CHARTTIME = notes.CHARTTIME.fillna(notes.CHARTDATE)

notes = notes[notes.CHARTTIME < notes.DISCHTIME - pd.DateOffset(hours=24)]

keep_cols = ['HADM_ID', 'SUBJECT_ID', 'TEXT', 'HOSPITAL_EXPIRE_FLAG']

pos_notes = notes.loc[notes.HOSPITAL_EXPIRE_FLAG == 1, keep_cols]

neg_notes = (
    notes
    .loc[:, keep_cols]
    .query("HOSPITAL_EXPIRE_FLAG == 0")
    .groupby("HADM_ID")
    .progress_apply(lambda df: df.sample(n=4) if df.shape[0] >= 4 else df)
    .reset_index(drop=True)
)

sampled_notes = pd.concat([pos_notes, neg_notes]).drop_duplicates()
sampled_notes.HOSPITAL_EXPIRE_FLAG.value_counts()


# ## Process text content and split data


def isolate(text, chars):
    for c in chars:
        text = text.replace(c, f" {c} ")
    return text

def replace(text, chars, new=""):
    for c in chars:
        text = text.replace(c, new)
    return text

def clean_text(text):
    text = replace(text, "[**")
    text = replace(text, "**]")
    text = isolate(text, "~!@#$%^&*()_+-={}:\";',./<>?\\|`'")
    text = text.lower()
    
    return text

sampled_notes.TEXT = sampled_notes.TEXT.progress_apply(clean_text)

subjects = sampled_notes[['SUBJECT_ID', "HOSPITAL_EXPIRE_FLAG"]].drop_duplicates()

train_subj, rest_subj = train_test_split(
    subjects, 
    test_size=0.25, 
    random_state=0,
    stratify=subjects.HOSPITAL_EXPIRE_FLAG
)

valid_subj, test_subj = train_test_split(
    rest_subj.SUBJECT_ID.values,
    test_size=0.6,
    random_state=1,
    stratify=rest_subj.HOSPITAL_EXPIRE_FLAG
)

train_subj = train_subj.SUBJECT_ID.values

train_notes = sampled_notes[sampled_notes.SUBJECT_ID.isin(train_subj)].reset_index(drop=True)
valid_notes = sampled_notes[sampled_notes.SUBJECT_ID.isin(valid_subj)].reset_index(drop=True)
test_notes = sampled_notes[sampled_notes.SUBJECT_ID.isin(test_subj)].reset_index(drop=True)


# ## Save



os.makedirs(args.save_dir, exist_ok=True)
train_notes.to_csv(os.path.join(args.save_dir, "train.csv"), index=False)
valid_notes.to_csv(os.path.join(args.save_dir, "valid.csv"), index=False)
test_notes.to_csv(os.path.join(args.save_dir, "test.csv"), index=False)

