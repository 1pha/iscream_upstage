import json
import numpy as np
import pandas as pd 
from datetime import datetime

class Preprocess:

    def __init__(self, path='iscream_public_edu_V3.txt'):

        self.path = path


    def load_iscream(self, fpath):

        with open(fpath, 'r') as f:
            lines = f.readlines()
            
        lines = [json.loads(line.strip().replace("'","\"")) for line in lines]

        df = pd.DataFrame(columns=lines[0].keys())
        for user in lines:
            df = df.append(pd.Series(user), ignore_index=True)
            
        return df
    

    def preprocess(self):

        df = self.load_iscream(self.path)
        # df['Timestamp'] = df['Timestamp'].apply(lambda x: list(datetime.fromtimestamp(_) for _ in x))
        df.apply(sortby_timestamp, axis=1)
        # df['studentAnswerRate'] = df['answerCode'].apply(answer_rate)
        df['Elapsed'] = df['Timestamp'].apply(calculate_elapsed)
        df['testConsecutive'] = df['testId'].apply(test_consecutive)
        df.drop('Timestamp', axis=1, inplace=True)

        return df



def convert_timestamp(df):

    return df['Timestamp'].apply(lambda x: list(datetime.fromtimestamp(_) for _ in x))


def sortby_timestamp(line):
        
    mask = np.array(line['Timestamp']).argsort()
    
    line['assessmentItemID'] = [line['assessmentItemID'][i] for i in mask]
    line['testId'] = [line['testId'][i] for i in mask]
    line['answerCode'] = [line['answerCode'][i] for i in mask]
    line['Timestamp'] = [line['Timestamp'][i] for i in mask]
        
    return line


def answer_rate(lst):

    answers = np.array(lst)
    return len(answers[answers == 1]) / len(answers)


def timedelta2float(td, tshold=86400):
    
    res = td.microseconds/float(1000000) + (td.seconds + td.days * 24 * 3600)
    if tshold:
        if res > tshold: res = 0
    return res


def calculate_elapsed_timedelta(lst):
    
    return [0.] + [timedelta2float(lst[i+1] - lst[i]) for i, _ in enumerate(lst[:-1])]

def calculate_elapsed(lst, tshold=86400):

    return [0.] + [(lst[i+1] - lst[i])*(lst[i+1] - lst[i] < tshold) for i, _ in enumerate(lst[:-1])]


def test_consecutive(lst):

    return [1] + [1 if lst[i] == lst[i-1] else 0 for i in range(1, len(lst))]