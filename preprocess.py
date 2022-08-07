import os
import pandas as pd
import re

from hanspell import spell_checker
from konlpy.tag import Okt
from tqdm import tqdm


def preprocess(data_path, train=True):
    file_result = "test_fix.csv"
    file_process = "test.csv"
    if train:
        file_result = "train_fix.csv"
        file_process = "train.csv"

    if os.path.exists(os.path.join(data_path, file_result)):
        print("Preprocess file already exist!")
        return pd.read_csv(os.path.join(data_path, file_result))
    else:
        before = pd.read_csv(os.path.join(data_path, file_process))
        after_id, after_review, after_target = [], [], []

        # Preparing Stopword
        f = open(os.path.join(data_path, "stopword.txt"), encoding="UTF-8")
        line = f.readlines()
        stopwords = []
        for l in line:
            # 개행문자 제거
            l = l.replace("\n", "")
            stopwords.append(l)

        print("---- START PREPROCESSING ----")
        for idx in tqdm(range(len(before))):

            # Regex
            review = before["reviews"].iloc[idx]
            pattern = re.compile("[^ 가-힣0-9a-zA-Z+]")
            pattern_check = pattern.sub("", review)

            # Stopword
            okt = Okt()
            s_morphs = okt.morphs(pattern_check)
            s_checked = [w for w in s_morphs if w not in stopwords]
            s_checked = " ".join(s_checked)
            if not train and len(s_checked) == 0:
                s_checked = pattern_check

            # grammer check
            spell_checked = spell_checker.check(s_checked)

            # Append result
            after_id.append(before["id"].iloc[idx])
            after_review.append(spell_checked.checked)
            if train:
                after_target.append(before["target"].iloc[idx])

        after = pd.DataFrame()
        if train:
            after = pd.DataFrame(
                {
                    "id": after_id,
                    "reviews": after_review,
                    "target": after_target,
                }
            )
        else:
            after = pd.DataFrame(
                {
                    "id": after_id,
                    "reviews": after_review,
                }
            )

        after.to_csv(os.path.join(data_path, file_result), index=False)
        print("---- FINISH PREPROCESSING ----")
        return after
