import os
import pandas as pd
import re

from hanspell import spell_checker
from konlpy.tag import Okt
from tqdm import tqdm


def preprocess(data_path, file_name):
    file_name = file_name.replace(".csv", "")

    if os.path.exists(os.path.join(data_path, file_name + "_fix.csv")):
        print("Preprocess file already exist!")
        return pd.read_csv(os.path.join(data_path, file_name + "_fix.csv"))

    else:
        before = pd.read_csv(os.path.join(data_path, file_name + ".csv"))
        after_id, after_review, after_target = [], [], []

        f = open("./dataset/stopword.txt", encoding="UTF-8")
        line = f.readlines()
        stopwords = []
        for l in line:
            l = l.replace("\n", "")
            stopwords.append(l)

        print("---- START PREPROCESSING ----")
        for idx in tqdm(range(len(before))):

            # Regex
            review = before["reviews"].iloc[idx]
            pattern = re.compile("[^ 가-힣0-9a-zA-Z+]")
            pattern_check = pattern.sub("", review)

            # Stopwords
            okt = Okt()
            s_morphs = okt.morphs(pattern_check)
            s_checked = [w for w in s_morphs if w not in stopwords]
            s_checked = " ".join(s_checked)
            if "train" in file_name and len(s_checked) == 0:
                continue

            # grammer check
            spell_checked = spell_checker.check(s_checked)

            after_id.append(before["id"].iloc[idx])
            after_review.append(spell_checked.checked)
            if "train" in file_name:
                after_target.append(before["target"].iloc[idx])

        after = pd.DataFrame()
        if "train" in file_name:
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
        after.to_csv(os.path.join(data_path, file_name + "_fix.csv"), index=False)
        print("---- FINISH PREPROCESSING ----")
        return after
