import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from tfidf import top_k_eval
from config import label_path

k = 1
test_segged_path = Path("./process_data/test_segged.csv")
clf_path = Path("tfidf.joblib")


clf = joblib.load(clf_path)
df_test = pd.read_csv(test_segged_path)
X_test_orign, X_test, y_test = df_test.content, df_test.segm,  df_test.label
y_pred = top_k_eval(k, X_test, y_test, clf)

label_dict_modified = {}
label_dict = np.load(label_path,
                     allow_pickle=True).tolist()
for key, val in label_dict.items():
    label_dict_modified[int(key)] = val
# label_dict_modified


df = pd.DataFrame(data=[y_test, y_pred]).T
df.set_axis(["test_label", "pred_label"], axis='columns', inplace=True)
df["test"] = df.test_label.map(label_dict_modified)
df["pred"] = df.pred_label.map(label_dict_modified)

df = pd.concat([df, X_test_orign, X_test, ], axis=1)

error_sample = df[["test_label", "test", "pred", "content",
                   "segm"]][df["test_label"] != df["pred_label"]]

error_sample[["pred", "content", "segm"]][error_sample["test"] == "急性上呼吸道感染"]
