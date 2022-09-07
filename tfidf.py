import pandas as pd
import numpy as np
from pathlib import Path
import pkuseg
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from config import train_path, test_path, label_path


def loadData(train_path, test_path, label_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    # df = pd.concat([df_train, df_test], ignore_index=True)
    # print(df)

    label_dict = np.load(label_path, allow_pickle=True).tolist()
    label_dict_reverse = {}
    for key, val in label_dict.items():
        label_dict_reverse[val] = int(key)
    # print(df)
    # print(label_dict_reverse)
    print("Load data successfully.")
    return df_train, df_test, label_dict_reverse


def get_text(texts):
    seg = pkuseg.pkuseg(model_name='medicine', postag=True)
    # 设置词性，保留名词、动词、状态词、形容词
    flags = ('n', 'v', 'z', 'a')
    stopwords = ("天", "周", "年", "月", "日", "半", "不",
                 "未", "无", "伴", "有", "上", "予", ",", ".")

    words_list = []
    for text in texts:
        words = seg.cut(text)
        words = [item[0] for item in seg.cut(
            text) if item[0] not in stopwords and item[1] in flags]
        words_list.append(' '.join(words))

    return words_list


def seg_word(df, label_dict_reverse):
    corpus = get_text(df.content)
    df = pd.concat([df, pd.DataFrame(corpus, columns=['segm'])], axis=1)
    df["label"] = df.label.map(label_dict_reverse)
    print("segmentation complished.")

    return df


def top_k_eval(k, X_test, y_test, clf):
    y_pred = clf.predict(X_test.values.astype('U'))

    y_score = clf.predict_proba(X_test.values.astype('U'))  # 输出概率
    sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[
        :, ::-1]  # 从高到低排序

    for i in range(26583):
        for item in sorted_pred[:, :int(k)][i]:
            if item == int(y_test[i:i+1]):
                y_pred[i] = item

    return y_pred


def report(k, y_test, y_pred):
    label_dict = np.load(label_path,
                         allow_pickle=True).tolist()
    target_names = []
    for key, val in label_dict.items():
        target_names.append(val)

    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"report_top{k}.csv")


def main():
    train_segged_path = Path("./process_data/train_segged.csv")
    test_segged_path = Path("./process_data/test_segged.csv")
    if train_segged_path.exists() and test_segged_path.exists():
        df_train = pd.read_csv(train_segged_path)
        df_test = pd.read_csv(test_segged_path)
    else:
        df_train, df_test, label_dict_reverse = loadData(
            train_path, test_path, label_path)
        df_train = seg_word(df_train, label_dict_reverse)
        df_train.to_csv(train_segged_path)
        df_test = seg_word(df_test, label_dict_reverse)
        df_test.to_csv(test_segged_path)

    X_train, X_test, y_train, y_test = df_train.segm, df_test.segm, df_train.label, df_test.label

    clf_path = Path("tfidf.joblib")
    if clf_path.exists():
        clf = joblib.load(clf_path)
    else:
        clf = Pipeline([
            ('vectorizer_tfidf', TfidfVectorizer()),
            ("LR", LogisticRegression())
        ])

        clf.fit(X_train.values.astype('U'), y_train)

        joblib.dump(clf, 'tfidf.joblib')
        print("dump pipeline successfully.")

    k = input("k:")
    y_pred = top_k_eval(k, X_test, y_test, clf)
    report(k, y_pred, y_test)


if __name__ == '__main__':
    main()
