import pandas as pd
import numpy as np
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
    df = pd.concat([df_train, df_test], ignore_index=True)
    # print(df)

    label_dict = np.load(label_path, allow_pickle=True).tolist()
    label_dict_reverse = {}
    for key, val in label_dict.items():
        label_dict_reverse[val] = int(key)
    # print(df)
    # print(label_dict_reverse)
    print("Load data successfully.")
    return df, label_dict_reverse


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
    df, label_dict_reverse = loadData(train_path, test_path, label_path)
    df = seg_word(df, label_dict_reverse)

    X_train, X_test, y_train, y_test = train_test_split(
        df.segm,
        df.label,
        test_size=0.2,
        stratify=df.label,
        random_state=1000
    )

    clf = Pipeline([
        ('vectorizer_tfidf', TfidfVectorizer()),
        ("LR", LogisticRegression())
    ])

    clf.fit(X_train.values.astype('U'), y_train)

    joblib.dump(clf, 'tfidf.joblib')
    print("dump pipeline successfully.")

    # clf = joblib.load("./tfidf.joblib")
    k = input("k:")
    y_pred = top_k_eval(k, X_test, y_test, clf)
    report(k, y_pred, y_test)


if __name__ == '__main__':
    main()
