import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv
import os
import numpy as np
import glob
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt


def load_model(filename: str = None, load_latest: bool = None):
    assert (filename is not None) ^ (load_latest is not None)

    model_folder = "model"

    if filename is not None:
        model = joblib.load(os.path.join(model_folder, filename))
    elif load_latest is not None and load_latest:
        all_model_filenames = glob.glob(os.path.join(model_folder, "*.joblib"))
        all_model_filenames = sorted(
            all_model_filenames,
            key=lambda x: [int(i) for i in Path(x).stem.split("-")],
            reverse=True,
        )
        latest_model_filename = all_model_filenames[0]
        model = joblib.load(latest_model_filename)
    else:
        raise NotImplementedError

    return model


def load_data(folders):
    update_every = 100
    accumulated_dict = None
    for folder_ind, folder in enumerate(folders):
        with open(os.path.join(folder, "accumulated_data.csv"), "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for line_ind, line in enumerate(csv_reader):
                if line_ind == 0:
                    if accumulated_dict is None:
                        accumulated_dict = {k: [] for k in line}
                    else:
                        assert len(line) == len(accumulated_dict.keys())
                        # reorder_keys = None if all(k1 == k2 for k1, k2 in zip(line, accumulated_dict.keys())) else {k2: k1_ind for (k1_ind, _), k2 in zip(enumerate(line), accumulated_dict.keys())}
                        continue
                else:
                    assert len(accumulated_dict) == len(line)
                    for k, item in zip(accumulated_dict.keys(), line):
                        accumulated_dict[k].append(
                            float(item) if k == "toi_skew_differential" else int(item)
                        )
        if (folder_ind + 1) % update_every == 0:
            print(f"{datetime.now()} Accumulated {folder_ind + 1} games.")

    return accumulated_dict


def validate(v_folders):
    # load and transform data appropriately
    accumulated_dict = load_data(v_folders)

    # load model
    clf = load_model(load_latest=True)

    # predict
    y_true = accumulated_dict["winner"]
    X = np.vstack(
        [
            v
            for k, v in accumulated_dict.items()
            if k
            not in [
                "winner",
                "gameId",
                "playId",
                "time_remaining",
            ]
        ],
    ).T
    y_predict = clf.predict(X)

    # report scores
    print(
        classification_report(y_true, y_predict, target_names=["Away win", "Home win"])
    )

    # todo validate like they do in the paper, by binning the win percentages of test set and then seeing within each
    #  bin how many of them actually correspond with wins, do correlation
    y_pred_prob = clf.predict_proba(X)
    bins = list(range(0, 105, 5))
    bin_true = []
    bin_pred_prob = []
    for e1, e2 in zip(bins[:-1], bins[1:]):
        bin_true.append(
            [
                y_true[i]
                for i in np.logical_and(
                    (e1 / 100) < y_pred_prob[:, 1], y_pred_prob[:, 1] <= (e2 / 100)
                ).nonzero()[0]
            ]
        )
        bin_pred_prob.append(
            [
                y_pred_prob[:, 1][i]
                for i in np.logical_and(
                    (e1 / 100) < y_pred_prob[:, 1], y_pred_prob[:, 1] <= (e2 / 100)
                ).nonzero()[0]
            ]
        )
    # find percentage in bin_true of wins and average percentage in bin_pred_prob then find correlation
    avg_true = [sum([x == 1 for x in ell]) / len(ell) for ell in bin_true]
    avg_pred_prob = [sum(ell) / len(ell) for ell in bin_pred_prob]

    corr = np.corrcoef(avg_true, avg_pred_prob)
    print(f"The correlation is {corr[0, 1]:.3f}")
    theta = np.polyfit(avg_true, avg_pred_prob, 1)

    plt.scatter(avg_true, avg_pred_prob, c="b")
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c="r")
    plt.show()

    from sklearn.inspection import permutation_importance

    result = permutation_importance(clf, X, accumulated_dict["winner"], n_repeats=10)

    pass


def train(t_folders):

    # load and transform data appropriately
    accumulated_dict = load_data(t_folders)

    # train model
    train_start = time.time()
    clf = RandomForestClassifier(random_state=None)
    y = accumulated_dict["winner"]
    X = np.vstack(
        [
            v
            for k, v in accumulated_dict.items()
            # todo investigate further the difference in training between using time_remaining and time_remaining_neg,
            #  I see some plots do not end at 100% or 0% and I believe because that is because there is data where
            #  time_remaining=0 (overtime) and the game is still undecided. Would be better if there was a better way to
            #  represent overtime and still know that it is sudden death.
            if k
            not in [
                "winner",
                "gameId",
                "playId",
                "time_remaining",
            ]
        ],
    ).T
    assert clf is not None
    clf.fit(X, y)
    train_end = time.time()
    print(f"Training took {train_end - train_start:.2f} seconds.")

    # save model
    if not os.path.isdir("model"):
        os.makedirs("model")

    joblib.dump(
        clf,
        os.path.join(
            "model",
            f"{datetime.now().year}-{datetime.now().month}-{datetime.now().day}-{datetime.now().hour}-{datetime.now().minute}.joblib",
        ),
    )


def train_val_split(train_ratio, season_year):
    season_year_folder = os.path.join("data", season_year)
    game_folders = glob.glob(os.path.join(season_year_folder, "*"))

    # only keep the regular season games
    game_folders = list(
        filter(
            lambda fol: len(glob.glob(os.path.join(fol, "*.txt")))
            and Path(glob.glob(os.path.join(fol, "*.txt"))[0]).stem[0] == "R",
            game_folders,
        )
    )

    np.random.seed(645)
    perm = np.random.permutation(len(game_folders))
    train_folders = [game_folders[i] for i in perm[: int(len(perm) * train_ratio)]]
    val_folders = [game_folders[i] for i in perm[int(len(perm) * train_ratio) :]]

    return train_folders, val_folders


def main():
    season_year = "20202021"
    train_ratio = 0.8

    # determine how training
    train_folders, val_folders = train_val_split(train_ratio, season_year)

    # train model
    # train(train_folders)

    # validate model
    validate(val_folders)


if __name__ == "__main__":
    main()
