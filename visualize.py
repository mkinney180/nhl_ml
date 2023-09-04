import os
import joblib
import glob
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# todo visualize single game prediction based on a trained model


def load_data(filename):
    accumulated_dict = None
    with open(filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line_ind, line in enumerate(csv_reader):
            if line_ind == 0:
                if accumulated_dict is None:
                    accumulated_dict = {k: [] for k in line}
                else:
                    continue
            else:
                assert len(accumulated_dict) == len(line)
                for k, item in zip(accumulated_dict.keys(), line):
                    accumulated_dict[k].append(
                        float(item) if k == "toi_skew_differential" else int(item)
                    )

    return accumulated_dict


def load_model(filename: str = None, load_latest: bool = None):
    assert (filename is not None) ^ (load_latest is not None)

    model_folder = "model"

    if filename is not None:
        model = joblib.load(os.path.join(model_folder, filename))
    elif load_latest is not None and load_latest:
        all_model_filenames = glob.glob(os.path.join(model_folder, "*.joblib"))
        all_model_filenames = sorted(
            all_model_filenames, key=lambda x: [int(i) for i in Path(x).stem.split("-")]
        )
        latest_model_filename = all_model_filenames[0]
        model = joblib.load(latest_model_filename)
    else:
        raise NotImplementedError

    return model


def predict_probabilities(clf, game_data):
    X = np.vstack(
        [
            v
            for k, v in game_data.items()
            if k not in ["winner", "gameId", "playId", "time_remaining"]
        ],
    ).T

    y_pred_prob = clf.predict_proba(X)
    home_win_prob = y_pred_prob[:, 1]

    return home_win_prob


def plot_probabilities(
    probabilities,
    time_remaining,
    home_goal_times,
    away_goal_times,
    home_team_name,
    away_team_name,
    home_final_goals,
    away_final_goals,
):
    assert len(probabilities) == len(time_remaining)
    timestamps = [max(time_remaining) - tr for tr in time_remaining]

    plt.plot(timestamps, probabilities, c="b")
    plt.vlines([1200, 2400, 3600], colors="r", ymin=0, ymax=1, label="Period end")
    plt.vlines(home_goal_times, colors="g", ymin=0, ymax=1, label="Home goal")
    plt.vlines(away_goal_times, colors="y", ymin=0, ymax=1, label="Away goal")
    plt.legend()
    plt.suptitle(
        f"{away_team_name} {away_final_goals} @ {home_team_name} {home_final_goals}"
    )
    plt.show()
    pass


def main(season, game_id):
    assert isinstance(season, str)
    season_year = f"{season}{int(season) + 1}"
    assert isinstance(game_id, str)
    full_game_id = season + game_id

    # load model
    clf = load_model(load_latest=True)

    # load game data
    game_directory_exists = os.path.isdir(
        os.path.join(os.path.dirname(__file__), "data", season_year, full_game_id)
    )
    if not game_directory_exists:
        raise Exception(f"Game folder for {season_year} {game_id} does not exist.")
    accumulated_data_exists = os.path.isfile(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            full_game_id,
            "accumulated_data.csv",
        )
    )
    if not accumulated_data_exists:
        raise Exception(f"Accumulated data for {season_year} {game_id} does not exist.")
    game_data = load_data(
        filename=os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            full_game_id,
            "accumulated_data.csv",
        )
    )

    probs = predict_probabilities(clf, game_data)

    if "time_remaining_neg" in game_data:
        time_remaining = game_data["time_remaining_neg"]
    elif "time_remaining" in game_data:
        time_remaining = game_data["time_remaining"]
    else:
        raise Exception(
            "One of time_remaining_neg or time_remaining needs to be in game data."
        )

    home_goal_times = [
        3600 - t
        for t, gd0, gd1 in zip(
            game_data["time_remaining"][1:],
            game_data["goal_differential"][:-1],
            game_data["goal_differential"][1:],
        )
        if gd1 - gd0 == 1
    ]
    away_goal_times = [
        3600 - t
        for t, gd0, gd1 in zip(
            game_data["time_remaining"][1:],
            game_data["goal_differential"][:-1],
            game_data["goal_differential"][1:],
        )
        if gd1 - gd0 == -1
    ]

    game_info_exists = bool(
        len(
            glob.glob(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    season_year,
                    full_game_id,
                    "*.txt",
                )
            )
        )
    )
    if not game_info_exists:
        raise Exception("The info .txt file for this game does not exists")

    game_info_filenames = glob.glob(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            full_game_id,
            "*.txt",
        )
    )
    assert (
        len(game_info_filenames) == 1
    ), "There should be only one .txt file in the folder"
    game_info_filename = game_info_filenames[0]
    game_info = Path(game_info_filename).stem
    game_type, away_team_name, _, home_team_name = game_info.split("_")
    home_team_name = home_team_name.replace("-", " ")
    away_team_name = away_team_name.replace("-", " ")

    plot_probabilities(
        probs,
        time_remaining,
        home_goal_times,
        away_goal_times,
        home_team_name,
        away_team_name,
        len(home_goal_times),
        len(away_goal_times),
    )


if __name__ == "__main__":
    season = "2020"
    game_id = "020088"
    main(season, game_id)
