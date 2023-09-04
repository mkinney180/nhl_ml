import os
from pathlib import Path
import time
import pandas as pd
import numpy as np
from collections import defaultdict
import glob


def shift_distribution(player_shifts, player_totals, timestamp):
    timestamp_shifts = []

    for playerId in player_totals.keys():
        assert playerId in player_shifts
        timestamp_shifts.append(0)
        while True:
            if not len(player_shifts[playerId]):
                break

            if timestamp <= player_shifts[playerId][0][1]:
                player_totals[playerId] += player_shifts[playerId][0][2]
                del player_shifts[playerId][0]
            elif timestamp <= player_shifts[playerId][0][0]:
                timestamp_shifts[-1] += player_shifts[playerId][0][0] - timestamp
                break
            else:
                break

        timestamp_shifts[-1] += player_totals[playerId]

    return player_shifts, player_totals, timestamp_shifts


def accumulate(season_year, game_id):
    data_directory_exists = os.path.isdir(
        os.path.join(os.path.dirname(__file__), "data")
    )
    season_year_directory_exists = os.path.isdir(
        os.path.join(os.path.dirname(__file__), "data", season_year)
    )
    game_directory_exists = os.path.isdir(
        os.path.join(os.path.dirname(__file__), "data", season_year, game_id)
    )
    live_data_exists = os.path.isfile(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "live_data.csv",
        )
    )
    pbp_data_exists = os.path.isfile(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "pbp_data.csv",
        )
    )
    away_shifts_data_exists = os.path.isfile(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "away_shifts_data.csv",
        )
    )
    home_shifts_data_exists = os.path.isfile(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "home_shifts_data.csv",
        )
    )

    assert (
        data_directory_exists
        and season_year_directory_exists
        and game_directory_exists
        and live_data_exists
        and pbp_data_exists
        and away_shifts_data_exists
        and home_shifts_data_exists
    ), f"Data required is missing for {game_id} in {season_year}!"

    live_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "live_data.csv",
        )
    )

    pbp_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "pbp_data.csv",
        )
    )

    home_shifts_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "home_shifts_data.csv",
        )
    )

    away_shifts_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            season_year,
            game_id,
            "away_shifts_data.csv",
        )
    )

    accumulate_dict = {
        "gameId": [],
        "playId": [],
        "time_remaining": [],  # 0 if in overtime
        "time_remaining_neg": [],  # can be negative if game goes into overtime CRICKETTTTTTT
        "goal_differential": [],  # all differentials will be home team - away team
        "goal_total": [],
        # "spread": [],
        "shot_differential": [],
        "shot_total": [],
        "faceoff_differential": [],
        "faceoff_total": [],
        "goalie_pulled": [],  # -1 away goalie pulled, 0 neither, 1 home goalie pulled
        "players_on_ice_differential": [],
        "players_on_ice_total": [],  # does include goalies
        "takeaway_differential": [],
        "takeaway_total": [],
        "hit_differential": [],
        "hit_total": [],
        "block_differential": [],
        "block_total": [],
        "giveaway_differential": [],
        "giveaway_total": [],
        "goalie_change": [],  # -1 away goalie change, 0 neither or even, 1 home goalie change
        "toi_skew_differential": [],
        "last_goal": [],  # -1 for away, 0 none, 1 home
        # hope is to add in the danger of shots by where they are taken and how often they go in
    }

    live_to_pbp_event = {
        "Game Official": "GEND",
        "Game End": "GEND",
        "Blocked Shot": "BLOCK",
        "Faceoff": "FAC",
        "Takeaway": "TAKE",
        "Hit": "HIT",
        "Shot": "SHOT",
        "Missed Shot": "MISS",
        "Penalty": "PENL",
        "Giveaway": "GIVE",
        "Goal": "GOAL",
    }

    home_shift_totals = {
        playerId: 0 for playerId in home_shifts_df["playerId"].unique()
    }
    away_shift_totals = {
        playerId: 0 for playerId in away_shifts_df["playerId"].unique()
    }

    home_shift_times = defaultdict(list)
    for _, row in home_shifts_df.iterrows():
        home_shift_times[row["playerId"]].append(
            (row["start_shift"], row["end_shift"], row["shift_length"])
        )
    away_shift_times = defaultdict(list)
    for _, row in away_shifts_df.iterrows():
        away_shift_times[row["playerId"]].append(
            (row["start_shift"], row["end_shift"], row["shift_length"])
        )

    start = time.time()
    skipped_rows = 0

    # go through the live_data and use that as the main data, reach out to others as needed
    for _, row in live_df.iterrows():
        # record gameId, playId
        gameId = row["gameId"]
        playId = row["playId"]

        # retrieve timestamp and event
        event = row["event"]
        time_remaining = row["timestamp"]

        # retrieve row from pbp
        pbp_row = pbp_df.loc[
            (pbp_df["timestamp"] == time_remaining)
            & (pbp_df["event"] == live_to_pbp_event[event])
        ]
        if len(pbp_row) > 1:
            pbp_row = pbp_row.iloc[[0]]
        elif len(pbp_row) == 0:
            skipped_rows += 1
            if skipped_rows >= 10:
                raise Exception("Too many skipped rows!")
            continue
        assert (
            len(pbp_row.index) == 1
        ), f"Actual length of play by play is {len(pbp_row.index)}"

        # info needed from pbp: players total/diff on ice, goalie pulled, goalie changed
        num_home_on_ice = sum(
            [int(i) for i in str(pbp_row["home_on_ice"].item()).split("_")]
        )
        num_away_on_ice = sum(
            [int(i) for i in str(pbp_row["away_on_ice"].item()).split("_")]
        )
        home_goalie_pulled = int(pbp_row["home_pulled_goalie"].item())
        away_goalie_pulled = int(pbp_row["away_pulled_goalie"].item())
        home_goalie_changed = 1 * (
            len(pbp_df["home_goalie_number"][: pbp_row.index.item() + 1].unique())
            > 1 + 1 * (-1 in pbp_df["home_goalie_number"][: pbp_row.index.item() + 1])
        )
        away_goalie_changed = 1 * (
            len(pbp_df["away_goalie_number"][: pbp_row.index.item() + 1].unique())
            > 1 + 1 * (-1 in pbp_df["away_goalie_number"][: pbp_row.index.item() + 1])
        )

        # using timestamp and home/ away shifts calculate skewness 3 * (mean - median) / sd of home and away and calculate difference
        home_shift_times, home_shift_totals, home_toi = shift_distribution(
            home_shift_times, home_shift_totals, time_remaining
        )
        away_shift_times, away_shift_totals, away_toi = shift_distribution(
            away_shift_times, away_shift_totals, time_remaining
        )
        home_skew = (
            3 * (np.mean(home_toi) - np.median(home_toi)) / (np.std(home_toi) + 1e-6)
        )
        away_skew = (
            3 * (np.mean(away_toi) - np.median(away_toi)) / (np.std(away_toi) + 1e-6)
        )

        # use live data to fill in the rest, most of them are running totals and diffs so need to rely on -1 index of things
        home_goal = int(row["home_goal"])
        away_goal = int(row["away_goal"])

        home_shot = row["home_shot"]
        away_shot = row["away_shot"]

        home_block = row["home_block"]
        away_block = row["away_block"]

        home_giveaway = row["home_giveaway"]
        away_giveaway = row["away_giveaway"]

        home_takeaway = row["home_takeaway"]
        away_takeaway = row["away_takeaway"]

        home_hit = row["home_hit"]
        away_hit = row["away_hit"]

        home_faceoff_won = row["home_faceoff_won"]
        away_faceoff_won = row["away_faceoff_won"]

        # put the information into accumulate dict
        accumulate_dict["gameId"].append(gameId)
        accumulate_dict["playId"].append(playId)
        accumulate_dict["time_remaining"].append(
            time_remaining if time_remaining > 0 else 0
        )
        accumulate_dict["time_remaining_neg"].append(time_remaining)

        accumulate_dict["goal_differential"].append(
            accumulate_dict["goal_differential"][-1] + home_goal - away_goal
            if len(accumulate_dict["goal_differential"])
            else 0
        )
        accumulate_dict["goal_total"].append(
            accumulate_dict["goal_total"][-1] + home_goal + away_goal
            if len(accumulate_dict["goal_total"])
            else 0
        )

        accumulate_dict["shot_differential"].append(
            accumulate_dict["shot_differential"][-1] + home_shot - away_shot
            if len(accumulate_dict["shot_differential"])
            else 0
        )
        accumulate_dict["shot_total"].append(
            accumulate_dict["shot_total"][-1] + home_shot + away_shot
            if len(accumulate_dict["shot_total"])
            else 0
        )

        accumulate_dict["faceoff_differential"].append(
            accumulate_dict["faceoff_differential"][-1]
            + home_faceoff_won
            - away_faceoff_won
            if len(accumulate_dict["faceoff_differential"])
            else 0
        )
        accumulate_dict["faceoff_total"].append(
            accumulate_dict["faceoff_total"][-1] + home_faceoff_won + away_faceoff_won
            if len(accumulate_dict["faceoff_total"])
            else 0
        )

        accumulate_dict["hit_differential"].append(
            accumulate_dict["hit_differential"][-1] + home_hit - away_hit
            if len(accumulate_dict["hit_differential"])
            else 0
        )
        accumulate_dict["hit_total"].append(
            accumulate_dict["hit_total"][-1] + home_hit + away_hit
            if len(accumulate_dict["hit_total"])
            else 0
        )

        accumulate_dict["block_differential"].append(
            accumulate_dict["block_differential"][-1] + home_block - away_block
            if len(accumulate_dict["block_differential"])
            else 0
        )
        accumulate_dict["block_total"].append(
            accumulate_dict["block_total"][-1] + home_block + away_block
            if len(accumulate_dict["block_total"])
            else 0
        )

        accumulate_dict["takeaway_differential"].append(
            accumulate_dict["takeaway_differential"][-1] + home_takeaway - away_takeaway
            if len(accumulate_dict["takeaway_differential"])
            else 0
        )
        accumulate_dict["takeaway_total"].append(
            accumulate_dict["takeaway_total"][-1] + home_takeaway + away_takeaway
            if len(accumulate_dict["takeaway_total"])
            else 0
        )

        accumulate_dict["giveaway_differential"].append(
            accumulate_dict["giveaway_differential"][-1] + home_giveaway - away_giveaway
            if len(accumulate_dict["giveaway_differential"])
            else 0
        )
        accumulate_dict["giveaway_total"].append(
            accumulate_dict["giveaway_total"][-1] + home_giveaway + away_giveaway
            if len(accumulate_dict["giveaway_total"])
            else 0
        )

        accumulate_dict["goalie_pulled"].append(home_goalie_pulled - away_goalie_pulled)
        accumulate_dict["goalie_change"].append(
            home_goalie_changed - away_goalie_changed
        )

        accumulate_dict["players_on_ice_differential"].append(
            num_home_on_ice - num_away_on_ice
        )
        accumulate_dict["players_on_ice_total"].append(
            num_home_on_ice + num_away_on_ice
        )

        accumulate_dict["toi_skew_differential"].append(home_skew - away_skew)
        if not len(accumulate_dict["last_goal"]):
            accumulate_dict["last_goal"].append(0)
        else:
            if home_goal:
                accumulate_dict["last_goal"].append(1)
            elif away_goal:
                accumulate_dict["last_goal"].append(-1)
            else:
                accumulate_dict["last_goal"].append(accumulate_dict["last_goal"][-1])

        assert all(
            len(vals) == len(accumulate_dict["gameId"])
            for vals in accumulate_dict.values()
        )

    end = time.time()
    print(f"Finished playId {game_id} in {end-start:.2f} seconds.")

    # add in home_win with 1 for true and 0 for false and same length as the rest of the columns
    if live_df["home_win"].to_list()[-1]:
        accumulate_dict["winner"] = [1] * len(accumulate_dict["gameId"])
    elif live_df["away_win"].to_list()[-1]:
        accumulate_dict["winner"] = [-1] * len(accumulate_dict["gameId"])
    else:
        raise NotImplementedError

    return accumulate_dict


def save_accumulation(accumulate_dict, season_year, game_id):
    # save this in the same folder as the other game dfs
    accumulate_df = pd.DataFrame(accumulate_dict)
    filename = os.path.join("data", season_year, game_id, "accumulated_data.csv")
    accumulate_df.to_csv(filename, index=False)


def accumulate_season(season_year):
    assert os.path.isdir("data")
    assert os.path.isdir(os.path.join("data", season_year))
    game_folders = glob.glob(os.path.join("data", season_year, "*"))
    for game_folder in game_folders:
        game_id = Path(game_folder).stem
        if os.path.isfile(os.path.join(game_folder, "accumulated_data.csv")):
            print(f"Accumulated data for {game_id} has been found. Continuing.")
            continue
        accumulate_dict = accumulate(season_year, game_id)
        save_accumulation(accumulate_dict, season_year, game_id)


if __name__ == "__main__":
    season_year = "20202021"
    game_id = "2019020010"

    start = time.time()
    accumulate_season(season_year)
    end = time.time()

    print(f"This took {end-start:.2f} seconds")
