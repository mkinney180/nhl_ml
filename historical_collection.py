import re
import time
import requests
import os
from pathlib import Path

from nhl_requests import (
    fetch_to_df_nhl_pbp,
    fetch_to_df_nhl_shifts,
    fetch_to_df_nhl_live_feed,
)


# minutes/seconds left in the game
# score difference / total goals
# before game spread
# shot difference / total
# faceoff win difference
# own / opp goalie pulled
# diff of players on ice / total
# takeaway difference / total
# hit difference / total
# block difference / total
# giveaway difference / total
# opp / own goalie change
# available skater difference x
# TOI skewness (mean - median) diff
# last goal

# season -> gameid -> [game id, timestamp, score_diff, score_total, spread, shot_diff, shot_total, faceoff_diff, goalie_pull (-1/0/1), players_diff,
# players_total, takeaway_diff, takeaway_total, hit_difference, hit_total, block_difference, block_total, giveaway_difference
# giveaway_total, opp_goalie_change, own_goalie_change, available_skater_diff, toi_skew_diff]
# ^ in perspective of home team when it comes to diff, check to make sure when away turns to negative the probability is inverted
# game info -> [game id, season, type, home team id, away team id, winning id]

# http://www.advancedfootballanalytics.com/2009/04/nhl-in-game-win-probability.html
# https://arxiv.org/pdf/1906.05029.pdf
# https://www.stephenpettigrew.com/articles/pettigrew_nhl_win_probs.pdf
# http://homepage.divms.uiowa.edu/~dzimmer/sports-statistics/nettletonandlock.pdf

# function to go through each season and each game (avoid preseason, all star game)
# this function will feed the json output
# save each game separately
def build_season_data(season_year):

    assert isinstance(season_year, str)
    assert len(season_year) == 8
    assert re.match(r"20[0|1|2]\d20[0|1|2]\d", season_year) is not None
    assert int(season_year[:4]) + 1 == int(season_year[4:])

    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "data")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "data"))

    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "data", season_year)):
        os.makedirs(os.path.join(os.path.dirname(__file__), "data", season_year))

    # pull all season games to get game ids
    season_request = requests.get(
        f"https://statsapi.web.nhl.com/api/v1/schedule?season={season_year}"
    ).json()

    print(f"Beginning data pull for season {season_year}")
    for i, game_date_dict in enumerate(season_request["dates"]):
        print(f'Starting day {i + 1} out of {len(season_request["dates"])}')
        games_data = game_date_dict["games"]
        for game_data in games_data:
            game_id = str(game_data["gamePk"])

            game_type = game_data["gameType"]
            if game_type not in ["R", "P"]:
                print(
                    f"gameId {game_id} is not a regular season or playoff game. Continuing."
                )
                continue

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

            if (
                game_directory_exists
                and live_data_exists
                and pbp_data_exists
                and away_shifts_data_exists
                and home_shifts_data_exists
            ):
                print(f"All data found for {game_id} in {season_year}. Continuing.")
                continue

            if not game_directory_exists:
                os.makedirs(
                    os.path.join(
                        os.path.dirname(__file__), "data", season_year, game_id
                    )
                )

            start = time.time()

            live_df = fetch_to_df_nhl_live_feed(game_id)
            live_df.to_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    season_year,
                    game_id,
                    "live_data.csv",
                ),
                index=False,
            )
            print(f"Saved live data for gameID {game_id}.")

            pbp_df = fetch_to_df_nhl_pbp(game_id)
            pbp_df.to_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    season_year,
                    game_id,
                    "pbp_data.csv",
                ),
                index=False,
            )
            print(f"Saved pbp data for gameID {game_id}.")

            away_shifts_df, home_shifts_df = fetch_to_df_nhl_shifts(game_id)
            away_shifts_df.to_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    season_year,
                    game_id,
                    "away_shifts_data.csv",
                ),
                index=False,
            )
            print(f"Saved away shift data for gameID {game_id}.")
            home_shifts_df.to_csv(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    season_year,
                    game_id,
                    "home_shifts_data.csv",
                ),
                index=False,
            )
            print(f"Saved home shift data for gameID {game_id}.")

            home_team = "-".join(game_data["teams"]["home"]["team"]["name"].split(" "))
            away_team = "-".join(game_data["teams"]["away"]["team"]["name"].split(" "))
            Path(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    season_year,
                    game_id,
                    f"{game_type}_{away_team}_at_{home_team}.txt",
                )
            ).touch()

            end = time.time()
            print(f"Finished gameId {game_id} in {end-start:.2f} seconds.")


def build_database():
    seasons = ["2017", "2018", "2019", "2020", "2021"]
    seasons = ["2020"]

    for season in seasons:
        season_year = f"{season}{int(season) + 1}"

        build_season_data(season_year)


if __name__ == "__main__":
    build_database()
