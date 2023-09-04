# idea is to create 5 models and then stack the outputs with a logistic regression
# five models are: player, shot situation, player x shot situation, goalie, goalie x shot situation
# I think there are low level interactions with player/ shot and goalie shot but not player goalie. So these base models
#  will all be random forests and the metamodel will be logistic regression
# player and goalie model should have past success statistics in general, streakiness, injury, descriptive statistics, - use random forests
# shot situation try out k nearest neighbors based on previous action and shot location and man advantage
# The combined models should be some sort of bayesian model with prior, try beta-binomial with prior being count of goals
#  from within some radius of "all" players and likelihood being count of goals from the specific player in a radius.
#  to help here are some resources:
#   calculate prior parameters from an empirical mean and variance:https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
#   beta-bionomial posterior distribution: https://www.bayesrulesbook.com/chapter-3.html
#   remember to weight by man advantage

# moneypuck about modeling: https://moneypuck.com/about.htm

import requests
from collections import defaultdict, Counter
from nhl_requests import nhl_live_feed_request, nhl_pbp_request
from bs4 import BeautifulSoup
import re


def parse_nhl_live_shots(live_game_json):
    df_dict = {
        "gameId": [],
        "playId": [],
        "event": [],
        "timestamp": [],
        "shot_x": [],
        "shot_y": [],
    }

    # get gameId gameData -> game -> pk
    gameId = live_game_json["gameData"]["game"]["pk"]

    allPlays = live_game_json["liveData"]["plays"]["allPlays"]
    for play_ind, play_dict in enumerate(allPlays):
        assert "event" in play_dict["result"]
        event = play_dict["result"]["event"]

        if event in [
            "Game Scheduled",
            "Period Ready",
            "Period Start",
            "Period End",
            "Period Official",
            "Stoppage",
            "Official Challenge",
            "Game End",
            "Shootout Complete",
            "Early Intermission Start",
            "Early Intermission End",
            "Emergency Goaltender",
            "Faceoff",
            "Takeaway",
            "Hit",
            "Penalty",
            "Giveaway",
            "Game Official",
            "Game End",
        ]:
            continue

        df_dict["gameId"].append(gameId)
        df_dict["event"].append(event)

        assert "about" in play_dict and "eventIdx" in play_dict["about"]
        playId = play_dict["about"]["eventIdx"]
        df_dict["playId"].append(playId)

        assert "periodTime" in play_dict["about"] and "period" in play_dict["about"]
        period = int(play_dict["about"]["period"])
        minutes_gone_in_period = int(play_dict["about"]["periodTime"].split(":")[0])
        seconds_gone_in_period = int(play_dict["about"]["periodTime"].split(":")[1])
        seconds_remaining_in_game = 20 * 3 * 60 - (
            20 * 60 * (period - 1)
            + 60 * minutes_gone_in_period
            + seconds_gone_in_period
        )
        df_dict["timestamp"].append(seconds_remaining_in_game)

        shot_xy = None
        if event == "Blocked Shot":
            event_ = "block"
        elif event in ["Shot", "Missed Shot"]:
            event_ = "shot"

            assert "coordinates" in play_dict
            shot_xy = play_dict["coordinates"]
        elif event == "Goal":
            event_ = "goal"

            assert "coordinates" in play_dict
            shot_xy = play_dict["coordinates"]
        else:
            raise NotImplementedError

        if shot_xy is not None and len(shot_xy):
            df_dict["shot_x"].append(shot_xy["x"])
            df_dict["shot_y"].append(shot_xy["y"])

        for col, vals in df_dict.items():
            if len(vals) < len(df_dict["gameId"]):
                df_dict[col].append(0)

        assert all(len(vals) == len(df_dict["gameId"]) for vals in df_dict.values())

    assert len(df_dict["gameId"]) > 0
    return df_dict


def fetch_to_df_nhl_shots(game_id):
    live_data = nhl_live_feed_request(game_id)
    live_data_df_dict = parse_nhl_live_shots(live_data)
    return live_data_df_dict


def parse_nhl_pbp_shots(nhl_event_soup, game_id):
    assert isinstance(nhl_event_soup, BeautifulSoup)
    # return line items in table similar to what is seen online: http://www.nhl.com/scores/htmlreports/20212022/PL030234.HTM
    df_dict = {
        "gameId": [],
        "playId": [],
        "strength": [],
        "timestamp": [],
        "event": [],
        "description": [],
        "off_goalie_pulled": [],
        "def_goalie_pulled": [],
    }

    # get the home and away team abbreviations
    abbreviations = {
        "Atlanta": ["AFM"],
        "Anaheim": ["ANA"],
        "Arizona": ["ARI"],
        "Boston": ["BOS"],
        "Brooklyn": ["BRK"],
        "Buffalo": ["BUF"],
        "Carolina": ["CAR"],
        "Columbus": ["CBJ"],
        "California": ["CGS"],
        "Calgary": ["CGY"],
        "Chicago": ["CHI"],
        "Cleveland": ["CLE"],
        "Colorado": ["COL"],
        "Dallas": ["DAL"],
        "Detroit": ["DET"],
        "Edmonton": ["EDM"],
        "Florida": ["FLA"],
        "Hamilton": ["HAM"],
        "Hartford": ["HFD"],
        "Kansas City": ["KCS"],
        "Los Angeles": ["LAK", "L.A"],
        "Minnesota": ["MIN"],
        "Montreal": ["MTL"],
        "New Jersey": ["NJD", "N.J"],
        "Nashville": ["NSH"],
        "New York Rangers": ["NYR"],
        "New York Islanders": ["NYI"],
        "Ottawa": ["OTT"],
        "Philadelphia": ["PHI"],
        "Phoenix": ["PHX"],
        "Pittsburgh": ["PIT"],
        "Quebec": ["QBD"],
        "Seattle": ["SEA"],
        "San Jose": ["SJS", "S.J"],
        "St. Louis": ["STL"],
        "Tampa Bay": ["TBL", "T.B"],
        "Toronto": ["TOR"],
        "Vancouver": ["VAN"],
        "Vegas": ["VGK"],
        "Winnipeg": ["WPG"],
        "Washington": ["WSH"],
    }

    visitor_table = nhl_event_soup.find("table", {"id": "Visitor"})
    home_table = nhl_event_soup.find("table", {"id": "Home"})

    away_abbrev = None
    home_abbrev = None

    for loc, abbr_list in abbreviations.items():
        if loc.lower() in visitor_table.text.lower():
            assert away_abbrev is None
            away_abbrev = abbr_list[0]
        if loc.lower() in home_table.text.lower():
            assert home_abbrev is None
            home_abbrev = abbr_list[0]

    assert away_abbrev is not None and home_abbrev is not None
    assert isinstance(away_abbrev, str) and isinstance(home_abbrev, str)

    # not all games have the id=PL-# format
    # ex game that does: 2021030234
    # ex game that doesn't: 2017020001
    PL_tr = nhl_event_soup.find_all("tr", {"id": re.compile(r"PL-\d")})
    color_tr = nhl_event_soup.find_all(
        "tr", {"class": re.compile(r"(?:even|odd)Color")}
    )

    all_tr = PL_tr if len(PL_tr) >= len(color_tr) else color_tr

    for tr in all_tr:
        line_items = [tag.text for tag in tr.find_all("td", recursive=False)]
        assert len(line_items) == 8

        event = line_items[4]
        if event not in ["SHOT", "GOAL", "MISS", "BLOCK"]:
            continue

        df_dict["gameId"].append(game_id)

        df_dict["playId"].append(line_items[0])
        df_dict["strength"].append(
            line_items[2] if not line_items[2] == "\xa0" else "-1"
        )

        period = int(line_items[1])
        minutes_gone_in_period = int(line_items[3].split(":")[0])
        seconds_gone_in_period = int(line_items[3].split(":")[1][:2])
        seconds_remaining_in_game = 20 * 3 * 60 - (
            20 * 60 * (period - 1)
            + 60 * minutes_gone_in_period
            + seconds_gone_in_period
        )
        df_dict["timestamp"].append(seconds_remaining_in_game)
        df_dict["event"].append(line_items[4])
        df_dict["description"].append(
            line_items[5].replace("\xa0", " ") if not line_items[5] == "\xa0" else "-1"
        )

        if not line_items[6] == "\xa0" and not line_items[7] == "\xa0":
            away_positions = [
                c[-1] for c in "".join(line_items[6].split("\n")).split("\xa0")
            ]
            home_positions = [
                c[-1] for c in "".join(line_items[7].split("\n")).split("\xa0")
            ]

            if df_dict["description"][-1].startswith(away_abbrev):
                df_dict["off_goalie_pulled"].append(
                    "0" if "G" in away_positions else "1"
                )
                df_dict["def_goalie_pulled"].append(
                    "0" if "G" in home_positions else "1"
                )
            elif df_dict["description"][-1].startswith(home_abbrev):
                df_dict["off_goalie_pulled"].append(
                    "0" if "G" in home_positions else "1"
                )
                df_dict["def_goalie_pulled"].append(
                    "0" if "G" in away_positions else "1"
                )
            else:
                raise NotImplementedError
        else:
            df_dict["off_goalie_pulled"].append("-1")
            df_dict["def_goalie_pulled"].append("-1")

        assert all(len(vals) == len(df_dict["gameId"]) for vals in df_dict.values())

    assert len(df_dict["gameId"]) > 0

    return df_dict


def fetch_to_df_nhl_pbp(game_id):
    pbp_data = nhl_pbp_request(game_id)
    pbp_df = parse_nhl_pbp_shots(pbp_data, game_id)
    return pbp_df


def data_main(args):
    season_ids = ["20202021", "20212022"]

    # data for player model is their profile stats (ht wt spd shot-top-speed), num goals in last x games & pos/neg trend,
    #  minutes trend in last x games (const/pos/neg - not going for total), number of pts/goals per game,
    #
    # data for goalie model is their profile stats, num goals allowed in last x games & pos/neg trend, number of starts
    #  over last x games, how many games they got pulled, number of goals allowed per game
    #
    # response variable for player will be the number of goals they scored that game, ultimately want probability they
    #  will score at least one goal in the game, response variable for goalie will be the save percentage of the game
    #  want to get for each shot in the game was the probability they will save it basically

    """
    teams_request = requests.get("https://statsapi.web.nhl.com/api/v1/teams").json()
    team_ids = [d["id"] for d in teams_request["teams"] if d["active"]]
    player_data = {}
    game_player_window = 4
    for team_id in team_ids:
        roster_request = requests.get(
            f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.roster"
        ).json()
        assert len(roster_request["teams"]) == 1
        team_dict = roster_request["teams"][0]
        roster = team_dict["roster"]["roster"]
        player_ids = [d["person"]["id"] for d in roster]
        for player_id in player_ids:
            if player_id not in player_data:
                base_stats_request = requests.get(
                    f"https://statsapi.web.nhl.com/api/v1/people/{player_id}"
                ).json()
                career_stats_request = requests.get(
                    f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=yearByYear"
                ).json()

                base_stats = base_stats_request["people"][0]
                player_data[player_id] = {
                    "height": base_stats["height"],
                    "weight": base_stats["weight"],
                    "position": base_stats["primaryPosition"]["abbreviation"],
                    "current_age": base_stats["currentAge"],
                    "game_data": {},  # keys will be season IDs and values will be list of ordered game dicts
                }
                is_goalie = player_data[player_id]["position"] in ["G"]
                num_games_played = defaultdict(int)
                career_splits = career_stats_request["stats"][0]["splits"]
                for year_split in career_splits:
                    if year_split["league"]["name"] == "National Hockey League":
                        if year_split["season"] in season_ids:
                            num_games_played[year_split["season"]] += year_split[
                                "stat"
                            ]["games"]
                        if (
                            2005
                            <= int(year_split["season"][:4])
                            < int(season_ids[-1][:4])
                        ):
                            if not is_goalie:
                                if (
                                    "career_shots_per_game"
                                    not in player_data[player_id]
                                ):
                                    assert (
                                        "career_goals_per_game"
                                        not in player_data[player_id]
                                    )
                                    player_data[player_id].update(
                                        {
                                            "career_shots_per_game": defaultdict(
                                                Counter
                                            ),
                                            "career_goals_per_game": defaultdict(
                                                Counter
                                            ),
                                        }
                                    )

                                player_data[player_id]["career_shots_per_game"][
                                    year_split["season"]
                                ].update(
                                    Counter(
                                        {
                                            "shots": year_split["stat"]["shots"],
                                            "games": year_split["stat"]["games"],
                                        }
                                    )
                                )
                                player_data[player_id]["career_goals_per_game"][
                                    year_split["season"]
                                ].update(
                                    Counter(
                                        {
                                            "goals": year_split["stat"]["goals"],
                                            "games": year_split["stat"]["games"],
                                        }
                                    )
                                )
                            else:
                                if (
                                    "career_goals_against_per_toi"
                                    not in player_data[player_id]
                                ):
                                    assert (
                                        "career_shots_saves"
                                        not in player_data[player_id]
                                    )
                                    player_data[player_id].update(
                                        {
                                            "career_goals_against_per_toi": defaultdict(
                                                Counter
                                            ),
                                            "career_shots_saves": defaultdict(Counter),
                                        }
                                    )

                                player_data[player_id]["career_goals_against_per_toi"][
                                    year_split["season"]
                                ].update(
                                    Counter(
                                        {
                                            "goals_against": year_split["stat"][
                                                "goalsAgainst"
                                            ],
                                            "toi": 60
                                            * int(
                                                year_split["stat"]
                                                .get("timeOnIce", "0:0")
                                                .split(":")[0]
                                            )
                                            + int(
                                                year_split["stat"]
                                                .get("timeOnIce", "0:0")
                                                .split(":")[1]
                                            ),
                                        }
                                    )
                                )
                                player_data[player_id]["career_shots_saves"][
                                    year_split["season"]
                                ].update(
                                    Counter(
                                        {
                                            "shots_against": year_split["stat"][
                                                "shotsAgainst"
                                            ],
                                            "saves": year_split["stat"]["saves"],
                                        }
                                    )
                                )
                    else:
                        num_games_played[year_split["season"]] = 0
            else:
                is_goalie = player_data[player_id]["position"] in ["G"]
                career_stats_request = requests.get(
                    f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=yearByYear"
                ).json()
                num_games_played = defaultdict(int)
                career_splits = career_stats_request["stats"][0]["splits"]
                for year_split in career_splits:
                    if (
                        year_split["league"]["name"] == "National Hockey League"
                        and year_split["season"] in season_ids
                    ):
                        num_games_played[year_split["season"]] += year_split["stat"][
                            "games"
                        ]
            for season_id in season_ids:
                if (
                    season_id not in num_games_played
                    or num_games_played[season_id] == 0
                ):
                    continue
                season_log_request = requests.get(
                    f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=gameLog&season={season_id}"
                ).json()
                assert len(season_log_request["stats"]) == 1
                season_split = season_log_request["stats"][0]["splits"]
                assert len(season_split) == num_games_played[season_id]
                season_log = []
                for log_ind, game_log in enumerate(season_split):
                    if not is_goalie:
                        game_stats = {
                            "avg_goals": 0.0,
                            "avg_shots": 0.0,
                            "avg_mins": 0.0,
                            # "cum_pm": 0,
                            "num_goals": game_log["stat"].get("goals", 0),
                            "num_shots": game_log["stat"].get("shots", 0),
                            "num_mins": 60
                            * int(
                                game_log["stat"].get("timeOnIce", "0:0").split(":")[0]
                            )
                            + int(
                                game_log["stat"].get("timeOnIce", "0:0").split(":")[1]
                            ),
                            # "pm": game_log["stat"].get("plusMinus", 0),
                            'gamePk': game_log['game']['gamePk'],
                        }

                        prev_game_stats = season_log[-game_player_window:]
                        game_stats["avg_goals"] = (
                            sum([pgs["num_goals"] for pgs in prev_game_stats])
                            / game_player_window
                        )
                        game_stats["avg_shots"] = (
                            sum([pgs["num_shots"] for pgs in prev_game_stats])
                            / game_player_window
                        )
                        game_stats["avg_mins"] = (
                            sum([pgs["num_mins"] for pgs in prev_game_stats])
                            / game_player_window
                        )
                        # game_stats["cum_pm"] = sum(
                        #     [pgs["pm"] for pgs in prev_game_stats]
                        # )

                    else:
                        game_stats = {
                            "avg_saves": 0.0,
                            "avg_shots": 0.0,
                            "cum_goals": 0.0,
                            "cum_toi": 0.0,
                            "num_saves": game_log["stat"].get("saves", 0),
                            "num_shots": game_log["stat"].get("shotsAgainst", 0),
                            "num_goals": game_log["stat"].get("goalsAgainst", 0),
                            "num_toi": 60
                            * int(
                                game_log["stat"].get("timeOnIce", "0:0").split(":")[0]
                            )
                            + int(
                                game_log["stat"].get("timeOnIce", "0:0").split(":")[1]
                            ),
                            'gamePk': game_log['game']['gamePk'],
                        }

                        prev_game_stats = season_log[-game_player_window:]
                        game_stats["avg_saves"] = (
                            sum([pgs["num_saves"] for pgs in prev_game_stats])
                            / game_player_window
                        )
                        game_stats["avg_shots"] = (
                            sum([pgs["num_shots"] for pgs in prev_game_stats])
                            / game_player_window
                        )
                        game_stats["cum_goals"] = sum(
                            [pgs["num_goals"] for pgs in prev_game_stats]
                        )
                        game_stats["cum_toi"] = sum(
                            [pgs["num_toi"] for pgs in prev_game_stats]
                        )

                    season_log.append(game_stats)
                player_data[player_id]["game_data"][season_id] = season_log
    """

    # data for shot situation model is previous x game plays - actions and locations and man advantage (power play, EN, 6 on 5, 4 on 4 etc.)
    #  and shot location itself
    gamedays_shot_window = 20
    for season_id in season_ids[:-1]:

        season_request = requests.get(
            f"https://statsapi.web.nhl.com/api/v1/schedule?season={season_id}"
        ).json()
        for i, game_date_dict in enumerate(season_request["dates"]):
            games_data = game_date_dict["games"]
            for game_data in games_data:
                game_id = str(game_data["gamePk"])

                game_type = game_data["gameType"]
                if game_type not in ["R", "P"]:
                    print(
                        f"gameId {game_id} is not a regular season or playoff game. Continuing."
                    )
                    continue

                live_df = fetch_to_df_nhl_shots(game_id)
                pbp_df = fetch_to_df_nhl_pbp(game_id)
                pass
                # todo merge two df on event and timestamp to determine if the shot taken was a powerplay/shorthanded/adv empty netter/disadv empty netter
                # ev5 -> 5on5
                # ev4 -> 4on4
                # ev3 -> 3on3
                # pp4 -> 5on4
                # pp3 -> 5on3
                # pp43 -> 4on3
                # en5 -> 6on5
                # en4 -> 6on4
                merged_df = {
                    "gameId": [],
                    "playId": [],
                    "event": [],
                    "timestamp": [],
                    "shot_x": [],
                    "shot_y": [],
                    "strength": [],
                }
                for j in range(len(live_df["gameId"])):
                    exact_match = True
                    if live_df["timestamp"][j] in pbp_df["timestamp"]:
                        time_ind = pbp_df["timestamp"].index(live_df["timestamp"][j])
                    elif live_df["timestamp"][j] - 1 in pbp_df["timestamp"]:
                        time_ind = pbp_df["timestamp"].index(
                            live_df["timestamp"][j] - 1
                        )
                    elif live_df["timestamp"][j] - 2 in pbp_df["timestamp"]:
                        time_ind = pbp_df["timestamp"].index(
                            live_df["timestamp"][j] - 2
                        )
                    elif live_df["timestamp"][j] + 1 in pbp_df["timestamp"]:
                        time_ind = pbp_df["timestamp"].index(
                            live_df["timestamp"][j] + 1
                        )
                    elif live_df["timestamp"][j] + 2 in pbp_df["timestamp"]:
                        time_ind = pbp_df["timestamp"].index(
                            live_df["timestamp"][j] + 2
                        )
                    else:
                        print("timestamp not found")
                        continue

                    def check_shot_type(live_event, pbp_event):
                        translation = {
                            "Shot": "SHOT",
                            "Blocked Shot": "BLOCK",
                            "Missed Shot": "MISS",
                            "Goal": "GOAL",
                        }
                        return translation[live_event] == pbp_event

                    if not check_shot_type(
                        live_df["event"][time_ind], pbp_df["event"][time_ind]
                    ):
                        continue

                # retrieve row from pbp
                #         pbp_row = pbp_df.loc[
                #             (pbp_df["timestamp"] == time_remaining)
                #             & (pbp_df["event"] == live_to_pbp_event[event])
                #         ]
                #         if len(pbp_row) > 1:
                #             pbp_row = pbp_row.iloc[[0]]
                #         elif len(pbp_row) == 0:
                #             skipped_rows += 1
                #             if skipped_rows >= 10:
                #                 raise Exception("Too many skipped rows!")
                #             continue
                #         assert (
                #             len(pbp_row.index) == 1
                #         ), f"Actual length of play by play is {len(pbp_row.index)}"

    # data for combined models is shot locations, whether they were goals or not, and the shooter & goalie involved
    pass


def train_main(args):
    pass


def main(args):
    data_main(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
