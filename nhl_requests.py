import datetime
import re

import requests
import yaml
import os
import json
from bs4 import BeautifulSoup
import pandas as pd
from nltk import edit_distance

if not os.path.isfile("nhl.yaml"):
    openapi_url = "https://raw.githubusercontent.com/erunion/sport-api-specifications/master/nhl/nhl.yaml"
    response = requests.get(openapi_url)
    with open("nhl.yaml", "wb") as wf:
        wf.write(response.content)

avalanche_team_id = 21
mckinnon_id = 8477492
landeskog_id = 8476455
kadri_id = 8475172
latest_game_id = 2021030243

# nhl_yaml = yaml.safe_load(open("nhl.yaml", "rb"))
# team_info = requests.get("https://statsapi.web.nhl.com/api/v1/teams")
# team_info = team_info.json()
# for team in team_info["teams"]:
#     if "colorado" in team["name"].lower():
#         print(json.dumps(team, indent=4))

# avalanche_roster_info = requests.get(
#     f"https://statsapi.web.nhl.com/api/v1/teams/{avalanche_team_id}/roster"
# )
# avalanche_roster_info = avalanche_roster_info.json()
# print(json.dumps(avalanche_roster_info, indent=4))

# play_types = requests.get("https://statsapi.web.nhl.com/api/v1/playTypes").json()
# print(json.dumps(play_types, indent=2))

# avalanche_schedule = requests.get(
#     f"https://statsapi.web.nhl.com/api/v1/schedule?teamId={avalanche_team_id}&startDate=2020-06-01&endDate=2022-05-20"
# ).json()
# print(json.dumps(avalanche_schedule, indent=2))

# mckinnon_data = requests.get(
#     f"https://statsapi.web.nhl.com/api/v1/people/{mckinnon_id}"
# ).json()
# print(json.dumps(mckinnon_data, indent=2))

# mckinnon_stats = requests.get(
#     f"https://statsapi.web.nhl.com/api/v1/people/{mckinnon_id}/stats?stats=statsSingleSeason&season=20212022"
# ).json()
# print(json.dumps(mckinnon_stats, indent=2))

# latest_game_info = requests.get(
#     f"https://statsapi.web.nhl.com/api/v1/game/{latest_game_id}/linescore"
# ).json()
# print(json.dumps(latest_game_info, indent=2))

# live_game_info = requests.get(
#     f"https://statsapi.web.nhl.com/api/v1/game/{latest_game_id}/feed/live/diffPatch?startTimecode=20220523_141900"
# ).json()
# live_game_info_liveData = live_game_info["liveData"]
# print(json.dumps(live_game_info_liveData, indent=2))


def nhl_live_feed_request(game_id):
    live_game_info = requests.get(
        f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
    ).json()

    if (
        "message" in live_game_info
        and live_game_info["message"] == "Game data couldn't be found"
    ):
        return None
    else:
        return live_game_info


def parse_nhl_live_feed(live_game_json):
    # should return a pandas with the play by play data
    df_dict = {
        "gameId": [],
        "playId": [],
        "event": [],
        "timestamp": [],
        "home_faceoff_won": [],
        "away_faceoff_won": [],
        "home_hit": [],
        "away_hit": [],
        "home_goal": [],
        "away_goal": [],
        "home_giveaway": [],
        "away_giveaway": [],
        "home_takeaway": [],
        "away_takeaway": [],
        "home_block": [],
        "away_block": [],
        "home_shot": [],
        "away_shot": [],
        "shot_x": [],
        "shot_y": [],
        "home_penalty": [],
        "away_penalty": [],
        "home_win": [],
        "away_win": [],
    }

    # get triCode of home and away team gameData -> teams -> away/home -> triCode
    away_triCode = live_game_json["gameData"]["teams"]["away"]["triCode"]
    home_triCode = live_game_json["gameData"]["teams"]["home"]["triCode"]

    # get gameId gameData -> game -> pk
    gameId = live_game_json["gameData"]["game"]["pk"]

    # go through each play of liveData -> plays -> allPlays and record relevant information
    # ignore stoppages and period start and ends
    allPlays = live_game_json["liveData"]["plays"]["allPlays"]
    for play_ind, play_dict in enumerate(allPlays):

        assert "result" in play_dict and "event" in play_dict["result"]
        event = play_dict["result"]["event"]

        if (
            event
            in [
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
            ]
            and play_ind < len(allPlays) - 1
        ):
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

        if (event == "Game Official" or event == "Game End") and play_ind == len(
            allPlays
        ) - 1:
            # end of game, record who won based on goals scored
            assert (
                not live_game_json["liveData"]["linescore"]["teams"]["home"]["goals"]
                == live_game_json["liveData"]["linescore"]["teams"]["away"]["goals"]
            )
            event_ = "win"
            team_ = (
                "home"
                if live_game_json["liveData"]["linescore"]["teams"]["home"]["goals"]
                > live_game_json["liveData"]["linescore"]["teams"]["away"]["goals"]
                else "away"
            )

        else:
            # normal play, record which team play is attributed to and the type of play
            assert "team" in play_dict and "triCode" in play_dict["team"]
            if play_dict["team"]["triCode"] == away_triCode:
                team_ = "away"
            elif play_dict["team"]["triCode"] == home_triCode:
                team_ = "home"
            else:
                raise NotImplementedError

            shot_xy = None
            if event == "Blocked Shot":
                event_ = "block"
            elif event == "Faceoff":
                event_ = "faceoff_won"
            elif event == "Takeaway":
                event_ = "takeaway"
            elif event == "Hit":
                event_ = "hit"
            elif event in ["Shot", "Missed Shot"]:
                event_ = "shot"

                assert "coordinates" in play_dict
                shot_xy = play_dict["coordinates"]
            elif event == "Penalty":
                event_ = "penalty"
            elif event == "Giveaway":
                event_ = "giveaway"
            elif event == "Goal":
                event_ = "goal"

                assert "coordinates" in play_dict
                shot_xy = play_dict["coordinates"]
            else:
                raise NotImplementedError

            if shot_xy is not None and len(shot_xy):
                df_dict["shot_x"].append(shot_xy["x"])
                df_dict["shot_y"].append(shot_xy["y"])

        df_dict["_".join([team_, event_])].append(1)
        if event_ == "goal":
            df_dict["_".join([team_, "shot"])].append(1)

        for col, vals in df_dict.items():
            if len(vals) < len(df_dict["gameId"]):
                df_dict[col].append(0)

        assert all(len(vals) == len(df_dict["gameId"]) for vals in df_dict.values())

    df = pd.DataFrame(df_dict)
    assert not df.empty
    assert sum(df_dict.get("home_win", 0)) == 1 or sum(df_dict.get("away_win", 0)) == 1
    return df


def fetch_to_df_nhl_live_feed(game_id):
    live_data = nhl_live_feed_request(game_id)
    live_data_df = parse_nhl_live_feed(live_data)
    return live_data_df


def nhl_pbp_request(game_id):
    season_id = f"{str(game_id)[:4]}{int(str(game_id)[:4]) + 1}"
    game_identifier = str(game_id)[-6:]

    try_count = 0
    iterating = True
    while iterating:
        try_count += 1
        event_info = requests.get(
            f"http://www.nhl.com/scores/htmlreports/{season_id}/PL{game_identifier}.HTM"
        )
        event_soup = BeautifulSoup(event_info.content, "html.parser")
        assert isinstance(event_soup, BeautifulSoup)

        if "404 Not Found" == event_soup.find("title").text:
            iterating = try_count <= 2
        else:
            return event_soup

    return None


def parse_nhl_pbp(nhl_event_soup, game_id):
    assert isinstance(nhl_event_soup, BeautifulSoup)
    # return line items in table similar to what is seen online: http://www.nhl.com/scores/htmlreports/20212022/PL030234.HTM
    df_dict = {
        "gameId": [],
        "playId": [],
        "strength": [],
        "timestamp": [],
        "event": [],
        "description": [],
        # the on ice should be formatted as "{# offense}_{# defense}_{# goalies}"
        "away_on_ice": [],
        "home_on_ice": [],
        "away_goalie_number": [],
        "home_goalie_number": [],
        "away_pulled_goalie": [],
        "home_pulled_goalie": [],
        "away_del_penalty": [],
        "home_del_penalty": [],
        "away_penalty": [],
        "home_penalty": [],
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
            away_abbrev = abbr_list
        if loc.lower() in home_table.text.lower():
            assert home_abbrev is None
            home_abbrev = abbr_list

    assert away_abbrev is not None and home_abbrev is not None

    # not all games have the id=PL-# format
    # ex game that does: 2021030234
    # ex game that doesn't: 2017020001
    PL_tr = nhl_event_soup.find_all("tr", {"id": re.compile(r"PL-\d")})
    color_tr = nhl_event_soup.find_all(
        "tr", {"class": re.compile(r"(?:even|odd)Color")}
    )

    all_tr = PL_tr if len(PL_tr) >= len(color_tr) else color_tr

    for tr in all_tr:
        df_dict["gameId"].append(game_id)

        line_items = [tag.text for tag in tr.find_all("td", recursive=False)]
        assert len(line_items) == 8
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

        if line_items[6] == "\xa0":
            df_dict["away_on_ice"].append("-1")
            df_dict["away_goalie_number"].append("-1")
            df_dict["away_pulled_goalie"].append("-1")
        else:
            away_positions = [
                c[-1] for c in "".join(line_items[6].split("\n")).split("\xa0")
            ]
            away_numbers = [
                c[:-1] for c in "".join(line_items[6].split("\n")).split("\xa0")
            ]
            df_dict["away_on_ice"].append(
                f'{sum([1 for p in away_positions if p in ["C", "R", "L"]])}_{sum([1 for p in away_positions if p == "D"])}_{sum([1 for p in away_positions if p == "G"])}'
            )
            df_dict["away_goalie_number"].append(
                str(away_numbers[away_positions.index("G")])
                if "G" in away_positions
                else "-1"
            )
            df_dict["away_pulled_goalie"].append("0" if "G" in away_positions else "1")

        if line_items[7] == "\xa0":
            df_dict["home_on_ice"].append("-1")
            df_dict["home_goalie_number"].append("-1")
            df_dict["home_pulled_goalie"].append("-1")
        else:
            home_positions = [
                c[-1] for c in "".join(line_items[7].split("\n")).split("\xa0")
            ]
            home_numbers = [
                c[:-1] for c in "".join(line_items[7].split("\n")).split("\xa0")
            ]
            df_dict["home_on_ice"].append(
                f'{sum([1 for p in home_positions if p in ["C", "R", "L"]])}_{sum([1 for p in home_positions if p == "D"])}_{sum([1 for p in home_positions if p == "G"])}'
            )
            df_dict["home_goalie_number"].append(
                str(home_numbers[home_positions.index("G")])
                if "G" in home_positions
                else "-1"
            )
            df_dict["home_pulled_goalie"].append("0" if "G" in home_positions else "1")

        if df_dict["event"][-1] == "PENL":
            assert not df_dict["description"][-1] == -1
            if df_dict["description"][-1][:3] in away_abbrev:
                df_dict["away_penalty"].append(1)
                df_dict["home_penalty"].append(0)
            elif df_dict["description"][-1][:3] in home_abbrev:
                df_dict["away_penalty"].append(0)
                df_dict["home_penalty"].append(1)
            else:
                print(f'Abbreviation not found is {df_dict["description"][-1][:3]}')
                raise NotImplementedError
        else:
            df_dict["away_penalty"].append(0)
            df_dict["home_penalty"].append(0)

        if df_dict["event"][-1] == "DELPEN":
            assert not df_dict["description"][-1] == -1
            if df_dict["description"][-1][:3] in away_abbrev:
                df_dict["away_del_penalty"].append(1)
                df_dict["home_del_penalty"].append(0)
            elif df_dict["description"][-1][:3] in home_abbrev:
                df_dict["away_del_penalty"].append(0)
                df_dict["home_del_penalty"].append(1)
            else:
                print(f'Abbreviation not found is {df_dict["description"][-1][:3]}')
                raise NotImplementedError
        else:
            df_dict["away_del_penalty"].append(0)
            df_dict["home_del_penalty"].append(0)

        assert all(len(vals) == len(df_dict["gameId"]) for vals in df_dict.values())

    df = pd.DataFrame(df_dict)
    assert not df.empty
    if not len(df.index) == int(df["playId"].to_list()[-1]):
        print(
            f'WARNING The length of the pbp df is {len(df.index)} and the last playId is {df["playId"].to_list()[-1]}.'
        )
    return df


def fetch_to_df_nhl_pbp(game_id):
    pbp_data = nhl_pbp_request(game_id)
    pbp_df = parse_nhl_pbp(pbp_data, game_id)
    return pbp_df


def nhl_home_shifts_request(game_id):
    season_id = f"{str(game_id)[:4]}{int(str(game_id)[:4]) + 1}"
    game_identifier = str(game_id)[-6:]

    try_count = 0
    iterating = True
    while iterating:
        try_count += 1
        home_shifts_info = requests.get(
            f"http://www.nhl.com/scores/htmlreports/{season_id}/TH{game_identifier}.HTM"
        )
        home_shifts_soup = BeautifulSoup(home_shifts_info.content, "html.parser")

        if "404 Not Found" == home_shifts_soup.find("title").text:
            iterating = try_count <= 2
        else:
            return home_shifts_soup

    return None


def nhl_away_shifts_request(game_id):
    season_id = f"{str(game_id)[:4]}{int(str(game_id)[:4]) + 1}"
    game_identifier = str(game_id)[-6:]

    try_count = 0
    iterating = True
    while iterating:
        try_count += 1
        away_shifts_info = requests.get(
            f"http://www.nhl.com/scores/htmlreports/{season_id}/TV{game_identifier}.HTM"
        )
        away_shifts_soup = BeautifulSoup(away_shifts_info.content, "html.parser")

        if "404 Not Found" == away_shifts_soup.find("title").text:
            iterating = try_count <= 2
        else:
            return away_shifts_soup

    return None


def nhl_shifts_request(game_id):
    home_shifts_soup = nhl_home_shifts_request(game_id)
    away_shifts_soup = nhl_away_shifts_request(game_id)

    return home_shifts_soup, away_shifts_soup


def parse_nhl_shifts(nhl_shifts_soup, game_id):
    # return for each player a list of tuples of (period, start shift time, end shift time)
    # idea will be when a period/time is inputted do a double for loop over player and tuples to calculate how long each
    #  player has been on the ice in the game
    assert isinstance(nhl_shifts_soup, BeautifulSoup)

    df_dict = {
        "gameId": [],
        "playerId": [],
        "start_shift": [],
        "end_shift": [],
        "shift_length": [],
        "event": [],
    }

    change_first_name = {
        "alexander": ["alex", "sasha"],
        "alex": ["alexander"],
        "gerald": ["gerry"],
        "gerry": ["gerald"],
        "nick": ["nicholas"],
        "nicholas": ["nick"],
        "christopher": ["chris"],
        "chris": ["christopher"],
        "cal": ["callan", "calvin"],
        "callan": ["cal"],
        "calvin": ["cal"],
        "egor": ["yegor"],
        "yegor": ["egor"],
        "sasha": ["alexander"],
        "william": ["will"],
        "will": ["william"],
    }
    change_last_name = dict()

    # get team name
    team_td = nhl_shifts_soup.find("td", {"class": "teamHeading + border"})
    team_text = team_td.text

    # get team id
    teams_request = requests.get("https://statsapi.web.nhl.com/api/v1/teams").json()
    team_ids = [
        d["id"]
        for d in teams_request["teams"]
        if d["name"].replace("é", "e").lower() == team_text.lower()
    ]
    assert len(team_ids) == 1
    team_id = team_ids[0]

    # get roster
    season_year = f"{str(game_id)[:4]}{int(str(game_id)[:4]) + 1}"
    roster_request = requests.get(
        f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/roster?season={season_year}"
    ).json()
    playername_to_id = {
        d["person"]["fullName"].lower().replace("é", "e"): d["person"]["id"]
        for d in roster_request["roster"]
    }

    all_td_playerHeading = nhl_shifts_soup.find_all(
        "td", {"class": "playerHeading + border"}
    )
    for td_playerHeading in all_td_playerHeading:
        number_player_name = td_playerHeading.text
        player_name = " ".join(number_player_name.split(" ")[1:])
        assert player_name.count(",") == 1
        last_name, first_name = player_name.split(",")
        first_name = first_name.strip().lower()
        last_name = last_name.strip().lower()
        if " ".join([first_name, last_name]) in playername_to_id:
            player_id = playername_to_id[" ".join([first_name, last_name])]
        elif first_name in change_first_name.keys() and any(
            " ".join([fn, last_name]) in playername_to_id
            for fn in change_first_name[first_name]
        ):
            player_id = [
                playername_to_id[" ".join([fn, last_name])]
                for fn in change_first_name[first_name]
                if " ".join([fn, last_name]) in playername_to_id
            ][0]
        elif (
            last_name in change_last_name.keys()
            and " ".join([first_name, change_last_name[last_name]]) in playername_to_id
        ):
            player_id = playername_to_id[
                " ".join([first_name, change_last_name[last_name]])
            ]
        elif (
            first_name in change_first_name.keys()
            and last_name in change_last_name.keys()
            and any(
                " ".join([fn, change_last_name[last_name]]) in playername_to_id
                for fn in change_first_name[first_name]
            )
        ):
            player_id = [
                playername_to_id[" ".join([fn, change_last_name[last_name]])]
                for fn in change_first_name[first_name]
                if " ".join([fn, change_last_name[last_name]]) in playername_to_id
            ][0]
        else:
            player_id = ".".join([first_name, last_name])
            print(
                f'{" ".join([first_name, last_name])} was not found in shift request.'
            )
            print(
                f'Closest found is "{sorted([(k, edit_distance(" ".join([first_name, last_name]), k)) for k in playername_to_id.keys()], key=lambda x: x[1], reverse=False)[0][0]}"'
            )
            assert False

        next_tr = td_playerHeading.find_next("tr").find_next("tr")
        while (
            hasattr(next_tr, "attrs")
            and "class" in next_tr.attrs
            and re.match(r"(?:odd|even)Color", next_tr.attrs["class"][0]) is not None
        ):
            line_info = next_tr.text.split("\n")
            assert len(line_info) == 8
            period = 4 if line_info[2] == "OT" else int(line_info[2])

            # easy enough to change these to counting up just by doing 20 * 3 * 60 - #
            start_shift_elapsed = line_info[3].split("/")[0].strip()
            start_shift_timestamp_elapsed = 20 * 3 * 60 - (
                20 * 60 * (period - 1)
                + 60 * int(start_shift_elapsed.split(":")[0])
                + int(start_shift_elapsed.split(":")[1])
            )
            start_shift_left = line_info[3].split("/")[1].strip()
            start_shift_timestamp_left = (
                20 * (3 - period) * 60
                + 60 * int(start_shift_left.split(":")[0])
                + int(start_shift_left.split(":")[1])
            )

            end_shift_elapsed = line_info[4].split("/")[0].strip()
            end_shift_timestamp_elapsed = 20 * 3 * 60 - (
                20 * 60 * (period - 1)
                + 60 * int(end_shift_elapsed.split(":")[0])
                + int(end_shift_elapsed.split(":")[1])
            )

            end_shift_left = line_info[4].split("/")[1].strip()
            end_shift_timestamp_left = (
                20 * (3 - period) * 60
                + 60 * int(end_shift_left.split(":")[0])
                + int(end_shift_left.split(":")[1])
            )

            shift_length = int(line_info[5].split(":")[0]) * 60 + int(
                line_info[5].split(":")[1]
            )
            if (
                start_shift_timestamp_elapsed - end_shift_timestamp_elapsed
                == shift_length
            ):
                start_shift_timestamp = start_shift_timestamp_elapsed
                end_shift_timestamp = end_shift_timestamp_elapsed
            elif start_shift_timestamp_left - end_shift_timestamp_left == shift_length:
                start_shift_timestamp = start_shift_timestamp_left
                end_shift_timestamp = end_shift_timestamp_left
            elif (
                start_shift_timestamp_left - end_shift_timestamp_left
                == start_shift_timestamp_elapsed - end_shift_timestamp_elapsed
                and start_shift_timestamp_left == start_shift_timestamp_elapsed
            ):
                start_shift_timestamp = start_shift_timestamp_left
                end_shift_timestamp = end_shift_timestamp_left
                shift_length = start_shift_timestamp_left - end_shift_timestamp_left
            else:
                # consider it missing data
                start_shift_timestamp = 0
                end_shift_timestamp = 0
                shift_length = 0

            assert shift_length == (start_shift_timestamp - end_shift_timestamp)

            if not line_info[6] == "\xa0":
                event = line_info[6]
            else:
                event = "0"

            df_dict["gameId"].append(game_id)
            df_dict["playerId"].append(player_id)
            df_dict["start_shift"].append(start_shift_timestamp)
            df_dict["end_shift"].append(end_shift_timestamp)
            df_dict["shift_length"].append(shift_length)
            df_dict["event"].append(event)

            assert all(len(vals) == len(df_dict["gameId"]) for vals in df_dict.values())

            next_tr = next_tr.find_next("tr")

    df = pd.DataFrame(df_dict)
    assert not df.empty
    return df


def fetch_to_df_nhl_shifts(game_id):

    home_shifts_soup, away_shifts_soup = nhl_shifts_request(game_id)
    away_shifts_df = parse_nhl_shifts(away_shifts_soup, game_id)
    home_shifts_df = parse_nhl_shifts(home_shifts_soup, game_id)

    return away_shifts_df, home_shifts_df


if __name__ == "__main__":
    nhl_shifts_request(2021030234)
    # live_data = nhl_live_feed_request(2021030324)
    # live_data_df = parse_nhl_live_feed(live_data)
    #
    # game_id = 2017020001
    # pbp_data = nhl_pbp_request(game_id)
    # pbp_df = parse_nhl_pbp(pbp_data, game_id)

    # home_shifts_soup, away_shifts_soup = nhl_shifts_request(game_id)
    # away_shifts_df = parse_nhl_shifts(away_shifts_soup, game_id)
    # home_shifts_df = parse_nhl_shifts(home_shifts_soup, game_id)
    pass

# https://hackernoon.com/retrieving-hockey-stats-from-the-nhls-undocumented-api-zz3003wrw
# https://gitlab.com/dword4/nhlapi
# https://towardsdatascience.com/nhl-analytics-with-python-6390c5d3206d
# https://www.dataquest.io/blog/python-api-tutorial/
# https://github.com/mhbw/evolving-hockey/blob/master/EH_scrape_functions.R
