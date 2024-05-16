"""
All utilities for logging and loading data from MongoDB
"""

import pymongo
import datetime
import pandas as pd

datastr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

epoch_csv_name = "Epochs_" + datastr + ".csv"
player_csv_name = "Players_" + datastr + ".csv"
team_csv_name = "Teams_" + datastr + ".csv"

if_first_time_writing_epoch_csv = True
logging_into_mongoDB = False

player_header_already_written_flag = False
team_header_already_written_flag = False


if logging_into_mongoDB:
    # connect to the MongoDB
    client = pymongo.MongoClient("localhost", 27017)

    # create a new database
    db = client["Revolution_" + datastr]

    # create collections: Players, Teams, Epochs
    players = db["Players"]
    teams = db["Teams"]
    epochs = db["Epochs"]


# push a list of epoch logging DataFrame to the Epochs collection
def push_epoch_logs_to_db(
    epoch_logging_dicts: list[dict], logging_into_csv: bool = False
):
    if logging_into_mongoDB:
        epochs.insert_many(epoch_logging_dicts)
    if logging_into_csv:
        epoch_logging_df = pd.DataFrame(epoch_logging_dicts)
        global if_first_time_writing_epoch_csv
        if if_first_time_writing_epoch_csv:
            epoch_logging_df.to_csv(epoch_csv_name, mode="w", header=True, index=False)
            if_first_time_writing_epoch_csv = False
        epoch_logging_df.to_csv(epoch_csv_name, mode="a", header=False, index=False)


# push a list of player logging DataFrame to the Players collection
def push_player_logs_to_db(
    player_logging_dicts: list[dict], logging_into_csv: bool = False
):
    if logging_into_mongoDB:
        players.insert_many(player_logging_dicts)
    if logging_into_csv:
        player_logging_df = pd.DataFrame(player_logging_dicts)
        global player_header_already_written_flag
        if player_header_already_written_flag:
            player_logging_df.to_csv(
                player_csv_name, mode="a", header=False, index=False
            )
        else:
            player_logging_df.to_csv(
                player_csv_name, mode="a", header=True, index=False
            )
            player_header_already_written_flag = True


# push a list of team logging DataFrame to the Teams collection
def push_team_logs_to_db(
    team_logging_dicts: list[dict], logging_into_csv: bool = False
):
    if logging_into_mongoDB:
        teams.insert_many(team_logging_dicts)
    if logging_into_csv:
        team_logging_df = pd.DataFrame(team_logging_dicts)
        global team_header_already_written_flag
        if team_header_already_written_flag:
            team_logging_df.to_csv(team_csv_name, mode="a", header=False, index=False)
        else:
            team_logging_df.to_csv(team_csv_name, mode="a", header=True, index=False)
            team_header_already_written_flag = True
