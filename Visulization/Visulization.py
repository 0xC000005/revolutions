import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROLLING_AVG = 1000

import ast


def analyze_player_log(csv_file, player_name):
    """
    Reads a CSV table of player data, isolates a single player's information,
    and visualizes trends.

    Args:
        csv_file (str): Path to the CSV file.
        player_name (str): Name of the player to analyze.
    """

    df = pd.read_csv(csv_file)
    player_data = {
        "Epoch": [],
        "Round": [],
        "Privilege": [],
        "Action": [],
        "Blocking": [],
        "Reward": [],
        "Pairing": [],
        "Great Chaos": [],
    }

    for index, row in df.iterrows():
        all_names = ast.literal_eval(row["Name"])

        if player_name in all_names:
            name_index = all_names.index(player_name)

            player_data["Epoch"].append(row["Epoch"])
            player_data["Round"].append(row["Round"])
            player_data["Privilege"].append(
                ast.literal_eval(row["Privilege"])[name_index]
            )
            player_data["Action"].append(ast.literal_eval(row["Action"])[name_index])
            player_data["Blocking"].append(
                ast.literal_eval(row["Blocking"])[name_index]
            )
            player_data["Reward"].append(ast.literal_eval(row["Reward"])[name_index])
            player_data["Pairing"].append(ast.literal_eval(row["Pairing"])[name_index])
            player_data["Great Chaos"].append(row["Great Chaos"])

    # Convert to a DataFrame
    player_df = pd.DataFrame(player_data)

    # Calculate Running Averages (Optional)
    window_size = 10
    player_df["Action_Avg"] = player_df["Action"].rolling(window=window_size).mean()
    player_df["Blocking_Avg"] = player_df["Blocking"].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(20, 12))
    # increase the space between subplots
    plt.subplots_adjust(hspace=1.2)

    # Subplot 1: Action
    plt.subplot(3, 1, 1)
    plt.plot(
        player_df.index, player_df["Action_Avg"], label="Action (Running Avg)"
    )  # Use index for x-axis
    plt.xlabel("Data Point Index")  # Update x-axis label
    plt.ylabel("Action")
    plt.title(f"Action Trends for {player_name}")
    plt.legend()

    # Subplot 2: Blocking
    plt.subplot(3, 1, 2)
    plt.plot(
        player_df.index, player_df["Blocking_Avg"], label="Blocking (Running Avg)"
    )  # Use index for x-axis
    plt.xlabel("Data Point Index")  # Update x-axis label
    plt.ylabel("Blocking")
    plt.title(f"Blocking Trends for {player_name}")
    plt.legend()

    # Subplot 3: Privilege
    plt.subplot(3, 1, 3)
    plt.plot(
        player_df.index, player_df["Privilege"], label="Privilege"
    )  # Use index for x-axis
    plt.xlabel("Data Point Index")  # Update x-axis label
    plt.ylabel("Privilege")
    plt.title(f"Privilege Trends for {player_name}")
    plt.legend()

    # save the figure
    plt.savefig("player_trends" + csv_file[5:-4] + ".png", dpi=100)

    plt.show()


def visualize_epoch_wise_log():
    filename = "../temp/backup/Epochs_20240304_230549.csv"
    logging_data_full = pd.read_csv(filename)
    # drop first 3 rows
    logging_data_full = logging_data_full.drop([0, 1, 2])

    # create a 2 rows 1 column figures
    fig, axs = plt.subplots(8, 1)
    # action statistics is the Action column of the logging_data_full
    action_statistics = logging_data_full["Action"]
    # convert the action statistics element from a string into a list of int
    action_statistics = [list(map(int, i[1:-1].split(", "))) for i in action_statistics]
    blocking_statistics = logging_data_full["Blocking"]
    # convert the blocking statistics element from a string into a list of int
    blocking_statistics = [
        list(map(int, i[1:-1].split(", "))) for i in blocking_statistics
    ]
    team_scores_statistics = logging_data_full["Teams"]
    # convert the team scores statistics element from a string into a list of int
    team_scores_statistics = [
        list(map(int, i[1:-1].split(", "))) for i in team_scores_statistics
    ]
    # convert Advantaged Agents Action Statistics into a list of int
    advantaged_agents_action_statistics = logging_data_full[
        "Advantaged Agents Action Statistics"
    ]
    advantaged_agents_action_statistics = [
        list(map(int, i[1:-1].split(", "))) for i in advantaged_agents_action_statistics
    ]
    # convert Disadvantaged Agents Action Statistics into a list of int
    disadvantaged_agents_action_statistics = logging_data_full[
        "Disadvantaged Agents Action Statistics"
    ]
    disadvantaged_agents_action_statistics = [
        list(map(int, i[1:-1].split(", ")))
        for i in disadvantaged_agents_action_statistics
    ]

    # society_distribution_table = get_the_society_distribution_statistics(logging_data_full)

    # plot the change of the first 3 rows over time
    # Plot the action statistics over time
    axs[0].plot(
        pd.Series([i[0] for i in advantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Cooperate",
    )
    axs[0].plot(
        pd.Series([i[1] for i in advantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Defect",
    )
    axs[0].plot(
        pd.Series([i[2] for i in advantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Revolution",
    )
    axs[0].set_title("Advantaged Agents Action Statistic Over Time")
    # set the legend position to be outside the plot
    axs[0].legend(loc="right")
    # add a x-axis label
    axs[0].set_xlabel("Epoch")

    # plot the change of the first 3 rows over time
    # Plot the action statistics over time
    axs[1].plot(
        pd.Series([i[0] for i in disadvantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Cooperate",
    )
    axs[1].plot(
        pd.Series([i[1] for i in disadvantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Defect",
    )
    axs[1].plot(
        pd.Series([i[2] for i in disadvantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Revolution",
    )
    axs[1].set_title("Disadvantaged Agents Action Statistic Over Time")
    # set the legend position to be outside the plot
    axs[1].legend(loc="right")
    # add a x-axis label
    axs[1].set_xlabel("Epoch")

    # plot the change of the last 3 rows over time
    # Plot the blocking statistics over time
    axs[2].plot(
        pd.Series([i[0] for i in blocking_statistics]).rolling(ROLLING_AVG).mean(),
        label="Blocking Cooperate",
    )
    axs[2].plot(
        pd.Series([i[1] for i in blocking_statistics]).rolling(ROLLING_AVG).mean(),
        label="Blocking Defect",
    )
    axs[2].plot(
        pd.Series([i[2] for i in blocking_statistics]).rolling(ROLLING_AVG).mean(),
        label="Blocking Revolution",
    )
    axs[2].set_title("Blocking Statistics Over Time")
    axs[2].legend(loc="right")
    axs[2].set_xlabel("Epoch")

    revolution_statistics = logging_data_full["Great Chaos"]
    # convert the revolution statistics into a int list
    revolution_statistics = [int(i) for i in revolution_statistics]
    # plot the moving average of the revolution of the logging_data_full over time
    axs[3].plot(pd.Series(revolution_statistics).rolling(ROLLING_AVG).mean())
    axs[3].set_title("Revolution Occurrence Over Time")
    axs[3].set_xlabel("Epoch")

    society_distribution_table = []

    for team_scores in team_scores_statistics:
        one_hot_team_scores = [0] * 5
        rich_team_counter = 0
        for i in range(len(team_scores)):
            if abs(team_scores[i]) >= 100:
                rich_team_counter += 1
        one_hot_team_scores[rich_team_counter] = 1
        society_distribution_table.append(one_hot_team_scores)

    # plot the society distribution statistics over time
    axs[4].plot(
        pd.Series([i[0] for i in society_distribution_table])
        .rolling(ROLLING_AVG)
        .mean(),
        label="0",
    )
    axs[4].plot(
        pd.Series([i[1] for i in society_distribution_table])
        .rolling(ROLLING_AVG)
        .mean(),
        label="1",
    )
    axs[4].plot(
        pd.Series([i[2] for i in society_distribution_table])
        .rolling(ROLLING_AVG)
        .mean(),
        label="2",
    )
    axs[4].plot(
        pd.Series([i[3] for i in society_distribution_table])
        .rolling(ROLLING_AVG)
        .mean(),
        label="3",
    )
    axs[4].plot(
        pd.Series([i[4] for i in society_distribution_table])
        .rolling(ROLLING_AVG)
        .mean(),
        label="4",
    )
    axs[4].set_title("How many teams are rich (>=100) at the end of each epoch?")
    axs[4].legend(loc="right")
    axs[4].set_xlabel("Epoch")

    # illegal_action_statistics = logging_data_full['Illegal Action']
    # # convert the revolution statistics into a a int list
    # illegal_action_statistics = [int(i) for i in illegal_action_statistics]
    # # plot the moving average of the revolution of the logging_data_full over time
    # axs[4].plot(pd.Series(illegal_action_statistics).rolling(ROLLING_AVG).mean())
    # axs[4].set_title('Frequency of choosing illegal action')
    # axs[4].set_xlabel('Epoch')

    axs[5].plot(
        pd.Series([i[0] for i in team_scores_statistics]).rolling(ROLLING_AVG).mean(),
        label="0",
    )
    axs[5].plot(
        pd.Series([i[1] for i in team_scores_statistics]).rolling(ROLLING_AVG).mean(),
        label="1",
    )
    axs[5].set_title("Team accumulative scores after each epoch?")
    axs[5].legend(loc="right")
    axs[5].set_xlabel("Epoch")

    total_reward = logging_data_full["Total Reward"]
    # plot the moving average of the revolution of the logging_data_full over time
    axs[6].plot(total_reward.rolling(ROLLING_AVG).mean())
    axs[6].set_title("Total reward after each epoch")
    axs[6].set_xlabel("Epoch")

    epsilons = logging_data_full["Epsilon"]
    # plot the moving average of the revolution of the logging_data_full over time
    axs[7].plot(epsilons.rolling(ROLLING_AVG).mean())
    axs[7].set_title("Epsilon for each epoch")
    axs[7].set_xlabel("Epoch")

    # increase the space between the two subplots
    fig.subplots_adjust(hspace=1.2)
    #
    # increase the size of the figure
    fig.set_size_inches(20, 12)

    plt.show()

    # save the figure
    fig.savefig("action_blocking_statistics" + filename[5:-4] + ".png", dpi=100)


if __name__ == "__main__":
    visualize_epoch_wise_log()
