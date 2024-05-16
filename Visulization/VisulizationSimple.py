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
    filename = "Epochs_20240310_225153.csv"
    logging_data_full = pd.read_csv(filename)
    # drop first 3 rows
    logging_data_full = logging_data_full.drop([0, 1, 2])

    # create a 2 rows 1 column figures
    fig, axs = plt.subplots(4, 1)
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
        label="Option 0 (Bach)",
    )
    axs[0].plot(
        pd.Series([i[1] for i in advantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Option 1 (Stravinsky, prefer)",
    )
    axs[0].set_title("Player 1 Action Statistic Over Time")
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
        label="Option 0 (Bach, prefer)",
    )
    axs[1].plot(
        pd.Series([i[1] for i in disadvantaged_agents_action_statistics])
        .rolling(ROLLING_AVG)
        .mean(),
        label="Option 1 (Stravinsky)",
    )
    axs[1].set_title("Player 2 Action Statistic Over Time")
    # set the legend position to be outside the plot
    axs[1].legend(loc="right")
    # add a x-axis label
    axs[1].set_xlabel("Epoch")

    axs[2].plot(
        pd.Series([i[0] for i in team_scores_statistics]).rolling(ROLLING_AVG).mean(),
        label="Player 1",
    )
    axs[2].plot(
        pd.Series([i[1] for i in team_scores_statistics]).rolling(ROLLING_AVG).mean(),
        label="Player 2",
    )
    axs[2].set_title("Players cumulative scores after each epoch")
    axs[2].legend(loc="right")
    axs[2].set_xlabel("Epoch")

    total_reward = logging_data_full["Total Reward"]
    # plot the moving average of the revolution of the logging_data_full over time
    axs[3].plot(total_reward.rolling(ROLLING_AVG).mean())
    axs[3].set_title("Society Total reward after each epoch")
    axs[3].set_xlabel("Epoch")

    # epsilons = logging_data_full["Epsilon"]
    # axs[4].plot(epsilons.rolling(ROLLING_AVG).mean())
    # axs[4].set_title("Epsilon for each epoch")
    # axs[4].set_xlabel("Epoch")

    # increase the space between the two subplots
    fig.subplots_adjust(hspace=1.2)
    # increase the size of the figure
    fig.set_size_inches(20, 12)

    # add title to the figure
    fig.suptitle(
        "Reactive training in battle of sexes using PPO (NPC always choose 0)",
        fontsize=20,
    )

    # add caption: infinite horizon with a mean horizon of 10

    plt.figtext(
        0.5,
        0.01,
        "Infinite horizon with a mean horizon of 10",
        ha="center",
        fontsize=18,
    )
    plt.show()

    # save the figure
    fig.savefig("action_blocking_statistics" + filename[5:-4] + ".png", dpi=100)


if __name__ == "__main__":
    visualize_epoch_wise_log()
