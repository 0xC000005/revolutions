import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def label_cooperates(action_pair):
    action_pair = eval(action_pair)
    """Labels action pairs as 'C' (cooperate) or 'D'(defect)."""
    if action_pair in [[0, 0], [1, 1]]:
        return "C"
    else:
        return "D"


def label_actions(action_pair):
    action_pair = eval(action_pair)
    if action_pair == [0, 0]:
        return 1
    elif action_pair == [0, 1]:
        return 2
    elif action_pair == [1, 0]:
        return 3
    elif action_pair == [1, 1]:
        return 4


def analyze_reciprocity(df):
    """Analyzes reciprocity patterns in the provided DataFrame."""
    # add a label column
    df["label"] = df["Action"].apply(label_cooperates)
    # drop all column except for the label, action and name
    df = df[["label", "Action", "Name"]]

    # if a label C has its previous action pair being the same as the current action pair, meaning the current label
    # should be D
    for i in range(1, len(df)):
        if (
            df.loc[i, "label"] == "C"
            and df.loc[i - 1, "label"] == "C"
            and df.loc[i, "Action"] == df.loc[i - 1, "Action"]
        ):
            df.loc[i, "label"] = "D"

    count_CC = 0
    count_CD = 0
    count_DC = 0
    count_DD = 0

    # count the number of CC, CD, DC, DD by linear scan
    for i in range(1, len(df)):
        if df.loc[i - 1, "label"] == "C" and df.loc[i, "label"] == "C":
            count_CC += 1
        elif df.loc[i - 1, "label"] == "C" and df.loc[i, "label"] == "D":
            count_CD += 1
        elif df.loc[i - 1, "label"] == "D" and df.loc[i, "label"] == "C":
            count_DC += 1
        else:
            count_DD += 1

    # convert all counts to percentage
    count_CC = count_CC / len(df)
    count_CD = count_CD / len(df)
    count_DC = count_DC / len(df)
    count_DD = count_DD / len(df)

    # plot the percentage of CC, CD, DC, DD using a bar plot
    plt.bar(["CC", "CD", "DC", "DD"], [count_CC, count_CD, count_DC, count_DD])
    plt.title("Reciprocity")
    plt.xlabel("Action Pair")
    plt.ylabel("Percentage")
    plt.savefig("Reciprocity.png")
    # add caption: the agent prefers keep cooperating with the TFT opponent

    # add name of the bar under each bar
    plt.show()


def analysis_action_pair(df):
    """
    There are 4 different action pairs per round: 0,0; 0,1; 1,0; 1,1
    We will calculate the percentage of each action pair through the entire game
    """
    # add 00, 01, 10, 11 columns to the dataframe
    df["00"] = 0
    df["01"] = 0
    df["10"] = 0
    df["11"] = 0

    # apply the function to the dataframe
    df["00"] = df["Action"].apply(lambda x: label_actions(x) == 1)
    df["01"] = df["Action"].apply(lambda x: label_actions(x) == 2)
    df["10"] = df["Action"].apply(lambda x: label_actions(x) == 3)
    df["11"] = df["Action"].apply(lambda x: label_actions(x) == 4)

    # calculate the percentage of each action pair
    count_00 = df["00"].sum() / len(df)
    count_01 = df["01"].sum() / len(df)
    count_10 = df["10"].sum() / len(df)
    count_11 = df["11"].sum() / len(df)

    # plot the percentage of each action pair using a bar plot
    plt.bar(["BB", "BS", "SB", "SS"], [count_00, count_01, count_10, count_11])
    plt.title(
        "Results from Bach vs Stravinsky when opponent always choose 1 (Stravinsky)"
    )
    plt.xlabel("Action Pair\nAgent prefers Stravinsky")
    plt.ylabel("Percentage")
    # make the fig size larger
    plt.gcf().set_size_inches(10, 6)
    plt.savefig("action_frequency" + filename[7:-4] + ".png")
    plt.show()


def analysis_defect_after_cooperation(df):
    count_00 = df["00"].sum() / len(df)
    count_01 = df["01"].sum() / len(df)
    count_10 = df["10"].sum() / len(df)
    count_11 = df["11"].sum() / len(df)

    # add a cooperation label
    df["label"] = 0

    # apply the label cooperation to the dataframe
    df["label"] = df["Action"].apply(label_cooperates)

    count_01_after_C = 0
    count_10_after_C = 0

    # check how many D is immediately after C
    for i in range(1, len(df)):
        if df.loc[i - 1, "label"] == "C" and df.loc[i, "label"] == "D":
            if df.loc[i, "Action"] == "[0, 1]":
                count_01_after_C += 1
            elif df.loc[i, "Action"] == "[1, 0]":
                count_10_after_C += 1

    count_01_after_C = count_01_after_C / len(df)
    count_10_after_C = count_10_after_C / len(df)

    # plot the percentage of count 10 and 01 vs those that are immediately after a C
    plt.bar(
        ["BS", "SB", "BS after C", "SB after C"],
        [count_01, count_10, count_01_after_C, count_10_after_C],
    )
    plt.title(
        "how many percentage of SB/BS is immediately after a successful coordination"
    )
    plt.xlabel("Action Pair\nAgent prefers Stravinsky, C is either BB or SS")
    plt.ylabel("Percentage")
    plt.gcf().set_size_inches(10, 6)
    plt.savefig("action_dynamics" + filename[7:-4] + ".png")
    plt.show()


# Load the CSV
filename = "Players_20240310_224922.csv"
df = pd.read_csv(filename)
# analyze_reciprocity(df)
analysis_action_pair(df)
# analysis_defect_after_cooperation(df)

if __name__ == "__main__":
    pass
