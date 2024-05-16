import Team
import Player
import Game
import random
import math
import copy
import pickle


def get_flatten_player_list(teams: list):
    """
    Flatten the list of players from all teams

    :param teams: a list of teams
    :return: a list of players from all teams
    """
    return [player for team in teams for player in team.team_players]


def random_pair_players(teams: list):
    """
    Return a list of randomly paired players from different teams ensuring all players are paired once.
    The idea is to flatten the list of players from all teams, randomly select two players from the flattened list,
    and remove the two players from the flattened list. Repeat the process until there are only two players left in
    the flattened list. If the last two players are from the same team, then recursively call the function. Otherwise,
    the last two players are considered to be a valid pair.

    :param teams: a list of teams, each containing a 'team_players' list
    :return: a list of player pairs
    """

    # TODO: This function is not very efficient, since it has to recursively generate random pairing schemas until a
    #  valid a valid one is found. A more efficient way is to generate a decision tree of all possible pairing
    #  schemas, and randomly select one from the decision tree.

    # # create a deep copy of the teams for reshuffling purposes
    # teams_copy = copy.deepcopy(teams)
    # # # load the teams from pickle for fast deep copy
    # # with open('teams.pickle', 'rb') as handle:
    # #     teams_copy = pickle.load(handle)
    #
    #
    # player_pairs_copy = []
    # while len(teams_copy[0].team_players) > 0:
    #     # assume 2 players per team, 4 teams; we pair 4 players from each team once, until there is no player left in '
    #     # all teams
    #     # TODO: we cannot pair odd number of teams
    #     remaining_teams_to_pair = teams_copy.copy()
    #     random.shuffle(remaining_teams_to_pair)
    #
    #     while len(remaining_teams_to_pair) > 0:
    #         # randomly pop a team from the remaining teams to pair
    #         team1 = remaining_teams_to_pair.pop(random.randint(0, len(remaining_teams_to_pair) - 1))
    #         # randomly pop another team from the remaining teams to pair
    #         team2 = remaining_teams_to_pair.pop(random.randint(0, len(remaining_teams_to_pair) - 1))
    #
    #         # randomly pop a player from team1
    #         random.shuffle(team1.team_players)
    #         player1 = team1.team_players.pop(random.randint(0, len(team1.team_players) - 1))
    #         # randomly pop another player from team2
    #         random.shuffle(team2.team_players)
    #         player2 = team2.team_players.pop(random.randint(0, len(team2.team_players) - 1))
    #
    #         # append the player pair to the player pairs list
    #         player_pairs_copy.append([player1, player2])
    #
    # # create a player pair from the player pair copy but with actual players from the teams
    # player_pairs = []
    # for player_pair_copy in player_pairs_copy:
    #     # get the actual players from the teams using team.get_player_from_team_by_name()
    #     player1 = teams[player_pair_copy[0].player_team].get_player_from_team_by_name(
    #         player_name=player_pair_copy[0].player_name)
    #     player2 = teams[player_pair_copy[1].player_team].get_player_from_team_by_name(
    #         player_name=player_pair_copy[1].player_name)
    #     player_pairs.append([player1, player2])
    #
    # return player_pairs

    num_player_per_team = len(teams[0].team_players)
    num_player_already_paired_per_team = 0
    paired_players = []
    player_pairs = []
    while num_player_already_paired_per_team < num_player_per_team:
        num_player_already_paired_per_team += 1
        num_of_teams = len(teams)
        teams_selected = []
        while len(teams_selected) < num_of_teams:
            teams_unselected = [team for team in teams if team not in teams_selected]
            # randomly select two teams from the teams, add the teams to the selected list
            team1 = teams_unselected.pop(random.randint(0, len(teams_unselected) - 1))
            team2 = teams_unselected.pop(random.randint(0, len(teams_unselected) - 1))
            teams_selected.append(team1)
            teams_selected.append(team2)

            player_unselected_team1 = [
                player for player in team1.team_players if player not in paired_players
            ]
            player_unselected_team2 = [
                player for player in team2.team_players if player not in paired_players
            ]

            # randomly select two players from the two teams, add the players to the paired list
            player1 = player_unselected_team1.pop(
                random.randint(0, len(player_unselected_team1) - 1)
            )
            player2 = player_unselected_team2.pop(
                random.randint(0, len(player_unselected_team2) - 1)
            )
            paired_players.append(player1)
            paired_players.append(player2)

            # append the player pair to the player pairs list
            player_pairs.append([player1, player2])

    return player_pairs


def get_current_privilege_from_team(player: Player.Player, teams: list):
    """
    Get the current privilege of the player from the team

    :param player: the player
    :param teams: a list of teams
    :return: the current privilege of the player
    """
    for team in teams:
        if player in team.team_players:
            return team.team_privilege


def generate_teams(
    number_of_players_per_team: int,
    init_privilege_per_team: list,
    use_ppo: bool,
    state_dim: int,
):
    number_of_teams = len(init_privilege_per_team)
    # check if the number of players * the number of teams is divisible by 2
    if number_of_players_per_team * number_of_teams % 2 != 0:
        raise ValueError(
            "The number of players * the number of teams must be divisible by 2"
        )

    # iterate through the teams, creating players, with the team ID equal to the index of the team privilege in the
    # initial privilege list
    teams = []
    for i in range(number_of_teams):
        players = []
        for j in range(number_of_players_per_team):
            players.append(
                Player.Player(
                    player_team=i,
                    use_ppo=use_ppo,
                    state_dim=state_dim,
                    use_model=True,
                )
            )
        teams.append(
            Team.Team(
                team_id=i,
                team_init_privilege=init_privilege_per_team[i],
                players=players,
            )
        )

    return teams


def get_teams_current_total_reward_list(teams: list):
    """
    Get the points of all teams

    :param teams: a list of teams
    :return: a list of points of all teams
    """
    return [team.team_current_total_reward for team in teams]


def get_relative_teams_reward(player: "Player", teams: list):
    # get the current total reward of all teams
    teams_current_total_reward_list = get_teams_current_total_reward_list(teams=teams)

    # if teams current total reward list is a zero list, check the privilege of all teams instead
    team_reward_all_zero = True
    for reward in teams_current_total_reward_list:
        if reward != 0:
            team_reward_all_zero = False
            break

    if team_reward_all_zero:
        teams_current_total_reward_list = [team.team_privilege for team in teams]

    # normalize the teams current total reward list
    min_val = min(teams_current_total_reward_list)
    max_val = max(teams_current_total_reward_list)
    normalized_team_rewards = []
    if max_val == min_val:
        normalized_team_rewards = [
            1 if x == max_val else 0 for x in teams_current_total_reward_list
        ]
    else:
        normalized_team_rewards = [
            (x - min_val) / (max_val - min_val) for x in teams_current_total_reward_list
        ]

    # Compute the log difference
    if max_val == min_val:
        log_diff = 0  # or some other predefined value
    else:
        log_diff = math.log(max_val - min_val)

    # Reorder the list
    relative_team_reward = normalized_team_rewards.pop(player.player_team)
    normalized_team_rewards.sort(reverse=True)
    reordered_scores = (
        [relative_team_reward] + normalized_team_rewards + [log_diff / 10]
    )  # think about diff norm

    return reordered_scores


def count_occurrence_of_revolution_within_current_round(players: list):
    """
    Count the occurrence of revolution within the current round_num

    :param Teams: a list of teams
    :return: the number of revolutions within the current round_num
    """

    revolution_counter = 0
    for player in players:
        if player.player_current_action == 2:
            revolution_counter += 1

    return revolution_counter


if __name__ == "__main__":
    # initialize the teams
    teams = generate_teams(
        number_of_players_per_team=10, init_privilege_per_team=[1, 1, 1, 1]
    )

    # # dump the teams in pickle for fast deep copy
    # with open('teams.pickle', 'wb') as handle:
    #     pickle.dump(teams, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # get the player pairs
    for i in range(100):
        player_pairs1 = random_pair_players(teams=teams)
    # player_pairs2 = random_pair_players(teams=teams)
    # player_pairs3 = random_pair_players(teams=teams)
    # player_pairs4 = random_pair_players(teams=teams)
    # player_pairs5 = random_pair_players(teams=teams)
    # player_pairs6 = random_pair_players(teams=teams)
    # player_pairs7 = random_pair_players(teams=teams)

    # for pair in player_pairs:
    #     player_1 = pair[0]
    #     player_2 = pair[1]
    #
    #     # create a game
    #     game = Game.Game(player_1=player_1, player_2=player_2)
    #
    #     # re-initialize the players before each game with the current privilege from the team
    #     player_1_current_privilege_from_team = get_current_privilege_from_team(player=player_1, teams=teams)
    #     player_2_current_privilege_from_team = get_current_privilege_from_team(player=player_2, teams=teams)
    #
    #     game.play(player_1_current_privilege_from_team=player_1_current_privilege_from_team,
    #               player_2_current_privilege_from_team=player_2_current_privilege_from_team,
    #               user_input=True)
    #
    # # iterate through all teams and update their histories after one round_num of the game
    # for team in teams:
    #     team.update_team_reward()
