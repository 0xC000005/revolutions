class Team:
    """
    Team class

    Attributes:
        team_id (int): Team's ID
        team_privilege (int): Team's privilege
        team_players (list): Team's players
        team_current_total_reward (int): Team's current total reward
        team_historical_total_rewards (list): Team's historical total rewards

    Methods:
        update_team_current_total_reward: update team's current total reward by summing up the current rewards of all
        players
        update_team_privilege: Update team's privilege
        update_team_historical_total_rewards: Update team's historical total rewards
        __str__: Print out the team's information
    """

    def __init__(self, team_id: int, team_init_privilege: int, players: list):
        self.team_id = team_id
        self.team_privilege = team_init_privilege
        self.team_players = players
        self.team_current_total_reward = 0
        self.team_historical_total_rewards = []

    def reset_team(self, init_privilege: int):
        self.team_privilege = init_privilege
        self.team_current_total_reward = 0
        self.team_historical_total_rewards = []
        for player in self.team_players:
            player.reset_player()

    # Explanation: total reward is the sum of rewards of all players in the team in one rounds of game
    # The history of these rewards are stored in team_historical_total_rewards
    # The privilege of the team is the sum of the historical total rewards of all players in the team or 0 if the sum is
    # negative
    def update_team_current_total_reward(self):
        self.team_current_total_reward = sum(
            [player.player_current_reward for player in self.team_players]
        )

    def update_team_historical_total_rewards(self):
        # TODO: currently the teams historical total rewards are the cumulative rewards of all players over time, and
        #  the privilege are normalized list of sum of cumulative rewards of all teams. Need confirmation from the PI

        self.team_historical_total_rewards.append(self.team_current_total_reward)

    def update_team_privilege(self, team_historical_reward_list: list):
        # self.team_privilege = max(sum(self.team_historical_total_rewards), 0)

        # self.team_privilege = sum(self.team_historical_total_rewards)

        team_total_reward_list = [
            sum(team_historical_reward)
            for team_historical_reward in team_historical_reward_list
        ]

        # normalized the team total reward list
        min_val = min(team_total_reward_list)
        max_val = max(team_total_reward_list)
        if max_val == min_val:
            normalized_team_rewards = [
                1 if x == max_val else 0 for x in team_total_reward_list
            ]
        else:
            normalized_team_rewards = [
                (x - min_val) / (max_val - min_val) for x in team_total_reward_list
            ]

        # return the current privilege of the team
        self.team_privilege = normalized_team_rewards[self.team_id]

    def update_team_reward(self):
        self.update_team_current_total_reward()
        self.update_team_historical_total_rewards()
        return self.team_historical_total_rewards

    def get_player_from_team_by_name(self, player_name: str):
        for player in self.team_players:
            if player.player_name == player_name:
                return player

    def __str__(self):
        # iterate through the players in the team and print out their information
        players_str = ""
        for player in self.team_players:
            players_str += str(player) + "\n"

        return (
            "Team ID: {}\nTeam Privilege: {}\nTeam Players: \n{}\nTeam Current Total Reward: {}\n"
            "Team Historical Total Rewards: {}\n".format(
                self.team_id,
                self.team_privilege,
                players_str,
                self.team_current_total_reward,
                self.team_historical_total_rewards,
            )
        )

    def __repr__(self):
        # just return the team ID
        return "Team ID: {}".format(self.team_id)
