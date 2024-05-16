import Revolution
import random
import math
import pandas as pd
import json
import MongoCSVLogging
import subprocess


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_wealth_disparity(teams_score_statistics_table: list) -> float:
    # calculate the wealth disparity
    #  = (max - min) / max
    return max(teams_score_statistics_table) - min(teams_score_statistics_table)


class MultiAgentEnv:
    def __init__(
        self,
        use_ppo: bool,
        num_agents: int,
        num_teams: int,
        advantage_bonus: float,
        failed_revolution_penalty: float,
        state_dim: int,
        team_init_privilege_upper_bound: int = 0,
        team_init_privilege_lower_bound: int = 0,
    ):
        self.num_agents = num_agents
        self.num_teams = num_teams
        self.init_privilege_per_team = [
            random.randint(
                team_init_privilege_lower_bound, team_init_privilege_upper_bound
            )
            for _ in range(self.num_teams)
        ]
        self.teams = Revolution.generate_teams(
            number_of_players_per_team=int(self.num_agents / self.num_teams),
            init_privilege_per_team=self.init_privilege_per_team,
            use_ppo=use_ppo,
            state_dim=state_dim,
        )
        self.agents = Revolution.get_flatten_player_list(teams=self.teams)
        self.pairings = []
        self.revolution_count = 0
        self.agent_rewards = [0] * self.num_agents
        self.advantage_bonus = advantage_bonus
        self.failed_revolution_penalty = failed_revolution_penalty
        # this is to store logging dataframe before it got pushed to the MongoDB
        self.logging = None
        self.great_chaos_count = 0

    def reset(self):
        """
        Reset the environment by re-initializing the teams
        Emptying pairings, revolution count, and agent rewards
        """
        self.pairings = []
        self.revolution_count = 0
        self.agent_rewards = [0] * self.num_agents
        for team_idx in range(len(self.teams)):
            team = self.teams[team_idx]
            init_privilege_of_current_team = random.randint(0, 0)
            team.reset_team(init_privilege=init_privilege_of_current_team)
            self.init_privilege_per_team[team_idx] = init_privilege_of_current_team

    def get_total_rewards_from_all_agents(self) -> int:
        """
        Get the total rewards from all agents
        """
        teams_current_total_reward_list = (
            Revolution.get_teams_current_total_reward_list(teams=self.teams)
        )
        return sum(teams_current_total_reward_list)

    def new_turn(
        self,
        reactive_training: bool = False,
        rl_agents: list = None,
        predefined_agents: list = None,
    ):
        if reactive_training:
            # create a list of tuple where each rl agent is paired with every predefined agent once
            self.pairings = []
            for rl_agent in rl_agents:
                for predefined_agent in predefined_agents:
                    self.pairings.append((rl_agent, predefined_agent))
        else:
            # get the player pairs
            self.pairings = Revolution.random_pair_players(teams=self.teams)

    def check_revolution(
        self, revolution_percentage: float = 0.5, revolution_cost: float = 1
    ) -> (bool, int):
        great_chaos = False
        redistribution = None
        flatten_player_list = Revolution.get_flatten_player_list(teams=self.teams)
        self.revolution_count = (
            Revolution.count_occurrence_of_revolution_within_current_round(
                players=flatten_player_list
            )
        )
        if self.revolution_count > self.num_agents * revolution_percentage:
            total_rewards = self.get_total_rewards_from_all_agents()
            redistributed_reward_for_each_agent = math.floor(
                total_rewards / self.num_agents
            )
            redistribution = redistributed_reward_for_each_agent // self.num_agents
            great_chaos = True
        self.revolution_count = 0

        return great_chaos, redistribution

    def resolve_revolutions(self):
        great_chaos, redistribution = self.check_revolution()
        if great_chaos:
            self.great_chaos_count += 1
            for agent in self.agents:
                if len(agent.replay_buffer) > 0:
                    agent.replay_buffer[-1][2] = (
                        redistribution - agent.player_current_reward
                    )
                    agent.player_current_reward = redistribution
        else:
            for agent in self.agents:
                if len(agent.replay_buffer) > 0:
                    if agent.replay_buffer[-1][1] == 2:
                        # a failed revolution gets punished
                        agent.replay_buffer[-1][2] = self.failed_revolution_penalty

        return great_chaos

    def conclude_trial(
        self,
        round_num: int,
        advantaged_agents: list,
        disadvantaged_agents: list,
        done: int,
        use_ppo: bool,
        both_agents_play_as_restricted: bool,
    ):

        for agent in advantaged_agents:
            if agent.model is not None:
                if use_ppo:
                    if round_num >= 1:
                        (
                            advantaged_agent_state,
                            advantaged_agent_current_action,
                            advantaged_agent_current_reward,
                            advantaged_agent_state,
                            adv_done,
                        ) = agent.replay_buffer[-2]
                        agent.model.buffer.rewards.append(
                            advantaged_agent_current_reward
                        )
                        agent.model.buffer.is_terminals.append(adv_done)
                    if done:
                        (
                            advantaged_agent_state,
                            advantaged_agent_current_action,
                            advantaged_agent_current_reward,
                            advantaged_agent_state,
                            adv_done,
                        ) = agent.replay_buffer[-1]
                        agent.model.buffer.rewards.append(
                            advantaged_agent_current_reward
                        )
                        agent.model.buffer.is_terminals.append(adv_done)
                else:
                    if round_num >= 1:
                        if both_agents_play_as_restricted:
                            agent.model.push_to_buffer(
                                agent.replay_buffer[-2], restricted=True
                            )
                        else:
                            agent.model.push_to_buffer(
                                agent.replay_buffer[-2], restricted=False
                            )
                    if done:
                        if both_agents_play_as_restricted:
                            agent.model.push_to_buffer(
                                agent.replay_buffer[-1], restricted=True
                            )
                        else:
                            agent.model.push_to_buffer(
                                agent.replay_buffer[-1], restricted=False
                            )

        for agent in disadvantaged_agents:
            if agent.model is not None:
                if use_ppo:
                    if round_num >= 1:
                        (
                            disadvantaged_agent_state,
                            disadvantaged_agent_current_action,
                            disadvantaged_agent_current_reward,
                            disadvantaged_agent_state,
                            disadvantaged_agent_done,
                        ) = agent.replay_buffer[-2]
                        agent.model.buffer.rewards.append(
                            disadvantaged_agent_current_reward
                        )
                        agent.model.buffer.is_terminals.append(disadvantaged_agent_done)
                    if done:
                        (
                            disadvantaged_agent_state,
                            disadvantaged_agent_current_action,
                            disadvantaged_agent_current_reward,
                            disadvantaged_agent_state,
                            disadvantaged_agent_done,
                        ) = agent.replay_buffer[-1]
                        agent.model.buffer.rewards.append(
                            disadvantaged_agent_current_reward
                        )
                        agent.model.buffer.is_terminals.append(disadvantaged_agent_done)

                else:
                    if round_num >= 1:
                        agent.model.push_to_buffer(
                            agent.replay_buffer[-2], restricted=True
                        )
                    if done:
                        agent.model.push_to_buffer(
                            agent.replay_buffer[-1], restricted=True
                        )

    def update_all_models(
        self, epoch: int, use_ppo: bool, update_epoch_interval: int = 100
    ):
        if epoch > 20:
            for agent in self.agents:
                if agent.model is not None:
                    if use_ppo:
                        pass
                    else:
                        loss1 = agent.model.train_td(restricted=True)
                        loss2 = agent.model.train_td(restricted=False)

        if epoch % update_epoch_interval == 0:
            if use_ppo:
                for agent in self.agents:
                    if agent.model is not None:
                        agent.model.update()
            else:
                for agent in self.agents:
                    if agent.model is not None:
                        agent.model.update_target()

    def logging_agent_score_board(
        self, epoch: int, round_num: int, great_chaos: bool, display: bool = False
    ) -> dict:
        # construct a table using dataframe with all players' attributes: player name, player team, player current
        # privilege, player current action, player current blocking action, player current reward
        player_name_list = []
        player_team_list = []
        player_current_privilege_list = []
        player_current_action_list = []
        player_current_blocking_action_list = []
        player_current_reward_list = []
        player_current_pairing_id_list = []
        for agent in self.agents:
            player_name_list.append(agent.player_name)
            player_team_list.append(agent.player_team)
            player_current_privilege_list.append(agent.player_current_privilege)
            player_current_action_list.append(int(agent.player_current_action))
            player_current_blocking_action_list.append(agent.player_currently_blocking)
            player_current_reward_list.append(agent.player_current_reward)
            # find the index of the current agent in the pairing list
            for pairing_id in range(len(self.pairings)):
                if (
                    self.pairings[pairing_id][0] == agent
                    or self.pairings[pairing_id][1] == agent
                ):
                    player_current_pairing_id_list.append(pairing_id)

        player_score_dict = {
            "Epoch": epoch,
            "Round": round_num,
            "Name": player_name_list,
            "Team": player_team_list,
            "Privilege": player_current_privilege_list,
            "Action": player_current_action_list,
            "Blocking": player_current_blocking_action_list,
            "Reward": player_current_reward_list,
            "Pairing": player_current_pairing_id_list,
            "Great Chaos": great_chaos,
        }
        if display:
            player_score_board = pd.DataFrame(player_score_dict)

            # rank the player score board by the reward
            player_score_board = player_score_board.sort_values(
                by=["Reward"], ascending=False
            )

            print(player_score_board.to_markdown())

        return player_score_dict

    def logging_team_score_board(
        self, epoch: int, round_num: int, display: bool = False
    ) -> dict:
        # construct a table using dataframe with all teams current attributes: team name, team current total reward
        team_name_list = []
        team_current_total_reward_list = []
        team_current_total_privilege_list = []
        team_historical_total_rewards_list_for_display = []
        team_historical_total_reward_list = []
        for team in self.teams:
            team_name_list.append(team.team_id)
            team_current_total_reward_list.append(team.team_current_total_reward)
            team_current_total_privilege_list.append(team.team_privilege)
            team_historical_total_reward_list.append(team.team_historical_total_rewards)
            if len(team.team_historical_total_rewards) <= 3:
                team_historical_total_rewards_list_for_display.append(
                    team.team_historical_total_rewards
                )
            else:
                team_historical_total_rewards_list_for_display.append(
                    team.team_historical_total_rewards[-3:]
                )

        team_historical_total_reward_list = [
            sum(team_historical_total_reward)
            for team_historical_total_reward in team_historical_total_reward_list
        ]

        team_score_dict = {
            "Epoch": epoch,
            "Round": round_num,
            "Team": team_name_list,
            "Reward": team_current_total_reward_list,
            "Privilege": team_current_total_privilege_list,
            "History": team_historical_total_rewards_list_for_display,
            "Accumulated Reward": team_historical_total_reward_list,
        }

        if display:
            team_score_board = pd.DataFrame(team_score_dict)

            # rank the dataframe with reward
            team_score_board = team_score_board.sort_values(
                by=["Reward"], ascending=False
            )
            print(team_score_board.to_markdown())

        return team_score_dict

    def get_action_blocking_and_illegal_action_statistics(self) -> (list, list, int):
        # get the flatten player list
        flatten_player_list = Revolution.get_flatten_player_list(teams=self.teams)

        action_statistics_table = [0] * 3
        blocking_statistics_table = [0] * 3
        illegal_action_count = 0

        for player in flatten_player_list:
            # iterate players history
            for history in player.player_history:
                # get the action and blocking action
                player_current_action = history["player_current_action"]
                player_current_blocking_action = history["player_currently_blocking"]
                illegal_action_count += history["player_current_illegal_action"]

                # update the action statistics table
                action_statistics_table[player_current_action] += 1

                # update the blocking statistics table\
                if player_current_blocking_action is not None:
                    blocking_statistics_table[player_current_blocking_action] += 1

        return action_statistics_table, blocking_statistics_table, illegal_action_count

    def recording_log(
        self,
        epoch: int,
        epsilon: float,
        total_epoch: int,
        rounds_of_game_per_epoch: int,
        advantaged_agents_action_statistics: list,
        disadvantaged_agents_action_statistics: list,
        logging_into_csv: bool = False,
    ):
        # construct a dataframe table with the following: epoch, epsilon, action_statistics_table,
        # blocking_statistics_table, great_chaos_count, teams_score_statistics_table and wealth_disparity
        action_statistics_table, blocking_statistics_table, illegal_action_count = (
            self.get_action_blocking_and_illegal_action_statistics()
        )
        teams_score_statistics_table = [
            sum(team.team_historical_total_rewards) for team in self.teams
        ]

        logging_table = {
            "Action": action_statistics_table,
            "Advantaged Agents Action Statistics": advantaged_agents_action_statistics,
            "Blocking": blocking_statistics_table,
            "Disadvantaged Agents Action Statistics": disadvantaged_agents_action_statistics,
            "Illegal Action": illegal_action_count,
            "Great Chaos": self.great_chaos_count,
            "Teams": teams_score_statistics_table,
            "Total Reward": sum(teams_score_statistics_table),
            "Wealth Disparity": get_wealth_disparity(teams_score_statistics_table),
            "Epsilon": epsilon,
            "metaData": None,
        }

        if self.logging is None:
            # create a metadata dictionary consist of the number of teams, number of agents, the total epoch,
            # and for each epoch how many rounds of game are there

            metadata = {
                "Number of Teams": self.num_teams,
                "Number of Agents": self.num_agents,
                "Total Epoch": total_epoch,
                "Rounds of Game per Epoch": rounds_of_game_per_epoch,
                "Current Git Sha": get_git_revision_short_hash(),
            }

            # convert the metadata dictionary into a json
            metadata = json.dumps(metadata)

            # convert the metadata into a string
            metadata = str(metadata)

            metadata_df = {
                "Action": 0,
                "Advantaged Agents Action Statistics": 0,
                "Blocking": 0,
                "Disadvantaged Agents Action Statistics": 0,
                "Illegal Action": 0,
                "Great Chaos": 0,
                "Teams": 0,
                "Total Reward": 0,
                "Wealth Disparity": 0,
                "Epsilon": 0,
                "metaData": metadata,
            }

            self.logging = [metadata_df, logging_table]

        else:
            self.logging.append(logging_table)

        self.great_chaos_count = 0

        # push the logging table to the MongoDB
        if epoch % 100 == 0:
            MongoCSVLogging.push_epoch_logs_to_db(
                epoch_logging_dicts=self.logging, logging_into_csv=logging_into_csv
            )
            # set logging to an empty list make it not None and release the memory of the logging table
            self.logging = []
