import torch
import Revolution
from MultiAgentEnv import MultiAgentEnv
from tqdm import tqdm
from Game import Game
import MongoCSVLogging
import numpy as np
from collections import deque
import Player

use_ppo = True
reactive_training = False

# this is now the mathematical mean of the number of rounds per epoch
rounds_of_game_per_epoch = 10
episodic_memory_length = rounds_of_game_per_epoch * 20
env = MultiAgentEnv(
    num_agents=2,
    num_teams=2,
    failed_revolution_penalty=0,
    advantage_bonus=0,
    use_ppo=use_ppo,
    state_dim=episodic_memory_length,
)

rl_agent = env.teams[0].team_players[0]
predefined_agent = Player.Player(
    player_team=1, use_model=False, use_ppo=False, state_dim=None
)
# # replace the rl agent with a predefined agent
# env.teams[1].team_players[0] = predefined_agent
rl_agents = [rl_agent]
predefined_agents = [predefined_agent]
env.agents = Revolution.get_flatten_player_list(teams=env.teams)


epsilon = 0.5
epochs = 180000
revolution_percentage = 0.15
total_rewards = 0
wealth_disparity = 0
display_round_information_within_one_epoch = False
# log the inner epoch data every N epochs
log_inner_epoch_data_interval = 50
# if there are more than 100 logs stored, push it to the database
push_to_db_interval_length = rounds_of_game_per_epoch * 10
agent_score_dicts = []
team_score_dicts = []


def action_space_to_one_hot(action_space):
    one_hot = [0] * 3
    for action in action_space:
        one_hot[action] = 1
    return one_hot


advantaged_agent = None
disadvantaged_agent = None

for epoch in tqdm(range(epochs)):
    env.reset()
    # if epoch <= 5000:
    #     epsilon = 0.5
    # if epoch > 7500:
    #     epsilon = epsilon * 0.99995
    if epoch >= epochs / 2:
        epsilon = 0.1

    advantaged_agents_action_statistics = [0, 0, 0]
    disadvantaged_agents_action_statistics = [0, 0, 0]

    rounds_of_game_per_epoch = np.random.geometric(p=0.1)

    advantaged_agent_episodic_memory = deque(maxlen=episodic_memory_length)
    disadvantaged_agent_episodic_memory = deque(maxlen=episodic_memory_length)
    # initialize the deque with -1
    for _ in range(episodic_memory_length):
        advantaged_agent_episodic_memory.append(-1)
        disadvantaged_agent_episodic_memory.append(-1)

    done = 0

    for round_num in range(rounds_of_game_per_epoch):

        if round_num == rounds_of_game_per_epoch - 1:
            done = 1

        if (
            epoch % log_inner_epoch_data_interval == 0
            and display_round_information_within_one_epoch
        ):
            print("Round {} of epoch {}".format(round_num, epoch))

        env.new_turn(
            reactive_training=reactive_training,
            rl_agents=rl_agents,
            predefined_agents=predefined_agents,
        )
        advantage_agents = []
        disadvantage_agents = []

        for pair in env.pairings:
            # agent_1 should always being a RL agent, agent_2 being a predefined agent, in the case of reactive training
            agent_1, agent_2 = pair
            agent_1_team = agent_1.player_team
            agent_2_team = agent_2.player_team
            agent_1_current_privilege_from_team = (
                Revolution.get_current_privilege_from_team(
                    player=agent_1, teams=env.teams
                )
            )
            agent_2_current_privilege_from_team = (
                Revolution.get_current_privilege_from_team(
                    player=agent_2, teams=env.teams
                )
            )

            game = Game(player_1=agent_1, player_2=agent_2)
            game.init_players_for_game(
                player_1_current_privilege_from_team=agent_1_current_privilege_from_team,
                player_2_current_privilege_from_team=agent_2_current_privilege_from_team,
            )

            if (
                advantaged_agent is None
                and disadvantaged_agent is None
                and reactive_training is False
            ):
                # randomly choose the advantaged and disadvantaged agents
                advantaged_agent = agent_1 if np.random.rand() > 0.5 else agent_2
                disadvantaged_agent = (
                    agent_2 if advantaged_agent == agent_1 else agent_1
                )
            elif reactive_training is True:
                # in the case of reactive training ONLY
                advantaged_agent = agent_1
                disadvantaged_agent = agent_2

            advantage_agents.append(advantaged_agent)
            disadvantage_agents.append(disadvantaged_agent)

            # one_hot_advantaged_action_space = action_space_to_one_hot(advantaged_agent.player_current_action_space)

            # relative_team_rewards = Revolution.get_relative_teams_reward(player=advantaged_agent, teams=env.teams)

            # advantaged_agent_state = one_hot_advantaged_action_space + relative_team_rewards

            advantaged_agent_state = advantaged_agent_episodic_memory

            advantaged_agent_state = (
                torch.tensor(advantaged_agent_state)
                .float()
                .reshape(
                    episodic_memory_length,
                )
            )

            if not use_ppo:
                advantaged_agent_action = advantaged_agent.model.take_action(
                    state=advantaged_agent_state,
                    restricted=True,
                    epsilon=epsilon,
                    action_mask=[1, 1, 0],
                )

            if use_ppo:
                advantaged_agent_action = advantaged_agent.model.select_action(
                    advantaged_agent_state
                )

            advantaged_illegal_current_move = advantaged_agent.take_action(
                action=advantaged_agent_action
            )

            if len(advantaged_agent.replay_buffer) != 0:
                # appending to the deque replay buffer
                advantaged_agent.replay_buffer[-1][3] = advantaged_agent_state

            advantaged_agent.blocking_players_action(
                player_to_block=disadvantaged_agent, action_to_block=2
            )

            # one_hot_disadvantaged_action_space = action_space_to_one_hot(
            #     disadvantaged_agent.player_current_action_space)

            # disadvantaged_agent_state = (one_hot_disadvantaged_action_space +
            #                              Revolution.get_relative_teams_reward(player=disadvantaged_agent,
            #                                                                   teams=env.teams))

            disadvantaged_agent_state = disadvantaged_agent_episodic_memory

            disadvantaged_agent_state = (
                torch.tensor(disadvantaged_agent_state)
                .float()
                .reshape(
                    episodic_memory_length,
                )
            )

            if disadvantaged_agent.model is not None:
                if not use_ppo:
                    disadvantaged_agent_action = disadvantaged_agent.model.take_action(
                        state=disadvantaged_agent_state,
                        restricted=True,
                        epsilon=epsilon,
                        action_mask=[1, 1, 0],
                    )

                if use_ppo:
                    disadvantaged_agent_action = (
                        disadvantaged_agent.model.select_action(
                            disadvantaged_agent_state
                        )
                    )
            else:
                if disadvantaged_agent is predefined_agent:
                    # # disadvantaged_agent_action = advantaged_agent_action
                    # # implement a tit-for-tat strategy
                    # if disadvantaged_agent_episodic_memory[-1] == -1:  # First round
                    #     disadvantaged_agent_action = 0  # Start by cooperating
                    # else:
                    #     # Mirror the opposite of the opponent's previous move
                    #     disadvantaged_agent_action = 1 if disadvantaged_agent_episodic_memory[-1] == 0 else 0
                    disadvantaged_agent_action = 0

            disadvantaged_agent_illegal_move = disadvantaged_agent.take_action(
                action=disadvantaged_agent_action
            )

            if len(disadvantaged_agent.replay_buffer) != 0:
                # appending to the deque replay buffer
                disadvantaged_agent.replay_buffer[-1][3] = disadvantaged_agent_state

            game.update_player_current_reward(
                first_hand=advantaged_agent,
                second_hand=disadvantaged_agent,
                advantaged_agent_illegal_move=advantaged_illegal_current_move,
                disadvantaged_agent_illegal_move=disadvantaged_agent_illegal_move,
            )

            # advantaged_agent_episodic_memory[round_num] = disadvantaged_agent_action
            advantaged_agent_episodic_memory.append(disadvantaged_agent_action)
            # disadvantaged_agent_episodic_memory[round_num] = advantaged_agent_action
            disadvantaged_agent_episodic_memory.append(advantaged_agent_action)

            # previous_advantaged_agent_action = advantaged_agent_action
            # previous_disadvantaged_agent_action = disadvantaged_agent_action

            advantaged_agent_reward = advantaged_agent.player_current_reward
            disadvantaged_agent_reward = disadvantaged_agent.player_current_reward

            # # amplify the advantaged players reward received, good or bad
            # advantaged_agent.player_current_reward *= env.advantage_bonus

            advantaged_agent.replay_buffer.append(
                [
                    advantaged_agent_state,
                    advantaged_agent.player_current_action,
                    advantaged_agent.player_current_reward,
                    advantaged_agent_state,
                    done,
                ]
            )

            disadvantaged_agent.replay_buffer.append(
                [
                    disadvantaged_agent_state,
                    disadvantaged_agent.player_current_action,
                    disadvantaged_agent.player_current_reward,
                    disadvantaged_agent_state,
                    done,
                ]
            )

            advantaged_agent.update_player_history()
            disadvantaged_agent.update_player_history()

        if epoch % log_inner_epoch_data_interval == 0:
            if display_round_information_within_one_epoch:
                print("\n\n")
                agent_score_dict = env.logging_agent_score_board(
                    epoch=epoch, round_num=round_num, great_chaos=False, display=True
                )
            else:
                agent_score_dict = env.logging_agent_score_board(
                    epoch=epoch, round_num=round_num, great_chaos=False, display=False
                )
            agent_score_dicts.append(agent_score_dict)

        # revolution = env.resolve_revolutions()
        # if revolution:
        #     if epoch % log_inner_epoch_data_interval == 0:
        #         if display_round_information_within_one_epoch:
        #             print("-------------------REVOLUTION-------------------")
        #             agent_score_dict = env.logging_agent_score_board(epoch=epoch, round_num=round_num, great_chaos=True, display=True)
        #         else:
        #             agent_score_dict = env.logging_agent_score_board(epoch=epoch, round_num=round_num, great_chaos=True, display=False)
        #         agent_score_dicts.append(agent_score_dict)

        # iterate through all teams and update their histories after one round_num of the game
        team_historical_reward_list = []
        for team in env.teams:
            team_historical_total_rewards = team.update_team_reward()
            team_historical_reward_list.append(team_historical_total_rewards)

        for team in env.teams:
            team.update_team_privilege(
                team_historical_reward_list=team_historical_reward_list
            )

        if epoch % log_inner_epoch_data_interval == 0:
            if display_round_information_within_one_epoch:
                team_score_dict = env.logging_team_score_board(
                    epoch, round_num, display=True
                )
            else:
                team_score_dict = env.logging_team_score_board(
                    epoch, round_num, display=False
                )
            team_score_dicts.append(team_score_dict)

        env.conclude_trial(
            round_num=round_num,
            advantaged_agents=advantage_agents,
            disadvantaged_agents=disadvantage_agents,
            done=done,
            use_ppo=use_ppo,
            both_agents_play_as_restricted=True,
        )

        for agent in advantage_agents:
            advantaged_agents_action_statistics[agent.player_current_action] += 1

        for agent in disadvantage_agents:
            disadvantaged_agents_action_statistics[agent.player_current_action] += 1

    env.update_all_models(
        epoch=epoch,
        use_ppo=use_ppo,
        update_epoch_interval=100,
    )

    env.recording_log(
        epoch=epoch,
        epsilon=epsilon,
        total_epoch=epochs,
        rounds_of_game_per_epoch=rounds_of_game_per_epoch,
        advantaged_agents_action_statistics=advantaged_agents_action_statistics,
        disadvantaged_agents_action_statistics=disadvantaged_agents_action_statistics,
        logging_into_csv=True,
    )

    if epoch % 100 == 0:
        MongoCSVLogging.push_team_logs_to_db(
            team_logging_dicts=team_score_dicts, logging_into_csv=True
        )
        MongoCSVLogging.push_player_logs_to_db(
            player_logging_dicts=agent_score_dicts, logging_into_csv=True
        )
        # clean the logs to free up memory
        team_score_dicts = []
        agent_score_dicts = []

if __name__ == "__main__":
    pass
