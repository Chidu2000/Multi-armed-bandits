from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def ucb(
    heroes: Heroes, 
    c: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Upper Confidence Bound (UCB) action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param c: The exploration coefficient that balances exploration vs. exploitation.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes   # Initial action values
    n_hero_quests = [0] * num_heroes
    rew_record = []                      # Rewards at each timestep
    avg_ret_record = []                  # Average reward up to each timestep
    tot_reg_record = []                  # Total regret up to each timestep
    opt_action_record = []               # Percentage of optimal actions selected
    
    total_rewards = 0
    total_regret = 0

    optimal_hero_index = np.argmax([hero['true_success_probability'] for hero in heroes.heroes])
    optimal_reward = heroes.heroes[optimal_hero_index]['true_success_probability']

    for t in range(heroes.total_quests):
        ucb_values = []
        for i in range(num_heroes):
            if n_hero_quests[i] == 0:
                ucb_value = float('inf')
            else:
                ucb_value = values[i] + c * np.sqrt(np.log(t + 1) / n_hero_quests[i])
            ucb_values.append(ucb_value)
        
        chosen_hero_index = np.argmax(ucb_values)
        
        reward = heroes.attempt_quest(chosen_hero_index)

        n_hero_quests[chosen_hero_index] += 1

        values[chosen_hero_index] += (reward - values[chosen_hero_index]) / n_hero_quests[chosen_hero_index]

        total_rewards += reward
        total_regret += (optimal_reward - reward)

        rew_record.append(reward)
        avg_ret_record.append(total_rewards / (t + 1))
        tot_reg_record.append(total_regret)
        opt_action_record.append(1 if chosen_hero_index == optimal_hero_index else 0)
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various c values
    c_values = [0.0, 0.5, 2.0]
    results_list = []
    for c in c_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=ucb, 
                                                                    c=c, init_value=0.0)
        
        results_list.append({
            'exp_name': f'c={c}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='UCB Experiment Results On Various C Values', 
                       results_folder='results', pdf_name='ucb_various_c_values.pdf')