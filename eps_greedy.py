from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def eps_greedy(
    heroes: Heroes, 
    eps: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform epsilon-greedy action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param eps: The epsilon value for exploration vs. exploitation.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: The average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """
    
    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes    # Initial action values
    n_hero_quests = [0] * num_heroes
    rew_record = []                       # Rewards at each timestep
    avg_ret_record = []                   # Average reward up to each timestep
    tot_reg_record = []                   # Total regret up to each timestep
    opt_action_record = []                # Percentage of optimal actions selected
    
    total_rewards = 0
    total_regret = 0

    optimal_hero_index = np.argmax([hero['true_success_probability'] for hero in heroes.heroes])
    optimal_reward = heroes.heroes[optimal_hero_index]['true_success_probability']
    
    for t in range(heroes.total_quests):
        if np.random.rand() < eps:
            chosen_hero_index = np.random.randint(0, num_heroes - 1)
        else:
            chosen_hero_index = np.argmax(values)
        
        reward = heroes.attempt_quest(chosen_hero_index)
        rew_record.append(reward)

        n_hero_quests[chosen_hero_index] += 1
        total_rewards += reward

        if n_hero_quests[chosen_hero_index] > 1:
            values[chosen_hero_index] += (reward - values[chosen_hero_index]) / n_hero_quests[chosen_hero_index]
        else:
            values[chosen_hero_index] = reward  # First quest for this hero
        
        avg_ret_record.append(total_rewards / (t + 1))

        regret = optimal_reward - reward
        total_regret += regret
        tot_reg_record.append(total_regret)

        if chosen_hero_index == optimal_hero_index:
            opt_action_record.append(1)  # Optimal hero selected
        else:
            opt_action_record.append(0)  # Non-optimal hero selected
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record


if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])


    # Test various epsilon values
    eps_values = [0.2, 0.1, 0.01, 0.]
    results_list = []
    for eps in eps_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=eps_greedy, 
                                                                    eps=eps, init_value=0.0)
        
        results_list.append({
            'exp_name': f'eps={eps}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='Epsilon-Greedy Experiment Results On Various Epsilons', 
                       results_folder='results', pdf_name='epsilon_greedy_various_epsilons.pdf')


    # Test various initial value settings with eps=0.0
    init_values = [0.0, 0.5, 1]
    results_list = []
    for init_val in init_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=eps_greedy, 
                                                                    eps=0.0, init_value=init_val)
        
        results_list.append({
            'exp_name': f'init_val={init_val}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })
    
    save_results_plots(results_list, plot_title='Epsilon-Greedy Experiment Results On Various Initial Values',
                       results_folder='results', pdf_name='epsilon_greedy_various_init_values.pdf')
    