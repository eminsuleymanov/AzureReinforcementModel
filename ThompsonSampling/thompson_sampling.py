import numpy as np # array yaratmaq ucun
import matplotlib.pyplot as plt # vizualization
import pandas as pd # datani importing
import random

def load_dataset(file_path):
    """
    Load the dataset from the specified file path.
    """
    return pd.read_csv(file_path)

def thompson_sampling(dataset, N, d):
    """
    Implement Thompson Sampling.
    
    Args:
        dataset: The dataset containing rewards.
        N: Number of rounds (rows).
        d: Number of ads (columns).
    
    Returns:
        ads_selected: List of ads selected in each round.
        total_reward: Total reward achieved.
    """
    ads_selected = []  # Initialize an empty list to track selected ads
    numbers_of_rewards_1 = [0] * d  # Count of rewards 1 for each ad
    numbers_of_rewards_0 = [0] * d  # Count of rewards 0 for each ad
    total_reward = 0  # Track total reward

    for n in range(N):
        ad = 0
        max_random = 0
        for i in range(d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                ad = i
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_rewards_1[ad] += 1
        else:
            numbers_of_rewards_0[ad] += 1
        total_reward += reward

    return ads_selected, total_reward

def plot_histogram(ads_selected, d):
    """
    Visualize the results with a histogram.
    
    Args:
        ads_selected: List of ads selected in each round.
        d: Number of ads (columns).
    """
    plt.figure(figsize=(10, 6))
    plt.hist(ads_selected, bins=np.arange(1, d + 2) - 0.5, rwidth=0.8, edgecolor="black")
    plt.title('Histogram of Ads Selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad was selected')
    plt.xticks(range(1, d + 1)) 
    plt.show()

def main():
    # Parameters
    dataset_path = 'Ads_CTR_Optimisation.csv'
    N = 10000  # Number of rounds
    d = 10     # Number of ads

    # Load dataset
    dataset = load_dataset(dataset_path)

    # Apply Thompson Sampling
    ads_selected, total_reward = thompson_sampling(dataset, N, d)

    # Adjust ads_selected to start from 1 instead of 0
    ads_selected = [ad + 1 for ad in ads_selected]

    # Print total reward
    print(f"Total Reward: {total_reward}")

    # Plot results
    plot_histogram(ads_selected, d)

if __name__ == "__main__":
    main()
