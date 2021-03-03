# Internal
import typing as T

# External
import numpy as np
from scipy.sparse.linalg import expm


def q_matrix_k1(
    user_count: int,
    internal_fake_transmission_rate: float,
    external_fake_transmission_rate: float,
    internal_genuine_transmission_rate: float,
    external_genuine_transmission_rate: float,
) -> np.ndarray:
    """
    Generates Q matrix for the k1 problem for any rates and user count
    """

    # Initialize a matrix O with size user_count + 1
    # Each state represents the number of users with fake_news on their feed
    q_matrix = np.zeros([user_count + 1, user_count + 1])

    # Loop to fill the matrix
    for i in range(user_count + 1):
        # Get infected count
        I = i
        # Get susceptible count
        S = user_count - i

        # Rates of good and bad news, used to update the diagonal of Q matrix
        g_tax = 0
        b_tax = 0

        # Check boundaries
        if i - 1 >= 0:
            # Set endogenous transmission for good news
            q_matrix[i, i - 1] = (S * I * internal_genuine_transmission_rate) / user_count
            # Add exogenous transmission rates for good news
            q_matrix[i, i - 1] += I * external_genuine_transmission_rate
            # Update good rate to calc diagonal latter
            g_tax = q_matrix[i, i - 1]

        # Check boundaries
        if i < user_count - 1:
            # Set endogenous transmission for bad news
            q_matrix[i, i + 1] = (S * I * internal_fake_transmission_rate) / user_count
            # Add exogenous transmission rates for bad news
            q_matrix[i, i + 1] += S * external_fake_transmission_rate
            # Update good rate to calc diagonal latter
            b_tax = q_matrix[i, i + 1]

        # Update diagonal based on good and bad transition rates
        q_matrix[i, i] = -(g_tax + b_tax)
    return q_matrix


def markov_timeline_probability_matrix(
    q_matrix: np.ndarray,
    initial_infected: int,
    simulation_time: float,
    simulation_steps: int = 100,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Generate matrix with probability timeline for each state and the timeline time steps
    """

    # Initialize empty matrix to be filled
    timeline_probability_matrix = np.zeros([simulation_steps + 1, q_matrix.shape[0]])
    timeline_timesteps = np.zeros([simulation_steps + 1])

    # Calculate the time increment for each step
    time_inc = simulation_time / simulation_steps

    # Initialize the first time step to be one time step in the future
    elapsed_time = time_inc

    # Loop to generate a probability timeline step for each possible state
    for step in range(simulation_steps + 1):
        # Progress tracker
        print(f"\rprogress {step}/{simulation_steps}    ", end="")

        # Calculate the probability distribution after 'elapsed_time' using expm
        trans = expm(q_matrix * elapsed_time)

        # Add probabilities for this timestep to our result
        timeline_probability_matrix[int(step), :] = trans[initial_infected, :]
        timeline_timesteps[step] = elapsed_time

        # Update elapsed time variable
        elapsed_time += time_inc

    return timeline_timesteps, timeline_probability_matrix


def evolution_from_markov(transient: np.ndarray) -> np.ndarray:
    """Generate evolution.

    Plot and compare with the infected and susceptible count of other methods.

    """
    # Initialize the evolution result matrix
    evolution = np.zeros(transient.shape)

    # Iterate calculating the infected user amount for each time step
    for j in range(1, transient.shape[1]):
        evolution[:, j] = j * transient[:, j]

    # Join the user amount for each of the infected stages,
    # to have an amount of infected for each time step
    return np.sum(evolution, axis=1)


__all__ = (
    "q_matrix_k1",
    "markov_timeline_probability_matrix",
    "evolution_from_markov",
)
