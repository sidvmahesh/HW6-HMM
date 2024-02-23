import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    hmm = HiddenMarkovModel(mini_hmm["observation_states"], mini_hmm["hidden_states"], mini_hmm["prior_p"], mini_hmm["transition_p"], mini_hmm["emission_p"])
    forward_result =  hmm.forward(mini_input["observation_state_sequence"])

    # Now, test the forward result with the ground truth
    input_observation_states = mini_input["observation_state_sequence"]
    num_obs = len(input_observation_states)
    num_states = len(hmm.hidden_states)
    probs = np.zeros((num_obs, num_states))
    for i in range(num_states):
        probs[0, i] = hmm.prior_p[i] * hmm.emission_p[i][hmm.observation_states_dict[input_observation_states[0]]]
    for t in range(1, num_obs):
        for j in range(num_states):
            for i in range(num_states):
                probs[t, j] += probs[t-1, i] * hmm.transition_p[i][j] * hmm.emission_p[j][hmm.observation_states_dict[input_observation_states[t]]]
    ground_truth_result = np.sum(probs[num_obs - 1, :])

    assert ground_truth_result == forward_result
    
    #print("hmm.viterbi(mini_input[observation_state_sequence]): ", hmm.viterbi(mini_input["observation_state_sequence"]))
    #print(mini_input["best_hidden_state_sequence"])
    
    #Now, test the viterbi result
    viterbi_result = hmm.viterbi(mini_input["observation_state_sequence"])
    assert len(viterbi_result) == len(mini_input["best_hidden_state_sequence"])
    for i in range(len(viterbi_result)):
        assert viterbi_result[i] == mini_input["best_hidden_state_sequence"][i]



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')
    hmm = HiddenMarkovModel(full_hmm["observation_states"], full_hmm["hidden_states"], full_hmm["prior_p"], full_hmm["transition_p"], full_hmm["emission_p"])
    print("\nforward_result: ", hmm.forward(full_input["observation_state_sequence"]))
    #print("hmm.forward(mini_input[observation_state_sequence]): ", hmm.forward(full_input["observation_state_sequence"]))
    #print("hmm.viterbi(mini_input[observation_state_sequence]): ", hmm.viterbi(full_input["observation_state_sequence"]))
    #print(full_input["best_hidden_state_sequence"])
    viterbi_result = hmm.viterbi(full_input["observation_state_sequence"])
    assert len(viterbi_result) == len(full_input["best_hidden_state_sequence"])
    for i in range(len(viterbi_result)):
        assert viterbi_result[i] == full_input["best_hidden_state_sequence"][i]












