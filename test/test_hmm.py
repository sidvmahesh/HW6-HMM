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
    #print(mini_hmm)
    hmm = HiddenMarkovModel(mini_hmm["observation_states"], mini_hmm["hidden_states"], mini_hmm["prior_p"], mini_hmm["transition_p"], mini_hmm["emission_p"])
    print("hmm.forward(mini_input[observation_state_sequence]): ", hmm.forward(mini_input["observation_state_sequence"]))
    #print(mini_input)
    print("hmm.viterbi(mini_input[observation_state_sequence]): ", hmm.viterbi(mini_input["observation_state_sequence"]))
    print(mini_input["best_hidden_state_sequence"])






    
   
    pass



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    pass













