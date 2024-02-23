import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        # Step 1. Initialize variables
        num_obs = len(input_observation_states)
        num_states = len(self.hidden_states)
        # Step 2. Calculate probabilities
        probs = np.zeros((num_obs, num_states))
        # Initialize start probabilities
        for i in range(num_states):
            probs[0, i] = self.prior_p[i] * self.emission_p[i][self.observation_states_dict[input_observation_states[0]]]
        for t in range(1, num_obs):
            for j in range(num_states):
                for i in range(num_states):
                    probs[t, j] += probs[t-1, i] * self.transition_p[i][j] * self.emission_p[j][self.observation_states_dict[input_observation_states[t]]]
        # Step 3. Return final probability 
        return np.sum(probs[num_obs - 1, :])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))
        # Step 2. Calculate Probabilities
        # Step 2: Initialize Variables
        viterbi_table = [[0.0 for i in range(len(self.hidden_states))] for i in range(len(decode_observation_states))]
        backpointer = [[0 for i in range(len(self.hidden_states))] for i in range(len(decode_observation_states))]
        # Step 3: Calculate Probabilities
        for t in range(len(decode_observation_states)):
            for s in range(len(self.hidden_states)):
                if t == 0:
                    viterbi_table[t][s] = self.prior_p[s] * self.emission_p[s][self.observation_states_dict[decode_observation_states[t]]]
                else:
                    max_prob = max(viterbi_table[t-1][prev_s] * self.transition_p[prev_s][s] for prev_s in range(len(self.hidden_states)))
                    viterbi_table[t][s] = max_prob * self.emission_p[s][self.observation_states_dict[decode_observation_states[t]]]
                    backpointer[t][s] = max(range(len(self.hidden_states)), key=lambda prev_s: viterbi_table[t-1][prev_s] * self.transition_p[prev_s][s])

        # Step 4: Traceback and Find Best Path
        best_path_prob = max(viterbi_table[-1])
        best_path_pointer = max(range(len(self.hidden_states)), key=lambda s: viterbi_table[-1][s])
        best_path = [best_path_pointer]
        for t in range(len(decode_observation_states)-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        # Step 5: Return Best Path
        #print("before decoding: ", self.hidden_states_dict)
        return [self.hidden_states_dict[i] for i in best_path]

        # Step 3. Traceback 


        # Step 4. Return best hidden state sequence 
        