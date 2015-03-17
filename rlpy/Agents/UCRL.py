__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Imanol Arrieta Ibarra", "Ian Osband"]

from rlpy.Agents.Agent import Agent, DescentAlgorithm
import numpy as np



class UCRL(DescentAlgorithm, Agent):
    """
    Upper Confidece algorithm.
    """
    episodeCap = None
    startState=None
    flag_ = True
    observed_rewards = None
    rewards = None
    var_requards = None
    observed = None
    observed_transitions = None
    bad_reward = None
    delta = None
    def __init__(self, policy, representation, discount_factor,
                 initial_learn_rate=0.1,bad_reward=-1,delta = 0.05, **kwargs):
        super(UCRL,self).__init__(policy=policy,
         representation=representation, discount_factor=discount_factor, **kwargs)
        self.logger.info("Initial learning rate:\t\t%0.2f" % initial_learn_rate)
        self.episodeCap = self.representation.domain.episodeCap
        self.observed_rewards = {}
        self.rewards= np.zeros((self.representation.features_num,self.representation.actions_num))
        self.var_rewards= np.zeros((self.representation.features_num,self.representation.actions_num))
        self.observed= np.zeros((self.representation.features_num,self.representation.actions_num))
        self.observed_transitions = np.zeros((self.representation.features_num,self.representation.actions_num,self.representation.agg_states_num))
        self.bad_reward = bad_reward
        self.delta = delta
        
    
    def learn(self, s, p_actions, a, r, ns, np_actions, na,terminal):
        
        if self.flag_:
            self.startState = self.representation.hashState(s)
            self.flag_ = False
        # The previous state could never be terminal
        # (otherwise the episode would have already terminated)
        prevStateTerminal = False
        # MUST call this at start of learn()
        self.representation.pre_discover(s, prevStateTerminal, a, ns, terminal)
         
        # Compute feature function values and next action to be taken
        discount_factor = self.discount_factor
        feat_weights    = self.representation.weight_vec # Value function, expressed as feature weights
        phi_s      = self.representation.phi(s, prevStateTerminal) # active feats in state
        phi       = self.representation.phi_sa(s, prevStateTerminal, a, phi_s) # active features or an (s,a) pair
        phi_prime_s= self.representation.phi(ns, terminal)
        phi_prime  = self.representation.phi_sa(ns, terminal, na, phi_prime_s)
        
        # Compute td-error
        td_error            = r + np.dot(discount_factor*phi_prime - phi, feat_weights)
        
        # Update counts in case ns is not terminal
        state = self.representation.hashState(s)
        next_state = self.representation.hashState(ns)
        # Number of times we have observed state i, action a

        self.observed[state,a] += 1.0
        nCounts = self.observed[state,a]
        self.rewards[state,a] += r
        self.observed_transitions[state,a,next_state] += 1
    
        # For the actions that were not available, we will assume that by taking
        # them, the agent gets a very bad reward and the transition will be to
        # the same state.
        nap_actions = set(range(self.representation.actions_num))-set(p_actions)
        for nact in nap_actions:
            self.observed[state,nact] += 1.0
            self.rewards[state,nact] += r
            self.observed_transitions[state,nact,state] += 1

            
        # MUST call this at end of learn() - add new features to representation as required.
        expanded = self.representation.post_discover(s, False, a, td_error, phi_s)
    
        # MUST call this at end of learn() - handle episode termination cleanup as required.
        # When the episode terminate we do the model sampling.
    
        
        if terminal:
       
            for a in xrange(self.representation.actions_num):
                self.observed[next_state,a] += 1.0
                self.rewards[next_state,a] +=1
                self.observed_transitions[next_state,a,self.startState] += 1
            self.representation.weight_vec = self.model_selection()
            self.episodeTerminated()
    
    def model_selection(self):
        '''
        Update the weights by inputing the values of Q(s,a) for every state
        and action
    
        Returns:
            Q - #features*#actions x 1 
        '''
        n_states = self.representation.features_num
        n_act = self.representation.actions_num
       
        self.UCrewards = np.zeros((n_states,n_act))
        self.UCtrans = np.zeros((n_states,n_act,n_states))
        #print sampled_trans[:,1]
        # Compute the Value Function        
        V = self.valueIteration(n_states,n_act)
        # Compute the Q(s,a) function        
        Q = self.Qfunction(V, n_states,n_act)
        
        
        # Q is an array of (#features,#actions) dimensions so, to update the 
        # weights we need to return it in the following format:
        # [Q(s1,a1),Q(s2,a1),Q(s3,a1),...,Q(sn,an)]
        return Q.T.reshape(-1,1).T[0]

    def Qfunction(self,V, n_states,n_act) :
        '''
        Given the value function and a model, computes the Q(s,a) function
        
        Intput:
        
        sample_rewards: np.array(#states,#actions)
            Rewards sampled from posterior distribution
            
        sample_trans: np.array(#states,#actions,#states)
            Transition probabilities sampled from posterior distribution
        
        V: np.array(#states)
            Value function
            
        n_states: int
            Number of states
        
        n_actions: int
            Number of actions
    
        Output:
            
        Q : np.array(#features,#actions)
            Q(s,a) function
            
        '''
        Q= np.zeros((n_states,n_act))
        for state in xrange(n_states):
            for action in xrange(n_act):
                Q[state,action] = self.UCrewards[state,action] + np.dot(self.UCtrans[state,action],V)
        return Q
    
   
        
    def valueIteration(self,n_states,n_act,tao=None):
        '''
        Computes value iteration to obtain the stationary V(s) value functions.
        
        Input
        
        sample_rewards: np.array((#states,#actions))
            Rewards sampled from posterior distribution.
        sample_trans: np.array((#states,#actions,#states))
            Transition probabilities sampled from posterior distribution
        n_states: int 
            Number of states
        n_act: int 
            Number of actions
        tao: int
            Length of episodes.
            
        Output
        
        V: np.array((#states))
            Optimal stationary value function
            
        '''
        
        oldV = np.zeros(n_states)
        complete = False
        err = 1
        nIter = 1
        epsilon = 1.e-4
        if (tao==None):
            maxIt = 10000
        else:
            maxIt = tao
        
        while (not complete):
            V = self.bellman(oldV,n_states,n_act)
            V -= min(V)
            err = np.max(V-oldV) - np.min(V-oldV)
            nIter += 1
            oldV = V
            
            if (err<epsilon):
                complete = True
                
            if (nIter>maxIt):
                complete = True
        return V
            
            
                
        
    
    def bellman(self, oldV,n_states,n_act):
        '''
        Apply the bellman operator to oldV
        
        Input
        
        oldV: np.array((#states,#actions))
            Old value function.
        sample_rewards: np.array((#states,#actions))
            Rewards sampled from posterior distribution.
        sample_trans: np.array((#states,#actions,#states))
            Transition probabilities sampled from posterior distribution
        n_states: int 
            Number of states
        n_act: int 
            Number of actions
            
        Output
        
        newVal: np.array((#states,#actions))
            New value function.
        '''
        qVals = np.zeros((n_states,n_act))
        r_const = np.sqrt(7*np.log(2*n_states*n_act/float(self.delta)))
        p_const = np.sqrt(14*n_states*np.log(2*n_states*n_act/float(self.delta)))
        
        
        for state in xrange(n_states):
            for action in xrange(n_act):
                self.UCrewards[state,action]  =(2*self.rewards[state,action] + r_const)/float(2*max(self.observed[state,action],1))
                p_dist= p_const/float(max(self.observed[state,action],1))
                p_hat = self.observed_transitions[state,action]/float(max(self.observed[state,action],1))
                self.UCtrans[state,action]  = self.linearprogram(oldV,p_hat,p_dist,n_states,n_act)  
                qVals[state,action] = self.UCrewards[state,action] + np.dot(self.UCtrans[state,action],oldV)
        qVals = qVals + (1.e-8)*self.random_state.rand(n_states,n_act)
        newVal = np.max(qVals,1)
        return newVal
        
    def linearprogram(self,oldV,p_hat,p_dist,n_states,n_act):
        pOpt = p_hat
        worse_states = list(np.argsort(oldV+self.random_state.rand(n_states)*0.00001))
        best_state = worse_states.pop()
        pOpt[best_state] = min(1,p_hat[best_state]+p_dist/2.0)
        i=0
    
        while sum(pOpt)>1 and i<len(worse_states):
            state = worse_states[i]
            pOpt[state]= max(0,1-sum(pOpt)+pOpt[state])
            i+=1
  
   
    
        return pOpt    