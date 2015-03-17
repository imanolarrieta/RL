__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Imanol Arrieta Ibarra", "Ian Osband"]

from rlpy.Agents.Agent import Agent, DescentAlgorithm
import numpy as np



class PosteriorSampling(DescentAlgorithm, Agent):
    """
    Standard Posterior Sampling algorithm with normal-gaussian prior for 
    rewards and dirichlet for transitions.
    """
    episodeCap = None
    startState=None
    flag_ = True
    observed_rewards = None
    mean_rewards = None
    var_requards = None
    observed = None
    observed_transitions = None
    bad_reward = None
    def __init__(self, policy, representation, discount_factor,
                 initial_learn_rate=0.1,bad_reward=-1, **kwargs):
        super(PosteriorSampling,self).__init__(policy=policy,
         representation=representation, discount_factor=discount_factor, **kwargs)
        self.logger.info("Initial learning rate:\t\t%0.2f" % initial_learn_rate)
        self.episodeCap = self.representation.domain.episodeCap
        self.observed_rewards = {}
        self.mean_rewards= np.zeros((self.representation.features_num,self.representation.actions_num))
        self.var_rewards= np.zeros((self.representation.features_num,self.representation.actions_num))
        self.observed= np.zeros((self.representation.features_num,self.representation.actions_num))
        self.observed_transitions = np.zeros((self.representation.features_num,self.representation.actions_num,self.representation.agg_states_num))
        self.bad_reward = bad_reward
    
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
        self.mean_rewards[state,a] = ((nCounts-1)*self.mean_rewards[state,a]+ r)/nCounts
        self.var_rewards[state,a] = ((nCounts-1)*self.var_rewards[state,a] + (r-self.mean_rewards[state,a])**2)/nCounts
        self.observed_transitions[state,a,next_state] += 1
    
        # For the actions that were not available, we will assume that by taking
        # them, the agent gets a very bad reward and the transition will be to
        # the same state.
        nap_actions = set(range(self.representation.actions_num))-set(p_actions)
        for nact in nap_actions:
            self.observed[state,nact] += 1.0
            nCounts = self.observed[state,nact]
            self.mean_rewards[state,nact] = ((nCounts-1)*self.mean_rewards[state,nact]+ self.bad_reward)/nCounts
            self.var_rewards[state,nact] = ((nCounts-1)*self.var_rewards[state,nact] + (self.bad_reward-self.mean_rewards[state,nact])**2)/nCounts
            self.observed_transitions[state,nact,state] += 1

            
        # MUST call this at end of learn() - add new features to representation as required.
        expanded = self.representation.post_discover(s, False, a, td_error, phi_s)
    
        # MUST call this at end of learn() - handle episode termination cleanup as required.
        # When the episode terminate we do the model sampling.
    
        
        if terminal:
       
            for a in xrange(self.representation.actions_num):
                self.observed[next_state,a] += 1.0
                nCounts = self.observed[next_state,a]
                self.mean_rewards[next_state,a] = ((nCounts-1)*self.mean_rewards[next_state,a]+ r)/nCounts
                self.var_rewards[next_state,a] = ((nCounts-1)*self.var_rewards[next_state,a] + (r-self.mean_rewards[next_state,a])**2)/nCounts
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
        # Make a very simple prior
        n_states = self.representation.features_num
        n_act = self.representation.actions_num
        
        # Sample a model from posterior distribution
        sampled_rewards,sampled_trans = self.sample_posterior(n_states,n_act)
        #print sampled_trans[:,1]
        # Compute the Value Function        
        V = self.valueIteration(sampled_rewards,sampled_trans,n_states,n_act)
        # Compute the Q(s,a) function        
        Q = self.Qfunction(sampled_rewards,sampled_trans,V, n_states,n_act)
        

        # Q is an array of (#features,#actions) dimensions so, to update the 
        # weights we need to return it in the following format:
        # [Q(s1,a1),Q(s2,a1),Q(s3,a1),...,Q(sn,an)]
        return Q.T.reshape(-1,1).T[0]

    def Qfunction(self,sample_rewards,sample_trans,V, n_states,n_act) :
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
                Q[state,action] = sample_rewards[state,action] + np.dot(sample_trans[state,action],V)
        return Q
    
    def sample_posterior(self,n_states,n_acts):
        '''
        Gives back a model sampled from the posterior distribution of 
        rewards and transition probablities
        
        Input
        
        n_states: int
            Number of states
        n_acts: int
            Number of actions
            
        Output
        
        sample_rewards: np.array((#states,#actions))
            Rewards sampled from posterior distribution.
        sample_trans: np.array((#states,#actions,#states))
            Transition probabilities sampled from posterior distribution
        '''
        mu = 0.
        n_mu = 1.
        tau = 1.
        n_tau = 1.
        prior_ng = self.convert_prior(mu, n_mu, tau, n_tau)
        prior_dir = np.ones(n_states)
        sampled_rewards = np.zeros((n_states,n_acts))
        sampled_trans = np.zeros((n_states,n_acts,n_states))
        obs_trans = self.observed_transitions

        
        for s in xrange(n_states):
            for a in xrange(n_acts):
                counts = obs_trans[s,a]
                # Posterior updating
                r_post = self.update_normal_ig(prior_ng, self.mean_rewards[s,a],self.var_rewards[s,a],self.observed[s,a])
                t_post = self.update_dirichlet(prior_dir, counts)
                    
                
                sampled_rewards[s,a] = self.sample_normal_ig(r_post)[0]
                sampled_trans[s,a] = self.sample_dirichlet(t_post)
                
        return sampled_rewards, sampled_trans
        
    
        
    def valueIteration(self,sample_rewards,sample_trans,n_states,n_act,tao=None):
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
            V = self.bellman(oldV,sample_rewards,sample_trans,n_states,n_act)
            V -= min(V)
            err = np.max(V-oldV) - np.min(V-oldV)
            nIter += 1
            oldV = V
            
            if (err<epsilon):
                complete = True
                
            if (nIter>maxIt):
                complete = True
        return V
            
            
                
        
    
    def bellman(self, oldV,sample_rewards,sample_trans,n_states,n_act):
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
        
        for state in xrange(n_states):
            for action in xrange(n_act):
                qVals[state,action] = sample_rewards[state,action] + np.dot(sample_trans[state,action],oldV)
                
        qVals = qVals + (1.e-8)*self.random_state.rand(n_states,n_act)
        newVal = np.max(qVals,1)
        return newVal
        
    # Sampling posterior functions
    #---------------------------------------------------------------------------
    # Rewards functions
    
    def convert_prior(self,mu, n_mu, tau, n_tau):
        '''
        Convert the natural way to speak about priors to our paramterization
    
        Args:
            mu - 1x1 - prior mean
            n_mu - 1x1 - number of observations of mean
            tau - 1x1 - prior precision (1 / variance)
            n_tau - 1x1 - number of observations of tau
    
        Returns:
            prior - 4x1 - (mu, lambda, alpha, beta)
        '''
        prior = (mu, n_mu, n_tau * 0.5, (0.5 * n_tau) / tau)
        return prior
    
    def update_normal_ig(self,prior, y_bar, y_var, n):
        '''
        Update the parameters of a normal gamma.
            T | a,b ~ Gamma(a, b)
            X | T   ~ Normal(mu, 1 / (lambda T))
    
        Args:
            prior - 4 x 1 - tuple containing (in this order)
                mu0 - prior mean
                lambda0 - pseudo observations for prior mean
                alpha0 - inverse gamma shape
                beta0 - inverse gamma scale
            y_bar - mean of observed rewards
            y_var - variance of observed rewards
            n - number of observations
    
        Returns:
            posterior - 4 x 1 - tuple containing updated posterior params.
                NB this is in the same format as the prior input.
        '''
        # Unpack the prior
        (mu0, lambda0, alpha0, beta0) = prior
    
    
        # Updating normal component
        lambda1 = lambda0 + n
        mu1 = (lambda0 * mu0 + n * y_bar) / lambda1
    
        # Updating Inverse-Gamma component
        alpha1 = alpha0 + (n * 0.5)
        ssq = n * y_var
        prior_disc = lambda0 * n * ((y_bar - mu0) ** 2) / lambda1
        beta1 = beta0 + 0.5 * (ssq + prior_disc)
    
        posterior = (mu1, lambda1, alpha1, beta1)
        return posterior
    
    def sample_normal_ig(self,prior):
        '''
        Sample a single normal distribution from a normal inverse gamma prior.
    
        Args:
            prior - 4 x 1 - tuple containing (in this order)
                mu - prior mean
                lambda0 - pseudo observations for prior mean
                alpha - inverse gamma shape
                beta - inverse gamma scale
    
        Returns:
            params - 2 x 1 - tuple, sampled mean and precision.
        '''
        # Unpack the prior
        (mu, lambda0, alpha, beta) = prior
    
        # Sample scaling tau from a gamma distribution
    
        tau = self.random_state.gamma(shape=alpha, scale=1. / beta)
        var = 1. / (lambda0 * tau)
        # Sample mean from normal mean mu, var
        mean = self.random_state.normal(loc=mu, scale=np.sqrt(var))
        return (mean, tau)
    
    
    #---------------------------------------------------------------------------
    # Transition functions
    
    def update_dirichlet(self,prior, data):
        '''
        Update the parameters of a dirichlet distribution.
        We assume that the data is drawn from multinomial over n discrete states.
    
        Args:
            prior - n x 1 - numpy array, pseudocounts of discrete observations.
            data - n x 1 - numpy array, counts of observations of each draw
    
        Returns:
            posterior - n x 1 - numpy array, overall pseudocounts.
        '''
        # Updating dirichlet is trivial
        posterior = prior + data
        return posterior
    
    def sample_dirichlet(self,prior):
        '''
        Sample a multinomial distribution from a Dirichlet prior.
    
        Args:
            prior - n x 1 - numpy array, pseudocounts of discrete observations.
    
        Returns:
            dist - n x 1 - numpy array, probability distribution over n discrete.
        '''
        n = len(prior)
        dist = np.zeros(n)
        for i in range(n):
            # Sample a gamma distribution for each entry
            dist[i] = self.random_state.gamma(prior[i])
    
        # Normalize the probability distribution
        dist = dist / sum(dist)
        return dist
