import copy,math
import numpy as np



def occupancies(M_sites,M_factors):
    """
    Produce all possible occupancy patterns with M_sites and M_factors.

    Output:
    r - a list of lists of length M_sites. The jth element is of each list is the
    occupancy at the jth binding site. If 0, then there is no factor at the site,
    while if k then the kth factor is bound.
    """
    if M_sites>1:
        next_lists = [[[j] + u for u in occupancies(M_sites-1,M_factors)] for j in range(M_factors+1)]
        return sum(next_lists,[]) # this joins the lists together
    else:
        return [[j] for j in range(M_factors+1)]



class tf_thermodynamic_model:
    def __init__(self,binding_energies,interaction_energies,rates,tf_concentrations,M_sites):
        self.L = len(binding_energies[0])
        self.M_sites = M_sites
        self.M_factors = len(tf_concentrations)
        self.rates = rates
        self.tf_concentrations = tf_concentrations
        self.interaction_energies = interaction_energies
        self.binding_energies = binding_energies

    def expression_level(self,g):
        """
        Compute the expression level given a genome

        Input:

        g - an M_sitesxL array in which the (i,j) component is 1 if there is a mismatch
            for the jth nucleotide of the ith binding site

        Output:

        expression level - the average expression level of the regulated gene under
                            the assumption of the thermodynamic model

        """
        patterns = occupancies(self.M_sites,self.M_factors)    # a list of possible occupancy patterns, has length (M_factors+1)^M_sites
        weights = []  # array to store statistical weights corresponding to each binding pattern
        orates = []   # array to store rates of each binding pattern
        for pattern in patterns:     # now we compute the rates and statistical weights for each binding pattern

            # here we compute the total energey of a given occupancy pattern
            weight = 1.
            orate = 0.

            for i in range(self.M_sites): # compute the energy of this state
                # use the energy matrix to get the energy for this site
                tf_ind = pattern[i]-1 # the index of tf bound at this site

                if tf_ind>-1: # if there is a tf bound at this site
                    E_binding = np.sum([self.binding_energies[tf_ind][k,g[i,k]] for k in range(self.L)])
                    # NOTE: np.sign((pattern[j]) = 1 if *any* factor is bound and 0 otherwise
                    E_interaction = - sum([self.interaction_energies[tf_ind,pattern[j]-1]*np.sign(pattern[j]) \
                        for j in range(i+1,self.M_sites)])
                    Etot = E_binding+E_interaction
                    # update weight and rate
                    weight = weight*np.exp(-Etot)*self.tf_concentrations[tf_ind]
                    orate += self.rates[i,tf_ind]

            weights.append(weight)
            orates.append(orate)

        Z = sum(weights)                       # partition function
        rate_sum = np.dot(orates,weights)      # the sum of rates time weights
        return rate_sum/Z

    def substitution_evolution(self,tmax,g,fitness_func,p_mut,ne,*,max_steps=10**6):
        """
        Simulate the evolutionary dynamics in the strong selection, weak mutation
        regime. In this limit we don't need to simulate the full moran process,
        instead, we simply compute the time until the next mutation
        arrises and sweeps the population. We will use Gillespie's algorithm to
        find the jump times and substitutions to make

        Input:

        g           - the initial genotype (we assume the population is always clonal)
        fitness_fun - maps the fitness to the genotype
        p_mut       - mutation rate, assumed to be the same for all sites
        ne          - the effective population size

        Output:

        t - the times at which new mutations fixate
        G - the genomes at each step

        """


        G = [g.reshape(self.L*self.M_sites)]
        t = [0]
        step = 0
        genome_length = self.M_sites*self.L

        while t[-1]<tmax and step<max_steps:
            g = G[-1]

            # find all possible mutations and
            # compute the corresponding substitution rates
            # using kimura's formula
            jump_rates = []
            mutations = []
            for k in range(genome_length):
                for base in range(1,4):
                    gm = copy.deepcopy(g)
                    gm[k] = (g[k]+base)%4 # update genome

                    # compute fixation probability of mutant using Kimura's formula
                    fm = fitness_func(self,gm.reshape((self.M_sites,self.L)))
                    f = fitness_func(self,g.reshape((self.M_sites,self.L)))
                    df = fm-f
                    if np.abs(df)<0.05/ne: # threshold for neutral
                        jump_rate = p_mut # neutral mutation has p_fix = 1/ne
                    else:
                        p_fix = (1.0-np.exp(-2*df))/(1.0-np.exp(-2*ne*df))
                        jump_rate = p_mut*ne*p_fix
                    jump_rates.append(jump_rate)
                    mutations.append(gm)


            # compute time until next substitution
            rate_tot = np.sum(jump_rates)
            t_next =np.random.exponential(1.0/rate_tot)

            # determine which substitution to make
            r = np.random.rand()
            ind = 0
            rate_sum = 0.0
            while r>rate_sum/rate_tot:
                rate_sum += jump_rates[ind]
                ind += 1

            g_new = mutations[ind-1]
            # make substitution and update trajectory
            G.append(g_new)
            step = step+1
            t.append(t[-1]+t_next)

        return t[:step],G[:step]

    # def moran_evolution(self,tmax,G,fitness_func,p_mut,*,max_steps=10**6):
    #     # initialize division times
    #     f = [fitness_func(self,g) for g in G]
    #     f_avg = np.zeros(max_steps) # fitnesses
    #     t = np.zeros(max_steps) # times
    #     num_cells = len(G)
    #     div_times = np.array([np.random.exponential(1./fitness_func(self,g)) for g in G])
    #
    #     step = 0
    #     t_cur = 0.
    #
    #     f_avg[0] = np.mean(f)
    #
    #     while t_cur<tmax and step<max_steps:
    #         # get index of dividing cell and find division time
    #         ind1 = np.argmin(div_times)
    #         t_next = div_times[ind1]
    #
    #         # produce daughter cells
    #         g1,g2 = mutate(self,G[ind1],p_mut),mutate(self,G[ind1],p_mut)
    #
    #         # insert daughter cells into population
    #         G[ind1] = g1
    #         ind2 = np.random.randint(num_cells)
    #         G[ind2] = g2
    #         f[ind1] = fitness_func(self,g1)
    #         f[ind2] = fitness_func(self,g2)
    #
    #
    #         # recompute times
    #         div_times[ind1] = t_next+np.random.exponential(1./fitness_func(self,g1))
    #         div_times[ind2] = t_next+np.random.exponential(1./fitness_func(self,g2))
    #
    #         # update time and fitness
    #         t_cur = t_next
    #         step = step+1
    #         t[step] = t_cur
    #         f_avg[step] = np.mean(f)
    #
    #     return t[:step],f_avg[:step]


# ----------------------------------------------------------------------------------------------




# def mutate(model,g,p_mut):
#     gm = copy.deepcopy(g) # deepcopy so that we don't change g when we change gm
#     for k in range(model.M_sites):
#         for s in range(model.L):
#             r = np.random.rand()
#             if r < p_mut:
#                 gm[k][s] = 1-gm[k][s]
#     return gm


# def run_moran_evolution(max_steps=10**6):
#     # initialize division times
#     fitnesses = [g.fitness() for g in population]
#     copies = [g.copies() for g in population]
#
#     f_avg = np.zeros(max_steps) # fitnesses
#     t = np.zeros(max_steps) # times
#
#     # jump times
#     rates = np.array([g.copies()*g.fitness()*(n_tot - g.copies())/n_tot for g in genomes])
#     div_times = rates = np.array([np.random.exponential(1.0/r) for r in rates])
#
#     step = 0
#     t_cur = 0.
#
#     f_avg[0] = np.mean(f)
#
#     while t_cur<tmax and step<max_steps:
#         ind1 = np.argmin(div_times)
#         t_next = div_times[ind1]
#
#         # generate mutants and add them to population
#         g1,g2 = mutate(population[ind1].sequence()),mutate(population[ind1].sequence())
#         for g in g1,g2:
#             if g1 =
#                 population[ind].copies +=1
#             else:
#                 population.append(Genotype(fitness,1))
#
#         # dilute population
#         r = np.random.rand()
#         k = 0
#         while r<(n_tot - population[k].copies())/n_tot:
#             k=+1
#         population[k].copies =-1
#         if population[k].copies==1:
#
#
#
#         # update division times
#         rates = np.array([g.copies()*g.fitness()*(n_tot - g.copies())/n_tot for g in genomes])
#         div_times = rates = np.array([np.random.exponential(1./fitness(r)) for r in rates])
#
#     return t[0:step],f_avg[0:step]
