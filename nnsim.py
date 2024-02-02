import numpy as np

class NNSim:
    """Replicates `nearest_neighbour_simulation.wls` from Kerr et al. 2022"""
    def __init__(self, num_sites: int, num_replicates: int, a: float, x: float, y: float):
        self.rng = np.random.default_rng(10023)
        self.num_sites = num_sites
        self.num_replicates = num_replicates
        self.a = a
        self.x = x
        self.y = y
        self.k_values = np.array([
            a*y, a*y, a, a, a*x*y, a*x*y, a*x*y, a*x*y, a*x, a*x, a*x*y, a*x*y
        ])
        self.rxn_prods = np.array([1, 2, 1, 0, 1, 1, 2, 2, 1, 1, 0, 0])
        self.rxn_ids = np.arange(len(self.rxn_prods))
        self.cpgs = self._initialize_cpgs()

    def run(self, timespan: tuple, num_samples: int):
        time_state = np.zeros(self.num_replicates)
        cpgs_traj = [self.cpgs.copy()]
        time_traj = [time_state.copy()]

        while np.any(time_state < timespan[1]):
            reaction_mask, neighbors_left, neighbors_right = self._create_reaction_mask()

            possible_rxns = np.where(reaction_mask, self.k_values[:, None, None], 0)
            propensity_bysite = possible_rxns.sum(axis=0)
            propensity_byrepl = propensity_bysite.sum(axis=0)

            tau = (1 / propensity_byrepl) * np.log(1 / self.rng.uniform(size=self.num_replicates))
            time_state += tau

            sites, reactions = self._select_reactions(propensity_bysite, possible_rxns)

            for i, (site, rxn_id) in enumerate(zip(sites, reactions)):
                self.cpgs[site, i] = self.rxn_prods[rxn_id]

            cpgs_traj.append(self.cpgs.copy())
            time_traj.append(time_state.copy())

        return np.array(cpgs_traj), np.array(time_traj)

    def _create_reaction_mask(self):
        reaction_mask = np.zeros((len(self.k_values), self.num_sites, self.num_replicates), dtype=bool)
        neighbors_left = np.roll(self.cpgs, shift=-1, axis=0)
        neighbors_right = np.roll(self.cpgs, shift=1, axis=0)

        for i, state in enumerate([0, 1, 2, 1]):
            reaction_mask[i] = self.cpgs == state

        for i in range(4, len(self.k_values)):
            left_condition = (i % 2 == 0)
            neighbor_state = i // 4
            reaction_mask[i] = np.logical_or(
                np.logical_and(self.cpgs == state, neighbors_left == neighbor_state),
                np.logical_and(self.cpgs == state, neighbors_right == neighbor_state)
            )
        return reaction_mask, neighbors_left, neighbors_right

    def _initialize_cpgs(self):
        cpgs_init = self.rng.multinomial(
            1, np.ones(3)/3, size=(self.num_sites, self.num_replicates)
        ).argmax(axis=2)
        return cpgs_init

    def _select_reactions(self, propensity_bysite, possible_rxns):
        sites = np.array([np.searchsorted(propensity_bysite[:, i].cumsum(), 
                    self.rng.uniform() * propensity_bysite[:, i].sum()) for i in range(self.num_replicates)])
        reactions = []
        for i, site in enumerate(sites):
            probs = possible_rxns[:, site, i] / possible_rxns[:, site, i].sum()
            rxn_id = self.rng.choice(self.rxn_ids, p=probs)
            reactions.append(rxn_id)
        return sites, reactions
