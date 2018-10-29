# -*- coding: utf-8 -*-

r"""
A Niching Particle Swarm Optimization (Niche PSO) algorithm.

TODO: Add more information here
"""

import logging
from itertools import combinations
from time import sleep
from typing import List

import numpy as np

from pyswarms.backend import compute_pbest, Swarm
from ..backend.topology import Star
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter


def euclidian_distance(x, y):
    return np.linalg.norm(x - y)


class NichePSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        velocity_clamp=None,
        center=1.00,
        ftol=-np.inf,
        init_pos=None
    ):
        super(NichePSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            init_pos=init_pos
        )

        # Initialize logger
        self.reporter = Reporter(logger=logging.getLogger(__name__))

        # Initialize the topology
        self.top = Star()
        self.name = __name__

        self.sub_swarms: List[Swarm] = []

        # Check that the number of particles in the main swarm is even as they are moved to sub-swarms in groups of 2
        assert n_particles % 2 == 0

    def optimize(self, objective_func, iters, fast=True, **kwargs):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        fast : bool (default is True)
            if True, time.sleep is not executed
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """

        self.reporter.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.reporter.log("Optimize for {} iters with {}".format(iters, self.options), lvl=logging.INFO)

        # Initialize the resettable attributes
        self.reset()

        for _ in self.reporter.pbar(iters, self.name):
            if not fast:
                sleep(0.01)

            # Train the main swarm using the cognition model for one iteration

            # Compute cost for current position and personal best
            if self.swarm_size[0] > 0:
                self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
                self.swarm.pbest_cost = objective_func(self.swarm.pbest_pos, **kwargs)

                self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
                best_cost_yet_found = np.min(self.swarm.best_cost)
                # Update gbest from neighborhood
                self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

                self.reporter.hook(best_cost=np.min(self.swarm.best_cost))
                # Save to history
                hist = self.ToHistory(
                    best_cost=self.swarm.best_cost,
                    mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                    mean_neighbor_cost=np.mean(self.swarm.best_cost),
                    position=self.swarm.position,
                    velocity=self.swarm.velocity,
                    cost=self.swarm.current_cost
                )
                self._populate_history(hist)

                # Verify stop criteria based on the relative acceptable cost ftol
                relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
                if np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure:
                    break

                # Perform position velocity update
                self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp)
                self.swarm.position = self.top.compute_position(self.swarm, self.bounds)

            for sub_swarm in self.sub_swarms:
                # Train sub_swarm for one iteration using the GBest model
                # FIXME: This needs to happen AFTER position update - This needs to be done across all optimisers
                # Update each particle's fitness
                sub_swarm.current_cost = objective_func(sub_swarm.position, **kwargs)
                sub_swarm.pbest_cost = objective_func(sub_swarm.pbest_pos, **kwargs)
                sub_swarm.pbest_pos, sub_swarm.pbest_cost = compute_pbest(sub_swarm)
                sub_swarm.best_pos, sub_swarm.best_cost = self.top.compute_gbest(sub_swarm)
                sub_swarm.velocity = self.top.compute_velocity(sub_swarm, self.velocity_clamp)
                sub_swarm.position = self.top.compute_position(sub_swarm, self.bounds)
                # Update swarm radius
                sub_swarm.radius = self.calculate_radius(sub_swarm)

            self.merge_sub_swarms()

            # Allow subswarms to absorb any particles from the main swarm that moved into them
            for sub_swarm in self.sub_swarms:
                partices_to_move_to_this_sub_swarm = []
                # Iterate in reverse order so that removal does not affect indices
                for n in range(1, self.n_particles + 1):
                    diff = euclidian_distance(self.swarm.position[-n], sub_swarm.best_pos)
                    if diff <= sub_swarm.radius:
                        partices_to_move_to_this_sub_swarm.append(-n)
                for n in partices_to_move_to_this_sub_swarm:
                    particle = self.remove_particle_from_swarm(n)
                    sub_swarm.add_particle(particle)

            # Search main swarm for any particle that meets the partitioning criteria and possibly create sub-swarm
            number_of_iterations_to_track = 3
            i = 0
            while i < self.n_particles:
                # Calculate standard deviation for the current particle
                std_dev = self.calculate_std_dev_of_cost(number_of_iterations_to_track, i)
                # If the particle's deviation is low, split it and it's topographical neighbour into a sub-swarm
                if std_dev < self.options["delta"]:
                    self.sub_swarms.append(self.create_new_sub_swarm(i))
                    # Increment the counter twice since the next particle will be moved to a sub-swarm
                    i += 2
                else:
                    i += 1

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.best_pos.copy()
        # Write report in log and return final cost and position
        self.reporter.log("Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos),
                          lvl=logging.INFO)
        return final_best_cost, final_best_pos

    def calculate_radius(self, swarm: Swarm = None) -> float:
        swarm = self.swarm if swarm is None else swarm
        return max([euclidian_distance(swarm.best_pos, swarm.get_particle(particle)[0])
                    for particle in range(swarm.n_particles)])

    def calculate_std_dev_of_cost(self, number_of_iterations: int, particle_index: int):
        # Only allow the standard deviation to be considered after 3 iterations, as it starts at 0.0
        length = len(self.fitness_history)
        if length < number_of_iterations:
            return np.Infinity

        values = []
        for i in range(1, number_of_iterations + 1):
            current_value = self.fitness_history[-i][particle_index]
            values.append(current_value)
        return np.std(np.array(values))

    def create_new_sub_swarm(self, index):
        particle_position, particle_velocity = self.remove_particle_from_swarm(index)

        # Find particle i's closest neighbour
        neighbour_index = index % self.n_particles
        neighbour_position, neighbour_velocity = self.remove_particle_from_swarm(neighbour_index)

        # Make a new sub-swarm from this particle and it's closest neighbour
        position = np.row_stack((particle_position, neighbour_position))
        velocity = np.row_stack((particle_velocity, neighbour_velocity))
        swarm = Swarm(position=position, velocity=velocity, options=self.options)
        # TODO: Investigate why this is being set to an empty list rather than a list of 2 values
        # swarm.pbest_cost = np.all(self.n_particles, np.inf)
        # Solution: pbest_pox should be initialised to the initial position and pbest_cost to it's cost
        # TODO: Change this so that the pbest_position is not the same as the initial position. Or is this only for
        # the GBEST?
        swarm.pbest_pos = position
        return swarm

    def merge_sub_swarms(self):
        if len(self.sub_swarms) < 2:
            return

        # Calculate list of unique pairs of sub-swarms, using indices
        pairs = combinations(range(len(self.sub_swarms)), 2)

        # Iterate through pairs and see if any can be merged
        for i, j in pairs:
            # Check if sub-swarm i can be merged with sub-swarm j
            distance = euclidian_distance(self.sub_swarms[i].best_pos, self.sub_swarms[j].best_pos)
            can_be_merged = distance < self.options["mu"]

            if can_be_merged:
                # Perform the merge of sub-swarm i and sub-swarm j
                self.sub_swarms[i].add_particle((self.sub_swarms[j].position, self.sub_swarms[j].velocity))
                del self.sub_swarms[j]
                # When a merge happens, re-start the process
                self.merge_sub_swarms()
                return

        # Finish after iterating through list of pairs without performing any merges
        return

    def remove_particle_from_swarm(self, index):
        particle = self.swarm.remove_particle(index)
        self.n_particles -= 1
        self.swarm_size = self.n_particles, self.dimensions
        for i in range(len(self.fitness_history)):
            self.fitness_history[i] = np.delete(self.fitness_history[i], index, axis=0)
        return particle
