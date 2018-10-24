# -*- coding: utf-8 -*-

r"""
A Niching Particle Swarm Optimization (Niche PSO) algorithm.

TODO: Add more information here
"""

import logging
from typing import List

import numpy as np

from pyswarms.backend import compute_pbest, Swarm
from ..backend.topology import Star
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter


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

    def optimize(self, objective_func, iters, fast=False, **kwargs):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        fast : bool (default is False)
            if True, time.sleep is not executed
            TODO: Implement this
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

        # TODO: Fix everything below this

        for _ in self.reporter.pbar(iters, self.name):
            # Train the main swarm using the cognition model for one iteration

            # Compute cost for current position and personal best
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
            )
            self._populate_history(hist)

            # Verify stop criteria based on the relative acceptable cost ftol
            # relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            # if np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure:
            #     break

            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp)
            self.swarm.position = self.top.compute_position(self.swarm, self.bounds)

            for sub_swarm in self.sub_swarms:
                # Train sub_swarm for one iteration using the GBest model
                sub_swarm.pbest_cost = objective_func(sub_swarm.pbest_pos, **kwargs)
                sub_swarm.pbest_pos, sub_swarm.pbest_cost = compute_pbest(sub_swarm)
                sub_swarm.best_pos, sub_swarm.best_cost = self.top.compute_gbest(sub_swarm)
                sub_swarm.velocity = self.top.compute_velocity(sub_swarm, self.velocity_clamp)
                sub_swarm.position = self.top.compute_position(sub_swarm, self.bounds)
                # TODO: Decide if I need a second function here, that evaluates it as a single ANN and not a NNE
                # Update each particle's fitness
                sub_swarm.current_cost = objective_func(sub_swarm.position, **kwargs)
                # Update swarm radius
                sub_swarm.radius = self.calculate_radius(sub_swarm)

            # TODO: If possible, merge sub-swarms

            # Allow subswarms to absorb any particles from the main swarm that moved into them
            for sub_swarm in self.sub_swarms:
                partices_to_move_to_this_sub_swarm = []
                for n in range(self.n_particles):
                    diff = np.linalg.norm(self.swarm.position[n] - sub_swarm.position)
                    if diff <= sub_swarm.radius:
                        partices_to_move_to_this_sub_swarm += n
                for n in partices_to_move_to_this_sub_swarm:
                    # TODO: Watch out for having rows and columns swapped around here
                    particle = np.delete(self.swarm.position[n], n, 1)  # delete column/particle n of main swarm
                    self.n_particles -= 1
                    # Move particle into sub_swarm
                    sub_swarm.position = np.c_[sub_swarm.position, particle]

            # TODO: Search main swarm for any particle that meets the partitioning criteria and possibly create subswarm

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.best_pos.copy()
        # Write report in log and return final cost and position
        self.reporter.log("Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos),
                          lvl=logging.INFO)
        return final_best_cost, final_best_pos

    def calculate_radius(self, swarm: Swarm) -> float:
        return max([np.linalg.norm(swarm.best_pos - particle) for particle in self.n_particles])
