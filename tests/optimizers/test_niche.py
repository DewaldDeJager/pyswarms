#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import abc

import pytest

# Import from pyswarms
from pyswarms.single import NichePSO
from pyswarms.utils.functions.single_obj import sphere


class TestNicheOptimizer(abc.ABC):
    @pytest.fixture
    def optimizer(self):
        return NichePSO

    @pytest.fixture
    def optimizer_history(self, options):
        opt = NichePSO(10, 2, options=options)
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture
    def optimizer_reset(self, options):
        opt = NichePSO(10, 2, options=options)
        opt.optimize(sphere, 10)
        opt.reset()
        return opt

    @pytest.fixture
    def options(self):
        """Default options dictionary for most PSO use-cases"""
        return {"c1": 0.3, "c2": 0.7, "w": 0.9, "k": 2, "p": 2, "r": 1}

    def test_obj_incorrect_kwargs(self, options):
        """Test if error is raised with wrong kwargs"""
        opt = NichePSO(10, 2, options=options)
        opt.optimize(sphere, 1000)
