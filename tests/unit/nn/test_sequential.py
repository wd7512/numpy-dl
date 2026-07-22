"""Tests for the Sequential container."""

from numpy_dl.nn.activations import ReLU
from numpy_dl.nn.layers import Dense
from numpy_dl.nn.sequential import Sequential


class TestSequentialParameters:
    def test_parameters_collects_from_dense_layers(self) -> None:
        model = Sequential(
            [Dense(4, 8), ReLU(), Dense(8, 2)]
        )
        params = model.parameters()
        assert len(params) == 4  # two Dense layers, two tuples each

    def test_parameters_returns_same_list_ref(self) -> None:
        model = Sequential([Dense(4, 8), ReLU(), Dense(8, 2)])
        first = model.parameters()
        second = model.parameters()
        assert first is second, "Sequential.parameters() should cache, not rebuild"

    def test_parameters_skips_non_param_layers(self) -> None:
        model = Sequential([Dense(3, 4), ReLU(), ReLU(), Dense(4, 1)])
        assert len(model.parameters()) == 4

    def test_parameters_refs_match_child_dense(self) -> None:
        d1 = Dense(4, 8)
        d2 = Dense(8, 2)
        model = Sequential([d1, ReLU(), d2])
        seq_params = model.parameters()
        child_params = d1.parameters() + d2.parameters()
        assert seq_params == child_params, "cached tuples must equal child tuples"
        assert seq_params[0][0] is d1.W
        assert seq_params[1][0] is d1.b
        assert seq_params[2][0] is d2.W
        assert seq_params[3][0] is d2.b
