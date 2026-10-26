"""Coverage for incomplete bulk compositions passed to MAGEMin."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from phasetools.core import base as base_module

OXIDES = ["SiO2", "Al2O3", "MgO", "FeO"]
VALUES_WT = [55.72, 16.16, 3.97, 10.76]


class _VectorType:
    def __class_getitem__(cls, _item):
        return list


class TestIncompleteBulkCompositionAdapter(unittest.TestCase):
    """The Python adapter forwards incomplete inputs and stores MAGEMin output."""

    def test_empty_phase_removal_list_skips_magemin_and_clears_removal_state(self):
        jl = SimpleNamespace(Vector=_VectorType, Float64=float, String=str)
        converter = Mock(return_value=(VALUES_WT, OXIDES))
        remove_phases = Mock(side_effect=AssertionError("empty list must be a no-op"))
        fake_magemin = SimpleNamespace(
            convertBulk4MAGEMin=converter,
            remove_phases=remove_phases,
        )

        with (
            patch.object(base_module, "jl", jl),
            patch.object(base_module, "jlconvert", side_effect=lambda _type, values: list(values)),
            patch.object(base_module, "MAGEMin_C", fake_magemin),
        ):
            calc = base_module.MAGEMinBase.__new__(base_module.MAGEMinBase)
            calc.db = "mp"
            calc.rm_list = "previous removal state"
            calc.setup_bulk_composition(OXIDES, VALUES_WT, sys_in="mol", rm_list=[])

        remove_phases.assert_not_called()
        self.assertIsNone(calc.rm_list)
        self.assertEqual(calc._X_mol_standardized, tuple(VALUES_WT))

    def test_nonempty_phase_removal_list_still_uses_magemin(self):
        jl = SimpleNamespace(Vector=_VectorType, Float64=float, String=str)
        converter = Mock(return_value=(VALUES_WT, OXIDES))
        removal_result = object()
        remove_phases = Mock(return_value=removal_result)
        fake_magemin = SimpleNamespace(
            convertBulk4MAGEMin=converter,
            remove_phases=remove_phases,
        )

        with (
            patch.object(base_module, "jl", jl),
            patch.object(base_module, "jlconvert", side_effect=lambda _type, values: list(values)),
            patch.object(base_module, "MAGEMin_C", fake_magemin),
        ):
            calc = base_module.MAGEMinBase.__new__(base_module.MAGEMinBase)
            calc.db = "mp"
            calc.setup_bulk_composition(OXIDES, VALUES_WT, sys_in="mol", rm_list=["g"])

        remove_phases.assert_called_once_with(["g"], "mp")
        self.assertIs(calc.rm_list, removal_result)

    def test_wt_input_retains_standardized_molar_basis(self):
        oxides = ["FeO", "O"]
        molar = [0.8, 0.0]
        jl = SimpleNamespace(Vector=_VectorType, Float64=float, String=str)
        fake_magemin = SimpleNamespace(
            convertBulk4MAGEMin=Mock(return_value=(molar, oxides)),
        )
        with (
            patch.object(base_module, "jl", jl),
            patch.object(base_module, "jlconvert", side_effect=lambda _type, values: list(values)),
            patch.object(base_module, "MAGEMin_C", fake_magemin),
            patch.object(base_module, "convert_mol_percent_to_wt_percent", return_value=[90.0, 10.0]),
        ):
            calc = base_module.MAGEMinBase.__new__(base_module.MAGEMinBase)
            calc.db = "mp"
            calc.setup_bulk_composition(oxides, [90.0, 10.0], sys_in="wt", rm_list=[])

        self.assertEqual(calc._X_mol_standardized, (0.8, 0.0))
        self.assertEqual(list(calc.X), [90.0, 10.0])

if __name__ == "__main__":
    unittest.main()
