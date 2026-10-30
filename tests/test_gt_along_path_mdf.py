"""Mock-based tests for MDF/PTM growth and fractionation in ``gt_along_path``.

``gt_along_path(mdf=True)`` consults a state-shared
:class:`GarnetMDFCalculator` at every garnet-bearing P-T point.  These tests
pin:

* the lazily constructed helper shares this calculator's state
  (``_copy_state_to``) and is constructed only once;
* per-step MDF setup uses the **pre-fractionation** bulk, and
  ``calculate_compositions`` receives the step's P, T and the calculator's
  ``fe_basis``;
* reported Mg-Mn-Fe-Ca proportions become the MDF composition (converted
  for ``sys_in='wt'``, raw atomic proportions for ``sys_in='mol'``);
* with ``fractionate=True`` the segregated composition is
  ``result.mdf_oxide_wt`` passed as ``phase_comp`` — weight fractions for
  ``'wt'``, molar (unit-sum) for ``'mol'`` — while the removed *amount*
  stays the equilibrium ``frac_amount``;
* garnet-absent steps skip the MDF solve and keep the equilibrium
  reporting;
* a non-converged solve warns and its composition is still used;
* ``mdf=False`` (the default) never touches the MDF machinery;
* ``GarnetGenerator.generate_garnet_data`` forwards ``mdf``.

No live Julia runtime is needed -- all MAGEMin calls are mocked.
"""

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from phasetools.calculators.garnet import MAGEMinGarnetCalculator
from phasetools.models.garnet_growth import GarnetGenerator
from phasetools.utils.bulk_rock import (
    atomic_frac_to_wt_frac,
    atomic_mass_dict,
    get_molar_mass_dict,
)

# Closed cation proportions returned by every stubbed MDF solve (sum = 1).
_MDF_CATIONS = {"Mg": 0.2, "Fe": 0.5, "Mn": 0.05, "Ca": 0.25}
# Unit-sum oxide weight vector aligned to the test bulk's two oxides.
_MDF_OXIDE_WT = (0.6, 0.4)


def _make_calc(sys_in="mol"):
    calc = MAGEMinGarnetCalculator.__new__(MAGEMinGarnetCalculator)
    calc.db = "mp"
    calc.dataset = 636
    calc.verbose = False
    calc.sys_in = sys_in
    calc.X = np.array([50.0, 50.0])
    calc.Xoxides = MagicMock()
    calc._Xoxides_py = ["SiO2", "Al2O3"]
    calc._X_mol_standardized = (50.0, 50.0)
    calc._stoich_map = {}
    calc.rm_list = None
    calc.data = MagicMock()
    calc.fe_basis = "feot"
    # Spy on the real state transfer so the helper-mocking stays honest.
    calc._copy_state_to = MagicMock(wraps=calc._copy_state_to)
    return calc


def _point(gt_mol, gt_wt, gt_vol, phases=("g", "bi")):
    """One mocked _gt_single_point_from_jl return value."""
    out = MagicMock()
    out.ph = list(phases)
    # Unit-sum bulks for the MDF removal cap (same oxide order as _Xoxides_py).
    out.bulk = np.array([0.6, 0.4])
    out.bulk_wt = np.array([0.6, 0.4])
    return (gt_mol, gt_wt, gt_vol, 0.30, 0.05, 0.40, 0.25, out)


def _mdf_result(converged=True, oxide_wt=_MDF_OXIDE_WT):
    return SimpleNamespace(
        mdf=dict(_MDF_CATIONS),
        converged=converged,
        residual_norm_kj_per_formula=0.0 if converged else 0.5,
        mdf_oxide_wt=tuple(oxide_wt),
        equilibrium_modes=(0.1, 0.1, 0.1),
    )


def _run(calc, points, *, oxide_wt=_MDF_OXIDE_WT, **kwargs):
    """Run gt_along_path with the MDF helper, phase functions and jlconvert mocked."""
    calc._gt_single_point_from_jl = MagicMock(side_effect=list(points))
    with (
        patch("phasetools.calculators.phase_search.PhaseFunctions") as pf_cls,
        patch("phasetools.calculators.garnet.calculator.jlconvert") as mock_jlconvert,
        patch("phasetools.calculators.garnet.calculator.GarnetMDFCalculator") as mdf_cls,
    ):
        helper = MagicMock()
        helper._owns_data = False  # let the real _copy_state_to skip Finalize
        helper.calculate_compositions.return_value = _mdf_result(oxide_wt=oxide_wt)
        mdf_cls.return_value = helper
        pf_cls.fractionate_phase = MagicMock(
            side_effect=[np.array([40.0, 60.0]), np.array([40.0, 60.0])]
        )
        mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)
        P = np.arange(len(points), dtype=float) + 10.0
        T = np.arange(len(points), dtype=float) + 800.0
        result = calc.gt_along_path(P, T, **kwargs)
    return result, pf_cls, mdf_cls, helper


class TestMdfStateSharing(unittest.TestCase):
    """The helper is constructed once and shares the parent's state."""

    def test_helper_constructed_once_and_state_copied(self):
        calc = _make_calc("wt")
        points = [_point(0.05, 0.04, 0.03), _point(0.12, 0.10, 0.08)]
        _, _pf_cls, mdf_cls, helper = _run(
            calc, points, fractionate=True, mdf=True
        )
        mdf_cls.assert_called_once_with(db="mp", dataset=636, verbose=False)
        calc._copy_state_to.assert_called_once_with(helper)
        self.assertIs(calc._mdf_calc, helper)
        self.assertTrue(mdf_cls.return_value is helper)

        # A second run reuses the cached helper (no new construction).
        calc._gt_single_point_from_jl = MagicMock(side_effect=list(points))
        with (
            patch("phasetools.calculators.phase_search.PhaseFunctions") as pf_cls2,
            patch("phasetools.calculators.garnet.calculator.jlconvert") as mock_jlconvert,
            patch("phasetools.calculators.garnet.calculator.GarnetMDFCalculator") as mdf_cls2,
        ):
            pf_cls2.fractionate_phase = MagicMock(
                return_value=np.array([40.0, 60.0])
            )
            mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)
            P = np.arange(len(points), dtype=float) + 10.0
            T = np.arange(len(points), dtype=float) + 800.0
            calc.gt_along_path(P, T, fractionate=True, mdf=True)
        mdf_cls2.assert_not_called()
        self.assertEqual(calc._copy_state_to.call_count, 1)


class TestMdfCationReporting(unittest.TestCase):
    """Reported cations follow the MDF composition with the sys_in mirror."""

    def test_mol_basis_reports_raw_atomic_proportions(self):
        calc = _make_calc("mol")
        points = [_point(0.05, 0.04, 0.03), _point(0.12, 0.10, 0.08)]
        result, _, _, helper = _run(calc, points, fractionate=True, mdf=True)
        Mgi, Mni, Fei, Cai = result[3], result[4], result[5], result[6]
        for i in range(len(points)):
            self.assertAlmostEqual(Mgi[i], _MDF_CATIONS["Mg"], places=12)
            self.assertAlmostEqual(Mni[i], _MDF_CATIONS["Mn"], places=12)
            self.assertAlmostEqual(Fei[i], _MDF_CATIONS["Fe"], places=12)
            self.assertAlmostEqual(Cai[i], _MDF_CATIONS["Ca"], places=12)
        self.assertEqual(helper.calculate_compositions.call_count, 2)

    def test_wt_basis_converts_cations_to_weight_proportions(self):
        calc = _make_calc("wt")
        points = [_point(0.05, 0.04, 0.03), _point(0.12, 0.10, 0.08)]
        result, *_ = _run(calc, points, fractionate=True, mdf=True)
        expected = atomic_frac_to_wt_frac(dict(_MDF_CATIONS), atomic_mass_dict)
        Mgi, Mni, Fei, Cai = result[3], result[4], result[5], result[6]
        for i in range(len(points)):
            self.assertAlmostEqual(Mgi[i], expected["Mg"], places=12)
            self.assertAlmostEqual(Mni[i], expected["Mn"], places=12)
            self.assertAlmostEqual(Fei[i], expected["Fe"], places=12)
            self.assertAlmostEqual(Cai[i], expected["Ca"], places=12)


class TestMdfPerStepSetup(unittest.TestCase):
    """MDF sees the pre-fractionation bulk and the step's P-T."""

    def test_setup_uses_current_bulk_and_step_conditions(self):
        calc = _make_calc("mol")
        points = [_point(0.05, 0.04, 0.03), _point(0.12, 0.10, 0.08)]
        # normalise_start=False so step 0 actually removes its seed and the
        # reactive bulk handed to step 1's MDF setup is the fractionated one.
        _, _, _, helper = _run(
            calc, points, fractionate=True, mdf=True, normalise_start=False
        )

        setup_calls = helper.setup_bulk_composition.call_args_list
        self.assertEqual(len(setup_calls), 2, "one bulk setup per step")
        first, second = setup_calls
        self.assertEqual(first.args[0], ["SiO2", "Al2O3"])
        np.testing.assert_allclose(first.args[1], [50.0, 50.0],
                                   err_msg="step 0 sees the initial bulk")
        self.assertEqual(first.args[2], "mol")
        self.assertEqual(first.kwargs, {"rm_list": None})
        np.testing.assert_allclose(
            second.args[1], [40.0, 60.0],
            err_msg="step 1 sees the bulk AFTER step 0's fractionation, "
                    "before step 1's own removal",
        )

        solve_calls = helper.calculate_compositions.call_args_list
        self.assertEqual(len(solve_calls), 2)
        self.assertEqual(solve_calls[0].args, (10.0, 800.0))
        self.assertEqual(solve_calls[1].args, (11.0, 801.0))
        for call in solve_calls:
            self.assertEqual(call.kwargs, {"fe_basis": "feot"})


class TestMdfSegregatedComposition(unittest.TestCase):
    """phase_comp carries the MDF oxide vector in the sys_in basis."""

    def test_wt_basis_passes_mdf_oxide_wt_unchanged(self):
        calc = _make_calc("wt")
        points = [_point(0.05, 0.04, 0.03), _point(0.12, 0.10, 0.08)]
        result, pf_cls, *_ = _run(calc, points, fractionate=True, mdf=True)
        calls = pf_cls.fractionate_phase.call_args_list
        self.assertEqual(len(calls), 1, "step 0 is the retained seed: no removal")
        np.testing.assert_allclose(
            calls[0].kwargs["phase_comp"], np.asarray(_MDF_OXIDE_WT),
            err_msg="wt basis: mdf_oxide_wt passed through unchanged",
        )
        # The removed amount stays the equilibrium fraction.
        self.assertAlmostEqual(calls[0].kwargs["frac_amount"], 0.10 - 0.04, places=12)
        # The pile still reports equilibrium amounts (seed excluded).
        self.assertAlmostEqual(result[1][1], 0.06, places=12)

    def test_mol_basis_converts_to_unit_sum_molar_composition(self):
        calc = _make_calc("mol")
        points = [_point(0.00, 0.00, 0.00), _point(0.12, 0.10, 0.08)]
        _, pf_cls, *_ = _run(calc, points, fractionate=True, mdf=True)
        calls = pf_cls.fractionate_phase.call_args_list
        self.assertEqual(len(calls), 1)
        comp = np.asarray(calls[0].kwargs["phase_comp"], dtype=float)
        masses = get_molar_mass_dict()
        m = np.array([masses[ox] for ox in calc._Xoxides_py], dtype=float)
        expected = np.asarray(_MDF_OXIDE_WT) / m
        expected /= expected.sum()
        np.testing.assert_allclose(comp, expected, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(comp.sum()), 1.0, places=12)


class TestMdfSkipsGarnetAbsentSteps(unittest.TestCase):
    """No garnet in equilibrium -> no MDF solve; equilibrium reporting kept."""

    def test_absent_step_skips_mdf_and_reports_equilibrium_cations(self):
        calc = _make_calc("mol")
        points = [
            _point(0.00, 0.00, 0.00, phases=("amp", "bi")),  # garnet absent
            _point(0.12, 0.10, 0.08, phases=("g", "bi")),
        ]
        result, pf_cls, _, helper = _run(
            calc, points, fractionate=True, mdf=True
        )
        self.assertEqual(helper.setup_bulk_composition.call_count, 1)
        self.assertEqual(helper.calculate_compositions.call_count, 1)
        solve_call = helper.calculate_compositions.call_args
        self.assertEqual(solve_call.args, (11.0, 801.0),
                         "only the garnet-bearing step is solved")
        Mgi = result[3]
        self.assertAlmostEqual(Mgi[0], 0.30, places=12,
                               msg="absent step keeps the equilibrium-leg value")
        self.assertAlmostEqual(Mgi[1], _MDF_CATIONS["Mg"], places=12)
        self.assertEqual(len(pf_cls.fractionate_phase.call_args_list), 1,
                         "absent step has zero mode: no removal")


class TestMdfNonConvergence(unittest.TestCase):
    """Non-converged solve warns but its composition is used."""

    def test_warns_and_uses_unconverged_composition(self):
        calc = _make_calc("mol")
        points = [_point(0.05, 0.04, 0.03)]
        calc._gt_single_point_from_jl = MagicMock(side_effect=list(points))
        with (
            patch("phasetools.calculators.phase_search.PhaseFunctions"),
            patch("phasetools.calculators.garnet.calculator.jlconvert") as mock_jlconvert,
            patch("phasetools.calculators.garnet.calculator.GarnetMDFCalculator") as mdf_cls,
        ):
            helper = MagicMock()
            helper._owns_data = False
            helper.calculate_compositions.return_value = _mdf_result(converged=False)
            mdf_cls.return_value = helper
            mock_jlconvert.side_effect = lambda t, v: np.array(v, dtype=float)
            P, T = np.array([10.0]), np.array([800.0])
            with self.assertWarnsRegex(UserWarning, "did not converge"):
                result = calc.gt_along_path(P, T, fractionate=True, mdf=True)
        self.assertAlmostEqual(result[3][0], _MDF_CATIONS["Mg"], places=12,
                               msg="unconverged composition is still reported")


class TestMdfOffByDefault(unittest.TestCase):
    """mdf=False never constructs or touches the MDF helper."""

    def test_default_path_leaves_mdf_machinery_untouched(self):
        calc = _make_calc("mol")
        points = [_point(0.05, 0.04, 0.03), _point(0.12, 0.10, 0.08)]
        _, pf_cls, mdf_cls, _ = _run(calc, points, fractionate=True)
        mdf_cls.assert_not_called()
        calc._copy_state_to.assert_not_called()
        self.assertFalse(hasattr(calc, "_mdf_calc"))
        self.assertEqual(len(pf_cls.fractionate_phase.call_args_list), 1)


class TestGarnetGeneratorForwarding(unittest.TestCase):
    """generate_garnet_data passes mdf through to gt_along_path."""

    def test_generate_garnet_data_forwards_mdf(self):
        with (
            patch(
                "phasetools.calculators.garnet.calculator.MAGEMinPTGridCalculator.__init__",
                return_value=None,
            ),
            patch(
                "phasetools.calculators.garnet.calculator.MAGEMinGarnetCalculator.gt_along_path"
            ) as path_mock,
        ):
            gen = GarnetGenerator()
            n = 3
            z = np.zeros(n)
            path_mock.return_value = (z, z, z, z, z, z, z, np.zeros((n, 2)))
            gen.generate_garnet_data(
                np.arange(n, dtype=float) + 10.0,
                np.arange(n, dtype=float) + 800.0,
                np.arange(n, dtype=float),
                mdf=True,
            )
        path_mock.assert_called_once()
        self.assertIs(path_mock.call_args.kwargs.get("mdf"), True)
        self.assertIs(gen.mdf, True)


class TestMdfRemovalCap(unittest.TestCase):
    """A Mn/oxide-rich MDF composition can outstrip the bulk's supply."""

    def test_removal_scaled_down_to_bulk_supply_and_warns(self):
        calc = _make_calc("wt")
        # MDF garnet nearly pure in oxide 0: requested 0.8 removal would take
        # 0.8*0.9 = 0.72 of an oxide the bulk only holds 0.6 of.
        points = [_point(0.5, 0.8, 0.3)]
        with self.assertWarnsRegex(UserWarning, "exceeds the reactive bulk"):
            result, pf_cls, *_ = _run(
                calc, points, oxide_wt=(0.9, 0.1),
                fractionate=True, mdf=True, normalise_start=False,
            )
        calls = pf_cls.fractionate_phase.call_args_list
        self.assertEqual(len(calls), 1)
        cap = 0.6 / 0.9
        self.assertAlmostEqual(calls[0].kwargs["frac_amount"], cap, places=10,
                               msg="amount must be capped by the limiting oxide")
        np.testing.assert_allclose(
            calls[0].kwargs["phase_comp"], np.array([0.9, 0.1]),
            err_msg="cap reduces the amount, never the MDF composition",
        )
        # The pile uses the same capped amount (consistent bookkeeping).
        self.assertAlmostEqual(
            result[1][0], cap * (0.8 / 0.8), places=10,
            msg="wt pile increment uses the capped fraction",
        )

    def test_no_cap_warning_when_bulk_can_supply_the_removal(self):
        calc = _make_calc("wt")
        points = [_point(0.05, 0.04, 0.03)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, pf_cls, *_ = _run(
                calc, points, fractionate=True, mdf=True, normalise_start=False
            )
        self.assertEqual(len(pf_cls.fractionate_phase.call_args_list), 1,
                         "a real removal must have been attempted")
        cap_warnings = [w for w in caught if "exceeds the reactive bulk" in str(w.message)]
        self.assertEqual(cap_warnings, [])


class TestAbsentGarnetReporting(unittest.TestCase):
    """Absent garnet on a wt basis must report zeros, not divide by zero."""

    def test_extract_returns_zero_cations_when_garnet_absent(self):
        calc = _make_calc("wt")
        out = MagicMock()
        out.ph = ["amp", "bi"]
        self.assertEqual(
            calc._extract_garnet_elements_from_oxides(out, "wt"),
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_atomic_frac_to_wt_frac_handles_all_zero_input(self):
        zeros = {"Mg": 0.0, "Mn": 0.0, "Fe": 0.0, "Ca": 0.0}
        self.assertEqual(
            atomic_frac_to_wt_frac(zeros, atomic_mass_dict),
            zeros,
        )


if __name__ == "__main__":
    unittest.main()
