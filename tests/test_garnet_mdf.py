"""Unit tests for garnet MDF/PTM and closed Fe-total cation proportions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import phasetools
from phasetools.calculators.garnet.composition import normalise_garnet_cations
from phasetools.calculators.garnet.mdf import (
    GarnetMDFCalculator,
    GarnetMDFResult,
    garnet_fractions_to_mg_fe_mn_ca,
    kho_aware_exchange_potentials,
)
from phasetools.core.base import MAGEMinBase

FOUR_ROLES = {
    "pyrope": "py",
    "almandine": "alm",
    "spessartine": "spss",
    "grossular": "gr",
}
THREE_ROLES = {"pyrope": "py", "almandine": "alm", "grossular": "gr"}


class TestKhoAwareProjections(unittest.TestCase):
    @staticmethod
    def configure_fake_bulk(calc, oxides=("FeO", "O"), values=(8.0, 0.5)):
        calc._configured_bulk_oxides = tuple(oxides)
        calc._configured_bulk_values = tuple(values)
        calc._configured_bulk_basis = "mol"
        standardized = dict(zip(oxides, values, strict=True))
        for oxide in ("MgO", "Al2O3", "CaO", "MnO", "FeO"):
            standardized.setdefault(oxide, 1.0)
        calc._standardized_bulk_molar_oxides = tuple(standardized)
        calc._standardized_bulk_molar_values = tuple(standardized.values())
        calc._standardized_bulk_oxygen_mol = (
            float(values[oxides.index("O")]) if "O" in oxides else 0.0
        )

    def test_calculator_inherits_directly_from_base(self):
        self.assertIs(GarnetMDFCalculator.__bases__[0], MAGEMinBase)

    def test_removed_legacy_api_is_not_exposed_by_package(self):
        removed = (
            "SpearMDFCationCalculator",
            "SpearMDFCationGridResult",
            "SpearMDFCationResult",
            "EBCInversionResult",
            "MDFGridResult",
            "garnet_fractions_from_apfu",
            "scan_mdf_composition_surfaces",
        )
        for name in removed:
            with self.subTest(name=name):
                self.assertFalse(hasattr(phasetools, name))

    def test_kho_aware_gamma_projection_uses_named_oxide_order(self):
        gamma = {"CaO": 8.0, "O": 7.0, "FeO": 4.0, "MnO": 5.0, "Al2O3": 2.0, "MgO": 1.0}
        exchanges = kho_aware_exchange_potentials(gamma, FOUR_ROLES)
        self.assertEqual(
            exchanges,
            {"alm": 9.0, "kho": 13.0, "spss": 12.0, "gr": 21.0},
        )
        # mb has no Mn component and does not require matrix MnO.
        mb = kho_aware_exchange_potentials(
            {key: value for key, value in gamma.items() if key != "MnO"},
            THREE_ROLES,
        )
        self.assertEqual(mb, {"alm": 9.0, "kho": 13.0, "gr": 21.0})

    def test_kho_fraction_conversion_counts_three_mg_and_two_total_fe(self):
        fractions = {"py": 0.2, "alm": 0.4, "spss": 0.1, "gr": 0.2, "kho": 0.1}
        composition = garnet_fractions_to_mg_fe_mn_ca(
            fractions, component_names=FOUR_ROLES
        )
        self.assertEqual(set(composition), {"Mg", "Fe", "Mn", "Ca"})
        expected = {"Mg": 0.28125, "Fe": 0.4375, "Mn": 0.09375, "Ca": 0.1875}
        for key, value in expected.items():
            self.assertAlmostEqual(composition[key], value)
        self.assertAlmostEqual(sum(composition.values()), 1.0)

    def test_endmember_projection_uses_shared_cation_normaliser(self):
        fractions = {"py": 0.2, "alm": 0.4, "spss": 0.1, "gr": 0.2, "kho": 0.1}
        with patch(
            "phasetools.calculators.garnet.composition.normalise_garnet_cations",
            wraps=normalise_garnet_cations,
        ) as normalise:
            composition = garnet_fractions_to_mg_fe_mn_ca(fractions)
        normalise.assert_called_once()
        np.testing.assert_allclose(
            list(normalise.call_args.args[0].values()), [0.9, 1.4, 0.3, 0.6]
        )
        self.assertEqual(tuple(composition), ("Mg", "Fe", "Mn", "Ca"))

    def test_fe2_basis_excludes_kho_iron_and_preserves_almandine_iron(self):
        fractions = {"py": 0.2, "alm": 0.4, "spss": 0.1, "gr": 0.2, "kho": 0.1}
        composition = garnet_fractions_to_mg_fe_mn_ca(
            fractions, component_names=FOUR_ROLES, fe_basis="fE2+"
        )
        expected = {"Mg": 0.3, "Fe": 0.4, "Mn": 0.1, "Ca": 0.2}
        for key, value in expected.items():
            self.assertAlmostEqual(composition[key], value)
        self.assertAlmostEqual(sum(composition.values()), 1.0)

    def test_projection_rejects_unsupported_fe_basis(self):
        with self.assertRaisesRegex(ValueError, "fe_basis must be"):
            garnet_fractions_to_mg_fe_mn_ca(
                {"py": 0.5, "alm": 0.2, "gr": 0.2, "kho": 0.1},
                component_names=THREE_ROLES,
                fe_basis="Fe3+",
            )

    def test_mn_free_projection_reports_zero_mn_and_closes_remaining_cations(self):
        result = garnet_fractions_to_mg_fe_mn_ca(
            {"py": 0.2, "alm": 0.4, "gr": 0.3, "kho": 0.1},
            component_names=THREE_ROLES,
        )
        self.assertEqual(result["Mn"], 0.0)
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_unknown_active_endmember_and_missing_kho_oxygen_fail_clearly(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ()
        calc_output = SimpleNamespace(
            ph=("g",),
            SS_vec=(
                SimpleNamespace(emNames=("py", "alm", "spss", "gr", "kho", "other")),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported active garnet endmember"):
            calc.discover_garnet_endmembers(calc_output)

        with self.assertRaisesRegex(ValueError, "missing required oxide"):
            kho_aware_exchange_potentials(
                {"MgO": 1.0, "FeO": 2.0, "Al2O3": 3.0}, THREE_ROLES
            )

    def test_projection_rejects_unknown_zero_valued_endmember(self):
        with self.assertRaisesRegex(ValueError, "Unsupported garnet endmember"):
            garnet_fractions_to_mg_fe_mn_ca(
                {"py": 0.5, "alm": 0.2, "gr": 0.2, "kho": 0.1, "unknown": 0.0},
                component_names=THREE_ROLES,
            )

    def test_absent_garnet_is_reported_before_active_model_validation(self):
        calc = object.__new__(GarnetMDFCalculator)
        with self.assertRaisesRegex(ValueError, r"garnet \('g'\) is absent"):
            calc.discover_garnet_endmembers(SimpleNamespace(ph=("bi", "q")))

    def test_mb_discovery_drops_unavailable_spessartine_role(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ()
        output = SimpleNamespace(
            ph=("g",),
            SS_vec=(SimpleNamespace(emNames=("py", "alm", "gr", "kho")),),
        )
        names = calc.discover_garnet_endmembers(output)
        self.assertEqual(names, ("py", "alm", "gr", "kho"))
        self.assertNotIn("spessartine", calc.cation_component_names)

    def test_single_point_api_returns_only_four_cation_keys(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "spss", "gr", "kho")
        calc._standardized_bulk_oxygen_mol = 0.5
        self.configure_fake_bulk(calc)
        equilibrium_output = SimpleNamespace(ph=("g", "bi"))
        calc.calculate_equilibrium = lambda *_args: equilibrium_output
        calc._read_equilibrium_endmember_fractions = lambda _out: {
            "py": 0.2,
            "alm": 0.4,
            "spss": 0.1,
            "gr": 0.2,
            "kho": 0.1,
        }
        calc.calculate_matrix_state = lambda *_args: (
            object(),
            {"FeO": 1.0, "O": 2.0, "Al2O3": 3.0},
            ("matrix",),
        )
        fractions = {"py": 0.1, "alm": 0.4, "spss": 0.2, "gr": 0.2, "kho": 0.1}
        calc.solve_kho_aware_mdf = lambda *_args, **_kwargs: (
            fractions,
            (0.01, -0.01, 0.0, 0.0),
            0.0142,
            True,
            "converged",
        )
        result = calc.calculate_compositions(10.0, 600.0)
        self.assertIsInstance(result, GarnetMDFResult)
        self.assertEqual(set(result.mdf), {"Mg", "Fe", "Mn", "Ca"})
        self.assertEqual(set(result.equilibrium), {"Mg", "Fe", "Mn", "Ca"})
        self.assertAlmostEqual(sum(result.mdf.values()), 1.0)
        self.assertAlmostEqual(sum(result.equilibrium.values()), 1.0)
        self.assertAlmostEqual(result.equilibrium["Fe"], 0.4375)
        self.assertEqual(result.fe_basis, "FeOt")
        self.assertFalse(hasattr(result, "endmember_fractions"))
        self.assertEqual(result.active_endmember_names, calc.garnet_endmember_names)
        self.assertEqual(
            result._equilibrium_endmember_fractions,
            {"py": 0.2, "alm": 0.4, "spss": 0.1, "gr": 0.2, "kho": 0.1},
        )
        self.assertEqual(result._mdf_endmember_fractions, fractions)

    def test_fe2_basis_is_used_for_both_single_point_projections(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "spss", "gr", "kho")
        calc._standardized_bulk_oxygen_mol = 0.5
        self.configure_fake_bulk(calc)
        calc.calculate_equilibrium = lambda *_args: SimpleNamespace(ph=("g",))
        calc._read_equilibrium_endmember_fractions = lambda _out: {
            "py": 0.2,
            "alm": 0.4,
            "spss": 0.1,
            "gr": 0.2,
            "kho": 0.1,
        }
        calc.calculate_matrix_state = lambda *_args: (
            object(),
            {"FeO": 1.0, "O": 2.0, "Al2O3": 3.0},
            ("matrix",),
        )
        fractions = {"py": 0.1, "alm": 0.4, "spss": 0.2, "gr": 0.2, "kho": 0.1}
        calc.solve_kho_aware_mdf = lambda *_args, **_kwargs: (
            fractions,
            (0.01, -0.01, 0.0, 0.0),
            0.0142,
            True,
            "converged",
        )
        projections = []
        project = garnet_fractions_to_mg_fe_mn_ca

        def record_projection(*args, **kwargs):
            projections.append(kwargs["fe_basis"])
            return project(*args, **kwargs)

        with patch(
            "phasetools.calculators.garnet.mdf.garnet_fractions_to_mg_fe_mn_ca",
            side_effect=record_projection,
        ):
            result = calc.calculate_compositions(10.0, 600.0, fe_basis="Fe2+")
        self.assertEqual(projections, ["Fe2+", "Fe2+"])
        self.assertEqual(result.fe_basis, "Fe2+")
        self.assertAlmostEqual(result.equilibrium["Fe"], 0.4)
        self.assertAlmostEqual(result.mdf["Fe"], 0.4)

    def test_missing_or_zero_standardized_oxygen_dispatches_restricted(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "spss", "gr", "kho")
        self.configure_fake_bulk(calc, ("FeO",), (8.0,))
        calc.calculate_equilibrium = lambda *_args: SimpleNamespace(ph=("g",))
        calc._read_equilibrium_endmember_fractions = lambda _out: {
            "py": 0.2,
            "alm": 0.4,
            "spss": 0.1,
            "gr": 0.2,
            "kho": 0.1,
        }
        calc.calculate_matrix_state = lambda *_args: (
            object(),
            {"MgO": 1.0, "FeO": 2.0, "MnO": 3.0, "CaO": 4.0},
            ("matrix",),
        )
        calls = []

        def restricted(*_args, **_kwargs):
            calls.append("restricted")
            return (
                {"py": 0.25, "alm": 0.25, "spss": 0.25, "gr": 0.25, "kho": 0.0},
                (0.0, 0.0, 0.0),
                0.0,
                True,
                "converged",
            )

        calc.solve_restricted_mdf = restricted
        calc.solve_kho_aware_mdf = lambda *_args, **_kwargs: self.fail(
            "zero standardised O must not invoke the kho-aware solver"
        )
        for oxygen in (None, 0.0):
            if oxygen is None:
                calc._standardized_bulk_molar_oxides = (
                    "FeO",
                    "MgO",
                    "CaO",
                    "MnO",
                )
                calc._standardized_bulk_molar_values = (8.0, 1.0, 1.0, 1.0)
                if hasattr(calc, "_standardized_bulk_oxygen_mol"):
                    del calc._standardized_bulk_oxygen_mol
            else:
                calc._standardized_bulk_oxygen_mol = oxygen
            result = calc.calculate_compositions(10.0, 600.0)
            self.assertEqual(result.solver_mode, "restricted")
            self.assertEqual(result._mdf_endmember_fractions["kho"], 0.0)
        self.assertEqual(calls, ["restricted", "restricted"])

    def test_positive_standardized_oxygen_dispatches_full(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "spss", "gr", "kho")
        calc._standardized_bulk_oxygen_mol = 1.0e-5
        self.configure_fake_bulk(calc, ("FeO", "O"), (8.0, 0.01))
        calc.calculate_equilibrium = lambda *_args: SimpleNamespace(ph=("g",))
        calc._read_equilibrium_endmember_fractions = lambda _out: {
            "py": 0.2,
            "alm": 0.4,
            "spss": 0.1,
            "gr": 0.2,
            "kho": 0.1,
        }
        calc.calculate_matrix_state = lambda *_args: (
            object(),
            {"FeO": 1.0, "O": 2.0, "Al2O3": 3.0},
            ("matrix",),
        )
        calls = []
        calc.solve_kho_aware_mdf = lambda *_args, **_kwargs: (
            calls.append("full")
            or (
                {"py": 0.2, "alm": 0.4, "spss": 0.1, "gr": 0.2, "kho": 0.1},
                (0.0, 0.0, 0.0, 0.0),
                0.0,
                True,
                "converged",
            )
        )
        calc.solve_restricted_mdf = lambda *_args, **_kwargs: self.fail(
            "positive standardised O must use the kho-aware solver"
        )
        result = calc.calculate_compositions(10.0, 600.0)
        self.assertEqual(result.solver_mode, "full")
        self.assertEqual(calls, ["full"])

    def test_restricted_solver_supports_model_without_kho(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = THREE_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "gr")
        seen = []

        def evaluate(_calc, _phase, fractions, *_args, **kwargs):
            seen.append((dict(fractions), tuple(kwargs["endmember_names"])))
            return SimpleNamespace(valid=True, mu=(0.0, 3.0, 9.0))

        with patch(
            "phasetools.calculators.garnet.mdf.evaluate_solution_phase_gibbs",
            side_effect=evaluate,
        ):
            fractions, _residual, _norm, _converged, _message = (
                calc.solve_restricted_mdf(
                    {"MgO": 0.0, "FeO": 1.0, "CaO": 3.0}, 10.0, 600.0
                )
            )
        self.assertEqual(set(fractions), set(calc.garnet_endmember_names))
        self.assertNotIn("kho", seen[0][0])
        self.assertEqual(seen[0][1], calc.garnet_endmember_names)

    def test_positive_oxygen_without_active_kho_is_not_restricted(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = THREE_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "gr")
        calc._standardized_bulk_oxygen_mol = 0.01
        self.configure_fake_bulk(calc, ("FeO", "O"), (8.0, 0.1))
        calc.calculate_equilibrium = lambda *_args: SimpleNamespace(ph=("g",))
        with self.assertRaisesRegex(ValueError, "Positive standardised bulk O"):
            calc.calculate_compositions(10.0, 600.0)

    def test_restricted_solver_passes_full_names_and_exact_zero_kho(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = FOUR_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "spss", "gr", "kho")
        seen = []

        def evaluate(_calc, _phase, fractions, *_args, **kwargs):
            seen.append((dict(fractions), tuple(kwargs["endmember_names"])))
            return SimpleNamespace(valid=True, mu=(0.0, 3.0, 6.0, 9.0, 0.0))

        gamma = {"MgO": 0.0, "FeO": 1.0, "MnO": 2.0, "CaO": 3.0}
        with patch(
            "phasetools.calculators.garnet.mdf.evaluate_solution_phase_gibbs",
            side_effect=evaluate,
        ):
            fractions, residual, norm, converged, _message = calc.solve_restricted_mdf(
                gamma, 10.0, 600.0
            )
        self.assertEqual(fractions["kho"], 0.0)
        self.assertTrue(converged)
        self.assertEqual(norm, 0.0)
        self.assertEqual(residual, (0.0, 0.0, 0.0))
        self.assertTrue(seen)
        self.assertEqual(seen[0][0]["kho"], 0.0)
        self.assertEqual(seen[0][1], calc.garnet_endmember_names)

    def test_calculate_equilibrium_runs_unrestricted_and_discovers_names(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.X = [1.0]
        calc.Xoxides = ["SiO2"]
        calc._configured_bulk_basis = "mol"
        calc._configured_bulk_oxides = ("SiO2",)
        calc._configured_bulk_values = (1.0,)
        calc.garnet_endmember_names = ()
        calc.rm_list = ["g"]
        output = SimpleNamespace(status=0, ph=("g", "bi"))
        calc._calculate_single_point = lambda *_args: output
        setups = []
        calc._setup_transient_bulk = lambda oxides, values, *, rm_list: setups.append(
            (tuple(oxides), tuple(values), rm_list)
        )
        discovered = []
        calc.discover_garnet_endmembers = lambda value: discovered.append(value)

        self.assertIs(calc.calculate_equilibrium(10.0, 600.0), output)
        self.assertEqual(discovered, [output])
        self.assertEqual(setups, [(("SiO2",), (1.0,), [])])

    def test_matrix_wrapper_rechecks_garnet_and_preserves_molar_basis(self):
        calc = object.__new__(GarnetMDFCalculator)
        self.configure_fake_bulk(calc, ("MgO", "FeO"), (50.0, 50.0))
        calc._configured_bulk_basis = "mol"
        calc.sys_in = "mol"
        calc.rm_list = None
        calc.garnet_endmember_names = ()
        equilibrium = SimpleNamespace(status=0, ph=("g", "bi"))
        matrix = SimpleNamespace(status=0, ph=("bi", "q"), oxides=("MgO", "FeO"))
        outputs = iter((equilibrium, matrix))
        calc._calculate_single_point = lambda *_args: next(outputs)
        setups = []
        calc.setup_bulk_composition = lambda ox, vals, sys_in, rm_list=None: (
            setups.append((tuple(ox), tuple(vals), sys_in, rm_list))
        )
        calc.discover_garnet_endmembers = lambda _output: None
        with patch(
            "phasetools.calculators.garnet.mdf.extract_gamma",
            return_value=(3.0, 4.0),
        ):
            output, gamma, phases = calc.calculate_matrix_state(10.0, 600.0)
        self.assertIs(output, matrix)
        self.assertEqual(gamma, {"MgO": 3.0, "FeO": 4.0})
        self.assertEqual(phases, ("bi", "q"))
        self.assertEqual([entry[2:] for entry in setups], [("mol", []), ("mol", ["g"])])
        self.assertIsNone(calc.rm_list)

    def test_single_point_helper_passes_configured_state_to_magemin(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.data = object()
        calc.db = "mp"
        calc.dataset = 636
        calc.X = object()
        calc.Xoxides = object()
        calc.sys_in = "mol"
        calc.rm_list = object()
        with (
            patch(
                "phasetools.calculators.garnet.mdf.jlconvert",
                side_effect=lambda _type, value: value,
            ),
            patch(
                "phasetools.calculators.garnet.mdf.MAGEMin_C",
                SimpleNamespace(
                    single_point_minimization=lambda *args, **kwargs: (
                        "output",
                        args,
                        kwargs,
                    )
                ),
            ),
        ):
            result, args, kwargs = calc._calculate_single_point(10.0, 600.0)
        self.assertEqual(result, "output")
        self.assertEqual(args[:2], (10.0, 600.0))
        self.assertIs(args[2], calc.data)
        self.assertIs(kwargs["X"], calc.X)
        self.assertIs(kwargs["Xoxides"], calc.Xoxides)
        self.assertEqual(kwargs["sys_in"], "mol")
        self.assertIs(kwargs["rm_list"], calc.rm_list)

    def test_grid_scan_shape_failure_and_nonconverged_data_are_retained(self):
        calc = object.__new__(GarnetMDFCalculator)
        requested_bases = []

        def fake_calculate(pressure_kbar, temperature_c, **kwargs):
            requested_bases.append(kwargs["fe_basis"])
            if pressure_kbar == 1.0 and temperature_c == 10.0:
                raise RuntimeError("mock point failed")
            return GarnetMDFResult(
                mdf={"Mg": 0.4, "Fe": 0.3, "Mn": 0.1, "Ca": 0.2},
                equilibrium={"Mg": 0.3, "Fe": 0.4, "Mn": 0.1, "Ca": 0.2},
                residual_kj_per_formula=(0.04, 0.0, 0.0),
                residual_norm_kj_per_formula=0.04,
                converged=False,
                status="no_convergence",
                matrix_phases=("ol", "opx"),
                equilibrium_phases=("g", "ol", "opx"),
                active_endmember_names=("py", "alm", "gr", "kho"),
                _mdf_endmember_fractions={"py": 0.2, "alm": 0.3, "gr": 0.2, "kho": 0.3},
                _equilibrium_endmember_fractions={
                    "py": 0.3,
                    "alm": 0.3,
                    "gr": 0.2,
                    "kho": 0.2,
                },
            )

        calc.calculate_compositions = fake_calculate
        self.configure_fake_bulk(calc)
        result = calc.scan_isopleths(
            (1.0, 2.0),
            (10.0, 20.0),
            n_P=2,
            n_T=2,
            fe_basis="Fe2+",
        )
        self.assertEqual(result.P_grid.shape, (2, 2))
        self.assertEqual(result.T_grid.shape, (2, 2))
        self.assertTrue(np.isnan(result.mdf["Mg"][0, 0]))
        self.assertIn("RuntimeError: mock point failed", result.status[0, 0])
        self.assertEqual(result.status[1, 1], "no_convergence")
        self.assertFalse(result.converged[1, 1])
        self.assertAlmostEqual(result.mdf["Mg"][1, 1], 0.4)
        self.assertEqual(
            result.active_endmember_names[1, 1], ("py", "alm", "gr", "kho")
        )
        self.assertEqual(result.matrix_assemblages[1, 1], ("ol", "opx"))
        self.assertEqual(result.fe_basis, "Fe2+")
        self.assertEqual(requested_bases, ["Fe2+"] * 4)

    def test_reaction_residual_aligns_active_names_and_includes_kho(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = THREE_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "gr", "kho")
        calc._standardized_bulk_oxygen_mol = 0.5

        def fake_eval(_calc, _phase, _fractions, *_args, **kwargs):
            self.assertEqual(kwargs["endmember_names"], calc.garnet_endmember_names)
            return SimpleNamespace(valid=True, mu=(10.0, 19.0, 31.0, 23.0))

        gamma = {"MgO": 1.0, "FeO": 4.0, "CaO": 8.0, "Al2O3": 2.0, "O": 7.0}
        # alm=9, gr=21, kho=13; mu exchanges are 9, 21, 13.
        with patch(
            "phasetools.calculators.garnet.mdf.evaluate_solution_phase_gibbs",
            side_effect=fake_eval,
        ):
            residual = calc._mdf_residual(
                (0.25, 0.25, 0.25, 0.25),
                calc._active_exchange(gamma),
                10.0,
                600.0,
            )
        np.testing.assert_allclose(residual, 0.0, atol=0.0)

    def test_calculation_requires_configured_bulk(self):
        calc = object.__new__(GarnetMDFCalculator)
        with self.assertRaisesRegex(ValueError, "Configure a bulk composition"):
            calc.calculate_compositions(10.0, 600.0)
        with self.assertRaisesRegex(ValueError, "Configure a bulk composition"):
            calc.scan_isopleths((10.0, 20.0), (500.0, 600.0), n_P=1, n_T=1)

    def test_fe2o3_raw_input_dispatches_from_standardized_oxygen(self):
        calc = object.__new__(GarnetMDFCalculator)
        calc.cation_component_names = THREE_ROLES.copy()
        calc.kho_name = "kho"
        calc.garnet_endmember_names = ("py", "alm", "gr", "kho")
        calls = []

        def configure(oxides, values, sys_in="wt", rm_list=None):
            self.assertEqual(tuple(oxides), ("SiO2", "Al2O3", "MgO", "Fe2O3", "CaO"))
            self.assertNotIn("FeO", oxides)
            self.assertNotIn("O", oxides)
            # Mock MAGEMin convertBulk4MAGEMin's Fe2O3 -> FeO + excess-O result.
            calc._configured_bulk_oxides = tuple(oxides)
            calc._configured_bulk_values = tuple(values)
            calc._configured_bulk_basis = sys_in
            calc._standardized_bulk_molar_oxides = (
                "SiO2",
                "Al2O3",
                "MgO",
                "FeO",
                "O",
                "CaO",
            )
            calc._standardized_bulk_molar_values = (50.0, 15.0, 10.0, 8.0, 1.0, 12.0)
            calc._standardized_bulk_oxygen_mol = 1.0

        calc.setup_bulk_composition = configure
        calc.setup_bulk_composition(
            ("SiO2", "Al2O3", "MgO", "Fe2O3", "CaO"),
            (50.0, 15.0, 10.0, 4.0, 12.0),
            sys_in="mol",
            rm_list=[],
        )
        calc.calculate_equilibrium = lambda *_args: SimpleNamespace(ph=("g",))
        calc._read_equilibrium_endmember_fractions = lambda _out: {
            "py": 0.3,
            "alm": 0.3,
            "gr": 0.2,
            "kho": 0.2,
        }
        calc.calculate_matrix_state = lambda *_args: (
            object(),
            {"FeO": 1.0, "O": 2.0, "Al2O3": 3.0},
            ("matrix",),
        )
        calc.solve_kho_aware_mdf = lambda *_args, **_kwargs: (
            calls.append("full")
            or (
                {"py": 0.3, "alm": 0.3, "gr": 0.2, "kho": 0.2},
                (0.0, 0.0, 0.0),
                0.0,
                True,
                "ok",
            )
        )
        result = calc.calculate_compositions(10.0, 600.0)
        self.assertEqual(result.solver_mode, "full")
        self.assertEqual(calls, ["full"])


if __name__ == "__main__":
    unittest.main()
