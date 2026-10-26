"""Mock-based tests for generic MAGEMin thermodynamic bridge helpers."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phasetools.core.engine import (
    SolutionPhaseGibbsEvaluation,
    _validate_solution_fractions,
    evaluate_solution_phase_gibbs,
)
from phasetools.core.thermodynamics import (
    extract_candidate_phase_gibbs,
    extract_gamma,
    extract_stable_phase_gibbs,
    extract_system_gibbs,
    extract_thermodynamic_snapshot,
)


class TestThermodynamicExtraction(unittest.TestCase):
    def test_extracts_raw_fields_as_plain_python_values(self):
        output = SimpleNamespace(
            Gamma=[-1.0, -2.5], G_system=12.5, ph=["g", "q"],
            SS_vec=[SimpleNamespace(ph_name="g", G=10.0)],
            PP_vec=[SimpleNamespace(ph_name="q", G=11.0)],
            mSS_vec=[SimpleNamespace(ph_name="liq", G=13.0, deltaG=-0.2,
                                     ph_id=4, em_id=2)],
        )
        snapshot = extract_thermodynamic_snapshot(output, ["SiO2", "MgO"])
        self.assertEqual(snapshot.gamma, (-1.0, -2.5))
        self.assertEqual(snapshot.gamma_oxides, ("SiO2", "MgO"))
        self.assertEqual(snapshot.system_gibbs_energy, 12.5)
        self.assertEqual(
            [(record.phase, record.gibbs_energy) for record in snapshot.stable_phase_gibbs],
            [("g", 10.0), ("q", 11.0)],
        )
        self.assertEqual(snapshot.candidate_phase_gibbs[0].phase, "liq")
        self.assertEqual(snapshot.candidate_phase_gibbs[0].delta_g, -0.2)
        self.assertEqual(snapshot.candidate_phase_gibbs[0].phase_id, 4)

    def test_gamma_falls_back_to_equilibrium_state(self):
        with patch(
            "phasetools.core.thermodynamics.MAGEMin_C",
            SimpleNamespace(equilibrium_state=lambda _value: {"gam": [3, 4]}),
        ):
            self.assertEqual(extract_gamma(SimpleNamespace()), (3.0, 4.0))

    def test_mapping_and_missing_fields_are_supported(self):
        output = {"G_system": "9.0", "mSS_vec": [{"deltaG": "-1.5"}]}
        self.assertEqual(extract_system_gibbs(output), 9.0)
        self.assertEqual(extract_stable_phase_gibbs(output), ())
        candidates = extract_candidate_phase_gibbs(output)
        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0].gibbs_energy)
        self.assertEqual(candidates[0].delta_g, -1.5)
        self.assertIsNone(extract_gamma(output))


class TestSolutionPhaseGibbsEvaluation(unittest.TestCase):
    def test_fraction_validation_is_phase_agnostic_and_normalised(self):
        values = {"a": 0.25, "b": 0.75}
        self.assertEqual(_validate_solution_fractions(values), values)
        invalid = ({}, {"a": 1.0, "b": -0.1}, {"a": 0.5}, {"a": float("nan")})
        for fractions in invalid:
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                _validate_solution_fractions(fractions)

    def test_bridge_converts_and_extracts_gibbs_fields(self):
        class MockJulia:
            class Vector:
                def __class_getitem__(cls, item):
                    return list

            class Dict:
                def __class_getitem__(cls, items):
                    return dict

            class Float64:
                pass

            class String:
                pass

            class Symbol:
                def __init__(self, value):
                    self.value = value

            @staticmethod
            def getproperty(obj, name):
                return getattr(obj, name.value)

            @staticmethod
            def unsafe_wrap(_vector_type, values, length, own=False):
                return values[:length]

        class MockBridge:
            @staticmethod
            def define_bulk_rock(gv, values, names, basis, db):
                return gv

            @staticmethod
            def pwm_init(p, t, gv, z_b, database, simplex):
                return gv, z_b, database, simplex

            @staticmethod
            def p2x_convert(gv, database, phase, fractions):
                if phase != "mica":
                    raise AssertionError(phase)
                return SimpleNamespace()

            @staticmethod
            def pc_convert(gv, z_b, database, phase, _state):
                return SimpleNamespace(df=-123.4, mu=[1, 2, 3], dfx=[4, 5],
                                       n_em=3, n_xeos=2, sf_ok=1)

        calculator = SimpleNamespace(
            X=[1.0], original_X=[100.0], original_Xoxides=["SiO2"],
            original_sys_in="wt", db="mp", data=SimpleNamespace(
                gv=["gv"], z_b=["zb"], DB=["db"], splx_data=["simplex"]
            ),
        )
        fractions = {"a": 0.25, "b": 0.75}
        with patch("phasetools.core.engine.jl", MockJulia), patch(
            "phasetools.core.engine.jlconvert", side_effect=lambda _type, value: value,
        ), patch("phasetools.core.engine.MAGEMin_C", MockBridge):
            result = evaluate_solution_phase_gibbs(
                calculator, "mica", fractions, 10.0, 600.0,
                endmember_names=("mica_a", "mica_b", "mica_c"),
            )

        self.assertIsInstance(result, SolutionPhaseGibbsEvaluation)
        self.assertEqual(result.phase, "mica")
        self.assertEqual(result.df, -123.4)
        self.assertEqual(result.mu, (1.0, 2.0, 3.0))
        self.assertEqual(result.dfx, (4.0, 5.0))
        self.assertTrue(result.valid)
        self.assertEqual(result.endmember_names, ("mica_a", "mica_b", "mica_c"))

    def test_bridge_rejects_discovered_name_count_mismatch(self):
        class MockJulia:
            class Vector:
                def __class_getitem__(cls, item):
                    return list

            class Dict:
                def __class_getitem__(cls, items):
                    return dict

            class Float64:
                pass

            class String:
                pass

            class Symbol:
                def __init__(self, value):
                    self.value = value

            @staticmethod
            def getproperty(obj, name):
                return getattr(obj, name.value)

            @staticmethod
            def unsafe_wrap(_vector_type, values, length, own=False):
                return values[:length]

        bridge = SimpleNamespace(
            define_bulk_rock=lambda gv, *_args: gv,
            pwm_init=lambda p, t, gv, z_b, database, simplex: (gv, z_b, database, simplex),
            p2x_convert=lambda *_args: SimpleNamespace(),
            pc_convert=lambda *_args: SimpleNamespace(
                df=0.0, mu=[1.0, 2.0], dfx=[], n_em=2, n_xeos=0, sf_ok=1
            ),
        )
        calculator = SimpleNamespace(
            X=[1.0], original_X=[100.0], original_Xoxides=["SiO2"],
            original_sys_in="wt", db="mp", data=SimpleNamespace(
                gv=["gv"], z_b=["zb"], DB=["db"], splx_data=["simplex"]
            ),
        )
        with patch("phasetools.core.engine.jl", MockJulia), patch(
            "phasetools.core.engine.jlconvert", side_effect=lambda _type, value: value,
        ), patch("phasetools.core.engine.MAGEMin_C", bridge):
            with self.assertRaisesRegex(ValueError, "match the returned"):
                evaluate_solution_phase_gibbs(
                    calculator, "mica", {"a": 0.5, "b": 0.5}, 10.0, 600.0,
                    endmember_names=("only_one",),
                )


if __name__ == "__main__":
    unittest.main()
