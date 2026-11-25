"""Mock-based tests for MAGEMin handle ownership: ``_copy_state_to`` / ``finalize``.

``Initialize_MAGEMin`` allocates ~40 MB of C-side memory per handle that
Julia's GC never reclaims; only ``Finalize_MAGEMin`` frees it.  These tests
pin the ownership rules:

* ``_copy_state_to`` copies ``db``/``dataset``/``verbose`` alongside the
  standardised state and marks the target a non-owner of the shared handle;
* ``_copy_state_to`` finalises a handle the target previously owned (the
  construct-then-copy pattern must not leak);
* ``finalize()`` frees exactly once on the owner and is idempotent;
* ``finalize()`` warns and does nothing on a non-owner.

No live Julia runtime is needed -- ``Finalize_MAGEMin`` is mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

from phasetools.core.base import MAGEMinBase


def _make(db="mp", owns=True):
    """A MAGEMinBase built via ``__new__`` (no live Julia initialisation)."""
    calc = MAGEMinBase.__new__(MAGEMinBase)
    calc.db = db
    calc.dataset = 636
    calc.verbose = False
    calc.data = MagicMock(name="handle")
    calc._owns_data = owns
    calc.X = None
    calc.Xoxides = None
    calc.sys_in = None
    calc.rm_list = None
    calc._Xoxides_py = None
    calc._X_mol_standardized = None
    calc._stoich_map = None
    return calc


class TestCopyStateToOwnership(unittest.TestCase):

    def test_copies_db_dataset_verbose_and_clears_ownership(self):
        owner = _make(db="mp")
        helper = _make(db="ig", owns=False)   # already a non-owner: no auto-free
        with patch("phasetools.core.base.MAGEMin_C") as mock_mod:
            owner._copy_state_to(helper)
        self.assertIs(helper.data, owner.data)
        self.assertEqual(helper.db, "mp")
        self.assertEqual(helper.dataset, 636)
        self.assertIs(helper.verbose, False)
        self.assertFalse(helper._owns_data)
        mock_mod.Finalize_MAGEMin.assert_not_called()

    def test_finalises_targets_previous_owned_handle(self):
        """Construct-then-copy must free the orphaned handle, not leak it."""
        owner = _make()
        helper = _make()          # owns its own (different) handle
        old_handle = helper.data
        with patch("phasetools.core.base.MAGEMin_C") as mock_mod:
            owner._copy_state_to(helper)
        mock_mod.Finalize_MAGEMin.assert_called_once_with(old_handle)
        self.assertIs(helper.data, owner.data)
        self.assertFalse(helper._owns_data)

    def test_does_not_finalise_shared_handle_on_recopy(self):
        owner = _make()
        helper = _make(owns=False)  # already a non-owner aliasing owner's handle
        helper.data = owner.data
        with patch("phasetools.core.base.MAGEMin_C") as mock_mod:
            owner._copy_state_to(helper)
        mock_mod.Finalize_MAGEMin.assert_not_called()

    def test_copy_to_self_is_noop(self):
        owner = _make()
        with patch("phasetools.core.base.MAGEMin_C") as mock_mod:
            owner._copy_state_to(owner)
        self.assertTrue(owner._owns_data)
        mock_mod.Finalize_MAGEMin.assert_not_called()


class TestFinalize(unittest.TestCase):

    def test_owner_finalises_once_and_is_idempotent(self):
        owner = _make()
        handle = owner.data
        with patch("phasetools.core.base.MAGEMin_C") as mock_mod:
            owner.finalize()
            mock_mod.Finalize_MAGEMin.assert_called_once_with(handle)
            self.assertIsNone(owner.data)
            self.assertFalse(owner._owns_data)
            owner.finalize()       # second call: no-op
            mock_mod.Finalize_MAGEMin.assert_called_once_with(handle)

    def test_non_owner_warns_and_does_not_finalise(self):
        owner = _make()
        helper = _make(owns=False)
        helper.data = owner.data
        with patch("phasetools.core.base.MAGEMin_C") as mock_mod:
            with self.assertWarns(UserWarning):
                helper.finalize()
            mock_mod.Finalize_MAGEMin.assert_not_called()
            self.assertIs(helper.data, owner.data)   # shared handle untouched


if __name__ == "__main__":
    unittest.main()
