import unittest

from wealthplan.validation import CHECKS


class TestClosedForms(unittest.TestCase):
    """The optimizer reproduces closed-form solutions (see wealthplan.validation)."""


def _make(name):
    def test(self) -> None:
        for r in CHECKS[name]():
            with self.subTest(check=r.name):
                self.assertTrue(r.passed, f"{r.name}: expected {r.expected:.6g}, model {r.model:.6g}, "
                                          f"rel. error {r.rel_error:.2e} > {r.tolerance:.1e}")
    return test


for _name in CHECKS:
    setattr(TestClosedForms, f"test_{_name}", _make(_name))


if __name__ == "__main__":
    unittest.main()
