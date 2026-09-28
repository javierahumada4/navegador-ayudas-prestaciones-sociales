from rlm.verifier import ExactMatchVerifier, IMVAmountVerifier, NumericVerifier


def test_numeric_verifier_exact():
    v = NumericVerifier()
    assert v.is_correct("42", "42")
    assert v.is_correct("$42.00", "42")
    assert not v.is_correct("41", "42")
    assert not v.is_correct(None, "42")
    assert not v.is_correct("forty-two", "42")


def test_numeric_verifier_with_tolerance():
    v = NumericVerifier(tolerance=0.05)
    assert v.is_correct("3.14", "3.1416")
    assert not v.is_correct("3.0", "3.1416")


def test_imv_verifier_accepts_spanish_and_currency_formats():
    v = IMVAmountVerifier()
    assert v.is_correct("431,26 €", "431.26")
    assert v.is_correct("1.173,76 €", "1173.76")
    assert not v.is_correct("431,25 €", "431.26")


def test_imv_verifier_uses_half_up_to_cents():
    v = IMVAmountVerifier()
    assert v.is_correct("474.005", "474.01")
    assert not v.is_correct("474.005", "474.00")
    assert v.is_correct("474,004", "474.00")


def test_verify_extracts_from_full_completion():
    result = IMVAmountVerifier().verify(
        "<think>...</think><answer>474.005</answer>", "474.01"
    )
    assert result.is_correct and result.predicted == "474.005"
    missing = IMVAmountVerifier().verify("no tags at all", "18.00")
    assert not missing.is_correct and "no <answer>" in missing.detail


def test_exact_match_ignores_case_and_spacing():
    v = ExactMatchVerifier()
    assert v.is_correct("  Madrid ", "madrid")
    assert not v.is_correct("Barcelona", "Madrid")
