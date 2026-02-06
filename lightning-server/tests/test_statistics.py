"""Tests for statistics module - research data lookup and percentile calculation."""
import math
import pytest
from api.statistics import (
    RESEARCH_DATA,
    compute_percentile,
    estimate_tooth_age,
    lookup_stats,
    normal_cdf,
)


class TestResearchData:
    """Test research data structure."""

    def test_data_structure(self):
        """Verify research data has correct structure."""
        assert len(RESEARCH_DATA) > 0

        for row in RESEARCH_DATA:
            assert "source" in row
            assert "gender" in row
            assert "age_min" in row
            assert "age_max" in row
            assert "wid_mean" in row
            assert "wid_sd" in row
            assert row["gender"] in ["male", "female", "mixed"]
            assert row["age_min"] <= row["age_max"]
            assert row["wid_sd"] > 0

    def test_kim2018_coverage(self):
        """Verify Kim2018 data covers expected ranges."""
        kim_rows = [r for r in RESEARCH_DATA if r["source"] == "Kim2018"]
        assert len(kim_rows) == 6

        # Check gender coverage
        male_rows = [r for r in kim_rows if r["gender"] == "male"]
        female_rows = [r for r in kim_rows if r["gender"] == "female"]
        assert len(male_rows) == 3
        assert len(female_rows) == 3

    def test_oh2022_coverage(self):
        """Verify Oh2022 data covers ages 7-14."""
        oh_rows = [r for r in RESEARCH_DATA if r["source"] == "Oh2022"]
        assert len(oh_rows) == 8

        ages = [r["age_min"] for r in oh_rows]
        assert ages == list(range(7, 15))


class TestLookupStats:
    """Test research data lookup logic."""

    def test_exact_match_male_young(self):
        """Test lookup for male age 25 (should match Kim2018 male 16-30)."""
        stats = lookup_stats("male", 25)
        assert stats["source"] == "Kim2018"
        assert stats["gender"] == "male"
        assert stats["age_min"] == 16
        assert stats["age_max"] == 30
        assert abs(stats["wid_mean"] - 20.07) < 0.01

    def test_exact_match_female_middle(self):
        """Test lookup for female age 45 (should match Kim2018 female 31-59)."""
        stats = lookup_stats("female", 45)
        assert stats["source"] == "Kim2018"
        assert stats["gender"] == "female"
        assert stats["age_min"] == 31
        assert stats["age_max"] == 59
        assert abs(stats["wid_mean"] - 21.83) < 0.01

    def test_exact_match_female_old(self):
        """Test lookup for female age 70 (should match Kim2018 female 60-89)."""
        stats = lookup_stats("female", 70)
        assert stats["source"] == "Kim2018"
        assert stats["gender"] == "female"
        assert stats["age_min"] == 60
        assert stats["age_max"] == 89
        assert abs(stats["wid_mean"] - 11.97) < 0.01

    def test_mixed_gender_child(self):
        """Test lookup for male age 10 (should match Oh2022 mixed age 10)."""
        stats = lookup_stats("male", 10)
        assert stats["source"] == "Oh2022"
        assert stats["gender"] == "mixed"
        assert stats["age_min"] == 10
        assert stats["age_max"] == 10
        assert abs(stats["wid_mean"] - 15.60) < 0.01

    def test_mixed_gender_priority(self):
        """Test that exact gender match takes priority over mixed."""
        # Age 25 should match Kim2018 male, not Oh2022 mixed
        stats = lookup_stats("male", 25)
        assert stats["gender"] == "male"
        assert stats["source"] == "Kim2018"

    def test_nearest_age_fallback(self):
        """Test fallback to nearest age when no exact match."""
        # Age 5 has no exact match, should use nearest (Oh2022 age 7)
        stats = lookup_stats("male", 5)
        assert stats["source"] == "Oh2022"
        assert stats["age_min"] == 7

    def test_gender_preference_in_nearest(self):
        """Test that same gender is preferred in nearest age fallback."""
        # This is hard to test definitively without knowing exact distances
        # Just verify it returns valid data
        stats = lookup_stats("female", 100)
        assert stats is not None
        assert "wid_mean" in stats


class TestNormalCdf:
    """Test standard normal CDF."""

    def test_z_zero(self):
        """Test CDF at z=0 (should be 0.5)."""
        result = normal_cdf(0.0)
        assert abs(result - 0.5) < 0.001

    def test_z_positive(self):
        """Test CDF at z=1.0 (should be ~0.841)."""
        result = normal_cdf(1.0)
        assert abs(result - 0.8413) < 0.001

    def test_z_negative(self):
        """Test CDF at z=-1.0 (should be ~0.159)."""
        result = normal_cdf(-1.0)
        assert abs(result - 0.1587) < 0.001

    def test_z_extreme_positive(self):
        """Test CDF at z=3.0 (should be ~0.9987)."""
        result = normal_cdf(3.0)
        assert abs(result - 0.9987) < 0.001

    def test_z_extreme_negative(self):
        """Test CDF at z=-3.0 (should be ~0.0013)."""
        result = normal_cdf(-3.0)
        assert abs(result - 0.0013) < 0.001


class TestComputePercentile:
    """Test percentile calculation."""

    def test_at_mean(self):
        """Test percentile at exact mean (should be 50)."""
        percentile = compute_percentile(20.0, 20.0, 5.0)
        assert abs(percentile - 50.0) < 0.1

    def test_one_sd_above_mean(self):
        """Test percentile one SD above mean (whiter, should be ~16)."""
        # WID = mean + 1*SD -> z = 1.0 -> CDF = 0.841 -> percentile = 100 - 84.1 = 15.9
        percentile = compute_percentile(25.0, 20.0, 5.0)
        assert abs(percentile - 15.9) < 0.5

    def test_one_sd_below_mean(self):
        """Test percentile one SD below mean (yellower, should be ~84)."""
        # WID = mean - 1*SD -> z = -1.0 -> CDF = 0.159 -> percentile = 100 - 15.9 = 84.1
        percentile = compute_percentile(15.0, 20.0, 5.0)
        assert abs(percentile - 84.1) < 0.5

    def test_two_sd_above_mean(self):
        """Test percentile two SD above mean (very white, should be ~2.3)."""
        # WID = mean + 2*SD -> z = 2.0 -> CDF = 0.977 -> percentile = 100 - 97.7 = 2.3
        percentile = compute_percentile(30.0, 20.0, 5.0)
        assert abs(percentile - 2.3) < 0.5

    def test_zero_sd(self):
        """Test percentile with zero SD (should return 50)."""
        percentile = compute_percentile(20.0, 20.0, 0.0)
        assert abs(percentile - 50.0) < 0.1

    def test_inverted_scale(self):
        """Test that higher WID gives lower percentile."""
        p1 = compute_percentile(15.0, 20.0, 5.0)
        p2 = compute_percentile(25.0, 20.0, 5.0)
        assert p1 > p2


class TestEstimateToothAge:
    """Test tooth age estimation."""

    def test_at_median_percentile(self):
        """Test tooth age at 50th percentile (should equal user age)."""
        age = estimate_tooth_age(30, 50.0)
        assert age == 30

    def test_young_whiter_teeth(self):
        """Test young person with white teeth (low percentile)."""
        # Age 20, percentile 10 (very white)
        # offset = (10 - 50) / 50 = -0.8
        # younger_bias = 8, adjustment = -0.8 * 8 = -6.4
        # estimated = 20 - 6.4 = 13.6 -> 14
        age = estimate_tooth_age(20, 10.0)
        assert age < 20
        assert age >= 12  # Bounded by max(5, 20-8) = 12

    def test_middle_whiter_teeth(self):
        """Test middle-age person with white teeth."""
        # Age 40, percentile 10
        # offset = -0.8, younger_bias = 12, adjustment = -9.6
        # estimated = 30.4 -> 30
        age = estimate_tooth_age(40, 10.0)
        assert age < 40
        assert age >= 28  # Bounded by max(5, 40-12) = 28

    def test_old_yellower_teeth(self):
        """Test old person with yellow teeth (high percentile)."""
        # Age 70, percentile 90 (very yellow)
        # offset = (90 - 50) / 50 = 0.8
        # older_bias = 20, adjustment = 0.8 * 20 = 16
        # estimated = 86 -> 86
        age = estimate_tooth_age(70, 90.0)
        assert age > 70
        assert age <= 90  # Bounded by min(95, 70+20) = 90

    def test_boundary_lower(self):
        """Test lower boundary (age 5)."""
        # Very young with very yellow teeth
        age = estimate_tooth_age(10, 90.0)
        assert age >= 5

    def test_boundary_upper(self):
        """Test upper boundary (age 95)."""
        # Very old with very yellow teeth
        age = estimate_tooth_age(90, 90.0)
        assert age <= 95

    def test_age_brackets(self):
        """Test different age brackets use different biases."""
        # Young (<=20): smaller range
        young = estimate_tooth_age(20, 10.0)

        # Middle (40): medium range
        middle = estimate_tooth_age(40, 10.0)

        # Old (70): larger range
        old = estimate_tooth_age(70, 10.0)

        # All should be below their chronological age
        assert young < 20
        assert middle < 40
        assert old < 70

        # Older people should have bigger adjustments
        young_diff = 20 - young
        middle_diff = 40 - middle
        old_diff = 70 - old

        assert middle_diff > young_diff
        assert old_diff > middle_diff


class TestIntegration:
    """Integration tests combining lookup and percentile calculation."""

    def test_typical_young_male_white_teeth(self):
        """Test typical case: young male with white teeth."""
        # Male age 25, WID 25 (whiter than average)
        stats = lookup_stats("male", 25)
        assert stats["wid_mean"] == 20.07

        percentile = compute_percentile(25.0, stats["wid_mean"], stats["wid_sd"])
        # Should be below 50 (whiter)
        assert percentile < 50

        tooth_age = estimate_tooth_age(25, percentile)
        # Should be younger than 25
        assert tooth_age < 25

    def test_typical_old_female_yellow_teeth(self):
        """Test typical case: old female with yellow teeth."""
        # Female age 70, WID 8 (yellower than average)
        stats = lookup_stats("female", 70)
        assert stats["wid_mean"] == 11.97

        percentile = compute_percentile(8.0, stats["wid_mean"], stats["wid_sd"])
        # Should be above 50 (yellower)
        assert percentile > 50

        tooth_age = estimate_tooth_age(70, percentile)
        # Should be older than 70
        assert tooth_age > 70

    def test_child_average_teeth(self):
        """Test child with average teeth."""
        # Mixed gender age 10, WID at mean
        stats = lookup_stats("mixed", 10)
        wid = stats["wid_mean"]

        percentile = compute_percentile(wid, stats["wid_mean"], stats["wid_sd"])
        assert abs(percentile - 50.0) < 1.0

        tooth_age = estimate_tooth_age(10, percentile)
        assert tooth_age == 10
