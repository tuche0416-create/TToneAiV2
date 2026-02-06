"""Research data, percentile calculation, and tooth age estimation.

CRITICAL: This data MUST match data.csv exactly.
Pre-computed WID statistics use error propagation from Lab component variances.
"""
import math

# Pre-computed from data.csv using error propagation:
# WID_mean = 0.511*L_mean + (-2.324)*a_mean + (-1.100)*b_mean
# WID_sd = sqrt((0.511*L_sd)^2 + (2.324*a_sd)^2 + (1.100*b_sd)^2)
RESEARCH_DATA = [
    # Kim (2018) - Primary source, gendered
    {"source": "Kim2018", "gender": "male",   "age_min": 16, "age_max": 30, "wid_mean": 20.07, "wid_sd": 6.04},
    {"source": "Kim2018", "gender": "male",   "age_min": 31, "age_max": 59, "wid_mean": 16.61, "wid_sd": 6.09},
    {"source": "Kim2018", "gender": "male",   "age_min": 60, "age_max": 89, "wid_mean": 5.15,  "wid_sd": 7.33},
    {"source": "Kim2018", "gender": "female", "age_min": 16, "age_max": 30, "wid_mean": 23.18, "wid_sd": 5.71},
    {"source": "Kim2018", "gender": "female", "age_min": 31, "age_max": 59, "wid_mean": 21.83, "wid_sd": 5.54},
    {"source": "Kim2018", "gender": "female", "age_min": 60, "age_max": 89, "wid_mean": 11.97, "wid_sd": 6.41},
    # Oh et al. (2022) - Ages 7-14, mixed gender
    {"source": "Oh2022", "gender": "mixed", "age_min": 7,  "age_max": 7,  "wid_mean": 11.65, "wid_sd": 4.94},
    {"source": "Oh2022", "gender": "mixed", "age_min": 8,  "age_max": 8,  "wid_mean": 15.37, "wid_sd": 4.54},
    {"source": "Oh2022", "gender": "mixed", "age_min": 9,  "age_max": 9,  "wid_mean": 15.63, "wid_sd": 4.80},
    {"source": "Oh2022", "gender": "mixed", "age_min": 10, "age_max": 10, "wid_mean": 15.60, "wid_sd": 4.88},
    {"source": "Oh2022", "gender": "mixed", "age_min": 11, "age_max": 11, "wid_mean": 14.41, "wid_sd": 5.03},
    {"source": "Oh2022", "gender": "mixed", "age_min": 12, "age_max": 12, "wid_mean": 14.94, "wid_sd": 4.55},
    {"source": "Oh2022", "gender": "mixed", "age_min": 13, "age_max": 13, "wid_mean": 15.02, "wid_sd": 5.13},
    {"source": "Oh2022", "gender": "mixed", "age_min": 14, "age_max": 14, "wid_mean": 16.07, "wid_sd": 5.33},
    # Han et al. (2023) - Only 50-59 and 80-89 are reliable (60-79 have anomalous a_SD)
    {"source": "Han2023", "gender": "mixed", "age_min": 50, "age_max": 59, "wid_mean": 3.50,  "wid_sd": 16.61},
    {"source": "Han2023", "gender": "mixed", "age_min": 80, "age_max": 89, "wid_mean": 7.62,  "wid_sd": 7.92},
]


def lookup_stats(gender: str, age: int) -> dict:
    """Find best matching research data for given gender and age.

    Priority order:
    1. Exact gender + age range match
    2. Mixed gender + age range match
    3. Nearest age (same gender preferred)

    Args:
        gender: "male" or "female"
        age: User's age in years

    Returns:
        Research data row (dict with source, gender, age_min, age_max, wid_mean, wid_sd)
    """
    # Priority 1: Exact gender + age range
    for row in RESEARCH_DATA:
        if row["gender"] == gender and row["age_min"] <= age <= row["age_max"]:
            return row

    # Priority 2: Mixed gender match
    for row in RESEARCH_DATA:
        if row["gender"] == "mixed" and row["age_min"] <= age <= row["age_max"]:
            return row

    # Priority 3: Nearest age, same gender first
    best = None
    best_dist = float('inf')
    for row in RESEARCH_DATA:
        mid = (row["age_min"] + row["age_max"]) / 2
        dist = abs(age - mid)
        gender_bonus = 0 if row["gender"] == gender else 10
        total = dist + gender_bonus
        if total < best_dist:
            best_dist = total
            best = row

    return best


def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function.

    Args:
        z: Standard score (z-score)

    Returns:
        Probability P(Z <= z) for standard normal distribution
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_percentile(wid: float, wid_mean: float, wid_sd: float) -> float:
    """Calculate percentile rank of WID value.

    INVERTED: Higher WID = whiter teeth = LOWER percentile number
    (e.g., 10th percentile = top 10% whitest)

    Args:
        wid: Measured WID value
        wid_mean: Population mean WID
        wid_sd: Population standard deviation

    Returns:
        Percentile (0-100), inverted so lower = whiter
    """
    if wid_sd <= 0:
        return 50.0

    z = (wid - wid_mean) / wid_sd
    return 100 - normal_cdf(z) * 100


def estimate_tooth_age(user_age: int, percentile: float) -> int:
    """Estimate tooth appearance age from chronological age and percentile.

    Percentile deviation from 50 indicates teeth appear younger/older:
    - Lower percentile (whiter) -> younger tooth age
    - Higher percentile (yellower) -> older tooth age

    Age-dependent biases reflect clinical reality:
    - Young people: smaller range (teeth look similar)
    - Older people: larger range (more lifestyle variation)

    Args:
        user_age: Chronological age in years
        percentile: WID percentile (0-100)

    Returns:
        Estimated tooth appearance age (bounded to [5, 95])
    """
    offset = (percentile - 50) / 50

    if user_age <= 20:
        younger_bias, older_bias = 8, 12
    elif user_age <= 40:
        younger_bias, older_bias = 12, 15
    elif user_age <= 60:
        younger_bias, older_bias = 15, 18
    else:
        younger_bias, older_bias = 18, 20

    adjustment = offset * younger_bias if offset < 0 else offset * older_bias
    estimated = user_age + adjustment

    lower = max(5, user_age - younger_bias)
    upper = min(95, user_age + older_bias)

    return int(round(max(lower, min(upper, estimated))))
