from datetime import date

from mlb_winners.bvp_hrr_recap import Candidate, format_recap


def _candidate(*, batter: str, pitcher_era: str) -> Candidate:
    return Candidate(
        batter_id=1,
        batter=batter,
        team="Away",
        opponent="Home",
        pitcher_id=2,
        pitcher="Pitcher",
        game_pk=3,
        game="Away at Home",
        lineup_source="posted",
        bvp_hits=3,
        bvp_at_bats=5,
        bvp_avg=".600",
        bvp_ops="1.000",
        season_avg=".260",
        season_ops=".700",
        pitcher_era=pitcher_era,
    )


def test_format_recap_reports_opposing_era_above_450_subset() -> None:
    high_era_hit = _candidate(batter="High ERA hit", pitcher_era="4.51")
    high_era_miss = _candidate(batter="High ERA miss", pitcher_era="5.00")
    cutoff_era_hit = _candidate(batter="At cutoff", pitcher_era="4.50")

    recap = format_recap(
        date(2026, 7, 27),
        [
            (high_era_hit, {"hits": 1, "runs": 1, "rbi": 0}, True),
            (high_era_miss, {"hits": 0, "runs": 0, "rbi": 1}, False),
            (cutoff_era_hit, {"hits": 2, "runs": 0, "rbi": 0}, True),
        ],
        [],
    )

    assert "Opposing ERA > 4.50 subset: 1/2 hit 1.5 H+R+RBI (50.0%)" in recap
    assert "- HIT: High ERA hit" in recap
    assert "- MISS: High ERA miss" in recap
    assert "At cutoff" not in recap.rsplit("Opposing ERA > 4.50 subset:", 1)[1]
