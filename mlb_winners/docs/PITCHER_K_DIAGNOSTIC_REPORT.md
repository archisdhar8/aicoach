# Pitcher K Diagnostic Report

This report audits the pitcher strikeout prop model after the constant-0.50 XGB prediction bug.

## Final Recommendation

- blend allowed
- Production pitcher-K predictions use the baseline projection unless the ML guardrail explicitly enables a blend.

## Target Distribution

| rows | min | p25 | mean | p75 | max | std |
| --- | --- | --- | --- | --- | --- | --- |
| 13400 | 0.0000 | 3.0000 | 4.8206 | 6.0000 | 16.0000 | 2.5556 |

## Metrics

| training_rows | holdout_rows | target_min | target_max | target_mean | target_std | baseline_mae | baseline_rmse | xgb_mae | xgb_rmse | xgb_prediction_std | baseline_correlation | xgb_correlation | xgb_enabled | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13400 | 2412 | 0.0000 | 16.0000 | 4.8206 | 2.5556 | 2.0029 | 2.4843 | 1.8874 | 2.3744 | 0.9761 | 0.3217 | 0.3759 | 1 | blend allowed |

## Feature Missing / Default-Filled Rates

| feature | missing_rate | default_like_rate |
| --- | --- | --- |
| starter_velocity | 0.0000 | 1.0000 |
| starter_spin_rate | 0.0000 | 1.0000 |
| temperature_f | 0.0000 | 1.0000 |
| wind_speed_mph | 0.0000 | 1.0000 |
| park_run_factor | 0.0000 | 0.4651 |
| starter_rest_days | 0.0000 | 0.3751 |
| starter_season_starts | 0.0000 | 0.1986 |
| starter_season_era | 0.0000 | 0.1173 |
| starter_workload_trend | 0.0000 | 0.1131 |
| starter_season_fip_proxy | 0.0000 | 0.0900 |
| pitcher_kbb | 0.0000 | 0.0713 |
| starter_fip_proxy | 0.0000 | 0.0507 |
| starter_last5_fip | 0.0000 | 0.0493 |
| starter_season_whip | 0.0000 | 0.0155 |
| projected_starter_ip | 0.0000 | 0.0001 |
| expected_batters_faced | 0.0000 | 0.0000 |
| projected_k_rate | 0.0000 | 0.0000 |
| pitcher_recent_k_rate | 0.0000 | 0.0000 |
| opponent_k_rate | 0.0000 | 0.0000 |
| lineup_k_rate | 0.0000 | 0.0000 |
| opponent_xwoba | 0.0000 | 0.0000 |

## Calibration By Projection Bucket

### Baseline

| bucket | rows | projected_avg | actual_avg |
| --- | --- | --- | --- |
| 0-3 | 0 |  |  |
| 3-4 | 48 | 3.7562 | 3.3333 |
| 4-5 | 669 | 4.6605 | 3.8296 |
| 5-6 | 1199 | 5.4838 | 5.0826 |
| 6-7 | 472 | 6.3147 | 5.9619 |
| 7+ | 24 | 7.2182 | 6.2917 |

### XGB

| bucket | rows | projected_avg | actual_avg |
| --- | --- | --- | --- |
| 0-3 | 34 | 2.4265 | 2.2941 |
| 3-4 | 445 | 3.6855 | 3.5551 |
| 4-5 | 865 | 4.4695 | 4.5445 |
| 5-6 | 705 | 5.5014 | 5.5532 |
| 6-7 | 337 | 6.3352 | 6.1662 |
| 7+ | 26 | 7.2838 | 7.5769 |

## Sample Predictions

| game_pk | game_date | player_id | player_name | side | target_strikeouts | baseline_projection | xgb_projection | baseline_error | xgb_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 717968 | 2023-05-29 00:00:00 | 543243 | Sonny Gray | away | 3.0000 | 6.1404 | 6.3539 | 3.1404 | 3.3539 |
| 717969 | 2023-05-29 00:00:00 | 682243 | Bryce Miller | home | 3.0000 | 5.8967 | 5.6416 | 2.8967 | 2.6416 |
| 717969 | 2023-05-29 00:00:00 | 593334 | Domingo Germán | away | 4.0000 | 4.7071 | 4.1687 | 0.7071 | 0.1687 |
| 717970 | 2023-05-29 00:00:00 | 621112 | Paul Blackburn | home | 6.0000 | 4.9525 | 3.6116 | 1.0475 | 2.3884 |
| 717970 | 2023-05-29 00:00:00 | 647336 | Michael Soroka | away | 3.0000 | 4.8084 | 3.6906 | 1.8084 | 0.6906 |
| 717971 | 2023-05-29 00:00:00 | 656629 | Michael Kopech | home | 10.0000 | 5.4554 | 4.8087 | 4.5446 | 5.1913 |
| 717971 | 2023-05-29 00:00:00 | 656288 | Griffin Canning | away | 9.0000 | 5.2826 | 4.2583 | 3.7174 | 4.7417 |
| 717973 | 2023-05-29 00:00:00 | 676272 | Bobby Miller | home | 4.0000 | 5.9029 | 5.2312 | 1.9029 | 1.2312 |
| 717973 | 2023-05-29 00:00:00 | 592866 | Trevor Williams | away | 3.0000 | 5.4169 | 4.3634 | 2.4169 | 1.3634 |
| 717974 | 2023-05-29 00:00:00 | 669330 | Tyler Wells | home | 7.0000 | 5.4352 | 5.3292 | 1.5648 | 1.6708 |
| 717974 | 2023-05-29 00:00:00 | 671106 | Logan Allen | away | 10.0000 | 5.7268 | 5.3779 | 4.2732 | 4.6221 |
| 717975 | 2023-05-29 00:00:00 | 425794 | Adam Wainwright | home | 6.0000 | 4.6031 | 3.8708 | 1.3969 | 2.1292 |
| 717975 | 2023-05-29 00:00:00 | 622251 | Josh Staumont | away | 2.0000 | 4.8084 | 3.6827 | 2.8084 | 1.6827 |
| 717976 | 2023-05-29 00:00:00 | 573186 | Marcus Stroman | home | 8.0000 | 5.7533 | 4.9780 | 2.2467 | 3.0220 |
| 717976 | 2023-05-29 00:00:00 | 671737 | Taj Bradley | away | 8.0000 | 6.1115 | 6.4759 | 1.8885 | 1.5241 |
| 717977 | 2023-05-29 00:00:00 | 669194 | Ryne Nelson | home | 1.0000 | 5.1788 | 4.5130 | 4.1788 | 3.5130 |
| 717977 | 2023-05-29 00:00:00 | 666154 | Karl Kauffmann | away | 1.0000 | 4.0641 | 3.6455 | 3.0641 | 2.6455 |
| 717978 | 2023-05-29 00:00:00 | 543101 | Anthony DeSclafani | home | 2.0000 | 4.9603 | 4.6942 | 2.9603 | 2.6942 |
| 717978 | 2023-05-29 00:00:00 | 448179 | Rich Hill | away | 3.0000 | 5.6633 | 5.3953 | 2.6633 | 2.3953 |
| 717979 | 2023-05-29 00:00:00 | 571510 | Matthew Boyd | home | 5.0000 | 5.5054 | 5.1095 | 0.5054 | 0.1095 |
