# PIVOT V9 Statistical Audit

Status: **PASS**

| Run | Artifact | Rows | Seeds | Clusters | Finite |
| --- | --- | ---: | ---: | ---: | --- |
| `e2c-confirmatory` | `transition` | 36000 | 30 | 4500 | True |
| `e2c-development` | `transition` | 9600 | 8 | 1200 | True |
| `e3c-confirmatory` | `transition` | 192000 | 30 | 600 | True |
| `e4c-confirmatory` | `ood_report` | 8 | 30 | 8 | True |
| `e5c-confirmatory` | `transition` | 90720 | 30 | 180 | True |
| `e7c-confirmatory` | `strategic` | 3600 | 30 | 30 | True |

Bootstrap unit: `seed_or_trajectory_cluster`; transition rows are not treated as independent seeds.
