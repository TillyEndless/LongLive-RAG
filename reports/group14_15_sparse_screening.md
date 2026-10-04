# Group14/15 sparse screening

GPU0 only; case01; W12; seed=0. retained_ratio means the fraction of routed interactions retained and computed in BF16. Skipped interactions are not computed. Order: 30%, 20%, 10%, 5%.

| group | retained ratio | status | actual retained | DINO | SSIM | PSNR |
|---:|---:|---|---:|---:|---:|---:|
| 14 | 30% | COMPLETED | 0.30085470085470084 | 0.4799855649471283 | 0.11584633247414079 | 9.21841655640877 |
| 14 | 20% | COMPLETED | 0.20170940170940171 | 0.2690114378929138 | 0.17712713695312474 | 11.474186258702066 |
| 14 | 10% | COMPLETED | 0.10256410256410256 | 0.21318432688713074 | 0.1788379124799723 | 12.358353652947276 |
| 14 | 5% | COMPLETED | 0.05811965811965812 | 0.5232377052307129 | 0.283350408688307 | 15.032237219757931 |
| 15 | 30% | COMPLETED | 0.30085470085470084 | 0.54521644115448 | 0.12387786910378662 | 9.05250709318618 |
| 15 | 20% | PENDING | — | — | — | — |
| 15 | 10% | PENDING | — | — | — | — |
| 15 | 5% | PENDING | — | — | — | — |
