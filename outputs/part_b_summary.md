Versions: python 3.13.15, gymnasium 1.0.0, numpy 2.5.5

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0 | 7.98 |
| no | random | 22 | 0 | 50 | 0 | 0 | 6.42 |
| no | random | 33 | 1 | 49 | 0 | 2 | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100 | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0 | 8.30 |
| yes | random | 22 | 2 | 48 | 0 | 4 | 7.76 |
| yes | random | 33 | 0 | 50 | 0 | 0 | 6.74 |
| yes | fixed | 11 | 6 | 44 | 0 | 12 | 7.28 |
| yes | fixed | 22 | 2 | 48 | 0 | 4 | 9.50 |
| yes | fixed | 33 | 2 | 48 | 0 | 4 | 7.20 |

Pooled over the 3 batches (150 episodes):
- non-slippery, random: success 1/150 = 0.7%
- non-slippery, fixed: success 150/150 = 100.0%
- slippery, random: success 2/150 = 1.3%
- slippery, fixed: success 10/150 = 6.7%
