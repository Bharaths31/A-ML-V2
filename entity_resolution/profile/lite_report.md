# Lite Profile Report

- generated: 2026-09-27T17:32:40Z
- stride: 50 (sampled field stats)

## Ground truth
- singleton rate: 0.0558 (123247)
- avg fanout: 3.666  max: 11
- S2 propensity: 0.9211  S3: 0.9314  both: 0.8524

## Sources
### train_source1
- rows: 2206821  bytes: 210069713
- countries: {'US': 26617, 'India': 17520}
- name empty: 0.0  addr empty: 0.0
- name len p50/p95: 24.0/37.0
- scripts: {'ascii_latin': 44137}

### train_source2
- rows: 5034616  bytes: 489301488
- countries: {'India': 40337, 'US': 60356}
- name empty: 0.0  addr empty: 0.03402
- name len p50/p95: 25.0/40.0
- scripts: {'devanagari': 5403, 'ascii_latin': 85398, 'extended_latin': 9892}

### train_source3
- rows: 5285603  bytes: 503705637
- countries: {'US': 63098, 'India': 42615}
- name empty: 0.0  addr empty: 0.033
- name len p50/p95: 25.0/42.0
- scripts: {'ascii_latin': 93721, 'extended_latin': 8842, 'devanagari': 3150}

### test_source1
- rows: 1732544  bytes: 175022086
- countries: {'US': 13247, 'India': 16195, 'France': 5209}
- name empty: 0.0  addr empty: 0.0
- name len p50/p95: 24.0/36.0
- scripts: {'ascii_latin': 33836, 'extended_latin': 815}

### test_source2
- rows: 4887273  bytes: 509456422
- countries: {'India': 46096, 'US': 37463, 'France': 14187}
- name empty: 0.0  addr empty: 0.02599
- name len p50/p95: 25.0/41.0
- scripts: {'ascii_latin': 79136, 'devanagari': 6270, 'extended_latin': 12340}

### test_source3
- rows: 5082316  bytes: 506002772
- countries: {'India': 48185, 'US': 38782, 'France': 14680}
- name empty: 0.0  addr empty: 0.02745
- name len p50/p95: 25.0/42.0
- scripts: {'devanagari': 3645, 'ascii_latin': 86900, 'extended_latin': 11102}
