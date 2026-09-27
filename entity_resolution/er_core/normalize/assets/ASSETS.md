# Normalization Assets — Provenance Statement

All files in this directory are **hand-curated by the team as common-language
knowledge of business naming and postal address conventions**. No external
dataset, database, API, geocoding service, or internet lookup was consulted to
produce them.

| Asset | Content | Rationale |
|---|---|---|
| `legal_suffixes.txt` | Legal-entity suffix families (US: corp/inc/ltd/llc/co; India: pvt/ltd; France: sarl/sas/sa/eurl) | Unify suffix variants so `ABC Corp` and `ABC Corporation` block together. Deliberately excludes generic business-type words (Traders, Enterprises, Stores, …) which are frequently part of the trading name rather than removable suffixes. |
| `street_types.txt` | Street-type abbreviation families (US: st/rd/ave/blvd; India: nagar/colony/sector; France: rue/chemin/quai) | Unify `Rd`/`Road`, `Rue`/`R.` style variants |
| `stopwords.txt` | Common address function words | Prevent `of`/`the`/`near` from polluting similarity features |

## Country-agnosticism guarantee

The canonicalizer never detects or filters by country. The tables are **unions
across conventions**; overlapping or redundant entries are harmless. The
pipeline degrades gracefully if any table is emptied: an ablation with truncated
asset tables must show only a small score drop. Every table entry is a
linguistic convention, not a record lookup.
