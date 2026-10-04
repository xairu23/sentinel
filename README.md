# Sentinel

Weekly probabilistic forecasts of respiratory-virus hospital admissions (influenza first, then COVID-19 and RSV) for all 50 US states, DC, Puerto Rico and the US, submitted to the CDC's public forecasting hubs.

**Research question:** does adding wastewater surveillance data improve state-level influenza hospitalization forecasts?

## Structure

| Folder | What goes there |
|---|---|
| `scripts/` | Data ingestion and plots |
| `data/` | Downloaded and cleaned data |
| `models/` | Forecasting models |
| `backtest/` | Testing models on past seasons and scoring them (WIS, coverage) |
| `submission/` | Building and validating the weekly hub file |
| `site/` | Public website with live forecasts and accuracy |
| `model-metadata/` | Draft team file for the CDC hub |
| `figures/` | Plots |

## Quick start

```bash
pip install -r requirements.txt
python scripts/ingest.py
python scripts/plot_states.py
```

## Data sources

- Hospital admissions: [CDC FluSight hub target data](https://github.com/cdcepi/FluSight-forecast-hub/tree/main/target-data)
- Wastewater: [CDC Wastewater Data for Influenza A](https://data.cdc.gov/Public-Health-Surveillance/CDC-Wastewater-Data-for-Influenza-A/ymmh-divb/about_data)

## Status

- [x] Repository structure, ingestion, example plots
- [ ] Baseline model + scoring (WIS) + backtest on 2023-24, 2024-25, 2025-26
- [ ] Checkpoint: valid backtested forecasts
- [ ] Hub registration + first submission
- [ ] Wastewater experiment, ensemble, COVID/RSV, website, automation
