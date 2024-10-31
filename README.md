# uvex-rate-updater
A simple command-line tool to provide updated UVEX ToO trigger count estimates, given a previous estimate and a new BNS rate. The tool uses the calculations laid out in "Electromagnetic Follow-up to Gravitational Wave Events with the UltraViolet EXplorer (UVEX)" (Criswell et al., 2024).

Requirements: numpy, scipy

Basic usage:
```
python3 uvex_rate_updater.py [Median ToO estimate] [[new BNS rate median, lower 90, upper 90]]
```

Additional documentation accessible via
```
python3 uvex_rate_updater.py -h
```