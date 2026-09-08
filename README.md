# dEA-clustering

## Quick Start

**Install**

```bash
uv pip install -e .
```

**Build STN with no clustering (Level-0)**

```bash
uv run cluster run-logs/<run-dir>
```

**Build STN with all stages enabled**

```bash
uv run cluster run-logs/<run-dir> --lsh --birch --denstream
```

**Custom config**

```bash
uv run cluster run-logs/<run-dir> --lsh --birch --denstream -c config/default.yaml
```

**Plot**

```bash
# 3D STN  →  out/<run>/stn.png
uv run cluster run-logs/<run-dir> --lsh --birch --denstream --plot

# Fitness-plotted STN  →  out/<run>/fitness-stn.png
uv run cluster run-logs/<run-dir> --lsh --birch --denstream --fplot
```
