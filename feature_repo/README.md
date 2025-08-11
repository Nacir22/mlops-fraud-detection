# Feast Feature Repo (local)

- **Offline store**: local parquet (`data/processed/features.parquet`)
- **Online store**: Redis (via Docker Compose)

Common commands:
```bash
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
feast registry-dump
```
