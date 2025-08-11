from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String
from feast import Feature
from feast.data_source import FileOptions

# Offline data created by src.fraud.features -> data/processed/features.parquet
source = FileSource(
    name="transactions_source",
    path="../data/processed/features.parquet",
    timestamp_field="event_timestamp",
    file_options=FileOptions(file_format="parquet"),
)

customer = Entity(name="customer_id", join_keys=["customer_id"])

transactions_fv = FeatureView(
    name="transactions_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="balance_delta", dtype=Float32),
        Field(name="type_cash_out", dtype=Int64),
        Field(name="type_payment", dtype=Int64),
        Field(name="type_transfer", dtype=Int64),
        Field(name="type_debit", dtype=Int64),
        Field(name="type_cash_in", dtype=Int64),
        Field(name="step", dtype=Int64),
    ],
    online=True,
    source=source,
    tags={"team": "fraud"},
)
