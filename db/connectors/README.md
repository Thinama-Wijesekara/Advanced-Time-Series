# Future connector
A future Maxies API connector should read validated rows into a pandas DataFrame with the store-level columns, then call db.add_new.add_batch(dataframe, source, original_file_name).
The connector must not insert sales_new directly if it needs automatic batch metadata and validation.
add_batch validates calendar continuation, store coverage, CSV/Excel columns and existing database history before one transaction inserts all rows.
Sales rows are inserted as pending and can be processed later by a future data job.
The source should identify the connector, for example `maxies-api`.
Do not bypass existing pipeline validation in pipeline.functions.
db/ingest.py owns the database transaction so failed validation leaves sales_new and batches unchanged.
Direct SQL inserts are also supported when supplied with the required business columns.
