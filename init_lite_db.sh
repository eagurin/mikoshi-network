#!/bin/bash
set -e
# Create application user
echo "Creating role ${LITELLM_APP_USER}..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE USER "$LITELLM_APP_USER" WITH PASSWORD '$LITELLM_APP_PASSWORD';
    CREATE DATABASE "$POSTGRES_DB" OWNER "$LITELLM_APP_USER";
    GRANT ALL PRIVILEGES ON DATABASE "$POSTGRES_DB" TO "$LITELLM_APP_USER";
EOSQL
echo "Role ${LITELLM_APP_USER} and database ${POSTGRES_DB} created successfully."
# Create the vector extension
echo "Creating vector extension..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL
echo "Vector extension created successfully."
