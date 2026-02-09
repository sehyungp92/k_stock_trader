-- Create roles
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_writer') THEN
        CREATE ROLE trading_writer WITH LOGIN PASSWORD 'CHANGE_ME';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_reader') THEN
        CREATE ROLE trading_reader WITH LOGIN PASSWORD 'CHANGE_ME';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_reader;

-- Default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO trading_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO trading_writer;

-- Create metabase database for Metabase's internal use
CREATE DATABASE metabase;
