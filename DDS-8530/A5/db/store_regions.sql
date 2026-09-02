CREATE TABLE store_regions (
    store_id INT PRIMARY KEY,
    store_name VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    latitude NUMERIC(9, 6),
    longitude NUMERIC(9, 6),
    region_code VARCHAR(10),
    region_name VARCHAR(50),
    risk_level VARCHAR(20)
);

-- Generate and insert 300,000 synthetic rows into store_regions
INSERT INTO store_regions (
    store_id, 
    store_name, 
    city, 
    state, 
    latitude, 
    longitude, 
    region_code, 
    region_name, 
    risk_level
)
SELECT 
    generate_series AS store_id,
    'Store_' || generate_series AS store_name,
    CASE (generate_series % 5) 
        WHEN 0 THEN 'Houston' 
        WHEN 1 THEN 'Dallas' 
        WHEN 2 THEN 'Austin' 
        WHEN 3 THEN 'San Antonio' 
        ELSE 'El Paso' 
    END AS city,
    'Texas' AS state,
    25.0 + (random() * 10.0) AS latitude,
    -106.0 + (random() * 12.0) AS longitude,
    'TX-' || (generate_series % 10) AS region_code,
    'Texas' AS region_name,
    CASE (generate_series % 3) 
        WHEN 0 THEN 'High' 
        WHEN 1 THEN 'Moderate' 
        ELSE 'Low' 
    END AS risk_level
FROM generate_series(1, 300000);