"""
Delhi 20-Node Zonal Hub Dataset for Geospatial Disaggregation.

Each hub represents a primary power distribution node / transformer cluster
with geo-coordinates, base load share weights, compound growth rates,
rated transformer capacity, and demographic zone classification.
"""

DELHI_ZONAL_HUBS = [
    # --- BRPL: South & West Delhi ---
    {
        "zone_id": "BRPL_01", "name": "Dwarka Sub-City", "discom": "BRPL",
        "lat": 28.5921, "lon": 77.0460, "base_weight": 0.085,
        "growth_rate": 0.078, "rated_capacity_mw": 850,
        "type": "Rapid Expansion Residential"
    },
    {
        "zone_id": "BRPL_02", "name": "Saket & Hauz Khas", "discom": "BRPL",
        "lat": 28.5244, "lon": 77.2066, "base_weight": 0.075,
        "growth_rate": 0.042, "rated_capacity_mw": 650,
        "type": "High-End Commercial/Residential"
    },
    {
        "zone_id": "BRPL_03", "name": "Janakpuri & Uttam Nagar", "discom": "BRPL",
        "lat": 28.6219, "lon": 77.0878, "base_weight": 0.080,
        "growth_rate": 0.055, "rated_capacity_mw": 720,
        "type": "Dense Residential"
    },
    {
        "zone_id": "BRPL_04", "name": "Vasant Kunj & Aerocity", "discom": "BRPL",
        "lat": 28.5293, "lon": 77.1528, "base_weight": 0.070,
        "growth_rate": 0.065, "rated_capacity_mw": 600,
        "type": "Commercial & Hospitality"
    },
    {
        "zone_id": "BRPL_05", "name": "Nehru Place & Kalkaji", "discom": "BRPL",
        "lat": 28.5492, "lon": 77.2533, "base_weight": 0.060,
        "growth_rate": 0.038, "rated_capacity_mw": 550,
        "type": "Commercial Hub"
    },
    {
        "zone_id": "BRPL_06", "name": "Najafgarh Rural Hub", "discom": "BRPL",
        "lat": 28.6092, "lon": 76.9798, "base_weight": 0.060,
        "growth_rate": 0.082, "rated_capacity_mw": 500,
        "type": "Peri-Urban / Mixed"
    },

    # --- TPDDL: North & North-West Delhi ---
    {
        "zone_id": "TPDDL_01", "name": "Rohini Sector Complex", "discom": "TPDDL",
        "lat": 28.7495, "lon": 77.0736, "base_weight": 0.080,
        "growth_rate": 0.072, "rated_capacity_mw": 750,
        "type": "High Expansion Residential"
    },
    {
        "zone_id": "TPDDL_02", "name": "Narela Industrial Area", "discom": "TPDDL",
        "lat": 28.8527, "lon": 77.0931, "base_weight": 0.065,
        "growth_rate": 0.085, "rated_capacity_mw": 680,
        "type": "Industrial & Mega Logistics"
    },
    {
        "zone_id": "TPDDL_03", "name": "Pitampura & Shalimar Bagh", "discom": "TPDDL",
        "lat": 28.6990, "lon": 77.1384, "base_weight": 0.055,
        "growth_rate": 0.040, "rated_capacity_mw": 520,
        "type": "Established Residential"
    },
    {
        "zone_id": "TPDDL_04", "name": "Badli & Bawana Corridor", "discom": "TPDDL",
        "lat": 28.7972, "lon": 77.0441, "base_weight": 0.050,
        "growth_rate": 0.080, "rated_capacity_mw": 500,
        "type": "Heavy Industrial Substation"
    },
    {
        "zone_id": "TPDDL_05", "name": "Model Town & Civil Lines", "discom": "TPDDL",
        "lat": 28.7027, "lon": 77.1937, "base_weight": 0.040,
        "growth_rate": 0.030, "rated_capacity_mw": 400,
        "type": "Old Urban Mature"
    },

    # --- BYPL: Central & East Delhi ---
    {
        "zone_id": "BYPL_01", "name": "Laxmi Nagar & Preet Vihar", "discom": "BYPL",
        "lat": 28.6312, "lon": 77.2776, "base_weight": 0.065,
        "growth_rate": 0.030, "rated_capacity_mw": 580,
        "type": "High-Density Commercial-Domestic"
    },
    {
        "zone_id": "BYPL_02", "name": "Mayur Vihar Complex", "discom": "BYPL",
        "lat": 28.6078, "lon": 77.2994, "base_weight": 0.055,
        "growth_rate": 0.032, "rated_capacity_mw": 480,
        "type": "Saturated Residential"
    },
    {
        "zone_id": "BYPL_03", "name": "Chandni Chowk & Daryaganj", "discom": "BYPL",
        "lat": 28.6506, "lon": 77.2303, "base_weight": 0.045,
        "growth_rate": 0.018, "rated_capacity_mw": 420,
        "type": "Walled City Commercial"
    },
    {
        "zone_id": "BYPL_04", "name": "Shahdara & Seelampur", "discom": "BYPL",
        "lat": 28.6734, "lon": 77.2868, "base_weight": 0.040,
        "growth_rate": 0.035, "rated_capacity_mw": 400,
        "type": "High-Density Mixed"
    },
    {
        "zone_id": "BYPL_05", "name": "Patparganj Industrial Estate", "discom": "BYPL",
        "lat": 28.6289, "lon": 77.3168, "base_weight": 0.025,
        "growth_rate": 0.025, "rated_capacity_mw": 260,
        "type": "Light Industrial"
    },

    # --- NDMC & MES: Institutional Core ---
    {
        "zone_id": "NDMC_01", "name": "Connaught Place & Parliament", "discom": "NDMC",
        "lat": 28.6315, "lon": 77.2167, "base_weight": 0.030,
        "growth_rate": 0.012, "rated_capacity_mw": 320,
        "type": "Central Business & Government"
    },
    {
        "zone_id": "NDMC_02", "name": "Chanakyapuri Diplomatic Enclave", "discom": "NDMC",
        "lat": 28.5983, "lon": 77.1894, "base_weight": 0.015,
        "growth_rate": 0.010, "rated_capacity_mw": 180,
        "type": "Diplomatic Low-Rise"
    },
    {
        "zone_id": "MES_01", "name": "Delhi Cantonment Base", "discom": "MES",
        "lat": 28.5961, "lon": 77.1332, "base_weight": 0.005,
        "growth_rate": 0.010, "rated_capacity_mw": 80,
        "type": "Defense Strategic Grid"
    },
]
