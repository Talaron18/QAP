"""WRDS table names. Every entry is a CANDIDATE until confirmed with scripts/wrds_inspect.py.

CRSP moved from SIZ (legacy: msf/dsf/msenames/msedelist, frozen after Dec 2024) to CIZ
(v2). CIZ column names were NOT verified; do not map legacy names (shrcd, exchcd, dlret...)
onto CIZ. CIZ monthly/daily returns appear to already include delisting payoffs, so do not
compound a legacy DLRET on top without checking the delisting flags.
"""

CANDIDATE_TABLES = {
    "crsp_monthly": ["crsp.msf_v2", "crsp.msf"],
    "crsp_daily": ["crsp.dsf_v2", "crsp.dsf"],
    "crsp_names": ["crsp.stksecurityinfohist", "crsp.msenames"],
    "crsp_delist_legacy": ["crsp.msedelist"],
    "crsp_flag_metadata": ["crsp.metaflaginfo", "crsp.metasiztociz"],
    "ccm_link": ["crsp.ccmxpf_lnkhist"],
    "compustat_annual": ["comp.funda"],
    "compustat_quarterly": ["comp.fundq"],
    "ibes_summary": ["ibes.statsum_epsus"],
    "ibes_detail": ["ibes.det_epsus"],
    "ibes_crsp_link": ["wrdsapps_link_crsp_ibes.ibcrsphist"],
}
