"""WRDS tables and columns, confirmed against the user's wrds_schema.txt (Oct 2026).

CRSP is in CIZ (v2) format. Legacy SIZ tables (crsp.msf, msenames, msedelist) still exist but
are frozen after Dec 2024; do not mix them with v2 and do not map legacy names (shrcd, exchcd,
dlret) onto v2 columns. CIZ monthly/daily returns appear to already include delisting payoffs
(secondary sources), so never compound a delisting return on top of mthret/dlyret. The meaning
of mthdelflg is CONFIRMED from crsp.metaflaginfo (flagtype DE, "Aggregate Delisting Flag"):
A/P/V = delisting return IS included in mthret; G (gap > 10 days) / M (missing) = delisting return
EXCLUDED from mthret; N = active. CRSP notes an exact equivalent of the legacy holding-period
delisting return (SFZ_MDEL) is no longer available, so G/M rows need explicit treatment.
"""

CRSP_MONTHLY = "crsp.msf_v2"
CRSP_DAILY = "crsp.dsf_v2"
CRSP_SECURITY_INFO = "crsp.stksecurityinfohist"
CRSP_DELISTINGS = "crsp.stkdelists"
CCM_LINKS = "crsp.ccmxpf_lnkhist"
COMPUSTAT_ANNUAL = "comp.funda"
COMPUSTAT_QUARTERLY = "comp.fundq"
SHORT_INTEREST = "comp.sec_shortint"
TOTAL_Q = "totalq.total_q"
FF_FACTORS_MONTHLY = "ff.factors_monthly"
FF_FACTORS_DAILY = "ff.factors_daily"

CRSP_MONTHLY_COLS = [
    "permno", "permco", "mthcaldt", "mthret", "mthretx", "mthprc", "mthcap", "mthvol",
    "mthdelflg", "mthcumfacpr", "mthcumfacshr", "shrout", "siccd",
    "securitytype", "securitysubtype", "sharetype", "usincflg", "issuertype",
    "primaryexch", "conditionaltype", "tradingstatusflg",
]

COMPUSTAT_ANNUAL_COLS = [
    "gvkey", "datadate", "fyear", "fyr", "apdedate", "fdate", "pdate",
    "at", "seq", "ceq", "pstk", "pstkrv", "pstkl", "txditc", "sale", "revt", "cogs", "xsga",
    "xint", "xrd", "ib", "oibdp", "oiadp", "ebitda", "dltt", "dlc", "che", "csho", "prcc_f",
    "ni", "oancf", "dp", "capx", "ppent", "act", "lct", "invt", "rect", "ap", "lt", "sstk",
    "prstkc", "dvt", "mib", "ivao", "ivst", "cik", "costat",
]

COMPUSTAT_QUARTERLY_COLS = ["gvkey", "datadate", "fyearq", "fqtr", "rdq", "atq", "ibq", "saleq", "cogsq", "ceqq", "seqq"]

CRSP_DELISTINGS_COLS = [
    "permno", "delistingdt", "delactiontype", "delstatustype", "delreasontype", "delpaymenttype",
    "delret", "delretmisstype", "deldtprc", "deldlydt",
]

CCM_LINK_COLS = ["gvkey", "liid", "linktype", "linkprim", "lpermno", "lpermco", "linkdt", "linkenddt"]

# Not available in this WRDS account (no library found): I/B/E/S, OptionMetrics, TAQ, RavenPack,
# Thomson 13F, Compustat Snapshot (only sample libraries). Precomputed helpers that DO exist:
# wrdsapps.signals_raw_basic/plus (WRDS characteristic signals), wrdsapps_finratio.firm_ratio,
# wrdsapps.ibcrsphist (link only, no IBES data).

CRSP_DAILY_AGG = "crsp.dsf_v2"  # aggregated server-side into per (permno, month) liquidity inputs
