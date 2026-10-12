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

SHORT_INTEREST_COLS = ["gvkey", "iid", "datadate", "shortint", "shortintadj"]   # shortintadj: ONLY for the v12 check line

TOTAL_Q_COLS = ["gvkey", "datadate", "fyear", "k_int_know", "k_int_org", "k_int_offbs", "k_int", "q_tot"]

CCM_LINK_COLS = ["gvkey", "liid", "linktype", "linkprim", "lpermno", "lpermco", "linkdt", "linkenddt"]

# Not available in this WRDS account (no library found): I/B/E/S, OptionMetrics, TAQ, RavenPack,
# Thomson 13F, Compustat Snapshot (only sample libraries). Precomputed helpers that DO exist:
# wrdsapps.signals_raw_basic/plus (WRDS characteristic signals), wrdsapps_finratio.firm_ratio,
# wrdsapps.ibcrsphist (link only, no IBES data).

CRSP_DAILY_AGG = "crsp.dsf_v2"  # aggregated server-side into per (permno, month) liquidity inputs

JKP_TABLE = "contrib_global_factor.global_factor"
JKP_SIZE_GROUPS = ("mega", "large", "small")        # JKP NYSE-breakpoint groups; micro and nano are dropped
JKP_EXTRA_COLS = ["ret_exc_lead1m", "me", "size_grp"]  # not theme members: used only for the pre-registered checks
RQ_TABLE = "rq_all.rq_data"
KPSS_TABLE = "contrib_kpss.kpss_patents"
PATENT_LEVEL_TABLE = "contrib_patent_firm_link.patent_level"

# Forward paper trading: Compustat monthly security file (runs past CRSP's annual-update end)
COMP_SECM = "comp.secm"
COMP_SECM_REQUIRED = ["gvkey", "iid", "datadate", "prccm", "cshom", "trt1m", "exchg", "tpci"]
COMP_SECM_OPTIONAL = ["fic", "curcdm"]

# Single-stock ETF check (pre-registration v5): CRSP name history, pulled with select * (column names confirmed from the output)
CRSP_NAMES = "crsp.stocknames_v2"

# Alternative-data probes (pre-registration v9); small contributed tables, columns confirmed by docs/wrds_probe_altdata.txt
ARC_TABLE = "contrib_general.arc"
RISK_TEXT_TABLE = "contrib_general.as_firm_risks"
BLOCKS_TABLE = "contrib_general.blocksv"
CLASS_BOARDS_TABLE = "contrib_general.classified_boards"
CULTURE_TABLE = "contrib_corporate_culture.corp_culture"
LITIGATION_TABLE = "contrib_corp_fed_litigation.firmyear_level"

# International replication of the JKP findings (pre-registration v10): the 22 developed ex-US countries, regions fixed from the
# Global Factor Data documentation's MSCI table. Emerging and frontier markets are excluded.
REGIONS = {
    "Developed Europe": ["AUT", "BEL", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR", "IRL", "ITA", "NLD", "NOR", "PRT", "SWE"],
    "Japan": ["JPN"],
    "Other developed": ["AUS", "CAN", "HKG", "ISR", "NZL", "SGP"],
}
DEV_EX_US = [c for cs in REGIONS.values() for c in cs]
