import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Endeavour Mining — Analytics Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME & CSS — LIGHT MODE ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #F7F5F0; color: #1A1A1A; }

[data-testid="stSidebar"] {
    background-color: #1A1A2E;
    border-right: 3px solid #C9A84C;
}
[data-testid="stSidebar"] * { color: #F0EDE8 !important; }
[data-testid="stSidebar"] .stRadio label { color: #CCCCCC !important; font-size: 0.85rem; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #F0EDE8 !important; }

.hero-banner {
    background: linear-gradient(135deg, #1A1A2E 0%, #2D2350 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2rem; font-weight: 800;
    color: #C9A84C; letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}
.hero-sub {
    font-size: 0.85rem; color: #AAAACC;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.04em;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E0DDD8;
    border-top: 4px solid #C9A84C;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-val {
    font-size: 1.7rem; font-weight: 800;
    color: #B8860B; line-height: 1.1;
}
.metric-label {
    font-size: 0.68rem; color: #888;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 0.3rem;
}
.metric-delta { font-size: 0.75rem; font-weight: 700; margin-top: 0.3rem; }
.delta-up { color: #1E7E34; }
.delta-down { color: #C0392B; }

.section-title {
    font-size: 1rem; font-weight: 700;
    color: #1A1A2E; letter-spacing: 0.04em;
    text-transform: uppercase;
    border-left: 4px solid #C9A84C;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

.insight-box {
    background: #F0FFF4;
    border: 1px solid #68D391;
    border-left: 4px solid #38A169;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: #1C4532;
    font-family: 'Space Mono', monospace;
    line-height: 1.7;
}
.warning-box {
    background: #FFFAF0;
    border: 1px solid #F6AD55;
    border-left: 4px solid #DD6B20;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: #652B19;
    font-family: 'Space Mono', monospace;
    line-height: 1.7;
}
.info-box {
    background: #EBF8FF;
    border: 1px solid #90CDF4;
    border-left: 4px solid #3182CE;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: #1A365D;
    font-family: 'Space Mono', monospace;
    line-height: 1.7;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #EEEAE4;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #555;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #1A1A2E !important;
    color: #C9A84C !important;
    font-weight: 700;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label { color: #444 !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── DATA ──────────────────────────────────────────────────────
MINE_COLORS = {
    'Houndé': '#C9A84C',
    'Ity': '#4CA8C9',
    'Mana': '#C94C4C',
    'Sabodala-Massawa': '#4CC97A',
    'Lafigué': '#A84CC9',
    'Assafou': '#C9784C',
    'Kalana': '#4C78C9',
}

@st.cache_data
def load_data():
    # ── Fichier 1: Mine Operations ──
    ops = pd.DataFrame([
        ['Houndé','Q4-2025','Quarterly',2025,'Q4',1285,11525,12810,8.97,1223,1.40,88.5,47470,49060,4.50,None,19.43,10.25,None,None,None,None,None,None,None,314100,8500,1707,1882],
        ['Houndé','Q3-2025','Quarterly',2025,'Q3',1246,11472,12718,9.20,1205,1.46,85.4,48806,48418,3.93,None,17.10,7.88,None,None,None,None,None,None,None,322100,2700,1420,1475],
        ['Houndé','Q4-2024','Quarterly',2024,'Q4',1526,9307,10833,6.10,1405,3.13,79.4,108688,108146,4.70,None,12.81,5.77,None,None,None,None,None,None,None,322100,11000,922,1024],
        ['Houndé','FY-2025','Annual',2025,'Annual',5550,44802,50352,8.07,5130,1.79,86.2,256862,258921,3.93,None,16.30,7.71,None,None,None,None,None,None,None,314100,36500,1213,1354],
        ['Houndé','FY-2024','Annual',2024,'Annual',4662,38454,43116,8.25,5148,2.10,84.0,287726,287220,3.99,None,13.93,6.02,None,None,None,None,None,None,None,322100,49500,1121,1294],
        ['Ity','Q4-2025','Quarterly',2025,'Q4',2272,5713,7985,2.51,1886,1.37,90.7,73757,74260,4.63,None,19.21,6.03,None,None,None,None,None,None,None,351600,12200,1359,1523],
        ['Ity','Q3-2025','Quarterly',2025,'Q3',1991,5958,7949,2.99,1840,1.43,90.4,76789,74765,4.51,None,19.21,4.90,None,None,None,None,None,None,None,351600,9500,1142,1269],
        ['Ity','Q4-2024','Quarterly',2024,'Q4',2262,5858,8120,2.59,1955,1.45,90.2,83743,79755,4.01,None,16.78,4.91,None,None,None,None,None,None,None,306000,3500,943,987],
        ['Ity','FY-2025','Annual',2025,'Annual',8392,23761,32152,2.83,7357,1.51,90.4,318659,321080,4.40,None,18.28,4.96,None,None,None,None,None,None,None,351600,32800,1095,1197],
        ['Ity','FY-2024','Annual',2024,'Annual',7954,22465,30419,2.82,7122,1.64,91.0,342864,343809,3.87,None,17.33,4.56,None,None,None,None,None,None,None,306000,9800,890,919],
        ['Mana','Q4-2025','Quarterly',2025,'Q4',None,None,0,None,602,3.05,87.0,46327,48295,None,68.81,22.46,13.60,None,None,None,None,None,None,None,286700,17800,1806,2174],
        ['Mana','Q3-2025','Quarterly',2025,'Q3',None,None,0,None,551,2.50,85.4,39120,38135,None,65.07,24.68,13.13,None,None,None,None,None,None,None,223900,23100,1772,2377],
        ['Mana','Q4-2024','Quarterly',2024,'Q4',None,None,0,None,603,2.49,85.9,40861,40756,None,60.79,19.73,10.45,None,None,None,None,None,None,None,223900,15400,1320,1698],
        ['Mana','FY-2025','Annual',2025,'Annual',None,None,0,None,2247,2.85,85.9,172877,173499,None,65.95,24.39,12.54,None,None,None,None,None,None,None,286700,88000,1653,2160],
        ['Mana','FY-2024','Annual',2024,'Annual',185,745,930,4.03,2294,2.27,87.0,147806,147924,7.81,64.31,23.00,10.49,None,None,None,None,None,None,None,223900,33500,1514,1740],
        ['Sabodala-Massawa','Q4-2025','Quarterly',2025,'Q4',1224,6812,8036,5.57,1417,2.26,80.6,78273,78577,3.65,None,20.41,10.03,None,None,None,None,None,None,None,299000,5400,1169,1237],
        ['Sabodala-Massawa','Q3-2025','Quarterly',2025,'Q3',971,6163,7134,6.39,1378,1.60,82.4,61441,59537,4.09,None,18.29,8.70,None,None,None,None,None,None,None,299000,9100,1172,1326],
        ['Sabodala-Massawa','Q4-2024','Quarterly',2024,'Q4',1573,10889,12463,6.92,1377,2.29,70.4,69694,68852,2.66,None,17.29,8.13,None,None,None,None,None,None,None,230900,10600,1107,1261],
        ['Sabodala-Massawa','FY-2025','Annual',2025,'Annual',4253,30355,34607,7.14,5530,1.93,80.4,273533,273755,3.53,None,18.50,8.69,None,None,None,None,None,None,None,299000,42600,1092,1248],
        ['Sabodala-Massawa','FY-2024','Annual',2024,'Annual',5692,37786,43478,6.64,5061,1.89,76.2,229114,229881,2.89,None,16.54,8.61,None,None,None,None,None,None,None,230900,25300,1044,1158],
        ['Lafigué','Q4-2025','Quarterly',2025,'Q4',1822,11229,13051,6.16,1007,1.69,93.9,52521,51661,3.45,None,17.70,5.92,None,None,None,None,None,None,None,228200,2900,1419,1476],
        ['Lafigué','Q3-2025','Quarterly',2025,'Q3',1870,12802,14672,6.85,1026,1.20,93.4,37623,36709,3.00,None,15.75,4.67,None,None,None,None,None,None,None,228200,3600,1433,1530],
        ['Lafigué','Q4-2024','Quarterly',2024,'Q4',1711,8439,10150,4.93,936,2.11,93.7,59524,58543,2.93,None,13.78,6.20,None,None,None,None,None,None,None,66400,3100,748,801],
        ['Lafigué','FY-2025','Annual',2025,'Annual',6063,47977,54040,7.91,4216,1.47,93.4,187030,188898,3.01,None,16.86,4.84,None,None,None,None,None,None,None,228200,8200,1208,1251],
        ['Lafigué','FY-2024','Annual',2024,'Annual',4801,32350,37151,6.74,1779,1.83,93.8,95660,90118,2.78,None,14.17,9.56,None,None,None,None,None,None,None,66400,6000,774,844],
    ], columns=['Mine','Period','Period_Type','Year','Quarter','Tonnes_Ore_Mined_kt',
        'Tonnes_Waste_Mined_kt','Total_Tonnes_Mined_kt','Strip_Ratio','Tonnes_Milled_kt',
        'Grade_g_t','Recovery_Rate_pct','Gold_Produced_oz','Gold_Sold_oz',
        'Mining_Cost_OP','Mining_Cost_UG','Processing_Cost','GA_Cost',
        'c1','c2','c3','c4','c5','c6','c7','Sustaining_Capital_USDk','c8',
        'TCC_USD_oz','AISC_USD_oz'])

    ops = ops[['Mine','Period','Period_Type','Year','Quarter','Tonnes_Ore_Mined_kt',
               'Tonnes_Waste_Mined_kt','Total_Tonnes_Mined_kt','Strip_Ratio',
               'Tonnes_Milled_kt','Grade_g_t','Recovery_Rate_pct',
               'Gold_Produced_oz','Gold_Sold_oz','Processing_Cost','GA_Cost',
               'Sustaining_Capital_USDk','TCC_USD_oz','AISC_USD_oz']]
    ops['Gold_Produced_koz'] = ops['Gold_Produced_oz'] / 1000
    ops['Sustaining_Capital_USDm'] = ops['Sustaining_Capital_USDk'] / 1000

    # Realized gold prices per mine (from annual report)
    realized_prices = {
        ('Sabodala-Massawa',2025):3423, ('Sabodala-Massawa',2024):2339,
        ('Ity',2025):3496, ('Ity',2024):2398,
        ('Lafigué',2025):3498, ('Lafigué',2024):2607,
        ('Houndé',2025):3408, ('Houndé',2024):2462,
        ('Mana',2025):3518, ('Mana',2024):2388,
    }
    ops['Realized_Price'] = ops.apply(
        lambda r: realized_prices.get((r['Mine'], r['Year']), None), axis=1)
    ops['AISC_Margin_USD_oz'] = ops['Realized_Price'] - ops['AISC_USD_oz']
    ops['AISC_Margin_pct'] = ops['AISC_Margin_USD_oz'] / ops['Realized_Price'] * 100

    # Revenue by mine ($M) — from annual report
    rev_map = {
        ('Sabodala-Massawa',2025):938,('Sabodala-Massawa',2024):538,
        ('Ity',2025):1139,('Ity',2024):838,
        ('Lafigué',2025):662,('Lafigué',2024):235,
        ('Houndé',2025):883,('Houndé',2024):708,
        ('Mana',2025):612,('Mana',2024):356,
    }
    ops['Revenue_USDm'] = ops.apply(
        lambda r: rev_map.get((r['Mine'], r['Year']), None), axis=1)

    # ── Fichier 2: Group Financials ──
    fin = pd.DataFrame([
        ['Q4-2025','Quarterly',2025,'Q4',1273.8,-341.4,-174.2,-103.0,655.2,-13.3,-44.4,-7.2,-193.4,-28.4,-9.7,358.8,-61.7,-23.7,273.4,-150.9,122.5,67.8,0.28,680.7,225.0,0.93,625.0,609.0,476.3,157.5],
        ['Q3-2025','Quarterly',2025,'Q3',910.1,-280.6,-134.4,-70.3,424.8,-11.4,-10.4,-1.7,0.0,-9.5,-5.5,386.3,-48.9,-26.4,311.0,-109.4,201.6,167.3,0.69,465.9,158.6,0.66,393.9,308.5,165.9,453.2],
        ['Q4-2024','Quarterly',2024,'Q4',940.5,-293.9,-225.6,-64.3,356.7,-14.0,-9.1,-22.3,-199.5,-8.5,-5.2,98.1,33.6,-32.6,99.1,-202.4,-103.3,-119.1,-0.49,545.9,110.1,0.45,356.3,381.4,268.2,731.6],
        ['FY-2025','Annual',2025,'Annual',4233.9,-1179.9,-633.9,-326.6,2093.5,-52.7,-88.3,-23.2,-193.4,-64.7,-32.7,1638.5,-193.3,-101.9,1343.3,-454.2,889.1,679.2,2.80,2315.6,781.9,3.23,1907.4,1663.7,1155.9,157.5],
        ['FY-2024','Annual',2024,'Annual',2675.9,-1007.4,-609.3,-190.5,868.7,-47.3,-62.5,-151.0,-199.5,-21.4,-19.2,367.8,-142.7,-111.2,113.9,-348.5,-234.6,-293.9,-1.20,1324.6,227.3,0.93,951.7,949.6,313.3,731.6],
    ], columns=['Period','Period_Type','Year','Quarter','Revenue','Cost_of_Sales',
        'Depreciation','Royalties','Earnings_Mine_Ops','Corporate_Costs','Other_Exp',
        'Credit_Loss','Impairment','Share_Based_Comp','Exploration_Costs',
        'Earnings_from_Ops','Fin_Instrument','Finance_Costs','EBT','Tax',
        'Net_Earnings','Net_Earn_Shareholders','Basic_EPS','Adj_EBITDA',
        'Adj_Net_Earnings','Adj_EPS','Op_Cash_Flow','Op_Cash_Flow2',
        'Free_Cash_Flow','Net_Debt'])
    fin['EBITDA_Margin'] = fin['Adj_EBITDA'] / fin['Revenue'] * 100

    # ── Fichier 3: Réserves & Ressources ──
    res = pd.DataFrame([
        ['Houndé',85.0,2025,'Reserves','Proven',2.4,1.10,85],
        ['Houndé',85.0,2025,'Reserves','Probable',39.5,1.43,1811],
        ['Houndé',85.0,2025,'Reserves','P_P',41.9,1.41,1896],
        ['Houndé',85.0,2025,'Resources','M_I',57.0,1.44,2639],
        ['Houndé',85.0,2025,'Resources','Inferred',9.2,1.54,453],
        ['Houndé',85.0,2024,'Reserves','P_P',58.5,1.41,2600],
        ['Houndé',85.0,2024,'Resources','M_I',67.5,1.51,3300],
        ['Ity',85.0,2025,'Reserves','Proven',12.3,0.95,374],
        ['Ity',85.0,2025,'Reserves','Probable',64.6,1.35,2803],
        ['Ity',85.0,2025,'Reserves','P_P',76.9,1.28,3177],
        ['Ity',85.0,2025,'Resources','M_I',119.4,1.43,5483],
        ['Ity',85.0,2025,'Resources','Inferred',11.2,1.56,560],
        ['Ity',85.0,2024,'Reserves','P_P',78.6,1.41,3600],
        ['Ity',85.0,2024,'Resources','M_I',109.1,1.55,5400],
        ['Mana',85.0,2025,'Reserves','P_P',7.5,2.49,603],
        ['Mana',85.0,2025,'Resources','M_I',11.5,3.24,1196],
        ['Mana',85.0,2025,'Resources','Inferred',8.7,3.16,884],
        ['Mana',85.0,2024,'Reserves','P_P',7.6,2.79,700],
        ['Mana',85.0,2024,'Resources','M_I',15.9,3.36,1700],
        ['Sabodala-Massawa',90.0,2025,'Reserves','P_P',42.8,2.01,2768],
        ['Sabodala-Massawa',90.0,2025,'Resources','M_I',80.0,2.02,5190],
        ['Sabodala-Massawa',90.0,2025,'Resources','Inferred',27.2,2.02,1766],
        ['Sabodala-Massawa',90.0,2024,'Reserves','P_P',50.7,2.00,3300],
        ['Sabodala-Massawa',90.0,2024,'Resources','M_I',80.4,2.01,5200],
        ['Lafigué',80.0,2025,'Reserves','P_P',40.1,1.49,1926],
        ['Lafigué',80.0,2025,'Resources','M_I',38.1,1.86,2277],
        ['Lafigué',80.0,2025,'Resources','Inferred',3.4,2.12,230],
        ['Lafigué',80.0,2024,'Reserves','P_P',44.4,1.65,2400],
        ['Lafigué',80.0,2024,'Resources','M_I',46.2,1.95,2900],
        ['Assafou',100.0,2025,'Reserves','P_P',77.4,1.76,4379],
        ['Assafou',100.0,2025,'Resources','M_I',84.8,1.91,5203],
        ['Assafou',100.0,2024,'Reserves','P_P',72.8,1.76,4115],
        ['Assafou',100.0,2024,'Resources','M_I',73.6,1.95,4604],
        ['Kalana',80.0,2025,'Reserves','Probable',35.6,1.60,1829],
        ['Kalana',80.0,2025,'Resources','Indicated',46.0,1.57,2318],
    ], columns=['Mine','Ownership_pct','Year','Category','Sub_Category',
                'Tonnage_Mt','Grade_Au_g_t','Content_Au_koz'])

    # ── Fichier 4: Guidance 2026 ──
    guid = pd.DataFrame([
        ['Houndé',2026,'Production_koz',220,255],
        ['Ity',2026,'Production_koz',285,330],
        ['Mana',2026,'Production_koz',155,180],
        ['Sabodala-Massawa',2026,'Production_koz',260,305],
        ['Lafigué',2026,'Production_koz',170,195],
        ['Group',2026,'Production_koz',1090,1265],
        ['Houndé',2026,'AISC_USD_oz',1800,2000],
        ['Ity',2026,'AISC_USD_oz',1300,1500],
        ['Mana',2026,'AISC_USD_oz',2000,2250],
        ['Sabodala-Massawa',2026,'AISC_USD_oz',1350,1550],
        ['Lafigué',2026,'AISC_USD_oz',1600,1800],
        ['Group',2026,'AISC_USD_oz',1600,1800],
    ], columns=['Mine','Year','Metric','Low','High'])
    guid['Mid'] = (guid['Low'] + guid['High']) / 2

    # ── Fichier 5: Capex ──
    capex = pd.DataFrame([
        ['Houndé',2025,'Sustaining_Capital',36.5],
        ['Houndé',2025,'Non_Sustaining_Capital',95.2],
        ['Houndé',2025,'Exploration',11.0],
        ['Ity',2025,'Sustaining_Capital',32.8],
        ['Ity',2025,'Non_Sustaining_Capital',23.5],
        ['Ity',2025,'Exploration',19.4],
        ['Mana',2025,'Sustaining_Capital',88.0],
        ['Mana',2025,'Non_Sustaining_Capital',17.8],
        ['Mana',2025,'Exploration',3.6],
        ['Sabodala-Massawa',2025,'Sustaining_Capital',42.6],
        ['Sabodala-Massawa',2025,'Non_Sustaining_Capital',35.0],
        ['Sabodala-Massawa',2025,'Exploration',27.7],
        ['Lafigué',2025,'Sustaining_Capital',8.2],
        ['Lafigué',2025,'Non_Sustaining_Capital',80.0],
        ['Lafigué',2025,'Exploration',1.3],
        ['Assafou',2025,'Exploration',7.3],
        ['Other/Greenfield',2025,'Exploration',20.8],
        ['Houndé','2026_Guide','Sustaining_Capital',50.0],
        ['Houndé','2026_Guide','Non_Sustaining_Capital',60.0],
        ['Houndé','2026_Guide','Exploration',10.0],
        ['Ity','2026_Guide','Sustaining_Capital',40.0],
        ['Ity','2026_Guide','Non_Sustaining_Capital',45.0],
        ['Ity','2026_Guide','Exploration',15.0],
        ['Mana','2026_Guide','Sustaining_Capital',60.0],
        ['Mana','2026_Guide','Non_Sustaining_Capital',10.0],
        ['Mana','2026_Guide','Exploration',5.0],
        ['Sabodala-Massawa','2026_Guide','Sustaining_Capital',50.0],
        ['Sabodala-Massawa','2026_Guide','Non_Sustaining_Capital',30.0],
        ['Sabodala-Massawa','2026_Guide','Exploration',15.0],
        ['Lafigué','2026_Guide','Sustaining_Capital',30.0],
        ['Lafigué','2026_Guide','Non_Sustaining_Capital',90.0],
        ['Lafigué','2026_Guide','Exploration',10.0],
    ], columns=['Mine','Year','Type','Amount_USDm'])

    return ops, fin, res, guid, capex

ops, fin, res, guid, capex = load_data()

PLOTLY_LAYOUT = dict(
    paper_bgcolor='#FFFFFF', plot_bgcolor='#F9F8F6',
    font=dict(family='Syne', color='#333333', size=11),
    title_font=dict(family='Syne', color='#1A1A2E', size=14),
    xaxis=dict(gridcolor='#E8E5E0', zerolinecolor='#CCCCCC',
               tickfont=dict(color='#444444'), title_font=dict(color='#333333')),
    yaxis=dict(gridcolor='#E8E5E0', zerolinecolor='#CCCCCC',
               tickfont=dict(color='#444444'), title_font=dict(color='#333333')),
    legend=dict(bgcolor='#FFFFFF', bordercolor='#DDDDDD', borderwidth=1,
                font=dict(color='#333333')),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2rem;'>⛏️</div>
        <div style='font-family: Space Mono; font-size:0.65rem; color:#AAAACC; letter-spacing:0.1em;'>
        ENDEAVOUR MINING<br>ANALYTICS DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:0.7rem; color:#666; font-family:Space Mono; text-transform:uppercase; letter-spacing:0.1em;'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", [
        "🏠 Vue Générale",
        "⛏️ Opérationnel",
        "💰 Financier",
        "🪨 Réserves & Mine Life",
        "🤖 Prédiction ML",
        "💡 Capex & Stratégie"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='font-size:0.65rem; color:#8888AA; font-family:Space Mono;'>Source: EDV Annual Reports 2024-2025<br>Réalisé avec Python + Streamlit</div>", unsafe_allow_html=True)

# ─── HELPERS ───────────────────────────────────────────────────
def annual_ops(year):
    return ops[(ops['Period_Type']=='Annual') & (ops['Year']==year)]

def card(val, label, delta=None, delta_positive=True):
    delta_html = ""
    if delta:
        cls = "delta-up" if delta_positive else "delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-val">{val}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>"""

# ══════════════════════════════════════════════════════════════
# PAGE: VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════
if page == "🏠 Vue Générale":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Endeavour Mining — Analytics 2025</div>
        <div class="hero-sub">West Africa Gold Portfolio · 5 Mines · Data Science Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    fy25 = fin[fin['Period']=='FY-2025'].iloc[0]
    fy24 = fin[fin['Period']=='FY-2024'].iloc[0]
    ops25 = annual_ops(2025)

    c1,c2,c3,c4,c5 = st.columns(5)
    metrics = [
        (c1, f"${fy25['Revenue']/1000:.2f}B", "Revenue 2025", "+58% vs 2024", True),
        (c2, f"${fy25['Free_Cash_Flow']/1000:.2f}B", "Free Cash Flow", "+269% vs 2024", True),
        (c3, f"{ops25['Gold_Produced_koz'].sum()/1000:.3f}Moz", "Production 2025", "+10% vs 2024", True),
        (c4, f"${fy25['Adj_EBITDA']/fy25['Revenue']*100:.0f}%", "EBITDA Margin", "+5pp vs 2024", True),
        (c5, f"${fy25['Net_Debt']:.0f}M", "Net Debt", "-78% vs 2024", True),
    ]
    for col, val, label, delta, pos in metrics:
        col.markdown(card(val, label, delta, pos), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Production par Mine 2025 vs 2024</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        df_bar = ops[ops['Period_Type']=='Annual'][['Mine','Year','Gold_Produced_koz']].copy()
        df_bar['Year'] = df_bar['Year'].astype(str)
        fig = px.bar(df_bar, x='Mine', y='Gold_Produced_koz', color='Year',
                     barmode='group', color_discrete_map={'2025':'#C9A84C','2024':'#444'},
                     labels={'Gold_Produced_koz':'Production (koz)'},
                     title='Production annuelle par mine (koz)')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(ops25, values='Gold_Produced_koz', names='Mine',
                      color='Mine', color_discrete_map=MINE_COLORS,
                      title='Part de production 2025',
                      hole=0.55)
        fig2.update_layout(**PLOTLY_LAYOUT)
        fig2.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    ✅ Ity est la mine la plus productive (319 koz) mais Houndé a chuté de 288→257 koz (-11%).
    Lafigué, nouvelle mine en 2024, a quasi-doublé sa production (96→187 koz) — moteur de croissance.
    Mana monte de 148→173 koz grâce à la transition underground complète.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: OPÉRATIONNEL
# ══════════════════════════════════════════════════════════════
elif page == "⛏️ Opérationnel":
    st.markdown("<div class='hero-banner'><div class='hero-title'>Analyse Opérationnelle</div><div class='hero-sub'>Grade · Recovery · Strip Ratio · Efficacité métallurgique</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Grade vs Production", "Recovery Rate", "Strip Ratio", "Analyse Trimestrielle"])

    with tab1:
        st.markdown("<div class='section-title'>Grade vs Production — pourquoi Houndé sous-performe ?</div>", unsafe_allow_html=True)
        df = annual_ops(2025).copy()
        fig = px.scatter(df, x='Grade_g_t', y='Gold_Produced_koz',
                         color='Mine', size='Recovery_Rate_pct',
                         color_discrete_map=MINE_COLORS,
                         text='Mine', size_max=40,
                         labels={'Grade_g_t':'Teneur en or (g/t)', 'Gold_Produced_koz':'Production (koz)'},
                         title='Grade vs Production 2025 (taille = taux de récupération)')
        fig.update_traces(textposition='top center')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='insight-box'>
            ✅ Mana a le grade le plus élevé (2.85 g/t) mais la plus petite production (173 koz)
            → contrainte de capacité de traitement (underground).<br><br>
            ⚠️ Houndé a un grade de 1.79 g/t mais produit 257 koz → compense par le volume.
            Sa production baisse car le grade a chuté de 2.10 → 1.79 g/t (-15%).
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='info-box'>
            💡 Lafigué a le meilleur taux de récupération (93.4%) avec un grade modeste (1.47 g/t)
            → excellent process métallurgique pour une mine qui vient d'ouvrir.<br><br>
            Sabodala-Massawa : recovery améliorée de 76.2% → 80.4% grâce à l'expansion BIOX.
            </div>""", unsafe_allow_html=True)

    with tab2:
        df_rec = ops[ops['Period_Type']=='Annual'][['Mine','Year','Recovery_Rate_pct','Grade_g_t','Gold_Produced_koz']]
        fig = px.bar(df_rec, x='Mine', y='Recovery_Rate_pct', color='Year',
                     barmode='group', color_discrete_map={2025:'#C9A84C', 2024:'#444'},
                     title='Taux de récupération par mine 2024 vs 2025 (%)',
                     labels={'Recovery_Rate_pct':'Recovery Rate (%)'})
        fig.add_hline(y=90, line_dash='dash', line_color='#4CC97A',
                      annotation_text='Benchmark 90%', annotation_position='top right')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        ✅ Lafigué (93.4%) et Ity (90.4%) dépassent le benchmark 90%.<br>
        ⚠️ Sabodala-Massawa reste sous 80.4% → le minerai réfractaire est plus difficile à traiter,
        d'où l'investissement dans l'expansion BIOX. C'est un vrai enjeu technique pour la mine.
        </div>""", unsafe_allow_html=True)

    with tab3:
        df_strip = ops[ops['Period_Type'].isin(['Quarterly','Annual'])].copy()
        df_strip = df_strip[df_strip['Strip_Ratio'].notna() & (df_strip['Mine'] != 'Mana')]
        fig = px.bar(df_strip[df_strip['Period_Type']=='Quarterly'],
                     x='Period', y='Strip_Ratio', color='Mine',
                     color_discrete_map=MINE_COLORS, barmode='group',
                     title='Strip Ratio trimestriel — évolution de la dureté des mines',
                     labels={'Strip_Ratio':'Strip Ratio (t waste / t ore)'})
        fig.add_hline(y=7, line_dash='dot', line_color='#C94C4C',
                      annotation_text='Seuil coût critique (~7x)', annotation_position='top left')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='warning-box'>
        ⚠️ Houndé : Strip Ratio de 8.97 en Q4-2025 → la mine extrait presque 9t de stériles
        pour 1t de minerai. C'est un signal fort : les coûts vont continuer à monter en 2026.
        La guidance AISC 2026 de $1,800-2,000/oz confirme cette tendance.
        </div>""", unsafe_allow_html=True)

    with tab4:
        mines_sel = st.multiselect("Sélectionner les mines", ops['Mine'].unique().tolist(),
                                   default=['Houndé','Ity','Mana','Sabodala-Massawa','Lafigué'])
        df_q = ops[(ops['Period_Type']=='Quarterly') & (ops['Mine'].isin(mines_sel))]
        fig = px.line(df_q, x='Period', y='Gold_Produced_koz', color='Mine',
                      color_discrete_map=MINE_COLORS, markers=True,
                      title='Production trimestrielle par mine (koz)',
                      labels={'Gold_Produced_koz':'Production (koz)'})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE: FINANCIER
# ══════════════════════════════════════════════════════════════
elif page == "💰 Financier":
    st.markdown("<div class='hero-banner'><div class='hero-title'>Analyse Financière</div><div class='hero-sub'>AISC · Rentabilité · Free Cash Flow · Revenue par Mine</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["AISC & Rentabilité", "Revenue par Mine", "Cash Flow"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            df_aisc = ops[ops['Period_Type']=='Annual'].copy()
            fig = px.bar(df_aisc, x='Mine', y='AISC_USD_oz', color='Year',
                         barmode='group', color_discrete_map={2025:'#C9A84C', 2024:'#444'},
                         title='AISC par mine 2024 vs 2025 ($/oz)',
                         labels={'AISC_USD_oz':'AISC ($/oz)'})
            fig.add_hline(y=1500, line_dash='dash', line_color='#C94C4C',
                          annotation_text='Seuil rentabilité $1,500', annotation_position='top right')
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_m = ops[(ops['Period_Type']=='Annual') & (ops['Year']==2025) & ops['AISC_Margin_USD_oz'].notna()].copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='AISC ($/oz)', x=df_m['Mine'], y=df_m['AISC_USD_oz'],
                                  marker_color='#C94C4C'))
            fig2.add_trace(go.Bar(name='Marge AISC ($/oz)', x=df_m['Mine'],
                                  y=df_m['AISC_Margin_USD_oz'], marker_color='#C9A84C'))
            fig2.update_layout(**PLOTLY_LAYOUT, barmode='stack',
                               title='AISC vs Marge par once 2025 ($/oz)')
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter: AISC vs Production
        fig3 = px.scatter(ops[ops['Period_Type']=='Annual'],
                          x='Gold_Produced_koz', y='AISC_USD_oz',
                          color='Mine', symbol='Year',
                          color_discrete_map=MINE_COLORS,
                          size='Revenue_USDm', size_max=35,
                          text='Mine',
                          title='Efficacité : Production vs AISC (taille = Revenue)',
                          labels={'Gold_Produced_koz':'Production (koz)', 'AISC_USD_oz':'AISC ($/oz)'})
        fig3.add_hline(y=1500, line_dash='dash', line_color='#C94C4C', opacity=0.5)
        fig3.update_traces(textposition='top center')
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        🏆 Sabodala-Massawa : AISC $1,248/oz — meilleur rapport coût/production parmi les grandes mines.<br>
        ✅ Ity : $1,197/oz avec 319 koz → mine la plus rentable du groupe.<br>
        ⚠️ Mana : AISC $2,160/oz — au-dessus du prix de $2,160 il n'y a quasi aucune marge.
        Toute correction du prix de l'or menacerait la rentabilité de Mana.
        </div>""", unsafe_allow_html=True)

    with tab2:
        rev_data = ops[ops['Period_Type']=='Annual'][['Mine','Year','Revenue_USDm','Gold_Produced_koz','AISC_USD_oz']].dropna()
        rev_data['Year'] = rev_data['Year'].astype(str)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(rev_data, x='Mine', y='Revenue_USDm', color='Year',
                         barmode='group', color_discrete_map={'2025':'#C9A84C','2024':'#444'},
                         title='Revenue par mine 2024 vs 2025 ($M)',
                         labels={'Revenue_USDm':'Revenue ($M)'})
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            df25 = rev_data[rev_data['Year']=='2025']
            fig2 = px.treemap(df25, path=['Mine'], values='Revenue_USDm',
                              color='AISC_USD_oz', color_continuous_scale='RdYlGn_r',
                              title='Treemap Revenue 2025 (couleur = AISC)',
                              labels={'Revenue_USDm':'Revenue ($M)'})
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fin_q = fin[fin['Period_Type']=='Quarterly'].copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=['Revenue trimestriel ($M)', 'Free Cash Flow ($M)'])
        fig.add_trace(go.Bar(x=fin_q['Period'], y=fin_q['Revenue'],
                             marker_color='#C9A84C', name='Revenue'), row=1, col=1)
        fig.add_trace(go.Bar(x=fin_q['Period'], y=fin_q['Free_Cash_Flow'],
                             marker_color=fin_q['Free_Cash_Flow'].apply(
                                 lambda x: '#4CC97A' if x > 0 else '#C94C4C'),
                             name='FCF'), row=2, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, title='Évolution trimestrielle Revenue & FCF', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        🚀 FCF : $165.9M en Q3 → $476.3M en Q4-2025 (+187% en un seul trimestre).<br>
        La montée en régime de Lafigué + le prix de l'or record ont créé un effet de levier massif.
        Total FCF 2025 : $1.155B vs $313M en 2024 (+269%).
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: RÉSERVES & MINE LIFE
# ══════════════════════════════════════════════════════════════
elif page == "🪨 Réserves & Mine Life":
    st.markdown("<div class='hero-banner'><div class='hero-title'>Réserves & Mine Life</div><div class='hero-sub'>Durée de vie · Qualité des gisements · Risque d'épuisement</div></div>", unsafe_allow_html=True)

    # Mine life calculation
    ops25 = annual_ops(2025)
    res25_pp = res[(res['Year']==2025) & (res['Sub_Category']=='P_P')].copy()
    mine_life = []
    for _, row in res25_pp.iterrows():
        mine = row['Mine']
        prod_row = ops25[ops25['Mine']==mine]
        if not prod_row.empty and mine != 'Assafou':
            annual_prod_koz = prod_row['Gold_Produced_koz'].values[0]
            life = row['Content_Au_koz'] / annual_prod_koz if annual_prod_koz > 0 else None
            mine_life.append({'Mine': mine, 'Reserves_koz': row['Content_Au_koz'],
                               'Annual_Prod_koz': annual_prod_koz, 'Mine_Life_Years': life,
                               'Grade': row['Grade_Au_g_t']})
        elif mine == 'Assafou':
            mine_life.append({'Mine': mine, 'Reserves_koz': row['Content_Au_koz'],
                               'Annual_Prod_koz': 329, 'Mine_Life_Years': row['Content_Au_koz']/329,
                               'Grade': row['Grade_Au_g_t']})
    ml_df = pd.DataFrame(mine_life)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ml_df.sort_values('Mine_Life_Years', ascending=True),
                     x='Mine_Life_Years', y='Mine', orientation='h',
                     color='Mine_Life_Years',
                     color_continuous_scale=[[0,'#C94C4C'],[0.4,'#C9A84C'],[1,'#4CC97A']],
                     title='Durée de vie estimée des mines (années)',
                     labels={'Mine_Life_Years':'Années'})
        fig.add_vline(x=10, line_dash='dash', line_color='#C9A84C',
                      annotation_text='Cible: 10 ans')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Reserves 2024 vs 2025
        res_comp = res[res['Sub_Category']=='P_P'][['Mine','Year','Content_Au_koz','Grade_Au_g_t']]
        fig2 = px.bar(res_comp, x='Mine', y='Content_Au_koz', color='Year',
                      barmode='group', color_discrete_map={2025:'#C9A84C', 2024:'#444'},
                      title='Réserves P&P 2024 vs 2025 (koz)',
                      labels={'Content_Au_koz':'Réserves (koz)'})
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Grade des réserves
    st.markdown("<div class='section-title'>Qualité des gisements — Grade des réserves & ressources</div>", unsafe_allow_html=True)
    res25_detail = res[(res['Year']==2025) & (res['Sub_Category'].isin(['P_P','M_I']))].copy()
    fig3 = px.scatter(res25_detail, x='Content_Au_koz', y='Grade_Au_g_t',
                      color='Mine', size='Tonnage_Mt',
                      color_discrete_map=MINE_COLORS, symbol='Sub_Category',
                      text='Mine', size_max=50,
                      title='Qualité vs Quantité des gisements 2025',
                      labels={'Content_Au_koz':'Contenu or (koz)', 'Grade_Au_g_t':'Grade (g/t)'})
    fig3.update_traces(textposition='top center')
    fig3.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='warning-box'>
        ⚠️ Houndé : Réserves chutent de 2,600 → 1,896 koz (-27%) en un an.
        Avec 257 koz/an de production → seulement 7.4 ans de mine life.
        Signal d'alarme pour la durabilité de la mine sans nouvelle découverte.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='insight-box'>
        🚀 Assafou (développement) : 4,379 koz de réserves avec grade 1.76 g/t.
        À 329 koz/an projetés → 13.3 ans de mine life. C'est le futur pilier de croissance.
        Lafigué stagne légèrement (2,400 → 1,926 koz) mais reste solide.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: PRÉDICTION ML
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Prédiction ML":
    st.markdown("<div class='hero-banner'><div class='hero-title'>Prédiction Machine Learning</div><div class='hero-sub'>Régression · Forecast 2026 · Comparaison vs Guidance</div></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    📌 Méthode : Régression polynomiale (degree 2) entraînée sur les données 2024-2025
    pour projeter production et AISC en 2026. Comparaison avec la guidance officielle d'Endeavour.
    </div>""", unsafe_allow_html=True)

    mines_ml = ['Houndé','Ity','Mana','Sabodala-Massawa','Lafigué']
    metric_sel = st.radio("Métrique à prédire", ["Production (koz)", "AISC ($/oz)"], horizontal=True)

    results = []
    for mine in mines_ml:
        df_m = ops[(ops['Mine']==mine) & (ops['Period_Type']=='Annual')].copy()
        df_m = df_m.sort_values('Year')
        if len(df_m) < 2:
            continue
        X = df_m['Year'].values.reshape(-1, 1)
        if metric_sel == "Production (koz)":
            y = df_m['Gold_Produced_koz'].values
            col_name = 'Production_koz'
        else:
            y = df_m['AISC_USD_oz'].values
            col_name = 'AISC_USD_oz'

        poly = PolynomialFeatures(degree=1)
        Xp = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        pred_2026 = model.predict(poly.transform([[2026]]))[0]

        # Get guidance
        guid_row = guid[(guid['Mine']==mine) & (guid['Metric']==('Production_koz' if metric_sel=='Production (koz)' else 'AISC_USD_oz'))]
        guid_mid = guid_row['Mid'].values[0] if not guid_row.empty else None
        guid_low = guid_row['Low'].values[0] if not guid_row.empty else None
        guid_high = guid_row['High'].values[0] if not guid_row.empty else None

        gap = ((pred_2026 - guid_mid) / guid_mid * 100) if guid_mid else None

        results.append({
            'Mine': mine,
            'Actual_2024': y[0], 'Actual_2025': y[1],
            'ML_Pred_2026': round(pred_2026, 1),
            'Guidance_Mid': guid_mid, 'Guidance_Low': guid_low, 'Guidance_High': guid_high,
            'Gap_pct': round(gap, 1) if gap else None
        })

    results_df = pd.DataFrame(results)

    fig = go.Figure()
    for _, row in results_df.iterrows():
        mine = row['Mine']
        color = MINE_COLORS.get(mine, '#888')
        # Actual points
        fig.add_trace(go.Scatter(x=[2024, 2025], y=[row['Actual_2024'], row['Actual_2025']],
                                 mode='lines+markers', name=mine, line=dict(color=color, width=2),
                                 marker=dict(size=8), legendgroup=mine))
        # ML prediction
        fig.add_trace(go.Scatter(x=[2025, 2026], y=[row['Actual_2025'], row['ML_Pred_2026']],
                                 mode='lines+markers', name=f"{mine} (ML)",
                                 line=dict(color=color, width=2, dash='dash'),
                                 marker=dict(size=10, symbol='star'), legendgroup=mine,
                                 showlegend=False))
        # Guidance range
        if row['Guidance_Low'] and row['Guidance_High']:
            fig.add_trace(go.Scatter(x=[2026, 2026], y=[row['Guidance_Low'], row['Guidance_High']],
                                     mode='lines', line=dict(color=color, width=6, dash='dot'),
                                     opacity=0.5, name=f"{mine} Guidance",
                                     legendgroup=mine, showlegend=False))

    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f'Prédiction ML 2026 vs Guidance — {metric_sel}',
                      xaxis_title='Année',
                      yaxis_title=metric_sel,
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown("<div class='section-title'>Écart ML vs Guidance officielle</div>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, row in results_df.iterrows():
        cols = [col1, col2, col3, col4, col5]
        gap = row['Gap_pct']
        if gap is not None:
            if abs(gap) < 10:
                color = "#4CC97A"; icon = "✅"
            elif abs(gap) < 20:
                color = "#C9A84C"; icon = "⚠️"
            else:
                color = "#C94C4C"; icon = "🔴"
            gap_str = f"{icon} ML {'above' if gap>0 else 'below'} guidance: {abs(gap):.1f}%"
        else:
            gap_str = "—"

        unit = "koz" if metric_sel == "Production (koz)" else "$/oz"
        cols[i].markdown(f"""
        <div class="metric-card">
            <div style='font-size:0.75rem; color:{MINE_COLORS.get(row['Mine'],'#888')}; font-weight:700; margin-bottom:8px;'>{row['Mine']}</div>
            <div style='font-size:0.7rem; color:#666; font-family:Space Mono;'>2025 Actual</div>
            <div style='font-size:1.1rem; font-weight:700; color:#F0EDE8;'>{row['Actual_2025']:.0f} {unit}</div>
            <div style='font-size:0.7rem; color:#666; font-family:Space Mono; margin-top:6px;'>ML Pred 2026</div>
            <div style='font-size:1.1rem; font-weight:700; color:#C9A84C;'>{row['ML_Pred_2026']:.0f} {unit}</div>
            <div style='font-size:0.7rem; color:#666; font-family:Space Mono; margin-top:6px;'>Guidance</div>
            <div style='font-size:0.85rem; color:#888;'>{row['Guidance_Low']:.0f}–{row['Guidance_High']:.0f} {unit}</div>
            <div style='font-size:0.68rem; margin-top:6px; color:{color};'>{gap_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-top:1rem;'>
    🤖 Note méthodologique : Avec seulement 2 points (2024-2025), le modèle est une régression linéaire
    simple. Pour une prédiction robuste, il faudrait 5+ années de données. L'intérêt ici est de
    montrer l'écart entre la tendance mathématique et la guidance managériale — ce gap révèle
    les hypothèses implicites du management sur les améliorations opérationnelles attendues.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: CAPEX & STRATÉGIE
# ══════════════════════════════════════════════════════════════
elif page == "💡 Capex & Stratégie":
    st.markdown("<div class='hero-banner'><div class='hero-title'>Capex & Stratégie d'Investissement</div><div class='hero-sub'>Allocation du capital · Exploration → Réserves · 2025 vs 2026</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Allocation 2025", "2025 vs 2026 Guide", "Exploration → Réserves"])

    with tab1:
        mines_capex = ['Houndé','Ity','Mana','Sabodala-Massawa','Lafigué']
        cap25 = capex[(capex['Year']==2025) & (capex['Mine'].isin(mines_capex))].copy()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(cap25, x='Mine', y='Amount_USDm', color='Type',
                         color_discrete_map={
                             'Sustaining_Capital':'#C9A84C',
                             'Non_Sustaining_Capital':'#4CA8C9',
                             'Exploration':'#4CC97A'},
                         title='Allocation Capex 2025 par mine ($M)',
                         labels={'Amount_USDm':'Montant ($M)'})
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cap25_total = cap25.groupby('Type')['Amount_USDm'].sum().reset_index()
            fig2 = px.pie(cap25_total, values='Amount_USDm', names='Type',
                          color_discrete_sequence=['#C9A84C','#4CA8C9','#4CC97A'],
                          title='Répartition Capex groupe 2025', hole=0.5)
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        # Capex vs Production efficiency
        ops25 = annual_ops(2025).copy()
        total_capex_mine = cap25.groupby('Mine')['Amount_USDm'].sum().reset_index()
        total_capex_mine.columns = ['Mine', 'Total_Capex']
        merged = ops25.merge(total_capex_mine, on='Mine')
        merged['Capex_per_oz'] = merged['Total_Capex'] * 1e6 / merged['Gold_Produced_oz']

        fig3 = px.scatter(merged, x='Total_Capex', y='Gold_Produced_koz',
                          color='Mine', size='Capex_per_oz',
                          color_discrete_map=MINE_COLORS, text='Mine',
                          title='Capex total vs Production (taille = $/oz investi)',
                          labels={'Total_Capex':'Total Capex ($M)', 'Gold_Produced_koz':'Production (koz)'})
        fig3.update_traces(textposition='top center')
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        cap_comp = capex[capex['Mine'].isin(mines_capex)].copy()
        cap_comp['Periode'] = cap_comp['Year'].astype(str)
        cap_comp = cap_comp.groupby(['Mine','Periode','Type'])['Amount_USDm'].sum().reset_index()

        fig = px.bar(cap_comp, x='Mine', y='Amount_USDm', color='Type',
                     facet_col='Periode',
                     color_discrete_map={
                         'Sustaining_Capital':'#C9A84C',
                         'Non_Sustaining_Capital':'#4CA8C9',
                         'Exploration':'#4CC97A'},
                     title='Capex 2025 vs Guidance 2026 par mine ($M)',
                     labels={'Amount_USDm':'Montant ($M)'})
        fig.update_layout(**PLOTLY_LAYOUT, height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        🔍 Lafigué : Non-Sustaining passe de $80M → $90M en 2026 → la mine continue d'investir
        massivement dans sa croissance (expansion de capacité).<br>
        ✅ Houndé réduit le Non-Sustaining ($95M → $60M) mais augmente le Sustaining ($37M → $50M)
        → la mine passe en mode maintenance/strip ratio.<br>
        🚀 Mana réduit les investissements ($106M → $75M) — la transition underground est terminée.
        </div>""", unsafe_allow_html=True)

    with tab3:
        # Est-ce que exploration → plus de réserves ?
        expl = capex[(capex['Type']=='Exploration') & (capex['Year']==2025) & (capex['Mine'].isin(mines_capex))].copy()
        res_change = []
        for mine in mines_capex:
            r24 = res[(res['Mine']==mine) & (res['Year']==2024) & (res['Sub_Category']=='P_P')]['Content_Au_koz'].values
            r25 = res[(res['Mine']==mine) & (res['Year']==2025) & (res['Sub_Category']=='P_P')]['Content_Au_koz'].values
            if len(r24) > 0 and len(r25) > 0:
                expl_row = expl[expl['Mine']==mine]['Amount_USDm'].values
                res_change.append({
                    'Mine': mine,
                    'Exploration_USDm': expl_row[0] if len(expl_row) > 0 else 0,
                    'Reserves_2024_koz': r24[0],
                    'Reserves_2025_koz': r25[0],
                    'Reserve_Change_koz': r25[0] - r24[0],
                    'Reserve_Change_pct': (r25[0]-r24[0])/r24[0]*100
                })
        rc_df = pd.DataFrame(res_change)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(rc_df, x='Exploration_USDm', y='Reserve_Change_pct',
                             color='Mine', size='Reserves_2025_koz',
                             color_discrete_map=MINE_COLORS, text='Mine',
                             title='Est-ce que l\'exploration paie ? Capex vs Variation réserves',
                             labels={'Exploration_USDm':'Exploration ($M)',
                                     'Reserve_Change_pct':'Variation réserves (%)'},
                             size_max=40)
            fig.add_hline(y=0, line_color='#666', line_dash='dash')
            fig.update_traces(textposition='top center')
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(rc_df, x='Mine', y='Reserve_Change_koz',
                          color='Reserve_Change_koz',
                          color_continuous_scale=[[0,'#C94C4C'],[0.5,'#C9A84C'],[1,'#4CC97A']],
                          title='Variation des réserves 2024→2025 (koz)',
                          labels={'Reserve_Change_koz':'Variation (koz)'})
            fig2.add_hline(y=0, line_color='#888')
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class='warning-box'>
        ⚠️ Résultat surprenant : malgré l'exploration, TOUTES les mines perdent des réserves en 2025.
        La production dépasse les nouvelles découvertes → taux de remplacement < 100%.
        C'est un signal stratégique majeur : Endeavour doit accélérer l'exploration ou acquérir
        de nouveaux actifs. C'est précisément pourquoi Assafou est si critique pour l'avenir.
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem; font-family: Space Mono;
font-size: 0.65rem; color: #888888; letter-spacing: 0.08em;'>
ENDEAVOUR MINING ANALYTICS · BUILT WITH PYTHON + STREAMLIT · DATA: EDV ANNUAL REPORTS 2024-2025
</div>""", unsafe_allow_html=True)
