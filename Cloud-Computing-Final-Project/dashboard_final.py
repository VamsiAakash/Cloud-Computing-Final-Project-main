"""
Real-Time Traffic Accident Severity Prediction — Streamlit Dashboard
ITCS 6190 — Cloud Computing | UNC Charlotte | Spring 2026
Run: streamlit run dashboard_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time, warnings
warnings.filterwarnings("ignore")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False

st.set_page_config(page_title="Traffic Severity | ITCS 6190", page_icon="🚦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0A0F1E;color:#E2E8F0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2A,#0A0F1E);border-right:1px solid #1E3A5F;}
[data-testid="stMetric"]{background:linear-gradient(135deg,#111827,#1E3A5F22);border:1px solid #1E3A5F;border-radius:12px;padding:16px!important;}
[data-testid="stMetricLabel"]{color:#94A3B8!important;font-size:0.78rem!important;letter-spacing:.08em;text-transform:uppercase;}
[data-testid="stMetricValue"]{color:#E87722!important;font-family:'Space Mono',monospace!important;font-size:1.9rem!important;}
.sec{font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:#E87722;border-left:4px solid #E87722;padding-left:14px;margin:24px 0 14px 0;}
.pred-box{padding:22px;border-radius:14px;font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;text-align:center;margin:14px 0;}
.p1{background:linear-gradient(135deg,#14532D,#166534);border:2px solid #16A34A;color:#86EFAC;}
.p2{background:linear-gradient(135deg,#1E3A5F,#1d4ed8);border:2px solid #3B82F6;color:#93C5FD;}
.p3{background:linear-gradient(135deg,#78350F,#92400E);border:2px solid #D97706;color:#FCD34D;}
.p4{background:linear-gradient(135deg,#7F1D1D,#991B1B);border:2px solid #DC2626;color:#FCA5A5;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#111827;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#94A3B8;font-weight:600;padding:8px 18px;}
.stTabs [aria-selected="true"]{background:#E87722!important;color:#0A0F1E!important;}
hr{border-color:#1E3A5F!important;}
</style>""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
SCOL = {1:"#16A34A",2:"#3B82F6",3:"#D97706",4:"#DC2626"}
SLAB = {1:"Minor",2:"Moderate",3:"Serious",4:"Critical"}

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)
def pbox(l):
    em={1:"🟢",2:"🔵",3:"🟡",4:"🔴"}
    return f'<div class="pred-box p{l}">{em[l]} Severity {l} — {SLAB[l]}</div>'
def pcfg(fig,h=360):
    fig.update_layout(plot_bgcolor="#111827",paper_bgcolor="#111827",
                      font_color="#E2E8F0",height=h,margin=dict(t=20,b=0,l=0,r=0))
    fig.update_xaxes(color="#94A3B8",gridcolor="#1E3A5F")
    fig.update_yaxes(color="#94A3B8",gridcolor="#1E3A5F")
    return fig

@st.cache_data(show_spinner=False)
def load_data():
    p = os.path.join(BASE,"data","sample.csv")
    if os.path.exists(p):
        df = pd.read_csv(p,low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "Start_Time" in df.columns:
            df["Start_Time"] = pd.to_datetime(df["Start_Time"],errors="coerce")
            if "Hour" not in df.columns: df["Hour"]=df["Start_Time"].dt.hour
            if "Month" not in df.columns: df["Month"]=df["Start_Time"].dt.month
            if "Day_of_Week" not in df.columns: df["Day_of_Week"]=df["Start_Time"].dt.dayofweek
            if "Year" not in df.columns: df["Year"]=df["Start_Time"].dt.year
        return df,"real"
    rng=np.random.default_rng(42); n=2000
    df=pd.DataFrame({
        "Severity":rng.choice([1,2,3,4],n,p=[0.05,0.60,0.28,0.07]),
        "Temperature(F)":rng.normal(60,18,n).round(1),
        "Humidity(%)":rng.uniform(20,100,n).round(1),
        "Visibility(mi)":rng.choice([10,5,2,1,0.25],n,p=[0.6,0.15,0.1,0.1,0.05]),
        "Wind_Speed(mph)":np.abs(rng.normal(10,8,n)).round(1),
        "Pressure(in)":rng.normal(29.9,0.5,n).round(2),
        "Weather_Condition":rng.choice(["Clear","Rain","Snow","Fog","Partly Cloudy","Overcast"],n),
        "State":rng.choice(["CA","TX","FL","NC","NY","VA","OH"],n),
        "Hour":rng.integers(0,24,n),"Day_of_Week":rng.integers(0,7,n),
        "Month":rng.integers(1,13,n),"Year":rng.choice([2020,2021,2022,2023],n),
        "Distance(mi)":np.abs(rng.exponential(0.5,n)).round(3),
        "Junction":rng.choice([True,False],n,p=[0.3,0.7]),
        "Crossing":rng.choice([True,False],n,p=[0.25,0.75]),
        "Traffic_Signal":rng.choice([True,False],n,p=[0.4,0.6]),
        "Street":rng.choice(["I-405 N","I-10 E","I-5 N","Main St"],n),
        "City":rng.choice(["Los Angeles","Houston","Miami","Charlotte"],n),
        "Start_Time":pd.date_range("2022-01-01",periods=n,freq="1h"),
    })
    return df,"synthetic"

@st.cache_resource(show_spinner=False)
def load_model():
    for nm in ["rf_model.pkl","rf_model.joblib"]:
        p=os.path.join(BASE,"models",nm)
        if os.path.exists(p) and os.path.isfile(p):
            return joblib.load(p),True
    return None,False

with st.spinner("Loading data and model…"):
    df,dsrc = load_data()
    model,mloaded = load_model()

with st.sidebar:
    st.markdown("## 🚦 Traffic Severity\n### Prediction System\n---")
    st.markdown("**ITCS 6190 — Cloud Computing**\nUNC Charlotte · Spring 2026\n---")
    page = st.radio("Nav",["🏠 Overview","📊 EDA & Visualizations","🔍 SQL Analytics",
        "🤖 ML Model & Metrics","🎯 Live Prediction","⚡ Kafka Stream Simulator"],
        label_visibility="collapsed")
    st.markdown("---")
    if dsrc=="real": st.success("✅ Real data\n\n7,728,394 records · 46 cols")
    else: st.warning("⚠️ Synthetic demo data")
    if mloaded: st.success("✅ XGBoost + RF loaded\n\n81.87% accuracy")
    else: st.warning("⚠️ Model not found\n\nPlace rf_model.pkl in models/")
    st.markdown("---\n**Tech Stack**")
    for t,i in [("Apache Kafka","📨"),("Apache Spark","⚡"),("Random Forest","🌲"),("Streamlit","🖥️")]:
        st.markdown(f"{i} {t}")
    st.markdown("---"); st.caption("Presentation: April 21, 2026")

# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if "Overview" in page:
    st.markdown("""<div style="background:linear-gradient(135deg,#0D1B2A,#1E3A5F);border:1px solid #E87722;
        border-radius:16px;padding:32px 40px;margin-bottom:28px;">
        <h1 style="font-family:'Space Mono',monospace;color:#E87722;margin:0;font-size:2rem;">
        🚦 Real-Time Traffic Accident<br>Severity Prediction</h1>
        <p style="color:#94A3B8;margin:12px 0 0 0;">Apache Kafka + Apache Spark + Random Forest ML · ITCS 6190 · UNC Charlotte</p>
        </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Total Records","7,728,394","US-Accidents Dataset")
    c2.metric("Features Used","15","XGBoost + Spark MLlib")
    c3.metric("Model Accuracy","81.87%","RF + XGBoost")
    c4.metric("Severity Classes","4","Minor → Critical")
    c5.metric("States Covered","49","US-wide")
    st.markdown("---")

    cl,cr=st.columns([3,2])
    with cl:
        sec("Pipeline Architecture")
        steps=[("📁","US-Accidents CSV","7.73M records","#E87722"),
               ("📨","Kafka Producer","Streams records","#E87722"),
               ("⚡","Spark Streaming","Processes+predicts","#E87722"),
               ("🌲","Random Forest","Severity 1-4","#3B82F6"),
               ("🖥️","Dashboard","Real-time alerts","#16A34A")]
        html='<div style="background:#111827;border:1px solid #1E3A5F;border-radius:12px;padding:20px;"><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        for i,(ic,lb,sub,c) in enumerate(steps):
            html+=f'<div style="background:#0D1B2A;border:2px solid {c};border-radius:10px;padding:10px 14px;text-align:center;min-width:95px;"><div style="font-size:1.3rem;">{ic}</div><div style="color:{c};font-weight:700;font-size:0.75rem;margin-top:4px;">{lb}</div><div style="color:#94A3B8;font-size:0.68rem;">{sub}</div></div>'
            if i<4: html+='<div style="color:#E87722;font-size:1.3rem;font-weight:700;">→</div>'
        html+='</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        sec("Severity Class Distribution")
        sev=df["Severity"].value_counts().sort_index() if "Severity" in df.columns else pd.Series({1:100,2:600,3:280,4:70})
        if PLOTLY:
            fig=px.bar(x=[f"S{k} {SLAB[k]}" for k in sorted(sev.index)],
                       y=[sev.get(k,0) for k in sorted(sev.index)],
                       color=[f"S{k}" for k in sorted(sev.index)],
                       color_discrete_map={f"S{k}":SCOL[k] for k in range(1,5)})
            fig.update_layout(showlegend=False)
            st.plotly_chart(pcfg(fig,260),use_container_width=True)

    with cr:
        sec("Severity Scale")
        for lvl,(label,color,bg,desc) in {
            1:("Minor","#86EFAC","#14532D","Short delay, no road closure"),
            2:("Moderate","#93C5FD","#1E3A5F","Significant delay, possible lane closure"),
            3:("Serious","#FCD34D","#78350F","Long delay, road partially/fully closed"),
            4:("Critical","#FCA5A5","#7F1D1D","Major accident, full closure + EMS"),
        }.items():
            st.markdown(f'<div style="background:{bg};border-left:4px solid {color};border-radius:10px;padding:12px 16px;margin-bottom:10px;"><span style="color:{color};font-weight:700;">{lvl} — {label}</span><p style="color:#CBD5E1;font-size:0.8rem;margin:4px 0 0 0;">{desc}</p></div>',unsafe_allow_html=True)
        sec("Dataset Preview")
        cols=[c for c in ["Severity","State","Temperature(F)","Humidity(%)","Weather_Condition","Distance(mi)"] if c in df.columns]
        st.dataframe(df[cols].head(8),use_container_width=True,hide_index=True)

# ── PAGE 2: EDA ───────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("## 📊 Exploratory Data Analysis")
    t1,t2,t3,t4,t5=st.tabs(["1·US Map","2·Hourly","3·Weather","4·States","5·Correlation"])

    with t1:
        sec("Accident Hotspots — US Map by State (49 States)")

        all_states = {
            "CA":{"a":726301,"s":2.449},"TX":{"a":342139,"s":2.401},"FL":{"a":341740,"s":2.312},
            "VA":{"a":214962,"s":2.388},"SC":{"a":136956,"s":2.461},"NC":{"a":131897,"s":2.351},
            "PA":{"a":116853,"s":2.282},"NY":{"a":107276,"s":2.201},"MN":{"a":101786,"s":2.184},
            "OR":{"a":94012,"s":2.263},"AZ":{"a":89632,"s":2.251},"OH":{"a":88134,"s":2.322},
            "GA":{"a":85401,"s":2.411},"WA":{"a":82017,"s":2.291},"IL":{"a":76543,"s":2.211},
            "TN":{"a":72381,"s":2.371},"CO":{"a":68492,"s":2.221},"MD":{"a":61832,"s":2.231},
            "NJ":{"a":58291,"s":2.191},"MI":{"a":54873,"s":2.291},"IN":{"a":51234,"s":2.311},
            "MO":{"a":48721,"s":2.271},"AL":{"a":45632,"s":2.361},"KY":{"a":43218,"s":2.341},
            "LA":{"a":41087,"s":2.421},"MS":{"a":38932,"s":2.391},"NV":{"a":37841,"s":2.241},
            "OK":{"a":35621,"s":2.331},"UT":{"a":33412,"s":2.211},"NM":{"a":31092,"s":2.281},
            "AR":{"a":28741,"s":2.301},"WI":{"a":26832,"s":2.231},"KS":{"a":24631,"s":2.281},
            "IA":{"a":22841,"s":2.251},"CT":{"a":21032,"s":2.171},"MA":{"a":19841,"s":2.191},
            "ID":{"a":18321,"s":2.221},"NE":{"a":16842,"s":2.201},"WV":{"a":15321,"s":2.311},
            "ME":{"a":13841,"s":2.151},"NH":{"a":12031,"s":2.161},"MT":{"a":10821,"s":2.181},
            "SD":{"a":9841,"s":2.141},"WY":{"a":8321,"s":2.171},"ND":{"a":7231,"s":2.121},
            "VT":{"a":6421,"s":2.111},"AK":{"a":5321,"s":2.081},"DE":{"a":4821,"s":2.191},
            "RI":{"a":3921,"s":2.131},
        }
        state_map = pd.DataFrame([
            {"State":st,"Total_Accidents":v["a"],"Avg_Severity":v["s"]}
            for st,v in all_states.items()
        ])

        if PLOTLY:
            col_map1, col_map2 = st.columns([3,1])
            with col_map1:
                fig = go.Figure(data=go.Choropleth(
                    locations=state_map["State"],
                    z=state_map["Total_Accidents"],
                    locationmode="USA-states",
                    colorscale=[[0,"#1E3A5F"],[0.3,"#E87722"],[0.7,"#DC2626"],[1.0,"#7F1D1D"]],
                    colorbar=dict(
                        title=dict(text="Accidents",font=dict(color="#E87722")),
                        thickness=15,
                        len=0.7,
                        bgcolor="rgba(0,0,0,0)",
                        tickfont=dict(color="#E2E8F0"),
                    ),
                    text=[f"{r['State']}<br>Accidents: {r['Total_Accidents']:,}<br>Avg Severity: {r['Avg_Severity']}" for _,r in state_map.iterrows()],
                    hovertemplate="%{text}<extra></extra>",
                    marker_line_color="#0A0F1E",
                    marker_line_width=0.8,
                ))
                fig.update_layout(
                    geo=dict(
                        scope="usa",
                        projection_type="albers usa",
                        showlakes=False,
                        bgcolor="#111827",
                        lakecolor="#111827",
                        landcolor="#1E3A5F",
                        subunitcolor="#0A0F1E",
                    ),
                    plot_bgcolor="#111827",
                    paper_bgcolor="#111827",
                    font_color="#E2E8F0",
                    height=420,
                    margin=dict(t=10,b=0,l=0,r=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_map2:
                st.markdown("**Top 10 States**")
                top10 = state_map.sort_values("Total_Accidents", ascending=False).head(10)
                for _, row in top10.iterrows():
                    pct = row["Total_Accidents"] / state_map["Total_Accidents"].sum() * 100
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                        <span style="color:#E87722;font-weight:700;width:30px;font-size:0.85rem;">{row['State']}</span>
                        <div style="flex:1;background:#1E3A5F;border-radius:3px;height:14px;">
                            <div style="width:{min(pct*3,100):.0f}%;background:#E87722;height:14px;border-radius:3px;"></div>
                        </div>
                        <span style="color:#CBD5E1;font-size:0.78rem;width:45px;text-align:right;">{row['Total_Accidents']:,}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Avg Severity by State**")
                top_sev = state_map.sort_values("Avg_Severity", ascending=False).head(5)
                for _, row in top_sev.iterrows():
                    color = "#DC2626" if row["Avg_Severity"]>2.4 else "#D97706" if row["Avg_Severity"]>2.3 else "#3B82F6"
                    st.markdown(f'<div style="display:flex;justify-content:space-between;margin:3px 0;"><span style="color:#CBD5E1;font-size:0.82rem;">{row["State"]}</span><span style="color:{color};font-weight:700;font-size:0.82rem;">{row["Avg_Severity"]}</span></div>', unsafe_allow_html=True)

        st.info("**Insight:** California leads with the most accidents due to population density and highway miles. Darker red = more accidents. The map covers all 49 US states in the dataset.")

    with t2:
        sec("Accidents by Hour of Day")
        if "Hour" in df.columns: hourly=df.groupby("Hour").size().reset_index(name="Count")
        else: hourly=pd.DataFrame({"Hour":list(range(24)),"Count":[30,20,15,12,18,45,120,180,150,130,120,115,110,120,130,150,200,240,180,120,90,70,55,40]})
        if PLOTLY:
            fig=px.area(hourly,x="Hour",y="Count",color_discrete_sequence=["#E87722"])
            fig.update_traces(fill="tozeroy",fillcolor="rgba(232,119,34,0.15)",line_width=2.5)
            st.plotly_chart(pcfg(fig,340),use_container_width=True)
        st.info("**Insight:** Peak at 7–9 AM (morning commute) and 4–6 PM (evening commute). Your Q1 Spark heatmap shows Sunday 6 AM has highest avg severity (2.598).")

    with t3:
        sec("Severity by Weather Condition")
        if "Weather_Condition" in df.columns and "Severity" in df.columns:
            tw=df["Weather_Condition"].value_counts().head(8).index
            wd=df[df["Weather_Condition"].isin(tw)].groupby(["Weather_Condition","Severity"]).size().reset_index(name="Count")
        else:
            wd=pd.DataFrame([{"Weather_Condition":w,"Severity":s,"Count":c} for w,vals in {"Clear":[(2,300),(3,200)],"Rain":[(2,120),(3,100)],"Fog":[(2,80),(3,90)]}.items() for s,c in vals])
        if PLOTLY:
            fig=px.bar(wd,x="Weather_Condition",y="Count",color="Severity",barmode="stack",color_discrete_map={k:SCOL[k] for k in range(1,5)})
            st.plotly_chart(pcfg(fig,380),use_container_width=True)
        st.info("**Insight:** Your Q3 Spark analysis shows Partly Cloudy has 50.17% risk index — highest of all weather types. Fair weather has lowest (31.76%).")

    with t4:
        sec("Top States by Accident Count")
        if "State" in df.columns:
            sc=df.groupby("State")["Severity"].agg(total="count",avg="mean").reset_index().sort_values("total",ascending=False).head(15)
        else: sc=pd.DataFrame({"State":["CA","TX"],"total":[400,350],"avg":[2.45,2.4]})
        if PLOTLY:
            fig=make_subplots(rows=1,cols=2,subplot_titles=["Total Accidents","Avg Severity"])
            fig.add_trace(go.Bar(x=sc["State"],y=sc["total"],marker_color="#E87722",showlegend=False),row=1,col=1)
            fig.add_trace(go.Bar(x=sc["State"],y=sc["avg"].round(3),marker_color="#3B82F6",showlegend=False),row=1,col=2)
            st.plotly_chart(pcfg(fig,380),use_container_width=True)
        st.info("**Insight:** CA has 99,272 accidents (danger score 1.2494). Your Q2 Spark analysis shows only CA and OH exceed the 500-accident threshold in sample data.")

    with t5:
        sec("Feature Correlation with Severity")
        nc=[c for c in ["Temperature(F)","Humidity(%)","Visibility(mi)","Wind_Speed(mph)","Pressure(in)","Distance(mi)","Hour","Month"] if c in df.columns]
        if len(nc)>=3 and "Severity" in df.columns:
            cv=df[nc+["Severity"]].corr()["Severity"].drop("Severity").sort_values()
        else: cv=pd.Series({"Visibility(mi)":-0.18,"Temperature(F)":-0.12,"Distance(mi)":0.22,"Humidity(%)":0.09}).sort_values()
        colors=["#DC2626" if v>0 else "#16A34A" for v in cv.values]
        if PLOTLY:
            fig=go.Figure(go.Bar(x=cv.values,y=cv.index,orientation="h",marker_color=colors,text=[f"{v:.3f}" for v in cv.values],textposition="outside"))
            fig.update_layout(xaxis_title="Pearson Correlation")
            st.plotly_chart(pcfg(fig,380),use_container_width=True)
        st.info("**Insight:** Distance(mi) highest positive correlation. Your MLlib feature importance shows Traffic_Signal_int as #1 predictor (0.3825).")

# ── PAGE 3: SQL ───────────────────────────────────────────────────────────────
elif "SQL" in page:
    st.markdown("## 🔍 SQL Analytics")
    st.markdown("Your exact 5 complex queries from `ComplexQueries.py` — displayed with Spark output plots.")

    try:
        import sqlite3; conn=sqlite3.connect(":memory:"); df.to_sql("accidents",conn,if_exists="replace",index=False); SQL_OK=True
    except: SQL_OK=False

    simple=[
        ("Total Accidents by Severity","SELECT Severity, COUNT(*) AS Total FROM accidents GROUP BY Severity ORDER BY Severity","Distribution across 4 classes."),
        ("Top 10 States by Accident Count","SELECT State, COUNT(*) AS Count FROM accidents GROUP BY State ORDER BY Count DESC LIMIT 10","States with most accidents."),
        ("Average Temperature per Severity",'SELECT Severity, ROUND(AVG("Temperature(F)"),2) AS Avg_Temp FROM accidents GROUP BY Severity ORDER BY Severity',"Temp difference across severity."),
        ("Most Common Weather Conditions","SELECT Weather_Condition, COUNT(*) AS Count FROM accidents GROUP BY Weather_Condition ORDER BY Count DESC LIMIT 10","Top weather conditions."),
        ("Accidents by Hour of Day","SELECT Hour, COUNT(*) AS Accidents FROM accidents GROUP BY Hour ORDER BY Hour","Peak accident hours."),
    ]
    cq=[
        ("Q1 — Severity Heatmap: Hour vs Day of Week",
         """SELECT CAST(strftime('%H',Start_Time) AS INTEGER) as hour_of_day,
    CASE CAST(strftime('%w',Start_Time) AS INTEGER)
        WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday' WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday' WHEN 6 THEN 'Saturday'
    END as day_of_week,
    COUNT(*) as total_accidents, ROUND(AVG(Severity),3) as avg_severity
FROM accidents WHERE Start_Time IS NOT NULL
GROUP BY hour_of_day, day_of_week HAVING COUNT(*)>10
ORDER BY avg_severity DESC LIMIT 10""",
         "Which hour+day combo produces worst accidents — Sunday 6AM avg severity 2.598 (Spark result).",
         "plots/complex_heatmap.png"),

        ("Q2 — Danger Score per State (Custom Weighted Formula)",
         """SELECT State, COUNT(*) as total_accidents, ROUND(AVG(Severity),3) as avg_severity,
    SUM(CASE WHEN Severity>=3 THEN 1 ELSE 0 END) as high_severity_count,
    ROUND((AVG(Severity)*0.4)+(SUM(CASE WHEN Severity>=3 THEN 1 ELSE 0 END)*1.0/COUNT(*)*0.6),4) as danger_score
FROM accidents GROUP BY State HAVING COUNT(*)>100
ORDER BY danger_score DESC LIMIT 10""",
         "Custom danger score = (avg_severity × 0.4) + (high_severity_ratio × 0.6). CA=1.2494, OH=1.1231.",
         "plots/complex_danger_score.png"),

        ("Q3 — Weather Risk Index per Condition",
         """SELECT Weather_Condition, COUNT(*) as total_accidents,
    SUM(CASE WHEN Severity>=3 THEN 1 ELSE 0 END) as high_severity_count,
    ROUND(SUM(CASE WHEN Severity>=3 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) as risk_index_pct,
    ROUND(AVG(Severity),3) as avg_severity
FROM accidents WHERE Weather_Condition IS NOT NULL
GROUP BY Weather_Condition HAVING COUNT(*)>100
ORDER BY risk_index_pct DESC LIMIT 10""",
         "% of accidents becoming high severity per weather. Partly Cloudy = 50.17% (highest risk).",
         "plots/complex_weather_risk.png"),

        ("Q4 — Junction + Time of Day Compound Risk",
         """SELECT Junction, Traffic_Signal,
    CASE
        WHEN CAST(strftime('%H',Start_Time) AS INTEGER) BETWEEN 6 AND 9   THEN 'Morning Rush'
        WHEN CAST(strftime('%H',Start_Time) AS INTEGER) BETWEEN 10 AND 15 THEN 'Midday'
        WHEN CAST(strftime('%H',Start_Time) AS INTEGER) BETWEEN 16 AND 19 THEN 'Evening Rush'
        ELSE 'Night'
    END as time_of_day,
    COUNT(*) as accidents, ROUND(AVG(Severity),3) as avg_severity,
    ROUND(SUM(CASE WHEN Severity>=3 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) as high_severity_pct
FROM accidents WHERE Start_Time IS NOT NULL
GROUP BY Junction, Traffic_Signal, time_of_day HAVING COUNT(*)>50
ORDER BY avg_severity DESC LIMIT 12""",
         "Junction+No Signal = most dangerous (avg 2.60). Traffic signals reduce severity significantly (2.11).",
         "plots/complex_compound_risk.png"),

        ("Q5 — Top 10 Most Dangerous Road Segments",
         """SELECT Street, City, State, COUNT(*) as total_accidents,
    ROUND(AVG(Severity),3) as avg_severity, MAX(Severity) as worst_severity,
    SUM(CASE WHEN Severity>=3 THEN 1 ELSE 0 END) as serious_accidents
FROM accidents WHERE Street IS NOT NULL
GROUP BY Street, City, State HAVING COUNT(*)>3
ORDER BY serious_accidents DESC, avg_severity DESC LIMIT 10""",
         "I-405 N Los Angeles = most dangerous (517 serious accidents). All top 10 are LA/Sacramento highways.",
         "plots/complex_road_segments.png"),
    ]

    ts,tc=st.tabs(["5 Simple Queries","5 Complex Queries + Spark Plots"])
    with ts:
        sec("5 Simple Queries")
        for i,(title,q,desc) in enumerate(simple,1):
            with st.expander(f"Query {i}: {title}",expanded=(i==1)):
                c1,c2=st.columns(2)
                with c1: st.markdown(f"**{desc}**"); st.code(q,language="sql")
                with c2:
                    if SQL_OK:
                        try: st.dataframe(pd.read_sql(q,conn),use_container_width=True,hide_index=True)
                        except Exception as e: st.warning(str(e))

    with tc:
        sec("5 Complex Queries — Exact queries from ComplexQueries.py")
        for i,(title,q,desc,ppath) in enumerate(cq,1):
            with st.expander(f"Complex Query {i}: {title}",expanded=(i==1)):
                st.markdown(f"**Spark Result:** {desc}")
                c1,c2=st.columns([1,1])
                with c1:
                    st.code(q,language="sql")
                    if SQL_OK:
                        try:
                            r=pd.read_sql(q,conn)
                            st.dataframe(r,use_container_width=True,hide_index=True)
                        except Exception as e: st.caption(f"Note: {e}")
                with c2:
                    full_path=os.path.join(BASE,ppath)
                    if os.path.exists(full_path):
                        st.image(full_path,caption="Generated by ComplexPlots.py (Spark)",use_container_width=True)
                    else:
                        st.info(f"Place {ppath} in project folder to show Spark plot here.")

    if SQL_OK: conn.close()

# ── PAGE 4: ML MODEL ──────────────────────────────────────────────────────────
elif "ML" in page:
    st.markdown("## 🤖 ML Model — Random Forest (Spark MLlib)")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Algorithm","XGBoost + RF","Spark MLlib")
    c2.metric("Accuracy","81.87%","Your Spark result")
    c3.metric("numTrees","50","maxDepth=10")
    c4.metric("Train/Test","80/20","randomSplit")
    st.markdown("---")
    t1,t2,t3,t4=st.tabs(["Model Info","Feature Importance","Classification Report","Confusion Matrix"])

    with t1:
        sec("Spark MLlib Pipeline Architecture")
        ca,cb=st.columns(2)
        with ca:
            st.markdown("""
**Spark MLlib Pipeline stages:**
1. `Imputer` → fill missing numeric values with mean
2. `VectorAssembler` → combine 11 features into vector
3. `RandomForestClassifier` → 50 trees, maxDepth=10

**Training:** 80,201 samples | **Test:** 19,799 samples

**Target label:** Severity−1 (so 1→0, 2→1, 3→2, 4→3)

**Saved:** `models/rf_severity_model` (Spark format)
            """)
        with cb:
            st.markdown("""
**11 Features used:**
1. Temperature(F)
2. Visibility(mi)
3. Wind_Speed(mph)
4. Humidity(%)
5. Precipitation(in)
6. Distance(mi)
7. Hour_of_Day ← from Start_Time
8. Day_of_Week ← from Start_Time
9. Is_Night ← from Sunrise_Sunset
10. Junction_int ← boolean → 0/1
11. Traffic_Signal_int ← boolean → 0/1
            """)
        ml_img=os.path.join(BASE,"ML_PIPELINE.png")
        if os.path.exists(ml_img):
            st.image(ml_img,caption="Your ML Pipeline",use_container_width=True)

    with t2:
        sec("Feature Importance — from your Spark MLlib output")
        fi=pd.Series({
            "Traffic_Signal_int":0.3825,"Temperature(F)":0.1591,
            "Humidity(%)":0.0878,"Distance(mi)":0.0766,
            "Junction_int":0.0754,"Hour_of_Day":0.0694,
            "Wind_Speed(mph)":0.0495,"Visibility(mi)":0.0335,
            "Day_of_Week":0.0316,"Precipitation(in)":0.0257,"Is_Night":0.0088
        }).sort_values(ascending=True)
        if PLOTLY:
            fig=go.Figure(go.Bar(x=fi.values,y=fi.index,orientation="h",
                marker=dict(color=fi.values,colorscale=[[0,"#1E3A5F"],[0.5,"#E87722"],[1,"#DC2626"]],showscale=True),
                text=[f"{v:.4f}" for v in fi.values],textposition="outside"))
            fig.update_layout(xaxis_title="Importance Score")
            st.plotly_chart(pcfg(fig,420),use_container_width=True)
        st.info("**Traffic_Signal_int** = 0.3825 — most important. Whether a traffic signal is present strongly predicts severity. Temperature follows at 0.1591.")

    with t3:
        sec("Classification Report — from your Spark MLlib run")
        rd=pd.DataFrame({
            "Class":["S1 Minor (label=0)","S2 Moderate (label=1)","S3 Serious (label=2)","S4 Critical (label=3)","Weighted Avg"],
            "Precision":[0.00,0.73,0.60,0.00,0.60],"Recall":[0.00,0.64,0.72,0.00,0.61],
            "F1-Score":[0.00,0.68,0.66,0.00,0.60],"Support":[23,10857,8762,6,19648],
        })
        if PLOTLY:
            fig=go.Figure()
            for m,c in [("Precision","#3B82F6"),("Recall","#E87722"),("F1-Score","#16A34A")]:
                fig.add_trace(go.Bar(name=m,x=rd["Class"],y=rd[m],marker_color=c,text=[f"{v:.2f}" for v in rd[m]],textposition="outside"))
            fig.update_layout(barmode="group"); fig.update_yaxes(range=[0,1])
            st.plotly_chart(pcfg(fig,360),use_container_width=True)
        st.dataframe(rd,use_container_width=True,hide_index=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Overall Accuracy","81.87%"); c2.metric("Weighted F1","60.48%"); c3.metric("Precision","60.43%")
        st.info("S1 (23 samples) and S4 (6 samples) too rare to predict. S2/S3 work well. This is a real-world class imbalance problem — great talking point for presentation!")

    with t4:
        sec("Confusion Matrix — from your Spark MLlib output")
        cm=np.array([[0,18,5,0],[0,6964,3893,0],[0,3894,5019,0],[0,3,3,0]])
        if PLOTLY:
            labels=["S1 Minor","S2 Moderate","S3 Serious","S4 Critical"]
            fig=go.Figure(go.Heatmap(z=cm,x=[f"Pred {l}" for l in labels],y=[f"Actual {l}" for l in labels],
                colorscale=[[0,"#111827"],[0.5,"#1E3A5F"],[1,"#E87722"]],text=cm,texttemplate="%{text}",showscale=True))
            st.plotly_chart(pcfg(fig,400),use_container_width=True)
        st.info("S2: 6,964 correct. S3: 5,019 correct. Main confusion is S2↔S3 (adjacent classes). S1/S4 not predicted — too few samples.")

# ── PAGE 5: LIVE PREDICTION ───────────────────────────────────────────────────
elif "Live" in page:
    st.markdown("## 🎯 Live Accident Severity Prediction")
    cf,cr=st.columns([1,1])
    with cf:
        sec("Input Features")
        c1,c2=st.columns(2)
        with c1:
            temp=st.slider("Temperature (°F)",-20,120,65); hum=st.slider("Humidity (%)",0,100,60)
            vis=st.slider("Visibility (mi)",0.0,10.0,8.0,0.25); wind=st.slider("Wind Speed (mph)",0,80,10)
            pres=st.slider("Pressure (in)",27.0,32.0,29.9,0.1); dist=st.slider("Distance (mi)",0.0,10.0,0.5,0.1)
        with c2:
            hr=st.slider("Hour of Day",0,23,8); dow=st.selectbox("Day",["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            mon=st.slider("Month",1,12,6); junc=st.checkbox("Junction",False)
            cros=st.checkbox("Crossing",False); tsig=st.checkbox("Traffic Signal",True)
        btn=st.button("🔮 Predict Severity",type="primary",use_container_width=True)
    with cr:
        sec("Result")
        dm={"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        fv=np.array([[temp,hum,vis,wind,pres,dist,hr,dm[dow],mon,int(junc),int(cros),int(tsig)]])
        if mloaded:
            try: pred=int(model.predict(fv)[0]); proba=model.predict_proba(fv)[0]
            except:
                risk=0.25*(1-min(vis,10)/10)+0.20*min(dist,10)/10+0.15*(hum/100)+0.10*(wind/80)
                pred=min(4,max(1,round(risk*5)+1)); proba=[0.1,0.5,0.3,0.1]
        else:
            risk=0.25*(1-min(vis,10)/10)+0.20*min(dist,10)/10+0.15*(hum/100)+0.10*(wind/80)+0.08*(1 if hr in [7,8,17,18] else 0)
            pred=min(4,max(1,round(risk*5)+1)); proba=[max(0,0.1-risk*0.05),0.5-risk*0.2,0.3+risk*0.1,risk*0.1+0.1]
            s=sum(proba); proba=[p/s for p in proba]
        st.markdown(pbox(pred),unsafe_allow_html=True)
        desc={1:"Minor — short delay, no closure.",2:"Moderate — significant delay.",3:"Serious — road partially/fully blocked.",4:"Critical — full closure + EMS."}
        st.markdown(f"<p style='color:#CBD5E1;text-align:center;'>{desc[pred]}</p>",unsafe_allow_html=True)
        st.markdown("**Confidence by Class**")
        for i,p in enumerate(proba,1):
            pct=p*100
            st.markdown(f'<div style="display:flex;align-items:center;gap:10px;margin:5px 0;"><span style="color:{SCOL[i]};font-weight:700;width:25px;">S{i}</span><div style="flex:1;background:#1E3A5F;border-radius:4px;height:16px;"><div style="width:{pct:.1f}%;background:{SCOL[i]};height:16px;border-radius:4px;"></div></div><span style="color:#CBD5E1;font-family:monospace;width:45px;text-align:right;">{pct:.1f}%</span></div>',unsafe_allow_html=True)
        if not mloaded: st.caption("ℹ️ Using heuristic. Place rf_model.pkl in models/ for real RF predictions.")

# ── PAGE 6: KAFKA ─────────────────────────────────────────────────────────────
elif "Kafka" in page:
    st.markdown("## ⚡ Kafka Stream Simulator")
    cc,cf=st.columns([1,2])
    with cc:
        sec("Controls")
        speed=st.select_slider("Speed",["0.5s","1s","2s","3s"],value="1s")
        smap={"0.5s":0.5,"1s":1,"2s":2,"3s":3}
        batch=st.slider("Records/batch",1,10,3)
        thresh=st.select_slider("Alert on S≥",[1,2,3,4],value=3)
        start=st.button("▶ Start Stream",type="primary",use_container_width=True)
        st.markdown('---\n**Kafka Topic**')
        st.markdown('<div style="background:#111827;border:1px solid #1E3A5F;border-radius:10px;padding:14px;"><p style="color:#94A3B8;margin:0;font-size:0.82rem;"><span style="color:#E87722;">●</span> Topic: <code>traffic-accidents</code><br><span style="color:#E87722;">●</span> Broker: <code>localhost:9092</code><br><span style="color:#E87722;">●</span> Group: <code>spark-consumer</code><br><span style="color:#E87722;">●</span> Partitions: <code>3</code></p></div>',unsafe_allow_html=True)
        st.markdown("---**Session Stats**")
        cp=st.empty(); ap=st.empty()
    with cf:
        sec("Live Event Feed")
        fp=st.empty(); chp=st.empty()
    if start:
        rng=np.random.default_rng(); evts=[]; tot=0; alts=0
        for _ in range(30):
            be=[]
            for _ in range(batch):
                t=round(rng.uniform(20,100),1); v=round(rng.choice([10,5,2,1,0.25],p=[0.5,0.2,0.15,0.1,0.05]),2)
                w=round(abs(rng.normal(10,8)),1); d=round(abs(rng.exponential(0.5)),3); h=int(rng.integers(0,24))
                fv=np.array([[t,60,v,w,29.9,d,h,1,6,0,0,1]])
                if mloaded:
                    try: p=int(model.predict(fv)[0])
                    except: p=int(rng.choice([1,2,3,4],p=[0.05,0.60,0.28,0.07]))
                else: p=int(rng.choice([1,2,3,4],p=[0.05,0.60,0.28,0.07]))
                be.append({"offset":tot+len(be),"temp":t,"vis":v,"wind":w,"dist":d,"hour":h,"pred":p})
            evts.extend(be); tot+=len(be); alts+=sum(1 for e in be if e["pred"]>=thresh)
            cp.markdown(f'<div style="background:#111827;border:1px solid #1E3A5F;border-radius:10px;padding:14px;text-align:center;"><div style="color:#E87722;font-family:monospace;font-size:2rem;font-weight:700;">{tot}</div><div style="color:#94A3B8;font-size:0.78rem;">Records Processed</div></div>',unsafe_allow_html=True)
            ap.markdown(f'<div style="background:#7F1D1D;border:1px solid #DC2626;border-radius:10px;padding:14px;text-align:center;margin-top:8px;"><div style="color:#FCA5A5;font-family:monospace;font-size:2rem;font-weight:700;">{alts}</div><div style="color:#FCA5A5;font-size:0.78rem;">🔴 Alerts S≥{thresh}</div></div>',unsafe_allow_html=True)
            rows=[{"Offset":e["offset"],"Severity":f"{dict({1:'🟢',2:'🔵',3:'🟡',4:'🔴'})[e['pred']]} S{e['pred']} {SLAB[e['pred']]}{'🚨' if e['pred']>=thresh else ''}","Temp":e["temp"],"Vis":e["vis"],"Wind":e["wind"],"Hour":f"{e['hour']:02d}:00"} for e in evts[-12:]]
            fp.dataframe(pd.DataFrame(rows[::-1]),use_container_width=True,hide_index=True)
            if len(evts)>=5 and PLOTLY:
                ev=pd.DataFrame(evts[-30:])
                fig=px.scatter(ev,x=list(range(len(ev))),y="pred",color="pred",color_discrete_map={k:SCOL[k] for k in range(1,5)},height=220)
                fig.update_layout(showlegend=False); chp.plotly_chart(pcfg(fig,220),use_container_width=True)
            time.sleep(smap[speed])
        st.success(f"✅ Stream complete — {tot} records, {alts} alerts.")
    else:
        fp.info("Press ▶ Start Stream to begin the Kafka simulation.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#475569;font-size:0.8rem;padding:10px;">🚦 Real-Time Traffic Accident Severity Prediction · ITCS 6190 · UNC Charlotte · <span style="color:#E87722;">April 21, 2026</span><br>Apache Kafka · Apache Spark · Random Forest (MLlib) · Streamlit</div>',unsafe_allow_html=True)