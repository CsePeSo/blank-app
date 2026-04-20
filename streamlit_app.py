"""
Corner Prediction Streamlit App — v13.3 FINE-TUNED
====================================================
Funkciók:
- Egy meccs elemzése 75 soros nyers blokkból
- Golden zone tippek (Total / Hazai / Vendég)
- Over/Under táblázat 4.5 - 14.5 küszöbökkel
- Eredmény rögzítés és statisztika (MAE, hit rate, stb.)
- Adatok mentése CSV-be
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from corner_model import CornerPredictionModel, MatchDataParser

# ======================================================================
# KONFIGURÁCIÓ
# ======================================================================

st.set_page_config(
    page_title="Corner Prediction v13.3",
    page_icon="⚽",
    layout="wide",
)

HISTORY_FILE = "match_history.json"


def load_history():
    """Betölti a korábbi elemzéseket."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    """Elmenti a történeti adatokat."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# Kezdeti állapot
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "last_calc" not in st.session_state:
    st.session_state.last_calc = None
if "last_name" not in st.session_state:
    st.session_state.last_name = ""


# ======================================================================
# FŐOLDAL
# ======================================================================

st.title("⚽ Corner Prediction Model v13.3 FINE-TUNED")
st.caption("In-sample MAE: 2.23 | Hit rate: 92% (25 meccs)")

tab_elemzes, tab_eredmeny, tab_statisztika = st.tabs([
    "🎯 Elemzés",
    "📝 Eredmény rögzítés",
    "📊 Statisztika",
])


# ======================================================================
# TAB 1 — ELEMZÉS
# ======================================================================

with tab_elemzes:
    st.header("Új meccs elemzése")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        match_name = st.text_input(
            "Mérkőzés neve",
            placeholder="pl. Magdeburg vs Düsseldorf",
            value=st.session_state.last_name,
        )
        text_block = st.text_area(
            "75 soros adatblokk (MakeYourStat formátum)",
            height=400,
            placeholder="1 Avg. Corners 10.16\n2 Home Avg. Corners 5.44\n...",
        )
        override_k = st.number_input(
            "Kézi k (opcionális, 0 = automatikus)",
            min_value=0.0,
            max_value=1.5,
            value=0.0,
            step=0.01,
        )

        if st.button("🚀 Elemzés futtatása", type="primary", use_container_width=True):
            if not match_name.strip():
                st.error("Adj meg mérkőzés nevet!")
            elif not text_block.strip():
                st.error("Illeszd be a 75 soros adatblokkot!")
            else:
                model = CornerPredictionModel()
                k_val = override_k if override_k > 0 else None
                output, calc = model.run(match_name, text_block=text_block, override_k=k_val)

                if calc is None:
                    st.error(output)
                else:
                    st.session_state.last_calc = calc
                    st.session_state.last_name = match_name
                    st.success("✅ Elemzés kész!")

    with col_right:
        if st.session_state.last_calc is not None:
            c = st.session_state.last_calc

            st.subheader(f"📊 {st.session_state.last_name}")

            # λ értékek
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("λ TOTAL", f"{c['lambda_total']:.2f}",
                         delta=f"{c['lambda_total'] - c['league_avg']:+.2f} vs liga")
            col_b.metric("λ HAZAI", f"{c['lambda_home']:.2f}")
            col_c.metric("λ VENDÉG", f"{c['lambda_away']:.2f}")

            # Főbb paraméterek
            with st.expander("ℹ️ Részletek"):
                st.write(f"**ISZ:** {c['isz']:.2f}")
                st.write(f"**Match momentum:** {c['match_momentum']:.2f}")
                st.write(f"**k_liga:** {c['k_calib']:.3f} ({c['k_source']})")
                st.write(f"**Dinamikus k:** {c['dynamic_k']:.3f}")
                st.write(f"**h_ai:** {c['h_ai']:.2f} | **a_ai:** {c['a_ai']:.2f}")
                st.write(f"**Liga átlag szöglet:** {c['league_avg']:.2f}")
                if c.get('reality_check'):
                    st.warning("⚠️ Reality check aktív volt (λ visszahúzva)")

            # Golden tippek
            st.markdown("### 🎯 Golden zone tippek")
            tips_data = []
            for label, key in [("🎯 TOTAL", "total_tips"), ("🏠 HAZAI", "home_tips"), ("✈️ VENDÉG", "away_tips")]:
                tips = c[key]
                blocked = (
                    label == "🏠 HAZAI" and c["home_blocked"]
                ) or (
                    label == "✈️ VENDÉG" and c["away_blocked"]
                )
                if blocked:
                    tips_data.append({"Típus": label, "Tipp": "❌ TILTVA (alacsony intenzitás)", "Valószínűség": ""})
                elif tips:
                    t = tips[-1]
                    tips_data.append({
                        "Típus": label,
                        "Tipp": f"Over {t['threshold']}",
                        "Valószínűség": f"{t['probability']*100:.1f}%"
                    })
                else:
                    tips_data.append({"Típus": label, "Tipp": "Nincs golden tipp", "Valószínűség": ""})
            st.table(pd.DataFrame(tips_data))

            # Over/Under táblázat
            st.markdown("### 📐 Over/Under táblázat (4.5 – 14.5)")
            ou_rows = []
            for r in c["over_under"]:
                t = r["threshold"]
                ou_rows.append({
                    "Küszöb": f"{t}.5",
                    "Over %": f"{r['over_pct']:.1f}%",
                    "Under %": f"{r['under_pct']:.1f}%",
                })
            ou_df = pd.DataFrame(ou_rows)
            st.dataframe(ou_df, use_container_width=True, hide_index=True)

            # Mentés gomb
            if st.button("💾 Elmentés a történeti adatokba (eredmény nélkül)"):
                entry = {
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "name": st.session_state.last_name,
                    "lambda_total": round(c["lambda_total"], 2),
                    "lambda_home": round(c["lambda_home"], 2),
                    "lambda_away": round(c["lambda_away"], 2),
                    "total_tip": c["total_tips"][-1]["threshold"] if c["total_tips"] else None,
                    "home_tip": c["home_tips"][-1]["threshold"] if c["home_tips"] else None,
                    "away_tip": c["away_tips"][-1]["threshold"] if c["away_tips"] else None,
                    "actual_total": None,
                    "actual_home": None,
                    "actual_away": None,
                }
                st.session_state.history.append(entry)
                save_history(st.session_state.history)
                st.success("Mentve! Eredmény később beírható az 'Eredmény rögzítés' tabon.")


# ======================================================================
# TAB 2 — EREDMÉNY RÖGZÍTÉS
# ======================================================================

with tab_eredmeny:
    st.header("Tény eredmények rögzítése")

    if not st.session_state.history:
        st.info("Még nincs elmentett elemzés. Futtass egy elemzést és mentsd el az 'Elemzés' tabon.")
    else:
        # Nem rögzített eredmények
        pending = [i for i, e in enumerate(st.session_state.history) if e.get("actual_total") is None]

        if pending:
            st.subheader(f"Várakozó eredmények ({len(pending)})")
            for idx in pending:
                entry = st.session_state.history[idx]
                with st.expander(f"📅 {entry['date'][:10]} — {entry['name']} (λ={entry['lambda_total']})"):
                    col1, col2, col3 = st.columns(3)
                    h = col1.number_input("Hazai szögletek", 0, 30, 0, key=f"h_{idx}")
                    a = col2.number_input("Vendég szögletek", 0, 30, 0, key=f"a_{idx}")
                    col3.metric("Összesen", h + a)
                    if st.button(f"💾 Mentés", key=f"save_{idx}"):
                        st.session_state.history[idx]["actual_home"] = h
                        st.session_state.history[idx]["actual_away"] = a
                        st.session_state.history[idx]["actual_total"] = h + a
                        save_history(st.session_state.history)
                        st.success("Rögzítve!")
                        st.rerun()
        else:
            st.success("✅ Minden elemzéshez rögzítve van már eredmény!")

        # Rögzített eredmények táblája
        done = [e for e in st.session_state.history if e.get("actual_total") is not None]
        if done:
            st.divider()
            st.subheader(f"Rögzített eredmények ({len(done)})")
            df = pd.DataFrame(done)
            df_display = df[["date", "name", "lambda_total", "actual_total",
                             "total_tip", "actual_home", "actual_away"]].copy()
            df_display["date"] = df_display["date"].str[:10]
            df_display["Δ"] = df_display["lambda_total"] - df_display["actual_total"]
            df_display["tipp ✓/✗"] = df.apply(
                lambda r: "✓" if r["total_tip"] is not None and r["actual_total"] > r["total_tip"]
                         else ("✗" if r["total_tip"] is not None else "—"),
                axis=1,
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # CSV letöltés
            csv = df_display.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Letöltés CSV-ben", csv, "corner_history.csv", "text/csv")


# ======================================================================
# TAB 3 — STATISZTIKA
# ======================================================================

with tab_statisztika:
    st.header("Modell pontossági statisztikák")

    done = [e for e in st.session_state.history if e.get("actual_total") is not None]

    if len(done) < 1:
        st.info("Még nincs elegendő rögzített eredmény. Rögzíts legalább egy meccset az 'Eredmény rögzítés' tabon.")
    else:
        # MAE, bias
        errs = [abs(e["lambda_total"] - e["actual_total"]) for e in done]
        deltas = [e["lambda_total"] - e["actual_total"] for e in done]
        mae = sum(errs) / len(errs)
        bias = sum(deltas) / len(deltas)

        # Tipp hit rate
        tips_with = [e for e in done if e.get("total_tip") is not None]
        wins = [e for e in tips_with if e["actual_total"] > e["total_tip"]]
        hit_rate = 100 * len(wins) / len(tips_with) if tips_with else 0

        # Fő metrikák
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rögzített meccs", len(done))
        col2.metric("MAE", f"{mae:.2f}",
                    help="Átlagos abszolút hiba. Kisebb = jobb. In-sample: 2.23")
        col3.metric("BIAS", f"{bias:+.2f}",
                    help="Rendszerszintű torzítás (+ = túlbecsül, - = alábecsül)")
        col4.metric("Tipp hit rate", f"{hit_rate:.1f}%",
                    help=f"{len(wins)}/{len(tips_with)}. Cél: 85-95% (golden zone)")

        # Hiba kategóriák
        st.divider()
        st.subheader("Hibák eloszlása")
        cat_good = sum(1 for e in errs if e <= 1.5)
        cat_ok = sum(1 for e in errs if 1.5 < e <= 3.0)
        cat_bad = sum(1 for e in errs if e > 3.0)
        n = len(errs)
        col_g, col_o, col_b = st.columns(3)
        col_g.metric("🟢 Jó (|Δ| ≤ 1.5)", f"{cat_good} ({100*cat_good/n:.0f}%)")
        col_o.metric("🟡 OK (1.5 < |Δ| ≤ 3.0)", f"{cat_ok} ({100*cat_ok/n:.0f}%)")
        col_b.metric("🔴 Rossz (|Δ| > 3.0)", f"{cat_bad} ({100*cat_bad/n:.0f}%)")

        # Részletes táblázat
        st.divider()
        st.subheader("Részletes táblázat")
        rows = []
        for e in done:
            tip = e.get("total_tip")
            rows.append({
                "Dátum": e["date"][:10],
                "Meccs": e["name"],
                "λ": round(e["lambda_total"], 2),
                "Tény": e["actual_total"],
                "Δ": round(e["lambda_total"] - e["actual_total"], 2),
                "|Δ|": round(abs(e["lambda_total"] - e["actual_total"]), 2),
                "Tipp": f"O{tip}" if tip is not None else "—",
                "OK?": "✓" if tip is not None and e["actual_total"] > tip else ("✗" if tip is not None else "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Veszélyjel
        if mae > 3.0:
            st.warning("⚠️ MAE magasabb a vártnál. Lehet szükség ligaspecifikus kalibrációra.")
        if abs(bias) > 1.5:
            st.warning(f"⚠️ Nagy rendszerszintű {'túlbecslés' if bias > 0 else 'alábecslés'} ({bias:+.2f}). A modell konzisztensen eltér.")

        # Adatok törlése
        st.divider()
        with st.expander("⚠️ Veszélyes: történet törlése"):
            if st.button("🗑️ Összes történet törlése", type="secondary"):
                st.session_state.history = []
                save_history([])
                st.rerun()


# Lábjegyzet
st.divider()
st.caption("Modell: 2.7 v13.3 FINE-TUNED | Grid search optimalizált | MakeYourStat kompatibilis")
