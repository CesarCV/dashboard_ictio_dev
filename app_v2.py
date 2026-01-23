# app.py
# Ictio · Dashboard (simplificado)
# - Filtros en sidebar (fecha, país, BL2, BL3, BL4, protocolo)
# - Métrica única: cantidad de registros (filas)
# - 1) Barras por especie (conteo filas)
# - 2) Mapa BL4 (polígonos por subcuenca + color tipo heat por conteo + etiqueta)
#
# Requisitos: streamlit, pandas, numpy, altair, pydeck
# UI opcional: st-ant-tree (pip install st-ant-tree)
# Shapefiles: geopandas (pip install geopandas)

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk

# ─────────────────────────────────────────────────────────────────────────────
# UI Component (Ant Design TreeSelect)
try:
    from st_ant_tree import st_ant_tree
    HAS_ANT_TREE = True
except Exception:
    HAS_ANT_TREE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuración
st.set_page_config(
    page_title="Ictio · Dashboard",
    page_icon="🐟",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

# ─────────────────────────────────────────────────────────────────────────────
# Compat wrappers (Streamlit >= 2026)
def _altair(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

def _df(df_: pd.DataFrame, **kwargs):
    try:
        st.dataframe(df_, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df_, use_container_width=True, **kwargs)

def _deck(deck_obj):
    try:
        st.pydeck_chart(deck_obj, width="stretch")
    except TypeError:
        st.pydeck_chart(deck_obj, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
def _norm_key(x: str) -> str:
    """Normaliza claves BL4 para empatar CSV vs shapefile (quita espacios, guiones, etc.)."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "").replace("_", "")
    s = re.sub(r"[^\w]", "", s)
    return s

def _pick_bl4_col(columns) -> str | None:
    """Encuentra columna BL4 en shapefile (case-insensitive / contains)."""
    cols = list(columns)
    for c in cols:
        if str(c).strip().lower() == "bl4":
            return c
    for c in cols:
        if "bl4" in str(c).strip().lower():
            return c
    return None

def color_for_count(n: int, fill_opacity: float = 0.85):
    """
    Color azul (claro→oscuro) con escala log.
    Retorna [r,g,b,a] para deck.gl.
    """
    if n is None or n <= 0:
        return [235, 235, 235, int(255 * 0.45)]  # gris claro
    v = math.log10(n + 1.0)  # 0..~5
    t = min(1.0, v / 4.0)
    r = int(200 * (1 - t) + 25 * t)
    g = int(225 * (1 - t) + 80 * t)
    b = int(245 * (1 - t) + 200 * t)
    a = int(255 * float(fill_opacity))
    return [r, g, b, a]

def safe_center(bounds_or_points, fallback=(-9.19, -74.44)):
    """
    bounds_or_points puede ser:
    - bounds: (minx, miny, maxx, maxy)
    - lista de (lat, lon)
    """
    try:
        if isinstance(bounds_or_points, (tuple, list)) and len(bounds_or_points) == 4:
            minx, miny, maxx, maxy = bounds_or_points
            return ((miny + maxy) / 2, (minx + maxx) / 2)
        pts = [(float(a), float(b)) for a, b in bounds_or_points if a is not None and b is not None]
        if not pts:
            return fallback
        lat = float(np.mean([p[0] for p in pts]))
        lon = float(np.mean([p[1] for p in pts]))
        if math.isfinite(lat) and math.isfinite(lon):
            return (lat, lon)
    except Exception:
        pass
    return fallback

# ─────────────────────────────────────────────────────────────────────────────
# UI helper: dropdown tipo Looker (checks + search + clear)
def ui_multiselect_lite(
    label: str,
    options,
    key: str,
    *,
    placeholder: str = "Seleccionar…",
    sidebar: bool = False,
    max_tag_count: int = 1,
    max_height: int = 320,
    width_dropdown: str = "100%",
):
    opts = sorted({str(o).strip() for o in options if pd.notna(o) and str(o).strip() != ""})
    if not opts:
        return []

    host = st.sidebar.container() if sidebar else st.container()
    with host:
        prev = st.session_state.get(key, [])
        if not isinstance(prev, list):
            prev = [] if prev is None else [prev]

        count = len(prev) if len(prev) > 0 else len(opts)
        st.markdown(f"**{label} ({count})**")

        if HAS_ANT_TREE:
            tree_data = [{"value": o, "title": o} for o in opts]
            selected = st_ant_tree(
                treeData=tree_data,
                key=key,
                placeholder=placeholder,
                treeCheckable=True,
                showSearch=True,
                allowClear=True,
                maxTagCount=max_tag_count,
                max_height=max_height,
                width_dropdown=width_dropdown,
            )
            if not selected:
                return []
            return [s for s in selected if s in opts]

        # fallback nativo
        selected_native = st.multiselect(
            label="",
            options=opts,
            default=[],
            key=key,
            label_visibility="collapsed",
            placeholder=placeholder,
        )
        return selected_native

# ─────────────────────────────────────────────────────────────────────────────
# Datos
#DATA_PATHS = ["ictio_worksheet_jun25_dahboard.csv"]

BASE_DIR = Path(__file__).resolve().parent
# DATA_PATHS debe ser una LISTA (o tu load_data() debe tratar el caso de un solo Path).
# En el código original se definía como Path pero luego se iteraba: "for p in DATA_PATHS",
# lo que termina iterando por partes del path (y puede intentar leer "/" como si fuese CSV).
DATA_PATHS = [
    BASE_DIR / "ictio_worksheet_jun25_dahboard.csv",
]
BL4_DIR = BASE_DIR / "BL4"

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    path = None
    for p in DATA_PATHS:
        if Path(p).exists():
            path = Path(p)
            break
    if path is None:
        st.error("No se encontró el archivo de datos. Coloque el CSV junto a este script o ajuste DATA_PATHS.")
        st.stop()

    df_ = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Fecha
    if "fecha" in df_.columns:
        df_["fecha"] = pd.to_datetime(df_["fecha"], errors="coerce")
    elif "obs_date_yyyymmdd" in df_.columns:
        s = df_["obs_date_yyyymmdd"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        df_["fecha"] = pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    # strings típicos
    for c in ["country_name", "basin_l2_corr", "basin_l3_corr", "protocol_name", "scientific_name", "BL4", "watershed_name_corr"]:
        if c in df_.columns:
            df_[c] = df_[c].astype(str).str.strip()

    return df_

df = load_data()

# BL4 en CSV (preferimos columna BL4; si no existe, caemos a watershed_name_corr)
BL4_COL = "BL4" if "BL4" in df.columns else ("watershed_name_corr" if "watershed_name_corr" in df.columns else None)

# ─────────────────────────────────────────────────────────────────────────────
# Shapefiles BL4
#BL4_DIR = Path("BL4")

@st.cache_data(show_spinner=True)
def find_bl4_shapefiles():
    if not BL4_DIR.exists():
        return []
    return sorted([p for p in BL4_DIR.rglob("*.shp")])

@st.cache_data(show_spinner=True)
def load_bl4_base_geojson(
    simplify_tolerance: float = 0.002,
    precision_round: int = 5,
    dissolve: bool = True,
):
    """
    Lee TODOS los shapefiles en BL4/, extrae columna BL4, une en un solo GeoDataFrame
    y devuelve GeoJSON base con:
      - bl4 (texto)
      - bl4_norm (clave normalizada)
      - label_lat / label_lon (punto representativo para etiqueta)
    """
    shp_paths = find_bl4_shapefiles()
    if not shp_paths:
        return {"__error__": "No se encontraron shapefiles en BL4/."}

    try:
        import geopandas as gpd
        from shapely.wkt import dumps as wkt_dumps
        from shapely.wkt import loads as wkt_loads
    except Exception as e:
        return {"__error__": f"No se pudo importar geopandas/shapely: {e}"}

    gdfs = []
    for shp in shp_paths:
        try:
            g = gpd.read_file(shp)
            bl4_col = _pick_bl4_col(g.columns)
            if bl4_col is None:
                continue

            g = g[[bl4_col, "geometry"]].copy()
            g.rename(columns={bl4_col: "bl4"}, inplace=True)
            g["bl4"] = g["bl4"].astype(str).str.strip()
            g = g[g["bl4"].notna() & (g["bl4"].astype(str).str.len() > 0)]
            g = g[g.geometry.notna()]

            # CRS -> 4326
            try:
                if g.crs is None:
                    g.set_crs(4326, inplace=True)
                else:
                    g = g.to_crs(4326)
            except Exception:
                pass

            # reparar geometría (best-effort)
            try:
                g["geometry"] = g.geometry.buffer(0)
            except Exception:
                pass

            gdfs.append(g)
        except Exception:
            continue

    if not gdfs:
        return {"__error__": "No se pudo leer ningún shapefile con columna BL4."}

    allg = pd.concat(gdfs, ignore_index=True)
    allg = gpd.GeoDataFrame(allg, geometry="geometry", crs=4326)

    # Dissolve para tener 1 geometría por BL4 (recomendado)
    if dissolve:
        try:
            allg = allg.dissolve(by="bl4", as_index=False)
        except Exception:
            # si falla, seguimos sin dissolve
            pass

    # Simplificación (reduce bytes)
    if simplify_tolerance and simplify_tolerance > 0:
        try:
            allg["geometry"] = allg.geometry.simplify(simplify_tolerance, preserve_topology=True)
        except Exception:
            pass

    # Recorte de precisión (reduce bytes)
    def _round_geom(geom, prec=precision_round):
        try:
            return wkt_loads(wkt_dumps(geom, rounding_precision=prec))
        except Exception:
            return geom

    try:
        allg["geometry"] = allg.geometry.apply(_round_geom)
    except Exception:
        pass

    # Keys + punto representativo para etiquetas
    allg["bl4_norm"] = allg["bl4"].apply(_norm_key)
    try:
        rp = allg.geometry.representative_point()
        allg["label_lon"] = rp.x.astype(float)
        allg["label_lat"] = rp.y.astype(float)
    except Exception:
        allg["label_lon"] = np.nan
        allg["label_lat"] = np.nan

    # bounds para centrar mapa
    try:
        bounds = allg.total_bounds  # (minx, miny, maxx, maxy)
    except Exception:
        bounds = (-80.0, -20.0, -65.0, 5.0)

    # GeoJSON
    try:
        gj = json.loads(allg.to_json())
        gj["__bounds__"] = list(map(float, bounds))
        return gj
    except Exception as e:
        return {"__error__": f"Fallo exportando a GeoJSON: {e}"}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar · Filtros dependientes
st.sidebar.header("🔎 Filtros")

# Fecha
if "fecha" in df.columns and df["fecha"].notna().any():
    min_date = pd.to_datetime(df["fecha"]).min()
    max_date = pd.to_datetime(df["fecha"]).max()
    r_fecha = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
else:
    r_fecha = None

# País
countries = sorted(df["country_name"].dropna().unique()) if "country_name" in df.columns else []
sel_country = ui_multiselect_lite("País", countries, key="country", sidebar=True, placeholder="Buscar…")
df1 = df[df["country_name"].isin(sel_country)] if sel_country else df.copy()

# BL2
bl2 = sorted(df1["basin_l2_corr"].dropna().unique()) if "basin_l2_corr" in df1.columns else []
sel_bl2 = ui_multiselect_lite("Basin L2", bl2, key="bl2", sidebar=True, placeholder="Buscar…")
df2 = df1[df1["basin_l2_corr"].isin(sel_bl2)] if sel_bl2 else df1

# BL3
bl3 = sorted(df2["basin_l3_corr"].dropna().unique()) if "basin_l3_corr" in df2.columns else []
sel_bl3 = ui_multiselect_lite("Basin L3", bl3, key="bl3", sidebar=True, placeholder="Buscar…")
df3 = df2[df2["basin_l3_corr"].isin(sel_bl3)] if sel_bl3 else df2

# BL4 (subcuenca)
if BL4_COL is not None and BL4_COL in df3.columns:
    bl4_opts = sorted(df3[BL4_COL].dropna().unique())
else:
    bl4_opts = []
sel_bl4 = ui_multiselect_lite("Subcuenca BL4", bl4_opts, key="bl4", sidebar=True, placeholder="Buscar…")
df4 = df3[df3[BL4_COL].isin(sel_bl4)] if (sel_bl4 and BL4_COL in df3.columns) else df3

# Protocolo
protos = sorted(df4["protocol_name"].dropna().unique()) if "protocol_name" in df4.columns else []
sel_proto = ui_multiselect_lite("Protocol", protos, key="proto", sidebar=True, placeholder="Buscar…")
df_f = df4[df4["protocol_name"].isin(sel_proto)] if sel_proto else df4

# Aplica fecha al final
if r_fecha is not None and "fecha" in df_f.columns:
    fi, ff = r_fecha if isinstance(r_fecha, tuple) else (r_fecha, r_fecha)
    df_f = df_f[(df_f["fecha"] >= pd.to_datetime(fi)) & (df_f["fecha"] <= pd.to_datetime(ff))]

st.sidebar.markdown("---")
st.sidebar.metric("Registros", f"{len(df_f):,}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
st.title("🐟 Ictio · Dashboard")
st.caption("Métrica: registros (conteo de filas).")

# Validaciones mínimas
if "scientific_name" not in df_f.columns or df_f["scientific_name"].dropna().empty:
    st.info("No hay especies disponibles con los filtros actuales.")
    st.stop()

if BL4_COL is None or BL4_COL not in df_f.columns:
    st.error("No se encontró columna BL4 en el CSV (ni alternativa watershed_name_corr).")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 1) Barras por especie (conteo filas)
st.subheader("Registros por especie")

species_opts = sorted(df_f["scientific_name"].dropna().unique())
sel_species = ui_multiselect_lite(
    "Especies",
    species_opts,
    key="species",
    placeholder="Buscar y marcar…",
    sidebar=False,
    max_tag_count=1,
    max_height=360,
)

df_view = df_f[df_f["scientific_name"].isin(sel_species)] if sel_species else df_f

topn = st.slider("Top N", 5, 60, 20, step=1)

agg_species = (
    df_view.groupby("scientific_name", as_index=False)
           .size()
           .rename(columns={"size": "registros"})
           .sort_values("registros", ascending=False)
           .head(topn)
)

if agg_species.empty:
    st.info("No hay registros para graficar con la selección actual.")
else:
    chart = (
        alt.Chart(agg_species)
        .mark_bar()
        .encode(
            x=alt.X("registros:Q", title="Registros (conteo de filas)"),
            y=alt.Y("scientific_name:N", sort="-x", title=""),
            tooltip=[
                alt.Tooltip("scientific_name:N", title="Especie"),
                alt.Tooltip("registros:Q", title="Registros", format=",.0f"),
            ],
        )
        .properties(height=max(320, 26 * len(agg_species)))
    )
    _altair(chart)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Mapa BL4 (choropleth + labels)
st.subheader("Mapa BL4 · Registros por subcuenca")

with st.expander("Ajustes de mapa", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        simplify_tol = st.slider("Simplificación", 0.0, 0.02, 0.002, step=0.001)
    with c2:
        fill_opacity = st.slider("Opacidad relleno", 0.15, 1.0, 0.85, step=0.05)
    with c3:
        label_min = st.number_input("Etiqueta desde", min_value=0, max_value=1_000_000, value=200, step=50)
    with c4:
        show_zero = st.checkbox("Mostrar subcuencas sin registros", value=False)

# GeoJSON base (polígonos por BL4)
base_gj = load_bl4_base_geojson(simplify_tolerance=float(simplify_tol), precision_round=5, dissolve=True)
if isinstance(base_gj, dict) and "__error__" in base_gj:
    st.error(base_gj["__error__"])
    st.stop()

base_features = base_gj.get("features", [])
if not base_features:
    st.error("No se obtuvieron polígonos BL4 desde los shapefiles.")
    st.stop()

# Conteo por BL4 (del CSV filtrado + especies seleccionadas)
tmp = df_view[[BL4_COL]].dropna().copy()
tmp["bl4_norm"] = tmp[BL4_COL].apply(_norm_key)
counts = tmp.groupby("bl4_norm").size().to_dict()  # {norm: n}

# Construye GeoJSON final con registros + color
features_out = []
label_points = []
for ft in base_features:
    props = ft.get("properties") or {}
    bl4_name = props.get("bl4") or props.get("BL4") or ""
    bl4_norm = props.get("bl4_norm") or _norm_key(bl4_name)
    n = int(counts.get(bl4_norm, 0))

    if (not show_zero) and n <= 0:
        continue

    props["bl4"] = bl4_name
    props["bl4_norm"] = bl4_norm
    props["registros"] = n
    props["fill_color"] = color_for_count(n, fill_opacity=float(fill_opacity))

    ft["properties"] = props
    features_out.append(ft)

    # Labels (usa label_lat/lon precomputado)
    if n >= int(label_min):
        lat = props.get("label_lat")
        lon = props.get("label_lon")
        if lat is not None and lon is not None:
            try:
                label_points.append(
                    {"latitude": float(lat), "longitude": float(lon), "text": f"{n:,}"}
                )
            except Exception:
                pass

if not features_out:
    st.info("No hay subcuencas para mostrar con los filtros actuales (y show_zero desactivado).")
    st.stop()

gj_final = {"type": "FeatureCollection", "features": features_out}

# Centro del mapa
bounds = base_gj.get("__bounds__", None)
center_lat, center_lon = safe_center(bounds if bounds else [], fallback=(-9.19, -74.44))
zoom = 5.0 if len(features_out) <= 18 else 4.2

view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0)

poly_layer = pdk.Layer(
    "GeoJsonLayer",
    gj_final,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[90, 90, 90, 190],
    lineWidthMinPixels=1,
)

layers = [poly_layer]

if label_points:
    layers.append(
        pdk.Layer(
            "TextLayer",
            data=label_points,
            get_position="[longitude, latitude]",
            get_text="text",
            get_size=14,
            size_units="pixels",
            get_color=[20, 20, 20, 220],
            pickable=False,
        )
    )

tooltip = {
    "html": "<b>Subcuenca BL4:</b> {bl4}<br/><b>Registros:</b> {registros}",
    "style": {"backgroundColor": "rgba(30,30,30,0.85)", "color": "white"},
}

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="light",
    tooltip=tooltip,
)

_deck(deck)

# Top BL4 (tabla limpia)
top_bl4 = (
    tmp.groupby(tmp[BL4_COL])
       .size()
       .sort_values(ascending=False)
       .head(25)
       .reset_index()
       .rename(columns={BL4_COL: "BL4", 0: "Registros"})
)
top_bl4.columns = ["BL4", "Registros"]
st.caption("Top 25 subcuencas BL4 por registros (según filtros y especies).")
_df(top_bl4, hide_index=True)
