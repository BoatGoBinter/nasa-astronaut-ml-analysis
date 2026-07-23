import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
df1 = pd.read_csv("International Astronaut Database.csv")
df1

df2 = pd.read_csv("astronauts.csv")
df2

def _norm(s):
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_name_col(df):
    for c in df.columns:
        cl = c.lower()
        if "name" in cl or "astronaut" in cl or "crew" in cl:
            return c
    return df.columns[0]

name1 = _pick_name_col(df1)
name2 = _pick_name_col(df2)

df1["_key"] = df1[name1].apply(_norm)
df2["_key"] = df2[name2].apply(_norm)

m = df1.merge(df2, how="outer", on="_key", suffixes=("_1", "_2"))

def _coalesce_pairs(frame):
    cols = frame.columns.tolist()
    bases = {}
    for c in cols:
        if c.endswith("_1"):
            bases.setdefault(c[:-2], []).append(c)
        elif c.endswith("_2"):
            bases.setdefault(c[:-2], []).append(c)
    out = frame.copy()
    for base in bases:
        c1, c2 = base + "_1", base + "_2"
        s1 = out[c1] if c1 in out else pd.Series([np.nan]*len(out))
        s2 = out[c2] if c2 in out else pd.Series([np.nan]*len(out))
        out[base] = s1.where(s1.notna(), s2)
    drop_cols = [c for c in out.columns if c.endswith("_1") or c.endswith("_2")]
    out = out.drop(columns=drop_cols)
    return out

df = _coalesce_pairs(m)

name_cols = [c for c in [name1, name2] if c in df.columns]
if name_cols:
    df["Name"] = df[name_cols[0]]
    for c in name_cols[1:]:
        df["Name"] = df["Name"].where(df["Name"].notna(), df[c])

if "_key" in df.columns:
    df = df.drop(columns=["_key"])

if "Name" in df.columns:
    df = df[["Name"] + [c for c in df.columns if c != "Name"]]

df
