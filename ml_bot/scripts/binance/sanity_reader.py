# sanity_reader.py (extrait)
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

@dataclass
class ExclusionWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    note: str = ""

def apply_sanity(config: Dict[str, Any], symbol: str, df_in: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Si df_in est fourni, on l'utilise directement; sinon on charge selon la config."""
    if df_in is None:
        # fallback éventuel si tu veux conserver l'ancien comportement
        raise ValueError("apply_sanity attend désormais df_in (DataFrame concaténé).")

    df = df_in.copy()
    # Normalisation colonnes + tri
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if config["sanity"].get("drop_duplicates", True):
        df = df.drop_duplicates(subset=["timestamp"])

    # Exclusions globales + par symbole (depuis config + exclusions_auto_file si présent)
    exclusions_cfg = config.get("exclusions", {})
    ex_windows = []

    def _collect(lst):
        out = []
        for x in lst or []:
            out.append(ExclusionWindow(pd.Timestamp(x["start"]), pd.Timestamp(x["end"]), x.get("note","")))
        return out

    # Exclusions manuelles dans config.yaml
    ex_windows += _collect(exclusions_cfg.get("global", []))
    ex_windows += _collect(exclusions_cfg.get("by_symbol", {}).get(symbol, []))


    # Applique le masque d'exclusion
    if ex_windows:
        mask = pd.Series(False, index=df.index)
        for w in ex_windows:
            mask |= df["timestamp"].between(w.start, w.end, inclusive="left")
        df = df.loc[~mask].copy()

    # Garde colonnes essentielles
    keep = ["timestamp","open","high","low","close","volume"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # Contrôle de continuité 1m
    diffs = df["timestamp"].diff().dt.total_seconds().fillna(60)
    gaps_count = int((diffs > 60).sum())
    dups_count = 0  # déjà dédoublonné

    sanity = {
        "gaps_count": gaps_count,
        "duplicates_count": dups_count,
        "first_ts": str(df["timestamp"].iloc[0]) if len(df) else None,
        "last_ts": str(df["timestamp"].iloc[-1]) if len(df) else None,
    }
    return df, sanity