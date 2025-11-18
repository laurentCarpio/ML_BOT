# next_bot/core/utils/constants.py

# --- Strategy ---
MY_FREQUENCY_LIST         = ["1m", "3m", "5m", "15m", "30m"]
STRATEGY_WINDOW           = -10

# --- Trading Sides ---
OPEN_LONG   = "buy"
OPEN_SHORT  = "sell"

ACCEPTABLE_MARGE = 1.05

RULE0_ATR_PARAMS_BY_FREQ = {
    # seuils ATR relatif (en proportion), lissés sur 'lookback' dernières barres
    # à ajuster selon ta paire & échantillon
    "1m":  {"atr_min_rel": 0.0028, "lookback": 3, "agg": "median"},  # 0.28%
    "3m":  {"atr_min_rel": 0.0026, "lookback": 3, "agg": "median"},
    "5m":  {"atr_min_rel": 0.0025, "lookback": 3, "agg": "median"},
    "15m": {"atr_min_rel": 0.0020, "lookback": 3, "agg": "median"},
    "30m": {"atr_min_rel": 0.0018, "lookback": 3, "agg": "median"},
    "1h":  {"atr_min_rel": 0.0015, "lookback": 3, "agg": "median"},
}

# Fenêtre, écart max entre événements, et stricte/souple par fréquence
RULE1_PARAMS_BY_FREQ = {
    "1m":  {"win": 8,  "max_gap": 4, "order_strict": False},
    "3m":  {"win": 8,  "max_gap": 4, "order_strict": False},
    "5m":  {"win": 8,  "max_gap": 3, "order_strict": False},
    "15m": {"win": 10, "max_gap": 3, "order_strict": True},
    "30m": {"win": 12, "max_gap": 3, "order_strict": True},
    "1h":  {"win": 12, "max_gap": 2, "order_strict": True},
}

# Règle 2 : paramètres par fréquence
RULE2_PARAMS_BY_FREQ = {
    # tolérance courte en basses TF (bruit microstruct.)
    "1m":  {"lookahead": 5, "max_touches": 1, "min_follow_atr": 0.20},
    "3m":  {"lookahead": 4, "max_touches": 1, "min_follow_atr": 0.22},
    "5m":  {"lookahead": 3, "max_touches": 1, "min_follow_atr": 0.25},
    # TF hautes : strict
    "15m": {"lookahead": 2, "max_touches": 0, "min_follow_atr": 0.30},
    "30m": {"lookahead": 2, "max_touches": 0, "min_follow_atr": 0.35},
    "1h":  {"lookahead": 1, "max_touches": 0, "min_follow_atr": 0.40},
}

MAX_CANDLES_BETWEEN_HMA_AND_BB_BY_FREQ = {
    "1m"  : 6,   # tolérance un peu plus large pour les signaux bruités
    "3m"  : 4,
    "5m"  : 4,
    "15m" : 3,
    "30m" : 2,
    "1h"  : 2,   # tu peux ajouter aussi si tu veux scaler plus haut
}

INDICATOR_PARAMS = {
    "1m":  {"hma":60, "bb_len":20, "bb_std":3.0, "kc_len":20, "kc_scalar":2.2, "kc_mamode":"ema", "bb_mamode":"sma", "atr_len":14},
    "3m":  {"hma":20, "bb_len":20, "bb_std":2.5, "kc_len":20, "kc_scalar":2.0, "kc_mamode":"ema", "bb_mamode":"sma", "atr_len":14},
    "5m":  {"hma":12, "bb_len":20, "bb_std":2.5, "kc_len":20, "kc_scalar":2.0, "kc_mamode":"ema", "bb_mamode":"sma", "atr_len":14},
    "15m": {"hma":4,  "bb_len":20, "bb_std":2.5, "kc_len":20, "kc_scalar":2.0, "kc_mamode":"ema", "bb_mamode":"sma", "atr_len":14},
    "30m": {"hma":3,  "bb_len":20, "bb_std":2.0, "kc_len":20, "kc_scalar":1.8, "kc_mamode":"ema", "bb_mamode":"sma", "atr_len":14},
    "1h":  {"hma":3,  "bb_len":20, "bb_std":2.0, "kc_len":20, "kc_scalar":1.8, "kc_mamode":"ema", "bb_mamode":"sma", "atr_len":14}
}