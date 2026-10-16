"""
portfolio_mix.py - Combina le equity curve del RAPS e della strategia BTC 1349.

Uso (da qualsiasi cartella, i percorsi sono argomenti):
    python portfolio_mix.py \
        --raps /root/hyperliquidRSPS/equity_curve.csv \
        --btc  /root/btc_1349/equity_curve.csv

Opzioni utili:
    --rebalance {D,M,Q,N}   ribilanciamento: giornaliero, mensile, trimestrale, mai (default M)
    --stress 1.5            fattore di stress sul drawdown (default 1.5)
    --target-dd 45          drawdown massimo accettato (%, dopo stress) -> peso BTC minimo consigliato

Requisiti:
  - RAPS: equity_curve.csv prodotto da backtest.py in modalita' COMPOUND (equity che parte dal capitale,
    non da 0). Colonne: date, equity.
  - BTC:  equity_curve.csv di btc_1349/backtest.py (colonne time, equity).
Il periodo usato e' solo quello in cui entrambe hanno dati.
"""
import argparse
import sys

import numpy as np
import pandas as pd


def load_curve(path, date_col_candidates, name):
    df = pd.read_csv(path)
    dcol = next((c for c in date_col_candidates if c in df.columns), None)
    if dcol is None or "equity" not in df.columns:
        sys.exit(f"[ERRORE] {path}: servono le colonne {date_col_candidates} ed 'equity'. Trovate: {list(df.columns)}")
    d = pd.to_datetime(df[dcol], utc=True).dt.tz_convert(None).dt.normalize()
    s = pd.Series(df["equity"].astype(float).values, index=d)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.iloc[0] <= 0 or (s <= 0).any():
        sys.exit(f"[ERRORE] {name}: l'equity contiene valori <= 0. Per il RAPS serve il backtest in modalita' compound "
                 f"(equity che parte dal capitale, non dal P&L cumulato da 0).")
    return s


def max_dd(eq):
    eq = np.asarray(eq, dtype=float)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def simulate(r_raps, r_btc, w_btc, rebalance):
    """Equity del portafoglio (parte da 1). Due 'sacche' che crescono ai propri rendimenti;
    ai giorni di ribilanciamento si riportano ai pesi target."""
    n = len(r_raps)
    a = 1.0 - w_btc     # sacca RAPS
    b = w_btc           # sacca BTC
    out = np.empty(n)
    idx = r_raps.index
    for i in range(n):
        a *= 1.0 + r_raps.iloc[i]
        b *= 1.0 + r_btc.iloc[i]
        tot = a + b
        out[i] = tot
        reb = False
        if rebalance == "D":
            reb = True
        elif i + 1 < n:
            nxt = idx[i + 1]
            if rebalance == "M" and nxt.month != idx[i].month:
                reb = True
            elif rebalance == "Q" and nxt.month != idx[i].month and nxt.month in (1, 4, 7, 10):
                reb = True
        if reb:
            a, b = tot * (1.0 - w_btc), tot * w_btc
    return pd.Series(out, index=idx)


def stats(eq):
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    tot = eq.iloc[-1] / 1.0 - 1.0
    cagr = eq.iloc[-1] ** (1 / years) - 1.0
    dd = max_dd(np.concatenate([[1.0], eq.values]))
    r = eq.pct_change().dropna()
    return tot, cagr, dd, (cagr / abs(dd) if dd < 0 else float("nan")), r.std() * np.sqrt(365)


def main():
    ap = argparse.ArgumentParser(description="Mix RAPS + BTC 1349")
    ap.add_argument("--raps", required=True, help="percorso equity_curve.csv del RAPS")
    ap.add_argument("--btc", required=True, help="percorso equity_curve.csv della strategia BTC")
    ap.add_argument("--rebalance", choices=["D", "M", "Q", "N"], default="M")
    ap.add_argument("--stress", type=float, default=1.5, help="fattore di stress sul drawdown (default 1.5)")
    ap.add_argument("--target-dd", type=float, default=None, help="drawdown massimo accettato in %% dopo stress")
    ap.add_argument("--out", default="portfolio_mix.csv")
    args = ap.parse_args()

    raps = load_curve(args.raps, ["date", "time", "data"], "RAPS")
    btc = load_curve(args.btc, ["time", "date", "data"], "BTC")

    both = pd.concat([raps.rename("raps"), btc.rename("btc")], axis=1, join="inner").dropna()
    if len(both) < 60:
        sys.exit(f"[ERRORE] solo {len(both)} giorni in comune: backtest su periodi che non si sovrappongono. "
                 f"Rilancia entrambi con la stessa --start.")
    # Rendimenti giornalieri sul periodo comune (primo giorno = base)
    rets = both.pct_change().dropna()
    r_raps, r_btc = rets["raps"], rets["btc"]
    start, end = both.index[0].date(), both.index[-1].date()
    print(f"\nPeriodo comune: {start} -> {end}  ({len(rets)} giorni)")

    # Correlazione
    corr = r_raps.corr(r_btc)
    bad = (r_raps < r_raps.quantile(0.2)) | (r_btc < r_btc.quantile(0.2))
    corr_bad = r_raps[bad].corr(r_btc[bad])
    both_down = ((r_raps < 0) & (r_btc < 0)).sum()
    either_down = ((r_raps < 0) | (r_btc < 0)).sum()
    print(f"Correlazione rendimenti giornalieri: {corr:+.2f}")
    print(f"Correlazione nei giorni cattivi (peggior 20% di almeno una): {corr_bad:+.2f}")
    # su finestre di 30 giorni (piu' rilevante per le strategie a trend)
    r30 = (1 + rets).rolling(30).apply(np.prod, raw=True) - 1
    print(f"Correlazione rendimenti a 30 giorni: {r30['raps'].corr(r30['btc']):+.2f}")
    print(f"Giorni con entrambe in perdita: {both_down} su {either_down} in cui almeno una perde")
    if rets["raps"].eq(0).mean() > 0.5 or rets["btc"].eq(0).mean() > 0.5:
        print("[NOTA] molti giorni a rendimento 0 (strategia flat): la correlazione giornaliera e' meno informativa.")

    print(f"\nRibilanciamento: {args.rebalance}   |   stress drawdown: x{args.stress}")
    hdr = f"{'%BTC':>5} {'%RAPS':>6} {'Totale':>10} {'CAGR':>8} {'MaxDD':>8} {'DD stress':>10} {'Calmar':>7} {'Vol ann.':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    rows, curves = [], {}
    for wb in np.arange(0, 1.0001, 0.1):
        eq = simulate(r_raps, r_btc, wb, args.rebalance)
        tot, cagr, dd, calmar, vol = stats(eq)
        rows.append((wb, tot, cagr, dd, calmar, vol))
        curves[f"btc_{int(round(wb*100))}"] = eq
        print(f"{wb*100:5.0f} {100-wb*100:6.0f} {tot*100:9.1f}% {cagr*100:7.1f}% {dd*100:7.1f}% "
              f"{max(dd*args.stress,-1.0)*100:9.1f}% {calmar:7.2f} {vol*100:8.1f}%")

    # Suggerimenti
    dd_btc = rows[-1][3]
    dd_raps = rows[0][3]
    w_inv = (1 / abs(dd_btc)) / (1 / abs(dd_btc) + 1 / abs(dd_raps))
    best = max(rows, key=lambda x: x[4] if not np.isnan(x[4]) else -1)
    print(f"\nPeso BTC con rischio uguale (inverso del drawdown): ~{w_inv*100:.0f}%")
    print(f"Peso BTC con Calmar migliore nel campione: {best[0]*100:.0f}% (attenzione: e' un ottimo in-sample)")
    if args.target_dd is not None:
        ok = [r for r in rows if abs(r[3]) * args.stress * 100 <= args.target_dd]
        if ok:
            r = min(ok, key=lambda x: x[0])
            print(f"Per stare sotto -{args.target_dd:.0f}% (dopo stress x{args.stress}): "
                  f"almeno {r[0]*100:.0f}% BTC / {100-r[0]*100:.0f}% RAPS "
                  f"(DD stress {r[3]*args.stress*100:.1f}%)")
        else:
            print(f"Nessun mix tra quelli testati resta sotto -{args.target_dd:.0f}%: "
                  f"serve ridurre il capitale totale esposto, non solo cambiare il mix.")

    # I momenti peggiori del mix suggerito
    wsel = min(rows, key=lambda x: abs(x[0] - w_inv))[0]
    eq = curves[f"btc_{int(round(wsel*100))}"]
    peak = eq.cummax()
    dd_s = eq / peak - 1
    t = dd_s.idxmin()
    print(f"\nMix {wsel*100:.0f}% BTC: drawdown peggiore {dd_s.min()*100:.1f}% il {t.date()} "
          f"(massimo precedente il {eq[:t].idxmax().date()})")

    pd.DataFrame(curves).to_csv(args.out)
    print(f"\nSalvato: {args.out} (equity del portafoglio per ogni mix)")
    print("\nCautele: non e' un test live. Niente fee/funding nel RAPS, campione BTC di soli 32 trade, "
          "periodo prevalentemente rialzista. Usalo per capire correlazione e ordine di grandezza, non per ottimizzare i pesi.")


if __name__ == "__main__":
    main()
