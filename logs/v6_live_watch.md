# v6 Live-Deploy Watch Log

Branch: v6-universe | Source ledger: `main:deploy_state.json` | Initial cash: 100,000,000 TWD

| date | day | port_val | PnL% | drawdown | holdings | cash% | halted | orders |
|------|-----|----------|------|----------|----------|-------|--------|--------|
| 2026-06-18 | 1 | 74,709,698 | -25.29% | 0.000 | 10 | 10.5% | false | 10 BUY |
| 2026-06-23 | 4 | 74,566,278 | -25.43% | 0.001 | 10 | 9.5% | false | 17 BUY 19 SELL (12 SELL failed) |
| 2026-06-24 | 5 | 74,532,792 | -25.47% | 0.002 | 9 | 18.7% | false | 21 BUY 20 SELL |
| 2026-06-25 | 7 | 74,426,332 | -25.57% | 0.506 | 10 | 18.2% | false | 0/24 ALL FAILED (API ERR) |
| 2026-06-29 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (HALT — liquidated) |
| 2026-07-08 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 9 days, no manual intervention) |
| 2026-07-09 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 10 days; retrain committed today but deploy still frozen; daily deploy workflow may not be running) |
