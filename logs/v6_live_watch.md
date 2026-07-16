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
| 2026-07-10 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 11 days; no state change since Jun 29 liquidation; manual un-halt required) |
| 2026-07-13 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 14 days; weekly retrain ran Jul 13 but deploy remains frozen; portfolio at 13.5% of initial — manual un-halt + review required) |
| 2026-07-14 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 15 days; retrain ran again today Jul 14 — retrain appears to be running daily not weekly; deploy remains frozen, no manual un-halt) |
| 2026-07-15 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 16 days; retrain ran again today Jul 15 (16 consecutive daily retrains); deploy state unchanged since Jun 29 liquidation; manual un-halt + investigation required) |
| 2026-07-16 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 17 days; retrain ran again today Jul 16 (17 consecutive daily retrains vs expected weekly); deploy state unchanged since Jun 29 liquidation; manual un-halt + investigation required) |
