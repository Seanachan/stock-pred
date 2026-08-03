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
| 2026-07-17 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 18 days; retrain ran again today Jul 17 (18 consecutive daily retrains); deploy state unchanged since Jun 29 liquidation; manual un-halt + investigation required) |
| 2026-07-20 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 21 days; deploy_state.json unchanged since Jun 29 liquidation; no manual un-halt detected; retrain workflow continues running; ACTION REQUIRED) |
| 2026-07-21 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 22 days; deploy_state.json unchanged since Jun 29 liquidation; retrain workflow continues running; manual un-halt + root-cause investigation required before resuming live trading) |
| 2026-07-22 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 23 days; deploy_state.json unchanged since Jun 29 liquidation; retrain workflow continues running; manual un-halt + root-cause investigation required before resuming live trading) |
| 2026-07-23 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 24 days; deploy_state.json unchanged since Jun 29 liquidation; retrain workflow continues running; manual un-halt + root-cause investigation required before resuming live trading) |
| 2026-07-24 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 25 days; deploy_state.json unchanged since Jun 29 liquidation; retrain workflow continues running; manual un-halt + root-cause investigation required before resuming live trading) |
| 2026-07-27 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 28 days; deploy_state.json unchanged since Jun 29 liquidation; retrain ran today Jul 27 (Sun weekly, expected); deploy remains frozen; manual un-halt + root-cause investigation required before resuming live trading) |
| 2026-07-28 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 29 days; deploy_state.json unchanged since Jun 29 liquidation; no manual intervention detected; immediate action required to un-halt or close out live sim) |
| 2026-07-29 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 30 days; deploy_state.json unchanged since Jun 29 liquidation; retrain ran today Jul 29 (weekly); portfolio remains frozen at 13.5% of initial; un-halt or shut down live sim required immediately) |
| 2026-07-30 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 31 days; deploy_state.json unchanged since Jun 29 liquidation; retrain ran today Jul 30 (weekly); portfolio remains frozen at 13.5% of initial; ACTION REQUIRED: un-halt or shut down live sim) |
| 2026-07-31 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 32 days; deploy_state.json unchanged since Jun 29 liquidation; no manual intervention in over a month; ACTION REQUIRED: un-halt or shut down live sim) |
| 2026-08-03 | 8 | 13,547,832 | -86.45% | 0.819 | 0 | 100.0% | **TRUE** | 0 (watch check — still halted, 35 days; deploy_state.json unchanged since Jun 29 liquidation; no manual intervention detected; retrain workflow continues running; ACTION REQUIRED: un-halt or shut down live sim) |
