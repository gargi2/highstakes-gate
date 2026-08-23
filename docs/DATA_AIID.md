# AIID corpus — provenance & aggregate stats

The AI Incident Database (AIID) export is the corpus for **Pillar A** (the
failure taxonomy). The data lives in `data/aiid/` and is **gitignored** — this
note is the committed, reproducible record of what was fetched.

## Source
Official AIID snapshots page: https://incidentdatabase.ai/research/snapshots
(weekly point-in-time dumps). Cloudflare R2 bucket base:
`https://pub-72b2b2fc36ec423189843747af98f80e.r2.dev`

## Downloaded 2026-08-23 (snapshot dated 2026-08-17)
| File | Size | Contents |
|---|---|---|
| `AIID_Excel_Export-20260817.xlsx` | 2.41 MB | Incidents + taxonomy classifications (5 sheets), no-code analysis |
| `backup-20260817101120.tar.bz2` | 108.53 MB | Full mongodump: BSON + CSV + MongoDB archive |

Reproduce:
```bash
BASE=https://pub-72b2b2fc36ec423189843747af98f80e.r2.dev
curl -sSL -o data/aiid/AIID_Excel_Export-20260817.xlsx  "$BASE/AIID_Excel_Export-20260817.xlsx"
curl -sSL -o data/aiid/backup-20260817101120.tar.bz2    "$BASE/backup-20260817101120.tar.bz2"
# top-level CSVs (no BSON):
tar -xjf data/aiid/backup-20260817101120.tar.bz2 -C data/aiid/snapshot_csv --strip-components=1 \
  mongodump_full_snapshot/incidents.csv mongodump_full_snapshot/reports.csv \
  mongodump_full_snapshot/classifications_CSETv1.csv mongodump_full_snapshot/classifications_GMF.csv \
  mongodump_full_snapshot/classifications_MIT.csv
```
Note: AIID publishes a *newer* weekly snapshot each Monday; pin to the
`20260817` files above for reproducibility, or re-record stats if you refresh.

## License
**CC BY-SA 4.0** (per `mongodump_full_snapshot/license.txt`). Report contents
are subject to their own IP rights. Attribute the AI Incident Database.

## Aggregate stats (schema confirmation — no incident text reproduced)
Full snapshot CSVs:
- **incidents**: 1,632 records · 9 fields — `incident_id, date, reports,
  Alleged deployer of AI system, Alleged developer of AI system,
  Alleged harmed or nearly harmed parties, description, title, _id`
- **reports**: 7,526 records · 21 fields (title, text, url, source_domain,
  dates, authors, language, tags, ...)
- **classifications_CSETv1**: 214 records · 65 fields (Harm Domain, AI Harm
  Level, Sector of Deployment, ...)
- **classifications_GMF**: 326 records · 21 fields (Known/Potential AI Goal,
  Technology, Technical Failure, ...)
- **classifications_MIT**: 1,498 records · 8 fields (Risk Domain, Risk
  Subdomain, Entity, Timing, Intent)

Excel export sheets: `Incidents` (1,630), `Reports` (7,526), `Entities`
(7,016), `Data Dictionary` (60 rows), `Coverage Map` (7 rows). The sheet's row
1 is a banner; real headers begin on row 2 (Session 2 reads with the header
offset).

## Pillar A relevance
`incidents.csv` (id, title, description) is the primary unit for the
data-readiness failure-mode coding in Session 2; `reports.csv` supplies the
longer text for keyword-assisted classification; the three taxonomy CSVs
(CSET, GMF, MIT) are AIID's *existing* taxonomies — we differentiate from them
(pre-training-detectability lens), not reinvent them.
