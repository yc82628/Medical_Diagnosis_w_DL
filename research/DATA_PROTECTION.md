# Data protection: input for the DPIA (template)

*For the hospital's data protection officer (DPO), as input to a data
protection impact assessment under GDPR Article 35. It describes what the
software does, so the DPO can assess the risks. It is not legal advice, and
it does not replace the hospital's own assessment.*

## What data, where

| Data | Where it lives | Protection |
|---|---|---|
| Retinal photographs (special-category health data, and arguably biometric) | The study server's disk, `artifacts/images/` | **Encrypted at rest** (Fernet: AES-128-CBC + HMAC-SHA256). Key outside the data folder (`secrets/data.key` or `DR_DATA_KEY`) |
| Patient ID, eye, visit ID, camera | Audit database on the server | Server disk (BitLocker recommended); access only through the authenticated service |
| Model outputs, grader decisions, notes | Audit database | Append-only, **hash-chained**: changes are detectable |
| Staff accounts | Audit database | PBKDF2-SHA256 password hashes (600,000 iterations); 15-minute inactivity sign-out |
| Research snapshots | `snapshots/` on hospital storage | **Pseudonymised** (below); images decrypted for training, so the folder must sit on an encrypted disk |

**Not stored by the platform at all:** patient names, birth dates, addresses,
or DICOM headers (only PatientID, laterality, study ID and camera are read
from a DICOM file).

**Nothing leaves the hospital.** There's no cloud service and no telemetry,
and the page loads no external resources. The server binds to the staff
network only (`DEPLOY.md`).

## Who can see what

| Role | Can |
|---|---|
| Photographer | Upload images, see the worklist |
| Grader | And record decisions, view attention maps, export a visit report |
| Admin | And verify audit-log integrity |
| Technical lead (server access) | Export research snapshots and run retraining. **Every export is in the audit log** |

Access is personal (one account per person). Network access is limited to
the staff network range, enforced by the application and by the Windows
firewall.

## Pseudonymisation of research snapshots

`learning/snapshot.py`:
- **Codes, not identifiers.** It replaces patient, visit and case identifiers
  with **HMAC-SHA256 codes**. The key is `secrets/pseudonym.key`, held by
  [role], and is never written into a snapshot.
- **Minimal data.** It reduces dates to the month and re-encodes images from
  pixels, so no metadata survives. It records how many graders took part,
  not who they were.
- **A fixed test set.** It assigns each patient permanently to training or
  the locked test set, as a function of the code.

This is **pseudonymisation, not anonymisation**: a retinal image can itself
identify a person. Snapshots therefore remain personal data and stay within
the study's legal basis, storage and retention rules.

## Retention and deletion

- Stored images: `python storage.py purge --days N`, scheduled to the study's
  retention period. The audit log keeps hashes, never pixels.
- Snapshots: delete the folder at the end of the retention period. They're
  never committed to version control (`.gitignore`).
- Backups (`deploy/backup.py`) hold still-encrypted images and **never the
  key**: store the key and the backups separately.

## Risks the DPO should weigh

| Risk | Mitigation in the software | What remains |
|---|---|---|
| Unauthorised access on the network | Staff-network allow-list, firewall rule, HTTPS, personal accounts, lockout | Shared or leaked passwords; a compromised staff device |
| Theft of the server or disk | Encrypted images; BitLocker recommended | The key file on the same disk if BitLocker is off; audit database fields (IDs) are not encrypted |
| Re-identification from snapshots | Keyed codes, month-only dates, no metadata | Images themselves; small cohorts |
| Undetected tampering with records | Hash-chained, append-only audit log; admin verification | Truncating the whole log. Periodically export the head hash off-site |
| Model errors influencing care | Not used for care in this study; DEMO / UNVALIDATED banners; acceptance gate | Human misuse outside the protocol |
| Loss of data | Verified backups; restore keeps previous data | Losing the encryption key makes images unrecoverable |

[DPO: legal basis, consent or objection handling, joint-controller
arrangements if a partner is involved, and the retention period.]
