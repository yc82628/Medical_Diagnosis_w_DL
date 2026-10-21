# Deploying on a hospital network

How to run Retinal Screening on **one machine on the hospital's network** —
typically a laptop or PC with an NVIDIA GPU — so that **only medical staff**,
on the **staff Wi-Fi / network**, can use it from their own devices' browsers.
Nothing is installed on staff devices except one certificate, and no data
leaves the hospital.

> **Research software — not a medical device.** Use it for research, such as
> a retrospective study approved by an ethics committee, where no output
> informs patient care. Before any patient data goes on it, agree a data
> protection impact assessment (DPIA) with the hospital's data protection
> officer. See `INTENDED_USE.md`, section 7.

```
 staff laptop / ward PC / tablet          the screening server (GPU laptop)
 ┌──────────────────────────┐   HTTPS   ┌────────────────────────────────────┐
 │ browser                  │──────────▶│ deploy\start_server.bat  port 8443  │
 │ trusts the local CA once │  staff    │  ├ refuses other networks (403)     │
 └──────────────────────────┘  Wi-Fi /  │  ├ sign-in, roles, 15-min sign-out  │
                               VLAN     │  ├ DR model + quality model on GPU  │
   guest Wi-Fi ────────✗──────────────▶ │  └ encrypted images, audit log      │
                                        └────────────────────────────────────┘
```

## What the server needs

| | Minimum | Notes |
|---|---|---|
| OS | Windows 10 or 11 | Linux works too, with the same commands minus `install_windows.ps1` |
| GPU | NVIDIA, 6 GB+ VRAM (RTX 3060 or newer; RTX 50-series supported) | The model uses about 3.6 GB. Without a GPU it still runs, a few seconds per image |
| RAM / disk | 16 GB / 20 GB free | Plus about 1 MB per stored image |
| Power | Mains, always | The install script stops it sleeping on mains |
| Network | A **fixed IP address** on the **staff** network | Ask IT for a DHCP reservation |
| Disk encryption | **BitLocker on** | The image-encryption key is a file on this disk |

## Setup, step by step

**1. Install the software** (first time only). Follow `SETUP.md`, steps 0–2,
using `requirements-gpu.txt` for the GPU build, then:

```bat
.venv\Scripts\pip install -r requirements-serve.txt
```

**2. Put the trained models in place**: `checkpoints\best_model.pt`,
`checkpoints\quality_model.pt` and `artifacts\evaluation.json`. They are not
on GitHub, so copy them from where the model was trained. Without them, the
service runs in a clearly labelled DEMO mode.

**3. Run the first-run setup** in the project folder:

```bat
.venv\Scripts\python deploy\setup.py
```

It checks the GPU and models, creates the image-encryption key and the first
admin account, and asks for the **staff network range**. Get this from IT, for
example `192.168.10.0/24`, and never include the guest Wi-Fi. It then creates
the HTTPS certificates and prints the address staff will use, for example
`https://ward3-screening.local:8443`.

For a scripted install with no prompts:

```bat
set DR_SETUP_ADMIN_PASSWORD=...
.venv\Scripts\python deploy\setup.py --non-interactive --admin-user it.admin --networks 192.168.10.0/24 --name retina.ward3.local
```

**4. Back up the encryption key right away.** Copy `secrets\data.key` to the
hospital's secret store or a safe, **not** onto the same backup drive as the
data. If it's lost, every stored image is unreadable. If it's stolen together
with a backup, every image is readable.

**5. Install the certificate on every staff device.** The hospital network
has no public domain name, so setup creates a small private certificate
authority. Each device trusts `deploy\certs\ca.crt` once. It isn't secret, so
email it or put it on a shared drive. Hospital IT can push it to every managed
device (Group Policy or Intune), which is the best option.

| Device | How |
|---|---|
| Windows | Double-click `ca.crt` → *Install Certificate* → *Local Machine* → *Trusted Root Certification Authorities* |
| macOS | Open `ca.crt` in Keychain Access → *System* → set *Always Trust* |
| iPad / iPhone | Open the file → *Settings › Profile Downloaded* → Install, then *Settings › General › About › Certificate Trust Settings* → enable it |
| Android | *Settings › Security › Encryption & credentials › Install a certificate › CA certificate* |

A device without it gets a security warning and can't connect. That's
intended. Never tell staff to click past the warning.

**6. Move `deploy\certs\ca.key` off the server** to offline storage. It's
only needed to issue a new server certificate (step 9), and anyone holding it
could impersonate the server.

**7. Let IT apply the Windows settings.** In PowerShell opened *as
Administrator*:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
```

It changes system settings, so read its header first. It:
- allows the port only from the staff networks, and never on Public networks
- starts the server at boot and restarts it if it stops
- stops the laptop sleeping on mains, including when the lid is closed

`-Uninstall` removes the firewall rule and the start-up task.

**8. Create staff accounts**, one per person and never shared:

```bat
.venv\Scripts\python auth.py create-user --username j.smith --role grader
```

Roles: `photographer` uploads images; `grader` also records decisions and
exports reports; `admin` can do everything, including checking audit-log
integrity. When someone leaves:
`python auth.py disable-user --username j.smith`, which also ends their
sessions immediately.

**9. Schedule daily backups** (Task Scheduler, as the same user):

```bat
.venv\Scripts\python.exe deploy\backup.py --to E:\retina-backups --keep 14
```

Each backup holds the audit database (copied safely while the server runs),
the still-encrypted images, the models and the evaluation files, plus a
checksum manifest, and is verified after writing. Restore, with the server
stopped:

```bat
.venv\Scripts\python deploy\backup.py restore E:\retina-backups\20261021-020000
```

Restoring keeps the replaced data in `artifacts.before-restore-…`, so nothing
is overwritten.

## Checks before staff use it

From a staff device on the staff Wi-Fi:

- [ ] `https://<server>:8443` opens with **no certificate warning**
- [ ] The header shows **Validated model** (not Demo or Unvalidated)
- [ ] You can sign in, upload a test image and see a result
- [ ] Leave it idle for 15 minutes: the next click asks you to sign in again

From a device on the **guest** Wi-Fi: the page says *"Not available on this
network"*, or doesn't load at all.

On the server: *Monitoring → Verify audit log* reports **Intact**.

## Who can get in — the layers

| Layer | What it stops |
|---|---|
| Staff network allow-list (`DR_ALLOWED_NETWORKS`) | Guests and visitors on other networks: refused before the sign-in page |
| Windows firewall rule (step 7) | The same, enforced by the operating system |
| HTTPS with a private CA | Eavesdropping on the Wi-Fi; fake servers |
| Personal accounts, roles | Anyone without an account; photographers recording diagnoses |
| 5 wrong passwords → 15-minute lock | Password guessing |
| 15-minute inactivity sign-out (`DR_IDLE_MINUTES`) | Someone using an unattended ward PC |
| Encrypted image store | Images read off a stolen disk or backup without the key |
| Hash-chained audit log | Silent edits to records; every view and export is logged |

## Day to day

| Task | How |
|---|---|
| Start / stop | `Start-ScheduledTask` / `Stop-ScheduledTask -TaskName 'Retinal Screening server'`, or run `deploy\start_server.bat` by hand |
| Change settings | Edit `deploy\server.env` (networks, port, idle minutes), then restart |
| New server IP or name | `python deploy\make_certs.py --ip NEW --name NEW` with `ca.key` back in `deploy\certs\`. Devices keep trusting it: the CA hasn't changed |
| Certificate expiry | The server certificate lasts 825 days (the most browsers accept). Re-run `make_certs.py` before then |
| Update the models | Stop, replace `checkpoints\*.pt` and `artifacts\evaluation.json`, start. A model that fails the acceptance gate is **not served**: the platform falls back to DEMO mode |
| Watch for problems | *Monitoring* shows grader agreement, ungradable rate and referral rate by week and by camera, with alerts |

## Limits — read before promising anything

- **One machine.** If the laptop is off, the service is off. There's no
  failover, and the database is SQLite, sized for one site and one server.
- **A new site needs its own calibration.** The model card explains it: the
  model ranks cases well across sites, but its decision threshold was
  calibrated on part of an external dataset. Each new hospital needs a
  labelled sample of its own images, graded to the reference standard,
  before results mean anything there.
- **The image-quality check is partial.** On held-out patients it catches
  about a third of the images human graders reject. Every result still
  needs human review.
- **Not a medical device.** Clinical use would require CE marking under EU
  MDR, with a quality management system (ISO 13485), a notified body and
  clinical evaluation.
