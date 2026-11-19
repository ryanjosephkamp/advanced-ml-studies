#!/usr/bin/env python3
"""
download_data.py - Download the raw EEG recordings this project uses and verify them.

Fetches exactly the 408 files that the pipeline (scripts/kamp_hw4.py) and the
study (study/run_study.py) read, from the data sets' official sources, into a
folder you choose, and checks each file's SHA-256 against data/raw_files.sha256:

    eeg_mmi/    EEG Motor Movement/Imagery 1.0.0 (PhysioNet), runs 4, 8, and 12 of
                all 109 subjects: 327 EDF files, 834 MB. The 9 files of S088, S092,
                and S100 are recorded at 128 Hz; the loader reads and skips them.
    eegmat/     EEG During Mental Arithmetic Tasks 1.0.0 (PhysioNet): 72 EDF files,
                184 MB.
    bci_iv_2a/  BCI Competition IV, data set 2a: the 9 training sessions A01T to
                A09T, 300 MB. The competition site serves them only inside
                BCICIV_2a_gdf.zip (440 MB, which also holds the evaluation sessions);
                the zip is downloaded, the 9 training files are extracted, and the
                zip is deleted.

The checksums for the PhysioNet files are PhysioNet's published SHA256SUMS. BCI
Competition IV publishes no checksums; those 9 are the sums of the files in its
zip as downloaded on November 11, 2026.

Usage, from the repository root:

    python scripts/download_data.py /path/to/raw              # everything, about 1.3 GB
    python scripts/download_data.py /path/to/raw --datasets eegmat
    python scripts/download_data.py /path/to/raw --dry-run    # list what would be fetched
    python scripts/download_data.py /path/to/raw --verify     # check files already there
    python scripts/download_data.py /path/to/raw --check-remote --dry-run
                                      # also compare the manifest with PhysioNet's
                                      # current SHA256SUMS and the zip's size

Files already present with the right checksum are skipped, so an interrupted run
can be restarted. Then run the code with --raw-dir /path/to/raw.

The data keep their own terms: the PhysioNet sets are under the Open Data Commons
Attribution License v1.0, and BCI Competition IV asks users to credit the Graz
group that recorded data set 2a and to cite a paper from its description. Full
citations are in data/README.md.

Only the Python standard library is used.
"""

import os
import sys
import time
import shutil
import hashlib
import zipfile
import argparse
import urllib.request

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_DIR, 'data', 'raw_files.sha256')

PHYSIONET = {
    'eeg_mmi': 'https://physionet.org/files/eegmmidb/1.0.0/',
    'eegmat': 'https://physionet.org/files/eegmat/1.0.0/',
}
BCI_ZIP_URL = 'https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip'
BCI_ZIP_NAME = 'BCICIV_2a_gdf.zip'
BCI_ZIP_BYTES = 439968864
DATASETS = ['eeg_mmi', 'eegmat', 'bci_iv_2a']

USER_AGENT = 'eeg-raw-download/1.0 (Python urllib)'
TIMEOUT = 60
ATTEMPTS = 3
CHUNK = 1 << 20


def read_manifest(path=MANIFEST):
    """Return {relative path: sha256} in file order."""
    files = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            files[rel.lstrip('*')] = digest.lower()
    return files


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def is_good(path, digest):
    return os.path.isfile(path) and sha256(path) == digest


def _request(url, method='GET'):
    req = urllib.request.Request(url, method=method, headers={'User-Agent': USER_AGENT})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def fetch(url, out_path, expected_bytes=None, show_progress=False):
    """Download url to out_path (through a .part file), with retries."""
    part = out_path + '.part'
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with _request(url) as resp, open(part, 'wb') as f:
                total = int(resp.headers.get('Content-Length') or 0)
                done, last = 0, 0.0
                for block in iter(lambda: resp.read(CHUNK), b''):
                    f.write(block)
                    done += len(block)
                    if show_progress and total and time.time() - last > 2:
                        print(f"    {done / 1e6:7.1f} of {total / 1e6:.1f} MB", flush=True)
                        last = time.time()
            if expected_bytes is not None and os.path.getsize(part) != expected_bytes:
                raise IOError(f"got {os.path.getsize(part)} bytes, expected {expected_bytes}")
            os.replace(part, out_path)
            return True
        except Exception as e:                       # network errors, short reads
            print(f"    attempt {attempt} failed for {url}: {e}")
            if os.path.exists(part):
                os.remove(part)
            time.sleep(2 * attempt)
    return False


def physionet_url(rel):
    dataset, name = rel.split('/', 1)
    return PHYSIONET[dataset] + name


def download_physionet(dest, todo):
    """Fetch and verify PhysioNet files; return the paths that failed."""
    failed = []
    for i, (rel, digest) in enumerate(todo, 1):
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"  [{i}/{len(todo)}] {rel}")
        if not fetch(physionet_url(rel), out):
            failed.append(rel)
        elif sha256(out) != digest:
            print(f"    checksum mismatch, removed: {rel}")
            os.remove(out)
            failed.append(rel)
    return failed


def download_bci(dest, todo, keep_zip=False):
    """Fetch the competition zip, extract the needed sessions, verify them."""
    folder = os.path.join(dest, 'bci_iv_2a')
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, BCI_ZIP_NAME)
    if not (os.path.isfile(zip_path) and os.path.getsize(zip_path) == BCI_ZIP_BYTES):
        print(f"  {BCI_ZIP_NAME} ({BCI_ZIP_BYTES / 1e6:.0f} MB) from {BCI_ZIP_URL}")
        if not fetch(BCI_ZIP_URL, zip_path, expected_bytes=BCI_ZIP_BYTES, show_progress=True):
            return [rel for rel, _ in todo]
    failed = []
    with zipfile.ZipFile(zip_path) as zf:
        members = {os.path.basename(n): n for n in zf.namelist()}
        for rel, digest in todo:
            name = os.path.basename(rel)
            out = os.path.join(dest, rel)
            if name not in members:
                print(f"    not in the zip: {name}")
                failed.append(rel)
                continue
            with zf.open(members[name]) as src, open(out + '.part', 'wb') as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            os.replace(out + '.part', out)
            if sha256(out) != digest:
                print(f"    checksum mismatch, removed: {rel}")
                os.remove(out)
                failed.append(rel)
            else:
                print(f"  extracted {rel}")
    if not keep_zip and not failed:
        os.remove(zip_path)
    return failed


def check_remote(selected):
    """Compare the manifest with PhysioNet's published SHA256SUMS and the zip size."""
    problems = 0
    for dataset, base in PHYSIONET.items():
        wanted = {r: d for r, d in selected.items() if r.startswith(dataset + '/')}
        if not wanted:
            continue
        with _request(base + 'SHA256SUMS.txt') as resp:
            published = {}
            for line in resp.read().decode().splitlines():
                if line.strip():
                    digest, name = line.split(None, 1)
                    published[f'{dataset}/{name.strip().lstrip("*")}'] = digest.lower()
        bad = [r for r, d in wanted.items() if published.get(r) != d]
        print(f"  {dataset}: {len(wanted) - len(bad)} of {len(wanted)} manifest entries match "
              f"PhysioNet's SHA256SUMS.txt")
        for r in bad:
            print(f"    differs or missing: {r}")
        problems += len(bad)
    if any(r.startswith('bci_iv_2a/') for r in selected):
        with _request(BCI_ZIP_URL, method='HEAD') as resp:
            size = int(resp.headers.get('Content-Length') or 0)
        ok = size == BCI_ZIP_BYTES
        print(f"  bci_iv_2a: {BCI_ZIP_NAME} is {size:,} bytes "
              f"({'as expected' if ok else f'expected {BCI_ZIP_BYTES:,}'})")
        problems += 0 if ok else 1
    return problems


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dest', help='Folder to download into (created if needed)')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS,
                   help='Data sets to fetch (default: all three)')
    p.add_argument('--files', nargs='+', default=None, metavar='PATH',
                   help='Fetch only these manifest paths, e.g. eegmat/Subject00_2.edf')
    p.add_argument('--dry-run', action='store_true',
                   help='List what would be downloaded; download nothing')
    p.add_argument('--verify', action='store_true',
                   help='Only check the files already in dest against the checksums')
    p.add_argument('--check-remote', action='store_true',
                   help="Compare the manifest with PhysioNet's SHA256SUMS and the zip's size")
    p.add_argument('--keep-zip', action='store_true',
                   help='Keep the BCI Competition IV zip after extracting')
    return p.parse_args()


def main():
    args = parse_args()
    manifest = read_manifest()
    selected = {r: d for r, d in manifest.items() if r.split('/', 1)[0] in args.datasets}
    if args.files:
        unknown = [f for f in args.files if f not in manifest]
        if unknown:
            sys.exit(f"Not in {os.path.relpath(MANIFEST, REPO_DIR)}: {', '.join(unknown)}")
        selected = {r: manifest[r] for r in args.files}
    dest = os.path.abspath(os.path.expanduser(args.dest))

    if args.check_remote:
        print("Checking the sources:")
        if check_remote(selected):
            sys.exit("The sources no longer match the manifest; nothing downloaded.")

    print(f"Checking {len(selected)} files in {dest}")
    status = {r: ('ok' if is_good(os.path.join(dest, r), d) else
                  'bad' if os.path.exists(os.path.join(dest, r)) else 'missing')
              for r, d in selected.items()}
    for dataset in DATASETS:
        rels = [r for r in selected if r.startswith(dataset + '/')]
        if rels:
            counts = {s: sum(status[r] == s for r in rels) for s in ('ok', 'missing', 'bad')}
            print(f"  {dataset}: {counts['ok']} verified, {counts['missing']} missing, "
                  f"{counts['bad']} with a wrong checksum")
    todo = [(r, selected[r]) for r in selected if status[r] != 'ok']

    if args.verify:
        sys.exit(1 if todo else 0)
    if not todo:
        print("Nothing to download.")
        return
    if args.dry_run:
        for rel, _ in todo:
            src = BCI_ZIP_URL + ' (extracted)' if rel.startswith('bci_iv_2a/') else physionet_url(rel)
            print(f"  would fetch {rel}  <-  {src}")
        return

    failed = download_physionet(dest, [t for t in todo if not t[0].startswith('bci_iv_2a/')])
    bci = [t for t in todo if t[0].startswith('bci_iv_2a/')]
    if bci:
        failed += download_bci(dest, bci, keep_zip=args.keep_zip)
    if failed:
        print(f"{len(failed)} files failed: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(selected)} files verified in {dest}")


if __name__ == '__main__':
    main()
