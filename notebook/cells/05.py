def download_once(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    print("downloading", url); tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp); os.replace(tmp, dst); return dst

def extract_once(zpath, out_dir):
    marker = os.path.join(out_dir, ".extracted")
    if not os.path.exists(marker):
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zpath) as z: z.extractall(out_dir)
        open(marker, "w").close()
    return out_dir

def synapse_sync_once(syn_id, out_dir):
    marker = os.path.join(out_dir, ".synced")
    if os.path.exists(marker): return out_dir
    import synapseclient, synapseutils
    try:
        from google.colab import userdata; token = userdata.get("SYNAPSE_AUTH_TOKEN")
    except Exception as e:
        raise RuntimeError("Add a Colab secret named SYNAPSE_AUTH_TOKEN (Synapse > Account Settings > Personal Access Tokens)") from e
    syn = synapseclient.Synapse(); syn.login(authToken=token, silent=True)
    os.makedirs(out_dir, exist_ok=True)
    synapseutils.syncFromSynapse(syn, syn_id, path=out_dir)
    open(marker, "w").close(); return out_dir

def inventory(root, max_files=40):
    # Print the first files and the arrays inside them, so a converter can be written for an unknown layout.
    files = sorted(p for p in glob.glob(f"{root}/**/*", recursive=True) if os.path.isfile(p) and not os.path.basename(p).startswith("."))
    print(f"{len(files)} files under {root}")
    for p in files[:max_files]:
        info = ""
        try:
            if p.endswith(".npz"):
                d = np.load(p, allow_pickle=False); info = {k: d[k].shape for k in d.files}
            elif p.endswith(".mat"):
                from scipy.io import whosmat; info = whosmat(p)
        except Exception as e:
            info = f"(unreadable: {e})"
        print(" ", os.path.relpath(p, root), os.path.getsize(p)//1024, "kB", info)

def windows_from_signal(sig, fs_in, fs_out, win_sec):
    # Non-overlapping windows at the native rate; drop NaN or flat windows; resample each kept window.
    n = int(round(win_sec*fs_in)); out = []
    g = math.gcd(int(round(fs_in)), fs_out); up, down = fs_out//g, int(round(fs_in))//g
    for i in range(len(sig)//n):
        w = np.asarray(sig[i*n:(i+1)*n], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.std() < 1e-6: continue
        out.append(resample_poly(w, up, down) if (up, down) != (1, 1) else w)
    return out

# ---------------- PhysioNet download helpers (D1, D3) ----------------
# Credentials come from Colab secrets and are only ever passed to wget as arguments. They are never printed:
# wget output is scrubbed before anything is shown.
FAKE_PHYSIONET = os.environ.get("WAVESPIKE_FAKE_PHYSIONET") if SMOKE else None   # local test: copy from a folder
STAGE_DIR = os.path.join(tempfile.gettempdir(), "wavespike_stage")               # local disk; complete files are then copied to Drive

def physionet_creds():
    if FAKE_PHYSIONET: return None
    try:
        from google.colab import userdata
        user, pw = userdata.get("PHYSIONET_USER"), userdata.get("PHYSIONET_PASS")
    except Exception as e:
        raise StopForUser("Add the Colab secrets PHYSIONET_USER and PHYSIONET_PASS (🔑 icon in the left sidebar) and enable "
                          f"notebook access for both ({type(e).__name__}).") from None
    if not user or not pw:
        raise StopForUser("The Colab secrets PHYSIONET_USER / PHYSIONET_PASS are empty.")
    return (user, pw)

def _nonempty(p): return os.path.isfile(p) and os.path.getsize(p) > 0

def _scrub(text, creds):
    text = text or ""
    for c in (creds or ()):
        if c: text = text.replace(c, "***")
    return text

def _project(base_url):                       # ".../files/<project>/<version>/" -> "<project>/<version>"
    return urllib.parse.urlparse(base_url).path.strip("/").split("/", 1)[1]

def _wget_auth(creds): return ["--user", creds[0], "--password", creds[1]] if creds else []

def physionet_listing(base_url, creds, sub=""):
    """File and folder names in one directory of a PhysioNet project, parsed from its HTML index page."""
    if FAKE_PHYSIONET:
        d = os.path.join(FAKE_PHYSIONET, _project(base_url), sub)
        return sorted(n + ("/" if os.path.isdir(os.path.join(d, n)) else "") for n in os.listdir(d))
    os.makedirs(STAGE_DIR, exist_ok=True); out = os.path.join(STAGE_DIR, "index.html")
    r = subprocess.run(["wget", "-q", "-O", out] + _wget_auth(creds) + [base_url + sub], capture_output=True, text=True)
    if r.returncode != 0:
        raise StopForUser(f"Listing {base_url + sub} failed (wget exit code {r.returncode}; 6 = authentication). Check "
                          "PHYSIONET_USER/PHYSIONET_PASS and that your account has signed this project's data use agreement. "
                          + _scrub(r.stderr, creds)[-300:])
    html = open(out, encoding="utf-8", errors="replace").read()
    prefix = urllib.parse.urlparse(base_url + sub).path
    names = set()
    for h in re.findall(r'href="([^"?#]+)"', html):
        h = urllib.parse.unquote(h)
        if h.startswith(prefix): h = h[len(prefix):]
        if not h or h.startswith(("/", "..", "#")) or "://" in h or h.startswith("mailto:"): continue
        names.add(h)
    return sorted(names)

def fetch_physionet(rel_paths, base_url, dest_root, creds, n_par=8, chunk=400, desc="files"):
    """Download base_url+rel to dest_root/rel for every rel path, skipping files that already exist.
    Uses `wget -N -c` with one URL list per process (keep-alive). Downloads go to local disk first, and a file is
    copied to Drive only once it is complete, so an interrupted run never leaves half-written files on Drive.
    Returns the paths that are still missing."""
    rel_paths = sorted(set(rel_paths))
    todo = [r for r in rel_paths if not _nonempty(os.path.join(dest_root, r))]
    if todo:
        t0 = time.time(); errors = []
        if FAKE_PHYSIONET:
            stage_root = os.path.join(FAKE_PHYSIONET, _project(base_url))
        else:
            stage_root = os.path.join(STAGE_DIR, _project(base_url).replace("/", "_")); os.makedirs(stage_root, exist_ok=True)
            cut = len(urllib.parse.urlparse(base_url).path.strip("/").split("/"))
            cmd = ["wget", "-N", "-c", "-q", "-x", "-nH", f"--cut-dirs={cut}", "-P", stage_root] + _wget_auth(creds)
        chunks = [todo[i:i+chunk] for i in range(0, len(todo), chunk)]
        def _job(ci):
            paths = chunks[ci]
            if not FAKE_PHYSIONET:
                lst = os.path.join(STAGE_DIR, f"urls_{os.getpid()}_{ci}.txt")
                with open(lst, "w") as f: f.write("\n".join(base_url + urllib.parse.quote(p) for p in paths) + "\n")
                r = subprocess.run(cmd + ["-i", lst], capture_output=True, text=True); os.remove(lst)
                if r.returncode not in (0, 8): errors.append(f"wget exit {r.returncode}: " + _scrub(r.stderr, creds)[-200:])
            for p in paths:
                s, d = os.path.join(stage_root, p), os.path.join(dest_root, p)
                if _nonempty(s):
                    os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
                    shutil.copyfile(s, d + ".part"); os.replace(d + ".part", d)
                    if not FAKE_PHYSIONET: os.remove(s)
            return len(paths)
        done = 0
        with ThreadPoolExecutor(max(1, n_par)) as ex:
            for n in ex.map(_job, range(len(chunks))):
                done += n
                if len(chunks) > 1: print(f"\r  {desc}: {done}/{len(todo)} requested ({time.time()-t0:.0f}s)", end="")
        if len(chunks) > 1: print()
        for e in errors[:5]: print("  ", e)
    missing = [r for r in todo if not _nonempty(os.path.join(dest_root, r))]
    print(f"{desc}: {len(rel_paths)-len(todo)} already on disk, {len(todo)-len(missing)} downloaded, {len(missing)} missing")
    return missing
