"""Render scene.html frame by frame in headless Chrome (CDP) and encode with ffmpeg.
usage: capture.py <scene_dir> <out_dir> stills t1,t2,...   |   capture.py <scene_dir> <out_dir> video <fps> <out.mp4>"""
import sys, os, json, time, base64, subprocess, urllib.request, tempfile, shutil
import websocket
scene_dir, out_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = int(os.environ.get("PORT", 9377))
prof = tempfile.mkdtemp(prefix="bvsr-chrome-", dir=out_dir)
proc = subprocess.Popen(["/usr/bin/arch", "-arm64", CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}", "--remote-allow-origins=*",
    "--window-size=1920,1080", "--hide-scrollbars", "--force-device-scale-factor=1", "--no-first-run", "about:blank"] + ([] if os.environ.get("GPU") else ["--disable-gpu"]),
    stdout=open(os.path.join(out_dir,f"chrome-{PORT}.log"),"w"), stderr=subprocess.STDOUT)
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tabs, err = None, None
    for _ in range(80):
        try:
            tabs = json.load(opener.open(f"http://127.0.0.1:{PORT}/json", timeout=3)); break
        except Exception as e:
            err = e; time.sleep(0.4)
            if proc.poll() is not None: raise RuntimeError(f"chrome exited {proc.returncode}")
    if tabs is None: raise RuntimeError(f"chrome not reachable: {err!r}")
    page = [t for t in tabs if t["type"] == "page"][0]
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], max_size=None)
    mid = [0]
    def cmd(method, **params):
        mid[0] += 1; ws.send(json.dumps({"id": mid[0], "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == mid[0]:
                if "error" in m: raise RuntimeError(m["error"])
                return m["result"]
    cmd("Page.enable")
    cmd("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
    cmd("Page.navigate", url="file://" + os.path.join(scene_dir, "scene.html"))
    # wait until the scene script, its fonts and its images are all loaded
    ready = None
    for _ in range(80):
        time.sleep(0.5)
        r = cmd("Runtime.evaluate", awaitPromise=True, expression="""(async () => {
            if (typeof renderAt !== 'function' || document.readyState !== 'complete') return '';
            await document.fonts.ready;
            await Promise.all([...document.images].map(i => i.decode().catch(() => {})));
            if ([...document.images].some(i => !i.naturalWidth)) return '';
            return typeof renderAt + ':' + document.fonts.size;
        })()""")
        ready = r["result"].get("value")
        if ready: break
    if not ready: raise RuntimeError("scene did not become ready")
    time.sleep(0.8)
    r = {"result": {"value": ready}}
    print("ready:", r["result"].get("value"), flush=True)
    def frame(t, fmt="jpeg", q=93):
        res = cmd("Runtime.evaluate", expression=f"renderAt({t:.5f})")
        if "exceptionDetails" in res: raise RuntimeError(f"renderAt failed at t={t}: {res['exceptionDetails'].get('text')}")
        args = dict(format=fmt)
        if fmt == "jpeg": args["quality"] = q
        if os.environ.get("FAST"): args["optimizeForSpeed"] = True
        return base64.b64decode(cmd("Page.captureScreenshot", **args)["data"])
    if mode == "stills":
        for t in [float(x) for x in sys.argv[4].split(",")]:
            open(os.path.join(out_dir, f"still_{t:05.1f}.jpg"), "wb").write(frame(t, "jpeg", 80))
        print("stills done")
    else:
        fps = int(sys.argv[4])
        dur = cmd("Runtime.evaluate", expression="DURATION")["result"]["value"]
        n = int(round(dur * fps)); fdir = os.path.join(out_dir, "frames"); os.makedirs(fdir, exist_ok=True)
        t0 = time.time()
        lo, hi = int(os.environ.get("LO", 0)), int(os.environ.get("HI", n))
        for i in range(lo, min(hi, n)):
            fp = os.path.join(fdir, f"f_{i:05d}.jpg")
            if os.path.exists(fp) and os.path.getsize(fp) > 0: continue
            open(fp + ".tmp", "wb").write(frame(i / fps, "jpeg", 96)); os.replace(fp + ".tmp", fp)
            if i % 200 == 0: print(f"frame {i}/{n}  {time.time()-t0:.0f}s", flush=True)
        print("frames done", n, f"{time.time()-t0:.0f}s")
finally:
    proc.terminate()
    try: proc.wait(timeout=5)
    except Exception: proc.kill()
    shutil.rmtree(prof, ignore_errors=True)
