"""Pin the frozen STEP A ruler: hash the artifact bytes (reproducible)."""
import hashlib
import json
from pathlib import Path

p = Path(__file__).resolve().parent / "results" / "step_a_ruler.json"
raw = p.read_bytes()
file_hash = hashlib.sha256(raw).hexdigest()
d = json.loads(raw.decode("utf-8"))
compact = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
content_hash = hashlib.sha256(compact).hexdigest()
pin = {
    "ruler_file": "results/step_a_ruler.json",
    "file_bytes_sha256": file_hash,
    "canonical_content_sha256": content_hash,
    "correction": "static is 42.6% of detections (3114/7431 at +2px, exclusive rule); "
                  "47.6% was Position's share OF THE STATICS",
    "stride_renderer_px": 12, "sites_renderer": 256,
    "stride_eth80_px": 16, "sites_eth80": 256,
    "residual_total_max": max([r["residual_total"] for b in d["renderer"] for r in b["rows"]]
                              + [r["residual_total"] for r in d["eth80"]["rows"]]),
}
out = Path(__file__).resolve().parent / "results" / "step_a_ruler_pin.json"
out.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(pin, indent=2, sort_keys=True))
