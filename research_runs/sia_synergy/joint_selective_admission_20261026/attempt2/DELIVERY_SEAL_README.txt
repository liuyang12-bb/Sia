Final delivery seal: DELIVERY_MANIFEST_FINAL.json.
The provisional delivery_manifest.json hashed its own still-empty open file. Independent checking found exactly that one mismatch; every source, input, raw result, test, report and other provisional artifact matched. The provisional file and tool are preserved.
FINAL_REVIEW.md references the provisional filename; use the final manifest named above. The final artifacts dictionary is computed before its output is opened, intentionally excluding itself. No experiment rerun, algorithm, configuration or metric was changed.
