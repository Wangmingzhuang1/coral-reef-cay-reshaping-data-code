import copernicusmarine as cm
d = cm.describe()
lines = []
for product in d.products:
    pid = str(getattr(product, "product_id", ""))
    if "INS" not in pid.upper():
        continue
    for ds in (getattr(product, "datasets", None) or []):
        names = set()
        for version in (getattr(ds, "versions", None) or []):
            for part in (getattr(version, "parts", None) or []):
                for service in (getattr(part, "services", None) or []):
                    for variable in (getattr(service, "variables", None) or []):
                        names.add(str(getattr(variable, "short_name", variable)).upper())
        hit = [n for n in names if "SEA_LEVEL" in n or "TIDE" in n]
        if hit:
            lines.append(pid + " | " + str(ds.dataset_id) + " | " + ",".join(sorted(hit)) + " | nvars=" + str(len(names)))
open("outputs/tide_gauge_sealevel_datasets.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("sealevel datasets", len(lines))