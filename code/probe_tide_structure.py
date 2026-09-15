import copernicusmarine as cm
d = cm.describe()
count = 0
out = []
for product in d.products:
    pid = str(getattr(product, "product_id", ""))
    if "INS" not in pid.upper():
        continue
    count += 1
    if count > 6:
        break
    out.append("PRODUCT " + pid)
    for ds in (getattr(product, "datasets", None) or [])[:2]:
        out.append("  DS " + str(ds.dataset_id))
        for version in (getattr(ds, "versions", None) or [])[:1]:
            out.append("    VER " + str(getattr(version, "label", None)))
            for part in (getattr(version, "parts", None) or [])[:2]:
                out.append("      PART " + str(getattr(part, "name", None)))
                for service in (getattr(part, "services", None) or [])[:2]:
                    vars_ = getattr(service, "variables", None)
                    out.append("        SVC " + str(getattr(service, "service_short_name", None)) + " nvars " + str(len(vars_) if vars_ else 0))
                    if vars_:
                        out.append("        sample " + str([str(getattr(v, "short_name", v)) for v in vars_[:10]]))
open("outputs/tide_gauge_structure_probe.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("probe done", count)