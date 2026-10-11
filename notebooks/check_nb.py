import json

with open("notebooks/01_eda.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
executed = sum(1 for c in code_cells if c.get("execution_count") is not None)
print(f"Total code cells : {len(code_cells)}")
print(f"Executed cells   : {executed}")

for i, cell in enumerate(code_cells):
    ec = cell.get("execution_count")
    outputs = cell.get("outputs", [])
    has_error = any(o.get("output_type") == "error" for o in outputs)
    status = "ERROR" if has_error else ("OK" if ec else "NOT RUN")
    src_preview = "".join(cell.get("source", []))[:60].replace("\n", " ")
    print(f"  Cell {i+1:2d} (exec={ec}) [{status}]: {src_preview}")
    if has_error:
        for o in outputs:
            if o.get("output_type") == "error":
                import re
                tb = "".join(o.get("traceback", []))
                tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                print(f"    >>> {o.get('ename')}: {o.get('evalue')}")
                print(tb[-500:])
