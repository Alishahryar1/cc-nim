import timeit

setup = """
d = {"a": 1, "b": 2}
o = object()
"""

stmt_1 = """
val = d.get("missing", {})
"""

stmt_2 = """
val = d.get("missing")
if val is None:
    val = {}
"""

stmt_3 = """
val = d.get("missing")
if not isinstance(val, dict):
    val = {}
"""

print("stmt_1:", timeit.timeit(stmt_1, setup=setup, number=10000000))
print("stmt_2:", timeit.timeit(stmt_2, setup=setup, number=10000000))
print("stmt_3:", timeit.timeit(stmt_3, setup=setup, number=10000000))
