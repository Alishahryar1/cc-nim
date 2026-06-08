import timeit

setup = """
d = {"a": 1, "b": 2}
"""

stmt_1 = """
val = d.get("missing", {})
"""

stmt_2 = """
val = d.get("missing")
if val is None:
    val = {}
"""

print(
    "stmt_1 (empty dict fallback allocation):",
    timeit.timeit(stmt_1, setup=setup, number=10000000),
)
print(
    "stmt_2 (strict None check):", timeit.timeit(stmt_2, setup=setup, number=10000000)
)
