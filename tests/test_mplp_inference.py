from itertools import combinations

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp
from pgmpy.models import MarkovNetwork

UNARY_TABLE = [0.5, 0.5]
TERNARY_TABLE = [1, 1, 1, 0, 1, 0, 0, 1]


def test_mplp_inference():
    mrf = MarkovNetwork()

    # Add variables and unary factors
    variable_names = ["R_1-2", "R_1-3", "R_1-4", "R_2-3", "R_2-4", "R_3-4"]

    for var_name in variable_names:
        mrf.add_node(var_name)

        unary_factor = DiscreteFactor(
            [var_name], cardinality=[2], values=UNARY_TABLE
        )
        mrf.add_factors(unary_factor)

    # Add ternary factors
    ternary_combos = combinations(range(1, 5), 3)

    for combo in ternary_combos:
        var1 = f"R_{combo[0]}-{combo[1]}"
        var2 = f"R_{combo[0]}-{combo[2]}"
        var3 = f"R_{combo[1]}-{combo[2]}"

        mrf.add_edges_from([(var1, var2), (var1, var3), (var2, var3)])
        ternary_factor = DiscreteFactor(
            [var1, var2, var3], cardinality=[2, 2, 2], values=TERNARY_TABLE
        )
        ternary_factor.normalize()

        mrf.add_factors(ternary_factor)

    mplp = Mplp(mrf)
    results = mplp.map_query()
    print(results)
