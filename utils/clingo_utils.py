from clingo import Control
import clingo

def run_clingo(program):
    """
    Runs clingo on the given ASP program and returns a list of answer sets.
    Each answer set is represented as a set of clingo symbols.
    """
    ctl = Control()
    ctl.configuration.solve.models = 0  # generate all models
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    models = []
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(set(model.symbols(shown=True)))
    return models