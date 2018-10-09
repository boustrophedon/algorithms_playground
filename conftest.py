import pytest

def pytest_collection_modifyitems(config, items):
    skip_bench = pytest.mark.skip(reason = "Skip benchmarks by default")
    skip_nonbench = pytest.mark.skip(reason = "Skip non-benchmarks when running with --bench flag")

    # if we are benching, skip all nonbenches
    if config.option.bench:
        for item in items:
            if "_bench_" not in item.nodeid:
                item.add_marker(skip_nonbench)

    # else, skip all benches by default
    else:
        for item in items:
            if "_bench_" in item.nodeid:
                item.add_marker(skip_bench)

# add bench option to disable/enable benches only
def pytest_addoption(parser):
    parser.addoption('--bench', action='store_true', dest="bench",
                 default=False, help="enable bench tests")
