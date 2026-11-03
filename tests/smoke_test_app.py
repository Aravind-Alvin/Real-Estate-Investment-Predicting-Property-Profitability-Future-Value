"""Runs app/streamlit_app.py against a stub `streamlit` (real Streamlit isn't required) to catch Python errors on every page."""
import sys, types, runpy
from pathlib import Path
from unittest.mock import MagicMock
import matplotlib; matplotlib.use("Agg")

def make_stub(page):
    st = types.ModuleType("streamlit")
    M = lambda: MagicMock()
    st.set_page_config = M()
    st.cache_resource = st.cache_data = lambda f=None, **k: (f if f else (lambda g: g))
    st.sidebar = MagicMock(); st.sidebar.radio = lambda label, opts: next(o for o in opts if page in o)
    st.selectbox = lambda l, o, index=0, **k: list(o)[index]
    st.multiselect = lambda l, o, default=None, **k: list(default or [])
    st.slider = lambda l, a=0, b=1, v=None, value=None, **k: (value if value is not None else v if v is not None else a)
    st.select_slider = lambda l, o, value=None, **k: value
    st.number_input = lambda l, a, b, v, **k: v
    st.checkbox = lambda l, v=False, **k: v
    st.form_submit_button = lambda *a, **k: True
    class Ctx:                      # column/tab/form object: delegates widgets to the stub, usable in `with`
        def __getattr__(self, n): return getattr(st, n)
        def __enter__(self): return self
        def __exit__(self, *a): return False
    st.columns = lambda n: [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda names: [Ctx() for _ in names]
    st.form = lambda *a, **k: Ctx()
    for n in ["title","caption","divider","subheader","metric","success","warning","error","info","pyplot","dataframe","download_button","markdown"]:
        setattr(st, n, M())
    return st

for page in ["Advisor", "Explore", "Market", "Model", "About"]:
    sys.modules["streamlit"] = make_stub(page)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"))
    print("OK:", page)
