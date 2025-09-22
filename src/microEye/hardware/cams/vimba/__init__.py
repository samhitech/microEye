try:
    import vmbpy as vb

    INSTANCE = vb.VmbSystem.get_instance()
except Exception:
    vb = None
    INSTANCE = None
