'''On/off kinetics (bright and dark times) of blinking molecules or DNA-PAINT
binding sites, grouped by cluster or track labels.'''

import numba
import numpy as np
import pandas as pd


def split_events(frames: np.ndarray, max_gap: int = 1):
    '''Bright and dark times, in frames, of one group's frame numbers.

    Up to `max_gap` missing frames inside an event keep it one event.
    '''
    frames = np.unique(frames)
    breaks = np.flatnonzero(np.diff(frames) - 1 > max_gap)
    starts = np.r_[frames[0], frames[breaks + 1]]
    ends = np.r_[frames[breaks], frames[-1]]
    return ends - starts + 1, starts[1:] - ends[:-1] - 1


@numba.njit(cache=True)
def _det3(m: np.ndarray) -> float:
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


@numba.njit(cache=True)
def _solve3(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    '''Cramer's rule for a 3x3 system; NaN if singular.'''
    det = _det3(m)
    x = np.full(3, np.nan)
    if det == 0.0 or not np.isfinite(det):
        return x
    for k in range(3):
        mk = m.copy()
        mk[:, k] = v
        x[k] = _det3(mk) / det
    return x


@numba.njit(cache=True)
def _cost(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> float:
    s = 0.0
    for i in range(len(t)):
        r = y[i] - (p[0] * (1.0 - np.exp(-t[i] / p[1])) + p[2])
        s += r * r
    return s


@numba.njit(cache=True)
def _fit_sorted(t: np.ndarray) -> float:
    '''Levenberg-Marquardt fit of a * (1 - exp(-t / tau)) + c to the CDF of
    the sorted times `t` (same least squares as scipy's curve_fit); NaN if
    fewer than 3 distinct times, no valid fit, or tau over 10x the longest
    time.'''
    n = len(t)
    u = np.empty(n)
    y = np.empty(n)
    k = 0
    for i in range(n):  # CDF at the top of each tie
        if i == n - 1 or t[i + 1] != t[i]:
            u[k] = t[i]
            y[k] = (i + 1) / n
            k += 1
    if k < 3:
        return np.nan
    u, y = u[:k], y[:k]

    p = np.array([1.0, t.mean(), 0.0])
    if not p[1] > 0:
        return np.nan
    cost = _cost(u, y, p)
    lam = 1e-3
    for _ in range(200):
        jtj = np.zeros((3, 3))
        jtr = np.zeros(3)
        jac = np.empty(3)
        for i in range(k):
            e = np.exp(-u[i] / p[1])
            r = y[i] - (p[0] * (1.0 - e) + p[2])
            jac[0] = 1.0 - e
            jac[1] = -p[0] * u[i] * e / (p[1] * p[1])
            jac[2] = 1.0
            for a in range(3):
                jtr[a] += jac[a] * r
                for b in range(3):
                    jtj[a, b] += jac[a] * jac[b]

        accepted = False
        step, new, new_cost = np.zeros(3), p.copy(), cost
        while lam < 1e12:
            m = jtj.copy()
            for a in range(3):
                m[a, a] += lam * max(jtj[a, a], 1e-12)
            step = _solve3(m, jtr)
            new = p + step
            if np.all(np.isfinite(new)) and new[1] > 0:
                new_cost = _cost(u, y, new)
                if new_cost < cost:
                    accepted = True
                    break
            lam *= 10.0
        if not accepted:
            break
        done = np.abs(step).max() <= 1e-10 * (np.abs(p).max() + 1e-10)
        p, cost = new, new_cost
        lam = max(lam / 10.0, 1e-12)
        if done:
            break

    # far beyond the longest time the CDF is a straight line: tau unmeasurable
    if p[0] > 0 and 0 < p[1] <= 10 * t[-1]:
        return p[1]
    return np.nan


@numba.njit(parallel=True, cache=True)
def _fit_groups(times: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    '''Fit each group times[offsets[g]:offsets[g + 1]] (sorted in a group).'''
    out = np.empty(len(offsets) - 1)
    for g in numba.prange(len(out)):
        out[g] = _fit_sorted(times[offsets[g] : offsets[g + 1]])
    return out


def fit_lifetime(times: np.ndarray) -> float:
    '''Fit a * (1 - exp(-t / tau)) + c to the empirical CDF of `times`.

    The free a and c absorb the shortest recordable time (one frame, the
    detection threshold, max_gap), so tau does not depend on the exposure.
    Returns NaN with fewer than 3 distinct times, if the fit fails, or if tau
    is over 10x the longest time (the CDF is then a straight line).
    '''
    return float(_fit_sorted(np.sort(np.asarray(times, dtype=float))))


def _events(frames: np.ndarray, labels: np.ndarray, max_gap: int):
    '''Bright and dark times, in frames, of all groups at once, each with the
    label of its group; both sorted by label.'''
    order = np.lexsort((frames, labels))
    f, lab = frames[order], labels[order]
    new_group = np.r_[True, lab[1:] != lab[:-1]]
    keep = new_group | np.r_[True, f[1:] != f[:-1]]  # one per frame
    f, lab, new_group = f[keep], lab[keep], new_group[keep]

    new_event = new_group | np.r_[True, np.diff(f) - 1 > max_gap]
    starts = np.flatnonzero(new_event)
    ends = np.r_[starts[1:], len(f)] - 1
    ev_label = lab[starts]
    same = ev_label[1:] == ev_label[:-1]
    dark = (f[starts[1:]] - f[ends[:-1]] - 1)[same]
    return ev_label, f[ends] - f[starts] + 1, ev_label[1:][same], dark


def _group_fits(times, labels, ids):
    '''Per-group fits of `times`, grouped by sorted `labels`.'''
    offsets = np.r_[np.searchsorted(labels, ids), len(times)]
    in_group = times[np.lexsort((times, labels))]
    return _fit_groups(in_group.astype(float), offsets.astype(np.int64)), offsets


def binding_kinetics(
    frames: np.ndarray,
    labels: np.ndarray,
    max_gap: int = 1,
    frame_time: float = 1.0,
    min_events: int = 3,
):
    '''Bright and dark times per group (labels < 0 are skipped).

    Parameters
    ----------
    frames, labels : np.ndarray
        Frame number and group label (cluster or track) of each localization.
    max_gap : int
        Missing frames tolerated inside one bright event.
    frame_time : float
        Time per frame; all times are returned in its unit.
    min_events : int
        Groups with fewer bright events get only means, no fits.

    Returns
    -------
    tuple
        A table with one row per group (label, events, mean and fitted bright
        and dark times), and all bright and dark times pooled.
    '''
    frames = np.asarray(frames, dtype=np.int64)
    labels = np.asarray(labels)
    keep = labels >= 0
    frames, labels = frames[keep], labels[keep]
    if len(frames) == 0:
        return pd.DataFrame(), np.empty(0), np.empty(0)

    b_label, bright, d_label, dark = _events(frames, labels, max_gap)
    bright, dark = bright * frame_time, dark * frame_time
    ids = np.unique(b_label)

    bright_fit, b_off = _group_fits(bright, b_label, ids)
    dark_fit, d_off = _group_fits(dark, d_label, ids)
    events, n_dark = np.diff(b_off), np.diff(d_off)
    bright_fit[events < min_events] = np.nan
    dark_fit[events < min_events] = np.nan
    group = np.arange(len(ids))
    bright_sum = np.bincount(np.repeat(group, events), bright, len(ids))
    dark_sum = np.bincount(np.repeat(group, n_dark), dark, len(ids))
    with np.errstate(invalid='ignore', divide='ignore'):
        dark_mean = np.where(n_dark > 0, dark_sum / n_dark, np.nan)

    table = pd.DataFrame(
        {
            'label': ids,
            'events': events,
            'bright_mean': bright_sum / events,
            'dark_mean': dark_mean,
            'bright_fit': bright_fit,
            'dark_fit': dark_fit,
        }
    )
    return table, bright, dark
