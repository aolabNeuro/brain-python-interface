'''
Helper functions for the DPS experiment notebook (DPS_experiment.ipynb):
3D plotting of planes and targets, PCA plane fitting of reach endpoints,
design of perturbation planes, and conversion to BMI3D perturbation params.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from IPython.display import HTML

from aopy import visualization as aovis


# ===========================================================================
# Constants
# ===========================================================================

# Unit vectors for each coordinate axis
_AXES = {'x': np.array([1., 0., 0.]),
         'y': np.array([0., 1., 0.]),
         'z': np.array([0., 0., 1.])}

# Swaps design-frame y and z to get BMI3D coordinates (its own inverse)
_S_DESIGN_TO_BMI3D = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


# ===========================================================================
# Plotting functions
# ===========================================================================

def plot_plane(plane, gain=1.0, color='grey', alpha=0.15, resolution=100,
               bounds=None, ax=None, **kwargs):
    '''
    Plot a plane as a translucent surface on a 3D axis. The plane can be given
    either as a mapping matrix (the xy-plane is transformed by it) or as a plane
    equation Ax + By + Cz + D = 0.

    Args:
        plane ((3, 3), (4, 4), or (4,) array): mapping matrix, or plane equation coefficients (A, B, C, D)
        gain (float, optional): scales the plotted extent of the plane (-10*gain to 10*gain). Default 1.0.
        color (str, optional): surface color. Default 'grey'.
        alpha (float, optional): surface transparency. Default 0.15.
        resolution (int, optional): number of grid points along each side of the plane. Default 100.
        bounds (tuple, optional): (xmin, xmax, ymin, ymax, zmin, zmax). If given, the surface
            is clipped to these bounds and the axis limits are set to them. Default None.
        ax (pyplot.Axes, optional): 3D axis to plot on. If None, creates a new figure. Default None.
        kwargs (dict, optional): additional keyword arguments passed to ax.plot_surface

    Returns:
        pyplot.Axes: the axis the plane was plotted on
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    grid_range = np.linspace(-10*gain, 10*gain, resolution)

    if plane.shape in [(3, 3), (4, 4)]:
        x, y = np.meshgrid(grid_range, grid_range)
        coords = np.column_stack((x.ravel(), y.ravel(), np.zeros_like(x.ravel())))
        rotated_coords = coords @ plane[:3, :3]
        x, y, z = rotated_coords.T.reshape(3, *x.shape)

    elif plane.shape == (4,):
        A, B, C, D = plane
        # Solve for whichever axis has the largest |normal component| so a
        # near-vertical plane (|C| ≈ 0) doesn't blow up the surface.
        dep = int(np.argmax(np.abs([A, B, C])))
        if dep == 2:
            x, y = np.meshgrid(grid_range, grid_range)
            z = (-A*x - B*y - D) / C
        elif dep == 1:
            x, z = np.meshgrid(grid_range, grid_range)
            y = (-A*x - C*z - D) / B
        else:
            y, z = np.meshgrid(grid_range, grid_range)
            x = (-B*y - C*z - D) / A
    else:
        raise ValueError("Plane must be (3,3), (4,4), or (4,).")

    # Apply bounds clipping
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        mask = (
            (x >= xmin) & (x <= xmax) &
            (y >= ymin) & (y <= ymax) &
            (z >= zmin) & (z <= zmax)
        )
        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        z = np.where(mask, z, np.nan)

    ax.plot_surface(x, y, z, alpha=alpha, color=color, **kwargs)

    if bounds is not None:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

    return ax

def color_targets(target_locations, target_idx, colors, target_radius, alpha=1, transform_matrix=None, bounds=None, ax=None, **kwargs):
    '''
    Plot each unique target as a filled disk on a 3D axis, colored by its target index.

    Args:
        target_locations (ntarget, 3): position of each target
        target_idx (ntarget): target index of each location, used to pick its color
        colors (list): colors indexed by target index. Must have at least as many
            entries as there are unique target indices.
        target_radius (float): radius of each disk
        alpha (float, optional): disk transparency. Default 1.
        transform_matrix ((3, 3) array, optional): rotation applied to each disk, which
            starts in the xy-plane. Default None (no rotation).
        bounds (tuple, optional): (xmin, xmax, ymin, ymax, zmin, zmax) axis limits. Default None.
        ax (pyplot.Axes, optional): 3D axis to plot on. If None, creates a new figure. Default None.
        kwargs (dict, optional): additional keyword arguments passed to Poly3DCollection

    Returns:
        None: targets are plotted on the axis
    '''
    assert len(target_locations) == len(target_idx), "Locations must be the same length as indices"

    target_locations = np.array(target_locations)
    target_idx = np.array(target_idx)

    unique_targets = np.unique(np.column_stack((target_idx, target_locations)), axis=0)
    assert len(colors) >= len(np.unique(target_idx)), "Not enough colors for unique indices"

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    for row in unique_targets:
        idx = int(row[0])
        loc = row[1:]
        plot_circles([loc], target_radius, colors[idx], alpha=alpha, transform_matrix=transform_matrix, bounds=bounds, ax=ax, **kwargs)

    ax.set_box_aspect([1, 1, 1])

def plot_circles(circle_positions, circle_radius, circle_color='b', transform_matrix=None, bounds=None, alpha=1, ax=None, **kwargs):
    '''
    Plot filled disks at the given positions on a 3D axis. Each disk starts in the
    xy-plane and is rotated by transform_matrix before being translated into place.

    Args:
        circle_positions (ncircle, 3): center of each disk
        circle_radius (float): radius of each disk
        circle_color (str, optional): disk color. Default 'b'.
        transform_matrix ((3, 3) array, optional): rotation applied to each disk. Default None (no rotation).
        bounds (tuple, optional): (xmin, xmax, ymin, ymax, zmin, zmax) axis limits. Default None.
        alpha (float, optional): disk transparency. Default 1.
        ax (pyplot.Axes, optional): 3D axis to plot on. Default None (current axis).
        kwargs (dict, optional): additional keyword arguments passed to Poly3DCollection

    Returns:
        None: disks are plotted on the axis
    '''
    if ax is None:
        ax = plt.gca(projection='3d')

    # Generate a 2D circle in the XY plane
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = circle_radius * np.cos(theta)
    y_circle = circle_radius * np.sin(theta)
    z_circle = np.zeros_like(x_circle)  # Initially in the xy plane

    # Stack into a (3, N) array for matrix multiplication
    circle_points = np.vstack((x_circle, y_circle, z_circle))

    # Apply transformation matrix
    if transform_matrix is None:
        transform_matrix = np.eye(3)  # Default to xy plane

    transformed_points = transform_matrix @ circle_points  # Apply transformation

    for pos in circle_positions:
        pos = np.array(pos)

        # Apply translation
        x, y, z = transformed_points + pos[:, np.newaxis]

        # Create a filled polygon (triangle mesh) for the circle
        verts = [list(zip(x, y, z))]  # Format required for Poly3DCollection

        # Add the filled circle to the plot
        poly = Poly3DCollection(verts, color=circle_color, alpha=alpha, **kwargs)
        ax.add_collection3d(poly)

    if bounds is not None:
        aovis.set_bounds(bounds, ax)


# ===========================================================================
# Rotations & geometry
# ===========================================================================

def _shortest_rotation(d, n):
    '''
    Shortest rotation R with R @ d = n. If d ≈ -n the rotation axis is
    ill-defined, so a 180 deg rotation about an axis perpendicular to d is returned.

    Args:
        d (3,): unit vector to rotate from
        n (3,): unit vector to rotate to

    Returns:
        (3, 3) array: rotation matrix
    '''
    cos_a = float(np.dot(d, n))
    if cos_a > 0.9999:
        return np.eye(3)
    if cos_a < -0.9999:
        axis = np.cross(d, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(d, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(d, n)
    sin_a = np.linalg.norm(axis)
    axis = axis / sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def _rot_axis(axis, angle):
    '''
    Right-handed rotation by angle about an arbitrary axis (Rodrigues).

    Args:
        axis (3,): rotation axis (normalized internally)
        angle (float): rotation angle in radians

    Returns:
        (3, 3) array: rotation matrix
    '''
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _display_normal(display):
    '''
    Normal vector of the display plane in design coordinates.

    Args:
        display (str): 'upright' or 'flat'

    Returns:
        (3,) array: +z for upright, +y for flat
    '''
    return np.array([0., 0., 1.]) if display == 'upright' else np.array([0., 1., 0.])


def _alt_axis(display):
    '''
    Alternative (perpendicular) perturbation axis.

    Args:
        display (str): 'upright' or 'flat'

    Returns:
        str: 'y' for upright, 'z' for flat
    '''
    return 'y' if display == 'upright' else 'z'


def _ang(a, b):
    '''
    Acute angle between two vectors (sign of either vector is ignored).

    Args:
        a (3,): first vector
        b (3,): second vector

    Returns:
        float: angle in degrees, between 0 and 90
    '''
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


# ===========================================================================
# PCA plane fitting
# ===========================================================================

def fit_PCA_plane(hand_traj):
    '''
    Fit a plane to the reach endpoints (last sample of each trajectory) with PCA.
    The plane normal is the direction of least variance, oriented into the
    +y/+z half-space.

    Args:
        hand_traj (list of (nt, 3) arrays): hand trajectory for each trial

    Returns:
        tuple: tuple containing:
            | **equation (4,):** plane equation coefficients (A, B, C, D)
            | **origin (3,):** mean endpoint, which the plane passes through
            | **normal (3,):** unit normal of the plane
            | **percent_explained (float):** fraction of endpoint variance within the plane
            | **var_normal (float):** endpoint variance along the normal
    '''
    endpoints = np.vstack([traj[-1] for traj in hand_traj])
    origin = np.mean(endpoints, axis=0)
    centered = endpoints - origin

    # Perform PCA using SVD
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Normal vector is the last principal component (smallest variance direction)
    normal = normal / np.linalg.norm(normal)

    # Compute plane offset D such that the plane goes through the origin
    D = -np.dot(normal, origin)
    equation = np.append(normal, D)

    # Compute variance explained
    projected = centered @ normal
    var_total = np.sum(np.var(centered, axis=0))
    var_normal = np.var(projected)
    percent_explained = (1 - (var_normal/var_total)) if (var_total > 0) else 0

    # Standardize normal direction so it points into the +y/+z half-space
    # the half containing the mapping plane normals for both upright (+z) and
    # flat (+y) displays plus their moderate perturbations.
    if normal[1] + normal[2] < 0:
        normal = -normal
        equation[:3] = -equation[:3]
        equation[3] = -equation[3]

    return equation, origin, normal, percent_explained, var_normal


def align_PC_plane(n_PC, hand_traj, target_idx, display_targets, display_normal):
    '''
    Align canonical display targets to subject reach data on the PC plane.

    Two steps:
      1. Pick the sign of n_PC by SSD between
         _shortest_rotation(display_normal, ±n_PC) @ canonical_targets and
         the reach centroids projected onto the PC plane. The chosen
         rotation is R_oop ("out of plane").
      2. Compute the in-plane rotation R_ip about the signed n_PC that
         minimizes SSD between R_ip @ R_oop @ canonical_targets and the
         projected centroids (closed-form 2D Procrustes).

    Args:
        n_PC (3,): unit normal of the PC plane (e.g. from fit_PCA_plane)
        hand_traj (list of (nt, 3) arrays): hand trajectory for each trial
        target_idx (ntr): target index of each trial
        display_targets (ntarget, 3): canonical target positions, indexed by target index
        display_normal (3,): normal of the display plane

    Returns:
        tuple: tuple containing:
            | **n_PC_signed (3,):** n_PC with the sign that best aligns the targets
            | **R_oop (3, 3):** out-of-plane rotation from display_normal to n_PC_signed
            | **R_ip (3, 3):** in-plane rotation about n_PC_signed
    '''
    n_PC = np.asarray(n_PC, dtype=float)
    target_idx = np.asarray(target_idx)
    endpoints = np.vstack([traj[-1] for traj in hand_traj])
    origin = endpoints.mean(axis=0)

    unique_idx = np.unique(target_idx)
    centroids = np.array([endpoints[target_idx == idx].mean(axis=0)
                          for idx in unique_idx])
    centroids_centered = centroids - origin
    projected = centroids_centered - (centroids_centered @ n_PC)[:, None] * n_PC

    canonical = np.asarray(display_targets)[unique_idx]

    # Step 1: pick sign and R_oop by SSD
    best_err = np.inf
    best_sign, R_oop = +1, None
    for s in (+1, -1):
        R = _shortest_rotation(display_normal, s * n_PC)
        rotated = canonical @ R.T
        err = float(np.sum((rotated - projected) ** 2))
        if err < best_err:
            best_err, best_sign, R_oop = err, s, R
    n_PC_signed = best_sign * n_PC

    # Step 2: in-plane rotation R_ip about n_PC_signed
    rotated_canonical = canonical @ R_oop.T
    crosses = np.cross(rotated_canonical, projected)
    sum_cross_n = float(crosses.sum(axis=0) @ n_PC_signed)
    sum_dot = float(np.sum(rotated_canonical * projected))
    phi = np.arctan2(sum_cross_n, sum_dot)

    K = np.array([[0, -n_PC_signed[2], n_PC_signed[1]],
                  [n_PC_signed[2], 0, -n_PC_signed[0]],
                  [-n_PC_signed[1], n_PC_signed[0], 0]])
    R_ip = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * (K @ K)

    return n_PC_signed, R_oop, R_ip


def _fit_signed_normal(display, targets, target_idx, hand_traj):
    '''
    Fit the PC plane to the reach endpoints and sign its normal so the display
    targets best align with the reaches.

    Args:
        display (str): 'upright' or 'flat'
        targets (ntarget, 3): canonical target positions, indexed by target index
        target_idx (ntr): target index of each trial
        hand_traj (list of (nt, 3) arrays): hand trajectory for each trial

    Returns:
        tuple: tuple containing:
            | **n_ref (3,):** signed unit normal of the PC plane
            | **dn (3,):** display normal
    '''
    dn = _display_normal(display)
    _, _, n_PC, _, _ = fit_PCA_plane(hand_traj)
    n_ref, _, _ = align_PC_plane(n_PC, hand_traj, target_idx, targets, dn)
    return n_ref / np.linalg.norm(n_ref), dn


# ===========================================================================
# No-mirror compliance
# ===========================================================================

def _targets_compliant(R, display, display_targets, tol=1e-9):
    '''
    No-mirror check: place targets by rotation R; every target must keep the
    sign of both its in-plane coords (right = x; up = y upright / forward = z flat).

    Args:
        R (3, 3): rotation that places the targets
        display (str): 'upright' or 'flat'
        display_targets (ntarget, 3): canonical target positions
        tol (float, optional): coordinates smaller than this are not checked. Default 1e-9.

    Returns:
        tuple: tuple containing:
            | **ok (bool):** True if no target is mirrored
            | **mirrored (list):** (target_idx, axis) for each mirrored coordinate
    '''
    inplane = (0, 1) if display == 'upright' else (0, 2)
    tgt = np.asarray(display_targets, float)
    mirrored = []
    for i, t in enumerate(tgt):
        h = R @ t
        for ax in inplane:
            if abs(t[ax]) > tol and np.sign(h[ax]) != np.sign(t[ax]):
                mirrored.append((i, ax))
    return (len(mirrored) == 0), mirrored


def _no_mirror_gap_deg(n, display_normal, display, display_targets, step=0.5, max_deg=90.0):
    '''
    Angular distance from normal n to the no-mirror compliant region: the
    smallest rotation of n (about any axis) whose roll-free placement is
    both-axis compliant.

    Args:
        n (3,): plane normal to test
        display_normal (3,): normal of the display plane
        display (str): 'upright' or 'flat'
        display_targets (ntarget, 3): canonical target positions
        step (float, optional): search step size in degrees. Default 0.5.
        max_deg (float, optional): largest rotation searched, in degrees. Default 90.0.

    Returns:
        float: gap in degrees. 0.0 if already compliant; nan if none within max_deg.
    '''
    if _targets_compliant(_shortest_rotation(display_normal, n), display, display_targets)[0]:
        return 0.0
    m = 48; k = np.arange(m) + 0.5
    pol = np.arccos(1 - 2 * k / m); azi = np.pi * (1 + 5 ** 0.5) * k
    axes = np.c_[np.cos(azi) * np.sin(pol), np.sin(azi) * np.sin(pol), np.cos(pol)]
    for deg in np.arange(step, max_deg + step, step):
        ang = np.radians(deg)
        for ax in axes:
            if _targets_compliant(_shortest_rotation(display_normal, _rot_axis(ax, ang) @ n),
                                  display, display_targets)[0]:
                return float(deg)
    return float('nan')


# ===========================================================================
# BMI3D angle conversion
# ===========================================================================

def bmi3d_angles(R_py):
    '''
    Convert a design-frame rotation to BMI3D perturbation params, so that BMI3D's
    row-vector pipeline v @ Ry @ Rz @ Rx reproduces R_py. (S swaps design y<->z;
    then extract YZX Euler of R_bmi^T.)

    Args:
        R_py (3, 3): rotation in design coordinates

    Returns:
        tuple: tuple containing:
            | **a_x (float):** perturbation_rotation_x, in degrees
            | **a_y (float):** pertubation_rotation (about y), in degrees
            | **a_z (float):** perturbation_rotation_z, in degrees
    '''
    R_bmi = _S_DESIGN_TO_BMI3D @ R_py @ _S_DESIGN_TO_BMI3D
    a_y, a_z, a_x = Rotation.from_matrix(R_bmi.T).as_euler('YZX', degrees=True)
    return a_x, a_y, a_z


# ===========================================================================
# Plane generation (initial & alt-axis)
# ===========================================================================

def parallel_plane(display, targets, target_idx, hand_traj):
    '''
    Dual-axis Prior-Aligned initial plane: roll-free placement onto the subject's
    PC plane (the designed plane overlaps the PC plane; no in-screen roll).

    Args:
        display (str): 'upright' or 'flat'
        targets (ntarget, 3): canonical target positions, indexed by target index
        target_idx (ntr): target index of each trial
        hand_traj (list of (nt, 3) arrays): hand trajectory for each trial

    Returns:
        dict: with keys
            | **n_ref (3,):** signed normal of the subject's PC plane
            | **n_design (3,):** normal of the designed plane
            | **R (3, 3):** rotation from the display plane to the designed plane
            | **compliant (bool):** True if no target is mirrored
            | **gap_deg (float):** distance in degrees to the no-mirror region (0.0 if compliant)
    '''
    n_ref, dn = _fit_signed_normal(display, targets, target_idx, hand_traj)
    R = _shortest_rotation(dn, n_ref)
    n_design = R @ dn
    ok = _targets_compliant(R, display, targets)[0]
    return dict(n_ref=n_ref, n_design=n_design, R=R, compliant=ok,
                gap_deg=0.0 if ok else _no_mirror_gap_deg(n_design, dn, display, targets))


def orthogonal_plane(display, targets, target_idx, hand_traj):
    '''
    Prior-orthogonal initial plane: 90 deg about the relative-x axis, roll-free,
    in the mirror-free region. If neither direction complies, keep the
    least-violating one (smallest no-mirror gap).

    Args:
        display (str): 'upright' or 'flat'
        targets (ntarget, 3): canonical target positions, indexed by target index
        target_idx (ntr): target index of each trial
        hand_traj (list of (nt, 3) arrays): hand trajectory for each trial

    Returns:
        dict: same keys as parallel_plane (n_ref, n_design, R, compliant, gap_deg)
    '''
    n_ref, dn = _fit_signed_normal(display, targets, target_idx, hand_traj)
    R_oop = _shortest_rotation(dn, n_ref)
    x_rel = R_oop @ np.array([1., 0., 0.])
    chosen = None
    for sgn in (1.0, -1.0):
        n_d = _rot_axis(x_rel, sgn * np.pi / 2) @ n_ref
        R_d = _shortest_rotation(dn, n_d)
        if _targets_compliant(R_d, display, targets)[0]:
            chosen = (n_d, R_d); break
    if chosen is None:
        cands = [(_no_mirror_gap_deg(_rot_axis(x_rel, s * np.pi / 2) @ n_ref, dn, display, targets), s)
                 for s in (1.0, -1.0)]
        s = min(cands, key=lambda c: (np.inf if np.isnan(c[0]) else c[0]))[1]
        n_design = _rot_axis(x_rel, s * np.pi / 2) @ n_ref
        R_design = _shortest_rotation(dn, n_design)
        ok = False
    else:
        n_design, R_design = chosen; ok = True
    return dict(n_ref=n_ref, n_design=n_design, R=R_design, compliant=ok,
                gap_deg=0.0 if ok else _no_mirror_gap_deg(n_design, dn, display, targets))


def alt_axis_plane(display, targets, target_idx, hand_traj, rotation_deg=60):
    '''
    Compound perturbation from the z2 PC plane: rotate rotation_deg about the
    relative-alt axis (y upright / z flat), roll-free, in the mirror-free region.
    When both directions comply keep the one further from the display; if neither
    complies keep the least-violating one (smallest no-mirror gap).

    Args:
        display (str): 'upright' or 'flat'
        targets (ntarget, 3): canonical target positions, indexed by target index
        target_idx (ntr): target index of each trial
        hand_traj (list of (nt, 3) arrays): hand trajectory for each trial
        rotation_deg (float, optional): rotation about the relative-alt axis, in degrees. Default 60.

    Returns:
        dict: same keys as parallel_plane (n_ref, n_design, R, compliant, gap_deg)
    '''
    n_ref, dn = _fit_signed_normal(display, targets, target_idx, hand_traj)
    R_oop = _shortest_rotation(dn, n_ref)
    alt_rel = R_oop @ _AXES[_alt_axis(display)]
    best = None
    for sgn in (1.0, -1.0):
        n_d = _rot_axis(alt_rel, sgn * np.radians(rotation_deg)) @ n_ref
        R_d = _shortest_rotation(dn, n_d)
        if _targets_compliant(R_d, display, targets)[0]:
            tilt = _ang(n_d, dn)
            if best is None or tilt > best[0]:
                best = (tilt, n_d, R_d)
    if best is None:
        cands = [(_no_mirror_gap_deg(_rot_axis(alt_rel, s * np.radians(rotation_deg)) @ n_ref,
                                     dn, display, targets), s) for s in (1.0, -1.0)]
        s = min(cands, key=lambda c: (np.inf if np.isnan(c[0]) else c[0]))[1]
        n_design = _rot_axis(alt_rel, s * np.radians(rotation_deg)) @ n_ref
        R_design = _shortest_rotation(dn, n_design)
        ok = False
    else:
        _, n_design, R_design = best; ok = True
    return dict(n_ref=n_ref, n_design=n_design, R=R_design, compliant=ok,
                gap_deg=0.0 if ok else _no_mirror_gap_deg(n_design, dn, display, targets))


# ===========================================================================
# Notebook utilities
# ===========================================================================

def play_sound(file):
    '''
    Play an audio file in a Jupyter notebook (the returned object must be the
    last expression in the cell to be displayed).

    Args:
        file (str): path to the audio file

    Returns:
        IPython.display.HTML: autoplaying audio element
    '''
    return HTML(f"""
    <audio autoplay>
      <source src="{file}" type="audio/mpeg">
    </audio>
    """)
