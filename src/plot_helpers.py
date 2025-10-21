import matplotlib.pyplot as plt
import networkx as nx
from .VQE_functions import get_sites_and_bonds
import numpy as np


def plot_topology(sites, node_size=1500, node_color="white", edge_color="blue", rad=0.25, labels=None):
    """
    Plot einer Spin-Topologie (MFIM) mit NetworkX und gebogenen Pfeilen.

    Args:
        sites (np.ndarray): Koordinaten der Sites (N,2).
        bonds (list[tuple]): Liste von Nachbarpaaren (i,j).
        node_size: Größe der Knoten.
        node_color: Farbe der Knoten.
        edge_color: Farbe der Pfeile (oder Liste).
        rad: Krümmungsradius für die Pfeile.
        labels: optionale Labels für die Nodes (Liste oder Dict).
    """
    if len(sites) == 0:
        raise ValueError("Es muss eine Topologie (sites) vorgegeben werden!")
    _ , bonds = get_sites_and_bonds(sites)
    # Graph aufbauen
    G = nx.DiGraph()
    G.add_nodes_from(range(len(sites)))
    G.add_edges_from(bonds + [(j,i) for i,j in bonds])

    # Position als dict für nx
    pos = {i: (sites[i,0], sites[i,1]) for i in range(len(sites))}

    fig, ax = plt.subplots(figsize=(5,5))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, edgecolors="black", ax=ax)
    if labels is None:
        labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", ax=ax)
    # Edges (gebogen)
    nx.draw_networkx_edges(
        G, pos,
        node_size=node_size,
        arrows=True,
        edge_color=edge_color,
        width=2,
        connectionstyle=f"arc3,rad={rad}",
        ax=ax
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return ax


def plot_2D(result, key="Rel_error", param_index=None):
    hx_vals = result["hx_vals"]
    hz_vals = result["hz_vals"]
    hx_max = result["max_error_coords"]["hx"]
    hz_max = result["max_error_coords"]["hz"]

    plt.figure(figsize=(6, 5))
    if key == "Opt_parameters":
        if param_index != None:
            parameter_map = result[key]
            plotmap = parameter_map[:,:,param_index]
            param_string = ["alpha","beta","gamma"]
            title = f"{param_string[param_index]} "
        else:
            raise ValueError("You need a parameter index for plotting the parameters")
    else:
        plotmap = result[key]
        title = f"VQE {key} "

    im = plt.imshow(plotmap.T, origin="lower",
                    extent=[hx_vals.min(), hx_vals.max(), hz_vals.min(), hz_vals.max()],
                    aspect="auto", cmap="viridis")
    plt.colorbar(im, label=title)
    plt.xlabel(r"$h_x$")
    plt.ylabel(r"$h_z$")
    plt.title(title)
    # Maximum markieren
    plt.scatter([hx_max], [hz_max], s=80, marker="x", linewidths=2, color="white")
    plt.show()
    return plt
    
def _imshow_ax(ax, data, extent, title, cmap, cbar_label=None, mark_max=False, max_coords=None):
    im = ax.imshow(data.T, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(r"$h_x$")
    ax.set_ylabel(r"$h_z$")
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    if mark_max and max_coords is not None:
        ax.scatter([max_coords["hx"]], [max_coords["hz"]],
                   s=80, marker="x", linewidths=2, color="white")
    return im

def plot_metrics(result, cmap="viridis", show_max=True):
    """
    Eine Figure mit 3 Subplots: Energy_exact, Energy_vqe, Rel_error.
    """
    hx_vals = result["hx_vals"]
    hz_vals = result["hz_vals"]
    extent = [hx_vals.min(), hx_vals.max(), hz_vals.min(), hz_vals.max()]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    _imshow_ax(axs[0], result["Energy_exact"], extent,
               "Energy_exact", cmap, cbar_label="Energy")
    _imshow_ax(axs[1], result["Energy_vqe"], extent,
               "Energy_vqe", cmap, cbar_label="Energy")
    _imshow_ax(axs[2], result["Rel_error"], extent,
               "Relative Error", cmap,
               mark_max=show_max, max_coords=result["max_error_coords"])

    plt.show()
    return fig, axs

def plot_params(result, layer=0, cmap="viridis"):
    """
    Eine Figure mit 3 Subplots: alpha, beta, gamma für den angegebenen Layer.
    """
    hx_vals = result["hx_vals"]
    hz_vals = result["hz_vals"]
    extent = [hx_vals.min(), hx_vals.max(), hz_vals.min(), hz_vals.max()]

    params = result["Opt_parameters"]  # shape (N, N, 3*num_layers)
    P = params.shape[-1]
    L = P // 3
    if not (0 <= layer < L):
        raise ValueError(f"layer außerhalb des Bereichs (0..{L-1}).")

    idx_alpha = 3*layer + 0
    idx_beta  = 3*layer + 1
    idx_gamma = 3*layer + 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    _imshow_ax(axs[0], params[:, :, idx_alpha], extent,
               f"alpha (layer {layer})", cmap, cbar_label="alpha [rad]")
    _imshow_ax(axs[1], params[:, :, idx_beta], extent,
               f"beta (layer {layer})", cmap, cbar_label="beta [rad]")
    _imshow_ax(axs[2], params[:, :, idx_gamma], extent,
               f"gamma (layer {layer})", cmap, cbar_label="gamma [rad]")

    plt.show()
    return fig, axs



def _bracket_1d(grid, x, tol=1e-12):
    """
    Finde die Indizes (i0,i1) in 'grid', so dass grid[i0] <= x <= grid[i1],
    sowie das Interpolationsgewicht t in [0,1] mit x = (1-t)*grid[i0] + t*grid[i1].
    Falls x exakt einem Gitterpunkt entspricht, wird (i0,i1,t) so gewählt,
    dass i1=None und t=0 zurückgegeben wird (Signal für 'exakter Treffer').
    Wirft ValueError, wenn x außerhalb des Gitters liegt.
    """
    grid = np.asarray(grid, dtype=float)
    if x < grid[0] - tol or x > grid[-1] + tol:
        raise ValueError(f"value {x} outside grid range [{grid[0]}, {grid[-1]}]")

    # Exakter Treffer am Rand?
    if abs(x - grid[0]) <= tol:
        return (0, None, 0.0)
    if abs(x - grid[-1]) <= tol:
        return (len(grid) - 1, None, 0.0)

    # Allgemeiner Fall: passende Klammer suchen
    i1 = int(np.searchsorted(grid, x, side="right"))
    i0 = i1 - 1
    # Exakter Treffer innerhalb
    if abs(x - grid[i0]) <= tol:
        return (i0, None, 0.0)
    if i1 < len(grid) and abs(x - grid[i1]) <= tol:
        return (i1, None, 0.0)

    if i1 >= len(grid) or i0 < 0:
        raise ValueError(f"value {x} outside grid range [{grid[0]}, {grid[-1]}]")

    t = (x - grid[i0]) / (grid[i1] - grid[i0])
    return (i0, i1, float(t))


def get_scan_value(result, hx, hz, key="Rel_error"):
    """
    Hole den Wert aus einem run_scan()-Result für beliebige (hx,hz).
    - Exakter Gitterpunkt -> exakter Matrixwert
    - Zwischen Gitterpunkten -> lineare/bilineare Interpolation
    - Außerhalb des Bereichs -> ValueError

    Args:
        result: dict wie von run_scan()
        hx, hz: Zielkoordinaten
        key:    "Energy_exact", "Energy_vqe", "Rel_error" oder "Opt_parameters"
        param_index: bei "Opt_parameters" benötigter Parameterindex (0..3*num_layers-1)

    Returns:
        float für Skalarfelder; bei "Opt_parameters" ebenfalls float (ein einzelner Parameter).
    """
    hx_vals = result["hx_vals"]
    hz_vals = result["hz_vals"]

    # Feld auswählen
    if key == "Opt_parameters":
        data = result["Opt_parameters"][:, :, :]  # (Nhx, Nhz)
    else:
        if key not in result:
            raise KeyError(f"Key '{key}' not found in result.")
        data = result[key]  # (Nhx, Nhz)

    # Indizes/Interpolationsgewichte bestimmen
    ix0, ix1, tx = _bracket_1d(hx_vals, hx)
    iz0, iz1, tz = _bracket_1d(hz_vals, hz)

    # Fälle: exakt / 1D linear / bilinear
    if ix1 is None and iz1 is None:
        # Exakter Gitterpunkt
        return float(data[ix0, iz0])

    if ix1 is None:
        # Nur in z Richtung interpolieren
        v0 = data[ix0, iz0]
        v1 = data[ix0, iz1]
        return float((1 - tz) * v0 + tz * v1)

    if iz1 is None:
        # Nur in x Richtung interpolieren
        v0 = data[ix0, iz0]
        v1 = data[ix1, iz0]
        return float((1 - tx) * v0 + tx * v1)

    # Bilinear
    v00 = data[ix0, iz0]
    v10 = data[ix1, iz0]
    v01 = data[ix0, iz1]
    v11 = data[ix1, iz1]
    res = (1 - tx) * (1 - tz) * v00 + tx * (1 - tz) * v10 +(1 - tx) *tz* v01 +tx*tz* v11
    return res
        
         
        
         