import numpy as np
from pathlib import Path
from typing import Dict, Iterable, Optional


def save_coordinations(coordinations: np.ndarray, path: str, timestep: int = 0) -> None:
    """Persist coordination histograms for crystalline atoms at a timestep."""

    with open(path, "a", encoding="utf-8") as handle:
        if timestep == 0:
            handle.write(
                "Timestep \t 0 Neighbours \t 1 Neighbour  \t 2 Neighbours  \t 3 Neighbours  \t 4 Neighbours \t 5 Neighbours \t 6 Neighbours \t "
                "0 Fraction \t 1 Fraction  \t 2 Fraction  \t 3 Fraction  \t 4 Fraction \t 5 Fraction \t 6 Fraction\n"
            )

        sum_coord = np.sum(coordinations)
        if sum_coord == 0:
            fractions = [0.0] * len(coordinations)
        else:
            fractions = [coord / sum_coord for coord in coordinations]

        handle.write(
            f"{timestep} \t {coordinations[0]} \t {coordinations[1]} \t {coordinations[2]} \t {coordinations[3]} \t {coordinations[4]} \t {coordinations[5]} \t {coordinations[6]} \t "
            f"{fractions[0]} \t {fractions[1]} \t {fractions[2]} \t {fractions[3]} \t {fractions[4]} \t {fractions[5]} \t {fractions[6]}\n"
        )


def save_states(states: np.ndarray, path: str, timestep: int = 0) -> None:
    """Summarize available surface states for the current simulation step."""

    with open(path, "a", encoding="utf-8") as handle:
        if timestep == 0:
            handle.write("Timestep \t On terrace \t At step  \t At kink" + "\n")

        handle.write(f"{timestep} \t {states[0]} \t {states[1]} \t {states[2]} \n")


def save_events(
    events: np.ndarray,
    path: str,
    event_types: Dict[str, Iterable[int]],
    timestep: int = 0,
) -> None:
    """Persist macro-level and per-rule event counts for diagnostics."""

    macro_path = f"{path}_macro_events.dat"
    detailed_path = f"{path}_all_events.dat"
    column_names_macro = "Timestep \t" + "\t ".join(event_types.keys()) + "\t all"
    column_names_detailed = ["Timestep"] + [f"Rule{i + 1}" for i in range(len(events))]

    with open(macro_path, "a", encoding="utf-8") as macro_handle, open(
        detailed_path, "a", encoding="utf-8"
    ) as detailed_handle:
        if timestep == 0:
            macro_handle.write(column_names_macro + "\n")
            detailed_handle.write(" \t ".join(column_names_detailed) + "\n")

        macro_handle.write(f"{timestep} \t ")
        detailed_handle.write(f"{timestep} \t ")
        for key in event_types.keys():
            macro_handle.write(f"{np.sum(events[event_types[key]])} \t ")
        macro_handle.write(f"{np.sum(events)} \t \n")
        for value in events:
            detailed_handle.write(f"{value} \t ")
        detailed_handle.write("\n")


def save_data(
    atoms: np.ndarray,
    path: str,
    timestep: int = 0,
    coordinations: Optional[np.ndarray] = None,
) -> int:
    """Export lattice atoms and aggregate population stats to XYZ-compatible files."""
    if coordinations is None:
        coordinations = np.zeros_like(atoms, dtype=np.int32)

    path_obj = Path(path)
    free_at = np.column_stack(np.nonzero(atoms))

    with path_obj.open("a", encoding="utf-8") as file_xyz:
        file_xyz.write(f"{len(free_at)}\n")
        file_xyz.write("Atoms in the system: 1 - mobile, 2 - crystalline" + "\n")

        stats = np.zeros((2))

        for i in range(len(free_at)):
            x, y, z = free_at[i]
            atom_state = atoms[x][y][z]
            coordination = coordinations[x][y][z]
            file_xyz.write(f"{atom_state} {x} {y} {z} {coordination}\n")
            stats[atom_state - 1] += 1

    stats_path = path_obj.with_name(f"{path_obj.stem}_stats.dat")
    with stats_path.open("a", encoding="utf-8") as file_stat:
        if timestep == 0:
            file_stat.write(
                "Timestep \t Mobile atoms \t Crystalline atoms \t Alpha \t Crystal size (if single crystal)" + "\n"
            )

        total_atoms = stats.sum()
        alpha = stats[1] / total_atoms if total_atoms else 0.0
        crystal_size = np.power(stats[1], 1 / 3) if stats[1] else 0.0
        file_stat.write(
            f"{timestep} \t {int(stats[0])} \t {int(stats[1])} \t {alpha} \t {crystal_size}\n"
        )

    return 0


def save_system_stats(
    atoms: np.ndarray,
    xyz_path: str,
    timestep: int = 0,
) -> None:
    """Write system_evolution_stats.dat using current atoms only (no XYZ write)."""
    path_obj = Path(xyz_path)
    stats_path = path_obj.with_name(f"{path_obj.stem}_stats.dat")
    # Count states 1 (mobile) and 2 (crystalline)
    # atoms is int32 array with 0/1/2
    mobile = int((atoms == 1).sum())
    crystalline = int((atoms == 2).sum())
    total = mobile + crystalline
    alpha = (crystalline / total) if total else 0.0
    crystal_size = (crystalline ** (1 / 3)) if crystalline else 0.0
    with stats_path.open("a", encoding="utf-8") as f:
        if timestep == 0:
            f.write(
                "Timestep \t Mobile atoms \t Crystalline atoms \t Alpha \t Crystal size (if single crystal)" + "\n"
            )
        f.write(f"{timestep} \t {mobile} \t {crystalline} \t {alpha} \t {crystal_size}\n")


def save_xyz_only(
    atoms: np.ndarray,
    path: str,
    timestep: int = 0,
    coordinations: Optional[np.ndarray] = None,
    cryst_age: Optional[np.ndarray] = None,
) -> int:
    """Write only the XYZ frame (no stats side file)."""
    if coordinations is None:
        coordinations = np.zeros_like(atoms, dtype=np.int32)
    path_obj = Path(path)
    free_at = np.column_stack(np.nonzero(atoms))
    with path_obj.open("a", encoding="utf-8") as file_xyz:
        file_xyz.write(f"{len(free_at)}\n")
        # If age is provided, append age to each row as the last column
        if cryst_age is not None:
            file_xyz.write("Atoms in the system: 1 - mobile, 2 - crystalline;  last column: crystal_age_steps" + "\n")
        else:
            file_xyz.write("Atoms in the system: 1 - mobile, 2 - crystalline" + "\n")
        for i in range(len(free_at)):
            x, y, z = free_at[i]
            atom_state = atoms[x][y][z]
            coordination = coordinations[x][y][z]
            if cryst_age is not None:
                age_val = int(cryst_age[x][y][z])
                file_xyz.write(f"{atom_state} {x} {y} {z} {coordination} {age_val}\n")
            else:
                file_xyz.write(f"{atom_state} {x} {y} {z} {coordination}\n")
    return 0


def save_states_3d(states: np.ndarray, path: str, timestep: int = 0, site_age: Optional[np.ndarray] = None) -> int:
    """Dump surface-state markers as dummy atoms into an XYZ file."""
    free_at = np.column_stack(np.nonzero(states))
    
    # Check if there are 0 atoms to save
    num_atoms = len(free_at)
    
    with open(path, "a", encoding="utf-8") as handle:
        # If no atoms found, add a dummy atom for OVITO compatibility
        if num_atoms == 0:
            handle.write("1\n")  # One atom
            if site_age is not None:
                handle.write("Dummy states on the crystal surface: 1 - On terrace, 2 - At step, 3 - At kink; last column: site_age_steps\n")
            else:
                handle.write("Dummy states on the crystal surface: 1 - On terrace, 2 - At step, 3 - At kink\n")
            # Add dummy atom of type 10 at position (0,0,0) for OVITO compatibility
            # OVITO doesn't work with empty snapshots, so this ensures the file is valid
            if site_age is not None:
                handle.write("10 0 0 0 0\n")
            else:
                handle.write("10 0 0 0\n")
        else:
            handle.write(f"{num_atoms}\n")
            if site_age is not None:
                handle.write("Dummy states on the crystal surface: 1 - On terrace, 2 - At step, 3 - At kink; last column: site_age_steps" + "\n")
            else:
                handle.write("Dummy states on the crystal surface: 1 - On terrace, 2 - At step, 3 - At kink" + "\n")

            for i in range(len(free_at)):
                x, y, z = free_at[i]
                if site_age is not None:
                    handle.write(f"{states[x][y][z]} {x} {y} {z} {int(site_age[x][y][z])}\n")
                else:
                    handle.write(f"{states[x][y][z]} {x} {y} {z}\n")

    return 0


__all__ = [
    "save_coordinations",
    "save_states",
    "save_events",
    "save_data",
    "save_system_stats",
    "save_xyz_only",
    "save_states_3d",
]


def save_crystalline_age_means(
    means_by_coord: list[float],
    path: str,
    timestep: int = 0,
) -> None:
    """Append mean crystalline ages per coordination number to a .dat file.

    Columns are ordered by coordination number starting from 1.
    """
    from pathlib import Path as _P
    p = _P(path)
    with p.open("a", encoding="utf-8") as f:
        if timestep == 0:
            header = ["Timestep"] + [f"coord_{k}_mean_age" for k in range(1, len(means_by_coord) + 1)]
            f.write("\t".join(header) + "\n")
        row = [str(timestep)] + [f"{float(v):.6f}" for v in means_by_coord]
        f.write("\t".join(row) + "\n")


def save_site_age_means(
    means_by_type: list[float],
    path: str,
    timestep: int = 0,
) -> None:
    """Append mean site ages per site type (1,2,3) to a .dat file.

    Columns order: type1(1-neighbour), type2(step), type3(kink)
    """
    from pathlib import Path as _P
    p = _P(path)
    with p.open("a", encoding="utf-8") as f:
        if timestep == 0:
            header = ["Timestep", "Site_on_terrace_mean_age", "Site_at_step_mean_age", "Site_at_kink_mean_age"]
            f.write("\t".join(header) + "\n")
        row = [str(timestep)] + [f"{float(v):.6f}" for v in means_by_type]
        f.write("\t".join(row) + "\n")
