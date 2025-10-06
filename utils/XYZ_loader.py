# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:00:46 2024

@author: Alexey
"""

from typing import List, Iterator, Union
import numpy as np
from tqdm import tqdm
import mmap
import os
import json
import hashlib
from datetime import datetime

class XYZLoader:
    """
    Memory-efficient XYZ file loader with persistent indexing capability.
    """
    def __init__(self, filename: str, index_dir: str = None):
        self.filename = filename
        self.index_dir = index_dir or os.path.join(os.path.dirname(filename), '.xyz_indices')
        self.file_size = os.path.getsize(filename)
        self._snapshot_positions = []
        self._n_atoms = 0
        self.types=[2]
        # Try to load existing index, create new if needed
        if not self._load_index():
            self._index_file()
            self._save_index()

    def _get_file_hash(self) -> str:
        """
        Generate a hash of the file's content and size for index verification.
        Only reads the first and last megabyte to speed up the process for large files.
        """
        hasher = hashlib.sha256()
        with open(self.filename, 'rb') as f:
            # Read first MB
            hasher.update(f.read(1024 * 1024))
            # Read last MB
            f.seek(max(0, self.file_size - 1024 * 1024))
            hasher.update(f.read())
        # Include file size and modification time in hash
        metadata = f"{self.file_size}_{os.path.getmtime(self.filename)}"
        hasher.update(metadata.encode())
        return hasher.hexdigest()

    def _get_index_filename(self) -> str:
        """Generate a unique index filename based on the XYZ file."""
        base_name = os.path.basename(self.filename)
        return os.path.join(self.index_dir, f"{base_name}.index")

    def _save_index(self):
        """Save the index data to a file."""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        index_data = {
            'file_hash': self._get_file_hash(),
            'n_atoms': self._n_atoms,
            'snapshot_positions': self._snapshot_positions,
            'created_at': datetime.now().isoformat()
        }

        with open(self._get_index_filename(), 'w') as f:
            json.dump(index_data, f)
        
        print(f"Index saved to {self._get_index_filename()}")

    def _load_index(self) -> bool:
        """
        Try to load existing index file.
        Returns True if successful, False if index needs to be recreated.
        """
        index_file = self._get_index_filename()
        if not os.path.exists(index_file):
            return False

        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)

            # Verify file hasn't changed
            if index_data['file_hash'] != self._get_file_hash():
                print("XYZ file has been modified, rebuilding index...")
                return False

            self._snapshot_positions = index_data['snapshot_positions']
            self._n_atoms = index_data['n_atoms']
            
            print(f"Loaded existing index from {index_file}")
            print(f"Index created at: {index_data['created_at']}")
            print(f"Found {len(self._snapshot_positions)} snapshots with {self._n_atoms} atoms each")
            return True

        except (json.JSONDecodeError, KeyError, TypeError):
            print("Invalid index file, rebuilding...")
            return False

    def _index_file(self):
        """Create an index of snapshot positions in the file."""
        print("Indexing XYZ file...")
        with open(self.filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            current_pos = 0
            with tqdm(total=self.file_size, unit='B', unit_scale=True) as pbar:
                while current_pos < mm.size():
                    self._snapshot_positions.append(current_pos)
                    
                    # Read number of atoms
                    n_atoms_str = mm[current_pos:current_pos+50].split(b'\n')[0].decode().strip()
                    n_atoms = int(n_atoms_str)
                    if self._n_atoms == 0:
                        self._n_atoms = n_atoms
                    # If atom count varies across snapshots, prefer first value silently
                    
                    # Skip to next snapshot
                    lines_to_skip = n_atoms + 2
                    for _ in range(lines_to_skip):
                        current_pos = mm.find(b'\n', current_pos) + 1
                        if current_pos == 0:  # EOF
                            break
                    
                    pbar.update(current_pos - pbar.n)
            
            mm.close()
        
        print(f"Found {len(self._snapshot_positions)} snapshots with {self._n_atoms} atoms each")

    def get_n_snapshots(self) -> int:
        """Return the total number of snapshots in the file."""
        return len(self._snapshot_positions)

    def get_n_atoms(self) -> int:
        """Return the number of atoms in each snapshot."""
        return self._n_atoms

    def load_snapshot(self, index: int) -> tuple:
        """
        Load a single snapshot by index.
        Returns: (symbols, coordinates, comment)
        """
        if index >= len(self._snapshot_positions):
            raise IndexError("Snapshot index out of range")

        with open(self.filename, 'r') as f:
            f.seek(self._snapshot_positions[index])
            
            n_atoms = int(f.readline().strip())
            comment = f.readline().strip()
            
            atoms=[]
            atom_types = []
            
            coordination = []
            
            for i in range(n_atoms):
                line = f.readline().split()
                
                atom_type=int(line[0])
                coordination=0
                coordinates = [float(x) for x in line[1:4]]
                if atom_type in self.types:
                    atoms.append((coordinates,atom_type,coordination))
        return atoms#atom_types, coordinates, coordination

    def load_snapshots(self, indices: Union[List[int], range]) -> Iterator[tuple]:
        """
        Load multiple snapshots by indices.
        Returns an iterator of (symbols, coordinates, comment) tuples.
        """
        for idx in tqdm(indices, desc="Loading snapshots"):
            yield self.load_snapshot(idx)

# Example usage

