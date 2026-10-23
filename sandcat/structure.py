from __future__ import annotations

import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from sandcat.ligand import LigFile


@dataclass(frozen=True)
class PdbFile:
    path: str
    cofactor_ligfile: LigFile | None = None
    fluorine_atoms_pkl_path: str = ''
    trj_path: str = ''
    use_in_pipeline: bool = True

    def return_pdb_id(self):
        return Path(self.path).stem[:4]

# Keyword only to avoid issues with default args in parent class
# See https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses


@dataclass(frozen=True, kw_only=True)
class PredictedPdbFile(PdbFile):
    pae_path: str

    def return_pdb_id(self):
        filename = Path(self.path).stem
        filename = filename.split('model')[0]
        return filename[:-1]  # Remove trailing _ or -


@dataclass(frozen=True)
class PocketFile:
    path: str = ''
    parent_pdb_path: str = ''
    pocket_stats: dict = field(default_factory=dict)
    use_in_pipeline: bool = False


class Processor(ABC):
    def __init__(self, output_dir: str | None = None):
        if type(output_dir) is str:
            self.output_dir = output_dir
        else:
            self.output_dir = tempfile.mkdtemp()

    @abstractmethod
    def process_one(
        self,
        input_file: PdbFile,
        ligand_input_file: LigFile | None = None,
    ) -> list[PdbFile]:
        return [input_file]

    def process_many(self, input_files: list[PdbFile]) -> list[list[PdbFile]]:
        return [self.process_one(file) for file in input_files]


class PocketProcessor(ABC):
    def __init__(self, output_dir: str | None = None, run_leopard: bool = True):
        if type(output_dir) is str:
            self.output_dir = output_dir
        else:
            self.output_dir = tempfile.mkdtemp()

        self.run_leopard = run_leopard

    @abstractmethod
    def process_one(
        self,
        input_file: PdbFile,
        relevant_chain_ids: list,
    ) -> list[PocketFile]:
        return [PocketFile(path=input_file.path, use_in_pipeline=True)]

    def process_many(
        self,
        input_files: list[PdbFile],
        relevant_chain_ids: list,
    ) -> list[list[PocketFile]]:

        return ([self.process_one(file, relevant_chain_ids) for file in input_files])
