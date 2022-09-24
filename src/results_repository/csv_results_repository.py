import csv
from typing import Dict, Any, List, Union

from src.results_repository.results_repository_base import ResultsRepositoryBase


class CSVResultsRepository(ResultsRepositoryBase):
    """
    This class is used to save results of a training in a csv file. Results are saved in
    """

    def __init__(self, results_dir: str):
        if not isinstance(results_dir, str):
            raise ValueError(f"results_dir parameter must be a string, not: {type(results_dir)}")

        self._results_dir = results_dir

    @property
    def results_dir(self):
        return self._results_dir

    def save(self, results: Union[List[Dict[str, Any]], Dict[str, Any]]):
        column_names = self._get_column_names(results=results)
        results_file_path = self._get_results_file_path()

        with open(results_file_path, 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writeheader()

            if isinstance(results, list):
                writer.writerows(results)
            else:
                writer.writerow(results)

    def get(self) -> dict:
        pass

    def _get_results_file_path(self) -> str:
        return f"{self.results_dir}/results.csv"

    @staticmethod
    def _get_column_names(results: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[str]:
        used_dict = results

        if isinstance(results, list):
            used_dict = results[0]

        return list(used_dict.keys())
