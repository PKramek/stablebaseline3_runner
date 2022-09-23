from src.Constants import Constants


class AlgorithmFactory:

    @staticmethod
    def get(name: str):
        algorithm = Constants.ALGORITHMS.get(name, None)
        if algorithm is None:
            raise ValueError(f"Unknown algorithm: {name}, possible choices are: {Constants.ALGORITHMS.keys()}")

        return algorithm
