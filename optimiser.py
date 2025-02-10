import multiprocessing as mp
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class SimulationResult:
    rise_time: float
    fall_time: float

    def objective_function(self) -> float:
        """
        Objective function to minimize. Modify this based on your requirements.
        Currently aims to minimize both rise and fall times equally.
        """
        return -(self.rise_time + self.fall_time)  # Negative because we're maximizing


class LTSpiceOptimizer:
    def __init__(
        self,
        asc_file: str,
        ltspice_path: str = r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
        epsilon: float = 0.01,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
    ):
        self.asc_file = Path(asc_file)
        self.ltspice_path = Path(ltspice_path)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.params = self._extract_params()

    def _extract_params(self) -> Dict[str, float]:
        """Extract .param definitions from the .asc file"""
        params = {}
        with open(self.asc_file, "r") as f:
            for line in f:
                if line.startswith(".param"):
                    # Parse param name and value
                    match = re.match(r"\.param\s+(\w+)\s*=\s*([\d.e+-]+)", line)
                    if match:
                        name, value = match.groups()
                        params[name] = float(value)
        return params

    def _create_modified_asc(self, params: Dict[str, float], temp_dir: Path) -> Path:
        """Create a modified .asc file with new parameter values"""
        output_file = temp_dir / f"modified_{self.asc_file.name}"
        with open(self.asc_file, "r") as fin, open(output_file, "w") as fout:
            for line in fin:
                if line.startswith(".param"):
                    param_name = re.match(r"\.param\s+(\w+)", line).group(1)
                    if param_name in params:
                        line = f".param {param_name}={params[param_name]}\n"
                fout.write(line)
        return output_file

    def _run_simulation(self, asc_file: Path) -> SimulationResult:
        """Run LTspice simulation and extract rise/fall times"""
        # Run LTspice
        subprocess.run(
            [str(self.ltspice_path), "-b", str(asc_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Parse the output log file to extract rise and fall times
        # This is a placeholder - you'll need to modify this based on your specific output format
        log_file = asc_file.with_suffix(".log")
        rise_time = 0
        fall_time = 0

        with open(log_file, "r") as f:
            for line in f:
                if "rise_time=" in line:
                    rise_time = float(
                        re.search(r"rise_time=([\d.e+-]+)", line).group(1)
                    )
                if "fall_time=" in line:
                    fall_time = float(
                        re.search(r"fall_time=([\d.e+-]+)", line).group(1)
                    )

        return SimulationResult(rise_time, fall_time)

    def _compute_gradient(
        self, current_params: Dict[str, float], temp_dir: Path
    ) -> Dict[str, float]:
        """Compute gradient for all parameters in parallel"""

        def compute_partial_derivative(args) -> Tuple[str, float]:
            param_name, current_value = args

            # Create parameter sets for +/- epsilon
            params_plus = current_params.copy()
            params_plus[param_name] = current_value + self.epsilon

            params_minus = current_params.copy()
            params_minus[param_name] = current_value - self.epsilon

            # Run simulations
            asc_plus = self._create_modified_asc(params_plus, temp_dir)
            asc_minus = self._create_modified_asc(params_minus, temp_dir)

            result_plus = self._run_simulation(asc_plus)
            result_minus = self._run_simulation(asc_minus)

            # Compute partial derivative
            partial = (
                result_plus.objective_function() - result_minus.objective_function()
            ) / (2 * self.epsilon)

            return param_name, partial

        # Run computations in parallel
        with mp.Pool() as pool:
            results = pool.map(compute_partial_derivative, current_params.items())

        return dict(results)

    def optimize(self) -> Dict[str, float]:
        """Run gradient descent optimization"""
        current_params = self.params.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            for iteration in range(self.max_iterations):
                # Compute gradient
                gradient = self._compute_gradient(current_params, temp_dir)

                # Update parameters
                for param_name, grad_value in gradient.items():
                    current_params[param_name] += self.learning_rate * grad_value

                # Run simulation with current parameters
                current_asc = self._create_modified_asc(current_params, temp_dir)
                result = self._run_simulation(current_asc)

                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"Current objective: {result.objective_function()}")
                print("Current parameters:", current_params)
                print()

        return current_params


if __name__ == "__main__":
    # Example usage
    optimizer = LTSpiceOptimizer(
        asc_file="your_circuit.asc",
        ltspice_path=r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
        epsilon=0.01,
        learning_rate=0.1,
        max_iterations=100,
    )

    optimal_params = optimizer.optimize()
    print("Optimization complete!")
    print("Optimal parameters:", optimal_params)
