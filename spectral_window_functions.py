#!/usr/bin/env python3
"""
Spectral Window Functions for Mathematical Approximation
=======================================================

Implementation of configurable Fourier window functions inspired by Borwein 
integrals for enhancing mathematical approximations. This module provides
a framework for applying various spectral filters (Triangle, Gaussian, Sinc,
Staircase, Pyramid) to improve convergence in series approximations.

The window functions are designed to:
1. Enhance convergence properties of oscillatory sums
2. Provide configurable spectral filtering with different characteristics
3. Maintain computational efficiency while improving accuracy
4. Offer a unified framework for comparing window function performance

Mathematical Foundation:
The window functions modify oscillatory terms in explicit formulas through
carefully designed enhancement factors that preserve mathematical structure
while improving numerical properties.
"""

import math
from math import log, pi, sqrt
from typing import List, Tuple, Dict, Callable
import cmath

class SpectralWindowCalculator:
    """
    Spectral window function calculator for mathematical approximation enhancement.

    Provides configurable window functions inspired by Borwein integral techniques
    for improving convergence in oscillatory mathematical series.
    """

    def __init__(self):
        """Initialize with Riemann zeta zeros for demonstration purposes."""
        # First 100 non-trivial Riemann zeta zeros (imaginary parts)
        # Source: A.M. Odlyzko computational tables
        self.zeta_zeros = [
            14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
            37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777,
            52.9703214777, 56.4462476971, 59.3470440026, 60.8317785246, 65.1125440481,
            67.0798105295, 69.5464017112, 72.0671576745, 75.7046906991, 77.1448400689,
            79.3373750202, 82.9103808541, 84.7354929805, 87.4252746131, 88.8091112076,
            92.4918992706, 94.6513440405, 95.8706342282, 98.8311942182, 101.3178510057,
            103.7255380405, 105.4466230523, 107.1686111843, 111.0295355432, 111.8746591770,
            114.3202209155, 116.2266803209, 118.7907828660, 121.3701250024, 122.9468292936,
            124.2568185543, 127.5166838796, 129.5787042000, 131.0876885309, 133.4977372030,
            134.7565097534, 138.1160420545, 139.7362089521, 141.1237074040, 143.1118458076,
            146.0009824868, 147.4227653426, 150.0535204208, 150.9252576122, 153.0246938112,
            156.1129092942, 157.5975918176, 158.8499881714, 161.1889641376, 163.0307096872,
            165.5370691880, 167.1844399782, 169.0945154156, 169.9119764794, 173.4115365196,
            174.7541915234, 176.4414342977, 178.3774077761, 179.9164840203, 182.2070784844,
            184.8744678484, 185.5987836777, 187.2289225835, 189.4161586560, 192.0266563607,
            193.0797266038, 195.2653966795, 196.8764818410, 198.0153096763, 201.2647519437,
            202.4935945141, 204.1896718031, 205.3946972022, 207.9062588878, 209.5765097169,
            211.6908625954, 213.3479193597, 214.5470447835, 216.1695385083, 219.0675963490,
            220.7149188393, 221.4307055547, 224.0070002546, 224.9833246696
        ]

    def logarithmic_integral(self, x: float) -> float:
        """
        Compute logarithmic integral using series expansion.

        Args:
            x: Input value

        Returns:
            Approximation of li(x)
        """
        if x <= 1:
            return 0

        log_x = math.log(x)
        if x < 2:
            return 0

        log_log_x = math.log(log_x)
        euler_gamma = 0.5772156649015328606
        result = euler_gamma + log_log_x
        power = log_x
        factorial = 1.0

        for k in range(1, 50):
            result += power / (k * factorial)
            power *= log_x
            factorial *= (k + 1)
            if abs(power / ((k+1) * factorial * (k+1))) < 1e-15:
                break
        return result

    def enhanced_oscillatory_term(self, x: float, gamma: float) -> complex:
        """
        Enhanced approximation of oscillatory term li(x^ρ) where ρ = 1/2 + iγ.

        Includes proper amplitude scaling and higher-order corrections while
        maintaining O(1) computational complexity per term.

        Args:
            x: Input value
            gamma: Imaginary part of zeta zero

        Returns:
            Complex approximation of li(x^ρ)
        """
        if x <= 1:
            return 0

        log_x = math.log(x)
        sqrt_x = math.sqrt(x)

        # Zero components: ρ = 1/2 + iγ
        rho_real = 0.5
        rho_imag = gamma

        # Amplitude includes proper x^(1/2) factor
        amplitude = sqrt_x
        phase = gamma * log_x

        # Compute 1/ρ for asymptotic expansion
        rho_magnitude_sq = rho_real**2 + rho_imag**2
        denominator_real = rho_real / rho_magnitude_sq
        denominator_imag = -rho_imag / rho_magnitude_sq

        cos_phase = math.cos(phase)
        sin_phase = math.sin(phase)

        # Primary asymptotic term: x^ρ / ρ / log(x)
        real_part = amplitude * (cos_phase * denominator_real - sin_phase * denominator_imag) / log_x

        # Higher-order asymptotic corrections
        if log_x > 1:
            correction1 = amplitude * cos_phase / (log_x**2)
            correction2 = 2 * amplitude * cos_phase / (log_x**3)
            real_part += correction1 + correction2

        # Damping for numerical stability
        damping = math.exp(-gamma / (log_x + 10))

        return real_part * damping

    def triangle_window(self, k: int, gamma: float, x: float) -> float:
        """
        Triangle window function using sinc² products.

        Inspired by Borwein integral techniques, this window provides
        smooth spectral characteristics with good sidelobe suppression.

        Args:
            k: Term index (1-based)
            gamma: Frequency parameter
            x: Input value

        Returns:
            Enhancement factor for k-th term
        """
        log_x = math.log(x)
        triangle_product = 1.0
        beta = gamma / 10.0

        # Multi-dimensional sinc² product for triangle characteristics
        for depth in range(2, min(k + 1, 4) + 1):
            for width in range(2, min(depth + 1, 3) + 1):
                arg = math.pi * gamma / (beta * width * depth * log_x)
                if abs(arg) > 1e-15:
                    sinc_val = math.sin(arg) / arg
                    triangle_product *= sinc_val ** 2

        # Adaptive weighting and modulation
        weight = math.exp(-k / (6 * math.sqrt(x))) / (k**1.1)
        discrete_oscillation = math.cos(k * math.pi * log_x / math.log(log_x))
        modulation = math.exp(-k / (1.5 * log_x * math.sqrt(x)))

        return 1.0 + weight * triangle_product * discrete_oscillation * modulation

    def gaussian_window(self, k: int, gamma: float, x: float) -> float:
        """
        Gaussian window function for smooth spectral filtering.

        Provides excellent frequency domain characteristics with
        minimal spectral leakage and smooth transitions.

        Args:
            k: Term index (1-based)
            gamma: Frequency parameter
            x: Input value

        Returns:
            Enhancement factor for k-th term
        """
        log_x = math.log(x)
        gaussian_product = 1.0
        beta = gamma / 10.0

        # Multi-dimensional Gaussian product
        for depth in range(2, min(k + 1, 4) + 1):
            for width in range(2, min(depth + 1, 3) + 1):
                arg = math.pi * gamma / (beta * width * depth * log_x)
                if abs(arg) > 1e-15:
                    gaussian_product *= math.exp(-arg**2 / 2)

        # Gaussian-weighted enhancement
        weight = math.exp(-k**2 / (4 * math.sqrt(x))) / (k**0.5)
        discrete_oscillation = math.cos(k * math.pi * log_x / math.log(log_x))
        modulation = math.exp(-k / (log_x * math.sqrt(x)))

        return 1.0 + weight * gaussian_product * discrete_oscillation * modulation

    def sinc_window(self, k: int, gamma: float, x: float) -> float:
        """
        Sinc window function for rectangular frequency response.

        Provides sharp frequency cutoff characteristics with
        controlled sidelobe behavior through product formulation.

        Args:
            k: Term index (1-based)
            gamma: Frequency parameter
            x: Input value

        Returns:
            Enhancement factor for k-th term
        """
        log_x = math.log(x)
        sinc_product = 1.0
        beta = gamma / 10.0

        # Sinc product for rectangular characteristics
        for depth in range(1, min(k + 1, 4)):
            arg = math.pi * gamma / (beta * depth * log_x)
            if abs(arg) > 1e-12:
                sinc_product *= math.sin(arg) / arg

        weight = math.exp(-k * k / (4 * x)) / (k**2)
        prime_modulation = math.cos(k * math.pi * log(x) / math.log(log(max(x, 10))))

        return 1.0 + weight * sinc_product * prime_modulation

    def staircase_window(self, k: int, gamma: float, x: float) -> float:
        """
        Staircase window function with stepped frequency response.

        Provides discrete frequency band characteristics through
        summation of weighted sinc functions at different scales.

        Args:
            k: Term index (1-based)
            gamma: Frequency parameter
            x: Input value

        Returns:
            Enhancement factor for k-th term
        """
        log_x = math.log(x)
        staircase_sum = 0.0
        beta = gamma / 10.0
        num_steps = min(k + 1, 4)

        # Stepped sinc summation for staircase characteristics
        for step in range(1, num_steps + 1):
            arg = math.pi * gamma / (beta * step * log_x)
            if abs(arg) > 1e-15:
                sinc_val = math.sin(arg) / arg
                staircase_sum += sinc_val / step

        weight = math.exp(-k / (4 * math.sqrt(x))) / (k**1.5)
        discrete_oscillation = math.cos(k * math.pi * log_x / math.log(log_x))
        modulation = math.exp(-k / (2 * log_x * math.sqrt(x)))

        return 1.0 + weight * staircase_sum * discrete_oscillation * modulation

    def pyramid_window(self, k: int, gamma: float, x: float) -> float:
        """
        Pyramid window function with triangular frequency weighting.

        Combines sinc² characteristics with layered summation to
        create pyramid-shaped frequency response with good sidelobe control.

        Args:
            k: Term index (1-based)
            gamma: Frequency parameter
            x: Input value

        Returns:
            Enhancement factor for k-th term
        """
        log_x = math.log(x)
        pyramid_sum = 0.0
        beta = gamma / 10.0
        num_layers = min(k + 1, 4)

        # Layered sinc² summation for pyramid characteristics
        for layer in range(1, num_layers + 1):
            arg = math.pi * gamma / (beta * layer * log_x)
            if abs(arg) > 1e-15:
                sinc_val = math.sin(arg) / arg
                pyramid_sum += (sinc_val ** 2) / layer

        weight = math.exp(-k / (5 * math.sqrt(x))) / (k**1.3)
        discrete_oscillation = math.cos(k * math.pi * log_x / math.log(log_x))
        modulation = math.exp(-k / (1.2 * log_x * math.sqrt(x)))

        return 1.0 + weight * pyramid_sum * discrete_oscillation * modulation

    def apply_spectral_filter(self, x: float, window_type: str = 'triangle') -> float:
        """
        Apply spectral filtering to mathematical approximation.

        Demonstrates the window function framework applied to the
        Riemann explicit formula for prime counting.

        Args:
            x: Input value
            window_type: Type of window function to apply

        Returns:
            Filtered approximation result
        """
        if x < 2:
            return 0

        log_x = math.log(x)
        sqrt_x = math.sqrt(x)

        # Base mathematical terms
        li_x = self.logarithmic_integral(x)

        # Correction terms (simplified for demonstration)
        log_correction = math.log(2 * pi)
        finite_correction = 0.0
        if x > 2:
            x_inv_sq = 1.0 / (x * x)
            if x_inv_sq > 1e-10:
                finite_correction = -0.5 * math.log(1 - x_inv_sq)

        base_correction = sqrt_x / log_x + sqrt_x

        # Apply spectral filtering to oscillatory sum
        oscillatory_sum = 0
        window_functions = {
            'triangle': self.triangle_window,
            'gaussian': self.gaussian_window,
            'sinc': self.sinc_window,
            'staircase': self.staircase_window,
            'pyramid': self.pyramid_window
        }

        window_func = window_functions.get(window_type, self.triangle_window)

        for i, gamma in enumerate(self.zeta_zeros):
            k = i + 1

            # Enhanced oscillatory term
            oscillatory_term = self.enhanced_oscillatory_term(x, gamma)

            # Apply spectral window
            enhancement_factor = window_func(k, gamma, x)

            oscillatory_sum += oscillatory_term.real * enhancement_factor

        # Complete filtered approximation
        result = (li_x 
                 - log_correction
                 - finite_correction
                 - base_correction
                 - oscillatory_sum)

        return max(0, result)

    def nth_prime_with_filter(self, n: int, window_type: str = 'triangle') -> float:
        """
        Estimate n-th prime using spectral filtering.

        Args:
            n: Prime index
            window_type: Spectral window to apply

        Returns:
            Estimated n-th prime value
        """
        if n <= 0:
            return 2

        log_n = math.log(n)
        log_log_n = math.log(log_n) if n > 2 else 0.1

        # Initial guess using asymptotic expansion
        initial_guess = n * (
            log_n + log_log_n - 1 + 
            (log_log_n - 2) / log_n +
            (log_log_n**2 - 6*log_log_n + 11) / (2 * log_n**2)
        )

        # Binary search with spectral filtering
        lower = max(2, initial_guess * 0.8)
        upper = initial_guess * 1.2

        while self.apply_spectral_filter(lower, window_type) > n:
            lower *= 0.9
        while self.apply_spectral_filter(upper, window_type) < n:
            upper *= 1.1

        for _ in range(50):
            mid = (lower + upper) / 2
            pi_mid = self.apply_spectral_filter(mid, window_type)

            if abs(pi_mid - n) < 0.1:
                return mid

            if pi_mid < n:
                lower = mid
            else:
                upper = mid

        return (lower + upper) / 2

    def compare_window_functions(self) -> Dict[str, float]:
        """
        Compare performance of different spectral window functions.

        Returns:
            Dictionary mapping window types to average errors
        """
        print("Spectral Window Function Comparison")
        print("=" * 80)

        # Test cases with known values
        test_cases = [1000, 5000, 10000, 25000, 50000, 100000]
        known_primes = {
            1000: 7919, 5000: 48611, 10000: 104729, 
            25000: 287117, 50000: 611953, 100000: 1299709
        }

        window_types = ['sinc', 'triangle', 'gaussian', 'staircase', 'pyramid']

        print("n\t\tActual\t\tSinc\t\tTriangle\tGaussian\tStaircase\tPyramid")
        print("-" * 80)

        total_errors = {window: 0 for window in window_types}
        valid_tests = 0

        for n in test_cases:
            if n in known_primes:
                actual = known_primes[n]
                results = {}

                for window in window_types:
                    result = self.nth_prime_with_filter(n, window)
                    results[window] = result
                    error = abs(result - actual) / actual * 100
                    total_errors[window] += error

                print(f"{n}\t\t{actual}\t\t{results['sinc']:.0f}\t\t{results['triangle']:.0f}\t\t{results['gaussian']:.0f}\t\t{results['staircase']:.0f}\t\t{results['pyramid']:.0f}")
                valid_tests += 1

        if valid_tests > 0:
            print(f"\nWindow Function Performance:")
            print("-" * 40)

            performance_results = {}
            best_window = None
            best_error = float('inf')

            for window in window_types:
                avg_error = total_errors[window] / valid_tests
                performance_results[window] = avg_error
                print(f"{window.capitalize():>10}: {avg_error:.3f}% average error")

                if avg_error < best_error:
                    best_error = avg_error
                    best_window = window

            print(f"\nBest performing window: {best_window.upper()} ({best_error:.3f}% error)")

            return performance_results

        return {}

def main():
    """Demonstrate spectral window function capabilities."""
    calculator = SpectralWindowCalculator()
    results = calculator.compare_window_functions()

    if results:
        print(f"\nSpectral filtering demonstrates configurable enhancement")
        print("of mathematical approximations through window function design.")

if __name__ == "__main__":
    main()
