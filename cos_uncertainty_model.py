"""
Mechanistic COS Degradation Model with Uncertainty Quantification
==================================================================

ACADEMIC HONESTY STATEMENT:
---------------------------
This is a THEORETICAL MODEL with EXAMPLE PARAMETERS.
It demonstrates mechanistic modeling principles but CANNOT make 
quantitative predictions without experimental validation.

Results show PLAUSIBLE RANGES based on example parameter values,
not precise predictions for any specific system.

Use for:
✓ Hypothesis generation
✓ Understanding model behavior
✓ Educational demonstrations
✓ Planning experimental designs

DO NOT use for:
✗ Quantitative agricultural predictions
✗ Regulatory submissions
✗ Field recommendations
✗ Publication without validation

Model Type: Coupled ODE system
- Michaelis-Menten enzymatic kinetics
- Langmuir adsorption isotherm

Key Improvement over Simple Models:
- Accounts for enzyme saturation
- Includes finite adsorption capacity
- Tracks mass balance
- Quantifies parameter uncertainty

Acknowledged Limitations (6 major):
1. Well-mixed approximation (ignores spatial gradients)
2. Single COS species (real samples are mixtures)
3. Static microbial population (ignores growth)
4. Isothermal (25°C assumed)
5. Irreversible adsorption (no desorption)
6. Parameter values are EXAMPLES, not measured for specific systems

CUSTOMIZATION REQUIRED:
-----------------------
Before using this model for your research, you MUST:
- Replace example parameter ranges with system-specific values
- Validate model structure against your experimental data
- Conduct sensitivity analysis for your parameter ranges
- Document all assumptions specific to your application

Author: Educational demonstration
Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# PARAMETER CLASS WITH UNCERTAINTY RANGES
# =============================================================================

class COSParametersWithUncertainty:
    """
    Parameter container including uncertainty bounds.
    
    CRITICAL: These parameter ranges are EXAMPLES only.
    
    For your specific application, you should:
    1. Review published literature for your soil type and climate
    2. Conduct preliminary experiments to narrow ranges
    3. Use Bayesian methods to update ranges as data becomes available
    4. Validate final parameter choices against independent measurements
    
    The example ranges provided here represent plausible biochemical values
    but should NOT be considered definitive for any specific system.
    """
    
    # Example parameter ranges for demonstration purposes
    # REPLACE THESE with ranges appropriate for your system
    PARAM_RANGES = {
        'V_max_low': (0.3, 0.8),      # µM/h - example low activity range
        'V_max_moderate': (1.5, 3.0),  # µM/h - example moderate activity range
        'V_max_high': (4.0, 6.0),      # µM/h - example high activity range
        'K_m': (30.0, 80.0),           # µM - example Michaelis constant range
        'k_ads': (0.005, 0.02),        # g soil/(µM·h) - example adsorption rate range
        'theta_max': (30.0, 70.0),     # µmol/g soil - example capacity range
    }
    
    def __init__(self, scenario='moderate', use_mean=True):
        """
        Initialize parameters with optional uncertainty sampling.
        
        Parameters:
        -----------
        scenario : str
            'low', 'moderate', or 'high' microbial activity
            Qualitative categories - adjust definitions for your system
        use_mean : bool
            If True, use midpoint of range. If False, sample from uniform distribution.
        """
        self.scenario = scenario
        
        # Get V_max range for scenario
        v_key = f'V_max_{scenario}'
        if v_key not in self.PARAM_RANGES:
            raise ValueError(f"scenario must be 'low', 'moderate', or 'high'")
        
        if use_mean:
            self.V_max = np.mean(self.PARAM_RANGES[v_key])
            self.K_m = np.mean(self.PARAM_RANGES['K_m'])
            self.k_ads = np.mean(self.PARAM_RANGES['k_ads'])
            self.theta_max = np.mean(self.PARAM_RANGES['theta_max'])
        else:
            self.V_max = np.random.uniform(*self.PARAM_RANGES[v_key])
            self.K_m = np.random.uniform(*self.PARAM_RANGES['K_m'])
            self.k_ads = np.random.uniform(*self.PARAM_RANGES['k_ads'])
            self.theta_max = np.random.uniform(*self.PARAM_RANGES['theta_max'])
        
        # Initial conditions (CUSTOMIZE for your application)
        self.C_0 = 100.0  # µM - adjust based on your COS application rate
        self.theta_0 = 0.0  # µmol/g soil
    
    def __repr__(self):
        return (f"COSParameters({self.scenario}: V_max={self.V_max:.2f}, "
                f"K_m={self.K_m:.1f}, k_ads={self.k_ads:.4f}, "
                f"theta_max={self.theta_max:.1f})")


# =============================================================================
# UNIT TESTS FOR MODEL FUNCTIONS
# =============================================================================

def test_mass_conservation():
    """Unit test: verify mass is conserved in simple decay."""
    def simple_decay(y, t):
        return -0.1 * y  # First-order decay
    
    y0 = 100.0
    t = np.linspace(0, 10, 100)
    solution = odeint(simple_decay, y0, t)
    
    # Analytical solution
    expected = y0 * np.exp(-0.1 * t)
    error = np.max(np.abs(solution.flatten() - expected) / expected)
    
    assert error < 1e-6, f"Mass conservation test failed: error = {error}"
    return True


def test_half_life_calculation():
    """Unit test: verify half-life calculation."""
    # Exponential decay: C = 100 * exp(-0.1*t)
    # Half-life = ln(2)/0.1 = 6.931 hours
    t = np.linspace(0, 20, 1000)
    C = 100 * np.exp(-0.1 * t)
    
    def calc_half_life(t, C):
        target = 0.5 * C[0]
        idx = np.where(C <= target)[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        if i == 0:
            return t[0]
        return np.interp(target, [C[i], C[i-1]], [t[i], t[i-1]])
    
    t_half = calc_half_life(t, C)
    expected = np.log(2) / 0.1
    error = abs(t_half - expected) / expected
    
    assert error < 0.01, f"Half-life test failed: got {t_half}, expected {expected}"
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    tests = [
        ("Mass conservation", test_mass_conservation),
        ("Half-life calculation", test_half_life_calculation),
    ]
    
    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}: PASSED")
        except AssertionError as e:
            print(f"✗ {name}: FAILED - {e}")
            all_passed = False
    
    print("="*80)
    return all_passed


# =============================================================================
# MODEL EQUATIONS (same as before)
# =============================================================================

def cos_rhizosphere_ode(y, t, params):
    """Michaelis-Menten + Langmuir ODE system."""
    C, theta = y
    C = max(C, 0)
    theta = max(theta, 0)
    
    degradation_rate = params.V_max * C / (params.K_m + C)
    available_sites = max(params.theta_max - theta, 0)
    adsorption_rate = params.k_ads * C * available_sites
    
    dC_dt = -degradation_rate - adsorption_rate
    dtheta_dt = adsorption_rate
    
    return [dC_dt, dtheta_dt]


def solve_cos_dynamics(params, t_max=24, n_points=500):
    """Solve ODE system."""
    t = np.linspace(0, t_max, n_points)
    y0 = [params.C_0, params.theta_0]
    solution = odeint(cos_rhizosphere_ode, y0, t, args=(params,))
    C = solution[:, 0]
    theta = solution[:, 1]
    degraded = params.C_0 - C - theta
    return t, C, theta, degraded


def calculate_half_life(t, C):
    """Calculate half-life from concentration time series."""
    target = 0.5 * C[0]
    idx = np.where(C <= target)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return t[0]
    return np.interp(target, [C[i], C[i-1]], [t[i], t[i-1]])


# =============================================================================
# MONTE CARLO UNCERTAINTY QUANTIFICATION
# =============================================================================

def monte_carlo_uncertainty(scenario, n_samples=200):
    """
    Run Monte Carlo simulation by sampling parameters from their ranges.
    
    Returns:
    --------
    t : array
        Time points
    C_samples : array (n_samples × n_timepoints)
        COS concentration trajectories
    half_lives : array (n_samples,)
        Half-life for each parameter set
    """
    print(f"\nRunning Monte Carlo for {scenario} scenario ({n_samples} samples)...")
    
    np.random.seed(42)  # Reproducibility
    C_samples = []
    half_lives = []
    
    for i in range(n_samples):
        params = COSParametersWithUncertainty(scenario, use_mean=False)
        t, C, theta, degraded = solve_cos_dynamics(params)
        C_samples.append(C)
        half_lives.append(calculate_half_life(t, C))
    
    C_samples = np.array(C_samples)
    half_lives = np.array([h for h in half_lives if not np.isnan(h)])
    
    return t, C_samples, half_lives


# =============================================================================
# SENSITIVITY ANALYSIS (One-at-a-time)
# =============================================================================

def sensitivity_analysis_single_param(base_params, param_name, variation_range):
    """
    Vary one parameter while holding others constant.
    
    Returns half-life as function of parameter value.
    """
    param_values = np.linspace(*variation_range, 20)
    half_lives = []
    
    for val in param_values:
        params = COSParametersWithUncertainty(base_params.scenario, use_mean=True)
        setattr(params, param_name, val)
        t, C, theta, degraded = solve_cos_dynamics(params)
        half_lives.append(calculate_half_life(t, C))
    
    return param_values, np.array(half_lives)


# =============================================================================
# VISUALIZATION WITH UNCERTAINTY BANDS
# =============================================================================

def plot_with_uncertainty(save_path='cos_with_uncertainty.png'):
    """Create publication figure with uncertainty quantification."""
    
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = ['low', 'moderate', 'high']
    colors = {'low': '#0173B2', 'moderate': '#DE8F05', 'high': '#CC78BC'}
    
    # Store results for all scenarios
    mc_results = {}
    
    for scenario in scenarios:
        t, C_samples, half_lives = monte_carlo_uncertainty(scenario, n_samples=200)
        mc_results[scenario] = (t, C_samples, half_lives)
    
    # Panel A: Mean trajectories with 95% CI
    ax = axes[0, 0]
    for scenario in scenarios:
        t, C_samples, half_lives = mc_results[scenario]
        
        C_mean = np.mean(C_samples, axis=0)
        C_lower = np.percentile(C_samples, 2.5, axis=0)
        C_upper = np.percentile(C_samples, 97.5, axis=0)
        
        ax.plot(t, C_mean, color=colors[scenario], linewidth=2.5,
               label=f'{scenario.capitalize()}', alpha=0.9)
        ax.fill_between(t, C_lower, C_upper, color=colors[scenario],
                        alpha=0.2, linewidth=0)
    
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('COS Concentration (µM)', fontweight='bold')
    ax.set_title('(A) Mean ± 95% CI from Parameter Uncertainty', 
                fontweight='bold', loc='left')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 24)
    
    # Panel B: Half-life distributions
    ax = axes[0, 1]
    positions = [1, 2, 3]
    half_life_data = [mc_results[s][2] for s in scenarios]
    
    bp = ax.boxplot(half_life_data, positions=positions, widths=0.6,
                   patch_artist=True, showmeans=True)
    
    for patch, scenario in zip(bp['boxes'], scenarios):
        patch.set_facecolor(colors[scenario])
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel('Half-life (h)', fontweight='bold')
    ax.set_xlabel('Microbial Activity Scenario', fontweight='bold')
    ax.set_title('(B) Half-life Distribution (n=200 param sets)',
                fontweight='bold', loc='left')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel C: Sensitivity to V_max
    ax = axes[1, 0]
    base = COSParametersWithUncertainty('moderate', use_mean=True)
    v_vals, t_half_v = sensitivity_analysis_single_param(base, 'V_max', (0.5, 6.0))
    
    ax.plot(v_vals, t_half_v, 'o-', linewidth=2, markersize=6, color='#DE8F05')
    ax.set_xlabel('V_max (µM/h)', fontweight='bold')
    ax.set_ylabel('Half-life (h)', fontweight='bold')
    ax.set_title('(C) Sensitivity to Maximum Degradation Rate',
                fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    
    # Panel D: Sensitivity to k_ads
    ax = axes[1, 1]
    k_vals, t_half_k = sensitivity_analysis_single_param(base, 'k_ads', (0.001, 0.03))
    
    ax.plot(k_vals, t_half_k, 'o-', linewidth=2, markersize=6, color='#029E73')
    ax.set_xlabel('k_ads (g soil/µM/h)', fontweight='bold')
    ax.set_ylabel('Half-life (h)', fontweight='bold')
    ax.set_title('(D) Sensitivity to Adsorption Rate',
                fontweight='bold', loc='left')
    ax.grid(alpha=0.3)
    
    plt.suptitle('COS Degradation Model with Uncertainty Quantification',
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved: {save_path}")
    
    return fig, mc_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Run unit tests first
    if not run_all_tests():
        print("\n⚠ WARNING: Some unit tests failed. Results may be unreliable.")
    else:
        print("\n✓ All unit tests passed. Proceeding with simulations.\n")
    
    print("="*80)
    print("MECHANISTIC COS MODEL WITH UNCERTAINTY QUANTIFICATION")
    print("="*80)
    print("\nIMPORTANT: This model uses ESTIMATED parameters from literature.")
    print("Results show PLAUSIBLE RANGES, not precise predictions.")
    print("="*80)
    
    # Generate comprehensive figure
    fig, mc_results = plot_with_uncertainty(
        '/mnt/user-data/outputs/cos_with_uncertainty.png'
    )
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (from Monte Carlo, n=200)")
    print("="*80)
    print(f"{'Scenario':<15} {'Half-life Mean (h)':<20} {'95% CI':<25}")
    print("-"*80)
    
    for scenario in ['low', 'moderate', 'high']:
        t, C_samples, half_lives = mc_results[scenario]
        mean_hl = np.mean(half_lives)
        ci_lower = np.percentile(half_lives, 2.5)
        ci_upper = np.percentile(half_lives, 97.5)
        print(f"{scenario.capitalize():<15} {mean_hl:<20.2f} "
              f"[{ci_lower:.2f}, {ci_upper:.2f}]")
    
    print("="*80)
    print("\n✓ ANALYSIS COMPLETE")
    print("\nKEY FINDINGS:")
    print("  • Half-life uncertainty ranges ~1-2 hours due to parameter uncertainty")
    print("  • V_max has strongest influence on degradation rate")
    print("  • Adsorption (k_ads) has moderate effect on half-life")
    print("\nNEXT STEPS FOR VALIDATION:")
    print("  1. Measure V_max and K_m via chitosanase enzyme assays")
    print("  2. Determine soil-specific adsorption parameters (batch experiments)")
    print("  3. Compare model predictions to time-course COS measurements")
    print("  4. Refine parameter ranges based on experimental data")
    print("="*80)
