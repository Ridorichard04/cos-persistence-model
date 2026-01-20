# COS Degradation Simulation - User Guide

## Overview

This package contains three Python simulations for modeling chitosan oligosaccharide (COS) degradation in rhizosphere soil:

1. **cos_degradation_simulation.py** - Simple first-order kinetic model
2. **cos_rigorous_simulation.py** - Mechanistic model with Michaelis-Menten kinetics
3. **cos_uncertainty_model.py** - Full uncertainty quantification with Monte Carlo

---

## Quick Start

### Prerequisites

**Required Software:**
- Python 3.8 or higher
- pip (Python package installer)

**Required Python Packages:**
```bash
pip install numpy matplotlib scipy seaborn
```

Or install all at once:
```bash
pip install numpy>=1.24 matplotlib>=3.7 scipy>=1.10 seaborn>=0.12
```

### Running the Simulations

**Option 1: Command Line**
```bash
python cos_degradation_simulation.py
python cos_rigorous_simulation.py
python cos_uncertainty_model.py
```

**Option 2: Python IDE**
- Open the .py file in your IDE (Spyder, PyCharm, VS Code, etc.)
- Click "Run" or press F5

**Option 3: Jupyter Notebook**
```python
%run cos_degradation_simulation.py
```

### Expected Outputs

Each script will:
1. Print progress messages to the console
2. Generate publication-quality figures (.png files)
3. Save figures to the current directory or `/mnt/user-data/outputs/`

---

## Choosing the Right Model

### Use `cos_degradation_simulation.py` if:
- ✓ You need a quick demonstration for teaching
- ✓ You're explaining basic degradation concepts
- ✓ You want simple, easy-to-understand code
- ✓ You're doing back-of-envelope calculations
- ✓ Your audience is undergraduates or general public

**Outputs:**
- Degradation curves for 4 microbial activity levels
- Sensitivity heatmap of half-life vs. parameters
- Combined analysis figure
- Half-life calculations table

### Use `cos_rigorous_simulation.py` if:
- ✓ You need biochemically accurate kinetics
- ✓ You're writing a graduate-level thesis
- ✓ You want to account for enzyme saturation
- ✓ You need to model finite adsorption capacity
- ✓ You want mass balance verification

**Outputs:**
- 6-panel comprehensive figure showing:
  - Dissolved COS concentration
  - Adsorbed COS on soil
  - Mass balance verification
  - Cumulative enzymatic degradation
  - Degradation rate over time
  - Half-life comparison bar chart

### Use `cos_uncertainty_model.py` if:
- ✓ You need to quantify parameter uncertainty
- ✓ You're preparing work for publication
- ✓ You want Monte Carlo simulations
- ✓ You need sensitivity analysis
- ✓ You want to show confidence intervals
- ✓ You need academic distinction-level quality

**Outputs:**
- 4-panel figure with uncertainty bands:
  - Mean trajectories with 95% confidence intervals
  - Half-life distribution boxplots
  - Sensitivity to V_max parameter
  - Sensitivity to k_ads parameter
- Statistical summary table
- Unit test results

---

## Customizing Parameters

### ⚠️ CRITICAL: Default Parameters Are EXAMPLES Only

**DO NOT use the default parameter values for:**
- Quantitative field predictions
- Agricultural recommendations
- Regulatory submissions
- Publication without validation

**You MUST customize parameters based on:**
- Your specific soil type
- Your microbial community
- Your COS characteristics
- Your experimental measurements

### How to Customize Parameters

#### Step 1: Locate the Parameter Section

All three scripts have a clearly marked section near the top:

**In `cos_degradation_simulation.py`:**
```python
# ============================================================================
# SIMULATION PARAMETERS (User-Adjustable)
# ============================================================================

C_0 = 100.0  # Initial COS concentration (µM)
k_a_fixed = 0.05  # Adsorption rate (h⁻¹)
k_d_values = [0.05, 0.1, 0.2, 0.4]  # Degradation rates (h⁻¹)
```

**In `cos_rigorous_simulation.py`:**
```python
class COSParameters:
    def __init__(self, scenario='moderate'):
        self.C_0 = 100.0  # Initial concentration
        self.V_max = 2.0  # Maximum degradation rate
        self.K_m = 50.0   # Michaelis constant
        # ... etc
```

**In `cos_uncertainty_model.py`:**
```python
PARAM_RANGES = {
    'V_max_low': (0.3, 0.8),      # Range for low activity
    'V_max_moderate': (1.5, 3.0),  # Range for moderate activity
    'K_m': (30.0, 80.0),           # Range for Michaelis constant
    # ... etc
}
```

#### Step 2: Replace with Your Values

**Example: You measured V_max = 3.5 µM/h in your soil**

Change this line:
```python
self.V_max = 2.0  # OLD - example value
```

To this:
```python
self.V_max = 3.5  # Measured via chitosanase assay, Smith et al. 2020 method
```

#### Step 3: Document Your Changes

Add a comment explaining where the value came from:
```python
# Measured experimentally, Lab Notebook p.47, 2024-01-15
# Method: Chitosanase enzyme assay, n=5 replicates
# Mean ± SD: 3.5 ± 0.4 µM/h
self.V_max = 3.5
```

---

## Parameter Guide

### What Each Parameter Means

| Parameter | Units | Meaning | How to Measure |
|-----------|-------|---------|----------------|
| **C_0** | µM | Initial COS concentration | Calculate from application rate, or measure via HPLC |
| **k_d** | h⁻¹ | Degradation rate constant (first-order) | Fit to COS disappearance data |
| **k_a** | h⁻¹ | Adsorption rate constant (first-order) | Batch adsorption kinetics |
| **V_max** | µM/h | Maximum enzymatic rate | Chitosanase enzyme assay |
| **K_m** | µM | Michaelis constant | Enzyme kinetics with varying [COS] |
| **k_ads** | g soil/µM/h | Adsorption rate (Langmuir) | Batch adsorption experiments |
| **θ_max** | µmol/g | Maximum adsorption capacity | Adsorption isotherm to saturation |

### Typical Value Ranges (Literature Examples)

These are **general guidance only** - your system may differ:

- **C_0**: 10-500 µM (depending on application rate)
- **k_d**: 0.01-0.5 h⁻¹ (low to high microbial activity)
- **V_max**: 0.5-10 µM/h (depends on enzyme concentration)
- **K_m**: 10-100 µM (typical for oligosaccharide substrates)
- **k_ads**: 0.001-0.05 g soil/µM/h (depends on clay content)
- **θ_max**: 10-100 µmol/g (depends on soil CEC)

---

## Modifying Output Settings

### Change Simulation Time

```python
# Default: 0 to 24 hours
t = np.linspace(0, 24, 241)

# Extended time: 0 to 48 hours
t = np.linspace(0, 48, 481)

# Shorter time: 0 to 12 hours
t = np.linspace(0, 12, 121)
```

### Change Figure Resolution

```python
# Default: 300 DPI (publication quality)
plt.rcParams['figure.dpi'] = 300

# Higher resolution: 600 DPI
plt.rcParams['figure.dpi'] = 600

# Screen display: 150 DPI
plt.rcParams['figure.dpi'] = 150
```

### Change Output Filename

Find the line:
```python
plt.savefig('/mnt/user-data/outputs/cos_degradation_curves.png', ...)
```

Change to:
```python
plt.savefig('my_custom_filename.png', ...)
```

### Add More Scenarios

In `cos_degradation_simulation.py`:
```python
# Default: 4 scenarios
k_d_values = [0.05, 0.1, 0.2, 0.4]

# Add more scenarios
k_d_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'numpy'"

**Solution:** Install required packages
```bash
pip install numpy matplotlib scipy seaborn
```

### Error: "Permission denied" when saving files

**Solution:** Change output directory to current folder
```python
# Change from:
plt.savefig('/mnt/user-data/outputs/figure.png', ...)

# To:
plt.savefig('figure.png', ...)
```

### Error: "ValueError: operands could not be broadcast together"

**Solution:** Check that parameter arrays have compatible dimensions
- Ensure `k_d_values` is a list or array
- Verify time array `t` is properly defined

### Figures not displaying

**Solution:** Add this at the end of the script:
```python
plt.show()
```

### Code runs but no output

**Solution:** Check your working directory
```python
import os
print(os.getcwd())  # Shows where files are being saved
```

### Monte Carlo simulation is slow

**Solution:** Reduce number of samples in `cos_uncertainty_model.py`
```python
# Default: 200 samples
monte_carlo_uncertainty(scenario, n_samples=200)

# Faster: 50 samples
monte_carlo_uncertainty(scenario, n_samples=50)
```

---

## Understanding the Output

### Console Output

**Simple Model (`cos_degradation_simulation.py`):**
```
✓ Figure 1 saved: cos_degradation_curves.png
======================================================================
COS HALF-LIFE CALCULATIONS
======================================================================
Microbial Activity        k_d (h⁻¹)    k_a (h⁻¹)    Half-life (h)  
----------------------------------------------------------------------
Low                       0.05         0.05         6.93           
```

**Interpretation:**
- Half-life = time for COS to reach 50% of initial concentration
- Shorter half-life = faster degradation
- Each row shows a different microbial activity scenario

### Figure Interpretation

**Panel A: Degradation Curves**
- X-axis: Time in hours
- Y-axis: COS concentration in µM
- Each line = different microbial activity level
- Steeper decline = faster degradation

**Panel B: Half-life Distribution (uncertainty model)**
- Box = interquartile range (25th-75th percentile)
- Line in box = median
- Whiskers = 95% range
- Wide boxes = high uncertainty

**Panel C & D: Sensitivity Analysis**
- Shows how changing one parameter affects half-life
- Steep slope = parameter has strong influence
- Flat line = parameter has weak influence

---

## Best Practices

### For Academic Work

1. **Always customize parameters** for your system
2. **Document all parameter sources** with references
3. **Include uncertainty** where known
4. **Validate predictions** against experimental data
5. **State all limitations** in your writeup
6. **Save your modified code** with your thesis/paper

### For Teaching/Demonstrations

1. **Acknowledge example values** are not system-specific
2. **Focus on model behavior** rather than exact numbers
3. **Use for hypothesis generation**, not quantitative claims
4. **Demonstrate sensitivity** to parameter choices
5. **Encourage critical thinking** about assumptions

### For Publication

1. **Measure parameters experimentally** whenever possible
2. **Perform formal uncertainty quantification** (use uncertainty model)
3. **Validate against independent data**
4. **Compare to alternative model structures**
5. **Submit code as supplementary material**
6. **Include all parameters** in methods section

---

## Advanced Usage

### Running Batch Simulations

Create a script to run multiple parameter sets:

```python
import numpy as np
from cos_rigorous_simulation import COSParameters, solve_cos_dynamics

# Define parameter sets to test
v_max_values = [1.0, 2.0, 3.0, 4.0, 5.0]
results = []

for v_max in v_max_values:
    params = COSParameters('moderate')
    params.V_max = v_max
    t, C, theta, degraded = solve_cos_dynamics(params)
    results.append((v_max, t, C))
    print(f"Completed V_max = {v_max}")

# Now analyze results...
```

### Exporting Data to CSV

```python
import pandas as pd

# After running simulation
df = pd.DataFrame({
    'Time_hours': t,
    'COS_concentration_uM': C,
    'Adsorbed_COS': theta,
    'Degraded_COS': degraded
})

df.to_csv('cos_simulation_results.csv', index=False)
print("Data exported to cos_simulation_results.csv")
```

### Creating Custom Figures

```python
import matplotlib.pyplot as plt

# After running simulation
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t, C, linewidth=2, color='darkblue', label='Dissolved COS')
ax.plot(t, theta, linewidth=2, color='brown', label='Adsorbed COS')

ax.set_xlabel('Time (h)', fontsize=14)
ax.set_ylabel('COS (µM or µmol/g)', fontsize=14)
ax.set_title('My Custom COS Dynamics', fontsize=16)
ax.legend()
ax.grid(alpha=0.3)

plt.savefig('my_custom_figure.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Integration with Your Workflow

### For Thesis/Dissertation

1. Run simulation with **your measured parameters**
2. Save figures in your thesis figures folder
3. Copy console output into your results section
4. Include modified .py file as appendix
5. Reference parameter sources in methods

### For Journal Article

1. Use **uncertainty model** for confidence intervals
2. Include sensitivity analysis figure
3. Compare predictions to experimental data
4. Provide code as **supplementary material**
5. State all limitations in discussion

### For Grant Proposal

1. Use to demonstrate **feasibility** of approach
2. Show predicted timescales justify sampling frequency
3. Identify which parameters need measurement
4. Use to justify experimental design
5. Show preliminary data support model assumptions

---

## File Structure

```
cos_simulations/
│
├── cos_degradation_simulation.py      # Simple first-order model
├── cos_rigorous_simulation.py          # Michaelis-Menten model
├── cos_uncertainty_model.py            # Uncertainty quantification
├── README.md                           # This file
│
├── outputs/                            # Generated figures (created automatically)
│   ├── cos_degradation_curves.png
│   ├── cos_halflife_heatmap.png
│   ├── cos_combined_analysis.png
│   ├── cos_rigorous_simulation.png
│   └── cos_with_uncertainty.png
│
└── documentation/                      # Supporting documents
    ├── parameter_ambiguity_summary.txt
    ├── rigorous_documentation.txt
    ├── final_academic_documentation.txt
    └── model_comparison.txt
```

---

## Getting Help

### Common Questions

**Q: Can I use these simulations for my agricultural field trial?**
A: No, not without experimental validation. Parameters must be measured for your specific soil and conditions.

**Q: Which model should I use for my undergraduate project?**
A: Start with `cos_degradation_simulation.py` for simplicity. Move to rigorous model if your supervisor requires it.

**Q: How do I cite these simulations?**
A: Cite as custom code with appropriate acknowledgment of limitations. Include the .py file as supplementary material.

**Q: The half-lives seem too short/long. Is something wrong?**
A: Probably not. Check that your parameters match your system. Default values are examples only.

**Q: Can I modify the code for other oligosaccharides?**
A: Yes! The model structure works for any degradable oligosaccharide. Just change parameters and labels.

### Need More Help?

1. Read the extensive comments in the code
2. Check the documentation files provided
3. Review the parameter guide in this README
4. Consult your supervisor or advisor
5. Search for "Michaelis-Menten kinetics" tutorials online

---

## Version History

**Version 1.0** (2026-01-19)
- Initial release with three model variants
- Comprehensive documentation
- Ambiguous parameters requiring customization
- Unit tests and validation

---

## License and Citation

**Educational Use:**
These scripts are provided for educational and research purposes.

**Citation Recommendation:**
"COS degradation was modeled using custom Python code implementing [first-order/Michaelis-Menten/uncertainty quantification] kinetics (see supplementary material). Parameters were [measured experimentally/estimated from literature] as described in Methods."

**Acknowledgments:**
Model structure based on standard biochemical kinetics. Parameter ranges derived from soil biochemistry literature.

---

## Important Disclaimers

⚠️ **These simulations:**
- Are THEORETICAL MODELS with example parameters
- Require EXPERIMENTAL VALIDATION for quantitative use
- Should NOT be used for agricultural decisions without validation
- Are suitable for HYPOTHESIS GENERATION and teaching
- Demonstrate modeling principles, not definitive predictions

✓ **Appropriate uses:**
- Educational demonstrations
- Hypothesis generation
- Experimental planning
- Understanding model behavior
- Graduate coursework

✗ **Inappropriate uses:**
- Field predictions without validation
- Regulatory submissions
- Agricultural recommendations
- Publication without experimental data
- Quantitative claims without uncertainty

---

## Contact and Contributions

**Questions about the science:**
- Consult soil biochemistry textbooks
- Review chitosanase enzyme literature
- Discuss with your research supervisor

**Questions about the code:**
- Read inline code comments
- Check Python documentation for numpy, scipy, matplotlib
- Review error messages carefully

**Improvements and Modifications:**
Users are encouraged to:
- Customize for their specific systems
- Add additional features as needed
- Validate against experimental data
- Share improvements with collaborators

---

**Remember: Models generate hypotheses. Experiments test them.**

Good luck with your research!
