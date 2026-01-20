"""
Chitosan Oligosaccharide (COS) Degradation Simulation
======================================================

This script models the degradation and adsorption of chitosan oligosaccharides 
in the rhizosphere using a first-order kinetic model that accounts for both 
microbial enzymatic degradation and soil adsorption.

Model equation: dC/dt = -(k_d + k_a) * C
Analytical solution: C(t) = C_0 * exp(-k_eff * t)

PARAMETER CUSTOMIZATION:
-----------------------
Users should adjust parameter values based on their specific:
- Soil type and properties
- Microbial community characteristics  
- COS application rates
- Environmental conditions (temperature, pH, moisture)

The default values provided are for demonstration purposes and represent
typical ranges from literature. For your specific application, these values
should be determined experimentally or adjusted based on your system.

Author: Generated for Science Case Section 5
Version: Customizable parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality plotting style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# SIMULATION PARAMETERS (User-Adjustable)
# ============================================================================

# Initial COS concentration (µM)
# Adjust based on your experimental conditions or application rate
C_0 = 100.0

# Adsorption rate constant (h⁻¹) - kept fixed for main figure
# This value should be determined experimentally for your specific soil type
# Range guidance: typically 0.03-0.1 h⁻¹ for oligosaccharides on clay minerals
k_a_fixed = 0.05

# Microbial degradation rate constants (h⁻¹) - representing different conditions
# These represent a range from low to high microbial chitosanase activity
# Adjust these values based on your soil microbial community characteristics
# Range guidance: 0.05-0.4 h⁻¹ based on published chitosanase activity studies
k_d_values = [0.05, 0.1, 0.2, 0.4]  # Low to high microbial activity

# Time array (hours)
# Adjust time range and resolution based on your experimental timeframe
t = np.linspace(0, 24, 241)  # 0 to 24 hours with 0.1 h steps

# ============================================================================
# DEGRADATION MODEL FUNCTION
# ============================================================================

def cos_degradation(t, C_0, k_d, k_a):
    """
    Calculate COS concentration over time using first-order decay kinetics.
    
    Parameters:
    -----------
    t : array
        Time points (hours)
    C_0 : float
        Initial COS concentration (µM)
    k_d : float
        Microbial degradation rate constant (h⁻¹)
    k_a : float
        Adsorption rate constant (h⁻¹)
    
    Returns:
    --------
    C : array
        COS concentration at each time point (µM)
    """
    k_eff = k_d + k_a  # Effective decay rate
    C = C_0 * np.exp(-k_eff * t)
    return C

def calculate_half_life(k_d, k_a):
    """
    Calculate COS half-life.
    
    Parameters:
    -----------
    k_d : float
        Microbial degradation rate constant (h⁻¹)
    k_a : float
        Adsorption rate constant (h⁻¹)
    
    Returns:
    --------
    t_half : float
        Half-life (hours)
    """
    k_eff = k_d + k_a
    t_half = np.log(2) / k_eff
    return t_half

# ============================================================================
# FIGURE 1: DEGRADATION CURVES UNDER VARYING MICROBIAL ACTIVITY
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(8, 6))

# Color scheme for different microbial activities
colors = ['#2E7D32', '#66BB6A', '#FFA726', '#D32F2F']
labels = ['Low (k_d = 0.05 h⁻¹)', 
          'Moderate (k_d = 0.1 h⁻¹)', 
          'High (k_d = 0.2 h⁻¹)', 
          'Very High (k_d = 0.4 h⁻¹)']

# Plot degradation curves
for i, k_d in enumerate(k_d_values):
    C = cos_degradation(t, C_0, k_d, k_a_fixed)
    ax1.plot(t, C, linewidth=2.5, color=colors[i], label=labels[i])

# Formatting
ax1.set_xlabel('Time (h)', fontweight='bold')
ax1.set_ylabel('COS Concentration (µM)', fontweight='bold')
ax1.set_title('Simulated COS Degradation Under Variable Microbial Activity\n(k_a = 0.05 h⁻¹)', 
              fontweight='bold', pad=15)
ax1.legend(title='Microbial Activity Level', frameon=True, loc='upper right')
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cos_degradation_curves.png', dpi=300, bbox_inches='tight')
print("✓ Figure 1 saved: cos_degradation_curves.png")

# ============================================================================
# HALF-LIFE CALCULATIONS AND DISPLAY
# ============================================================================

print("\n" + "="*70)
print("COS HALF-LIFE CALCULATIONS")
print("="*70)
print(f"{'Microbial Activity':<25} {'k_d (h⁻¹)':<12} {'k_a (h⁻¹)':<12} {'Half-life (h)':<15}")
print("-"*70)

for i, k_d in enumerate(k_d_values):
    t_half = calculate_half_life(k_d, k_a_fixed)
    activity_level = labels[i].split('(')[0].strip()
    print(f"{activity_level:<25} {k_d:<12.2f} {k_a_fixed:<12.2f} {t_half:<15.2f}")

print("="*70 + "\n")

# ============================================================================
# FIGURE 2: SENSITIVITY HEATMAP (Half-life vs k_d and k_a)
# ============================================================================

# Create parameter grids for sensitivity analysis
k_d_range = np.linspace(0.05, 0.4, 30)
k_a_range = np.linspace(0.03, 0.1, 30)
K_d, K_a = np.meshgrid(k_d_range, k_a_range)

# Calculate half-life for each combination
T_half = np.log(2) / (K_d + K_a)

# Create heatmap
fig2, ax2 = plt.subplots(figsize=(9, 7))

# Plot heatmap with colorbar
im = ax2.contourf(K_d, K_a, T_half, levels=20, cmap='YlOrRd_r')
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('COS Half-life (h)', fontweight='bold', rotation=270, labelpad=20)

# Add contour lines for clarity
contours = ax2.contour(K_d, K_a, T_half, levels=10, colors='black', alpha=0.3, linewidths=0.5)
ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f h')

# Formatting
ax2.set_xlabel('Microbial Degradation Rate, k_d (h⁻¹)', fontweight='bold')
ax2.set_ylabel('Adsorption Rate Constant, k_a (h⁻¹)', fontweight='bold')
ax2.set_title('COS Half-life Sensitivity to Degradation and Adsorption Rates', 
              fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cos_halflife_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2 saved: cos_halflife_heatmap.png")

# ============================================================================
# FIGURE 3: COMBINED VISUALIZATION (Optional)
# ============================================================================

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Degradation curves
for i, k_d in enumerate(k_d_values):
    C = cos_degradation(t, C_0, k_d, k_a_fixed)
    ax3a.plot(t, C, linewidth=2.5, color=colors[i], label=labels[i])

ax3a.set_xlabel('Time (h)', fontweight='bold')
ax3a.set_ylabel('COS Concentration (µM)', fontweight='bold')
ax3a.set_title('(A) Degradation Kinetics', fontweight='bold', loc='left')
ax3a.legend(title='Microbial Activity', frameon=True, fontsize=8)
ax3a.grid(True, alpha=0.2, linestyle='--')
ax3a.set_xlim(0, 24)
ax3a.set_ylim(0, 105)

# Right panel: Half-life bar chart
half_lives = [calculate_half_life(k_d, k_a_fixed) for k_d in k_d_values]
activity_labels = ['Low\n(0.05)', 'Moderate\n(0.1)', 'High\n(0.2)', 'Very High\n(0.4)']

bars = ax3b.bar(activity_labels, half_lives, color=colors, edgecolor='black', linewidth=1.2)
ax3b.set_ylabel('COS Half-life (h)', fontweight='bold')
ax3b.set_xlabel('Microbial Activity Level (k_d, h⁻¹)', fontweight='bold')
ax3b.set_title('(B) Half-life Comparison', fontweight='bold', loc='left')
ax3b.grid(True, alpha=0.2, linestyle='--', axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, half_lives)):
    height = bar.get_height()
    ax3b.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f} h', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cos_combined_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3 saved: cos_combined_analysis.png")

# ============================================================================
# SIMULATION SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"Initial COS concentration: {C_0} µM")
print(f"Adsorption rate (k_a): {k_a_fixed} h⁻¹")
print(f"Time range simulated: 0-24 hours")
print(f"Number of microbial conditions tested: {len(k_d_values)}")
print("\nKey Findings:")
print(f"  • Fastest degradation (k_d = {max(k_d_values)} h⁻¹): t₁/₂ = {min(half_lives):.1f} h")
print(f"  • Slowest degradation (k_d = {min(k_d_values)} h⁻¹): t₁/₂ = {max(half_lives):.1f} h")
print(f"  • Fold-change in half-life: {max(half_lives)/min(half_lives):.1f}×")
print("="*70)

print("\n📊 All figures saved to outputs directory")
print("📁 Files generated:")
print("   1. cos_degradation_curves.png")
print("   2. cos_halflife_heatmap.png")
print("   3. cos_combined_analysis.png")
