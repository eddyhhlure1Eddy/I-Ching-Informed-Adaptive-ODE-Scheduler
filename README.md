![lv_0_20251021160044](https://github.com/user-attachments/assets/4646a879-93f8-4823-9bd2-ae7a4e219efa)

# ODEv2 Technical Overview

**Version**: ODEv2  
**Author**: eddy  
**Date**: October 2025  
**Status**: Public Release (Confidential Internal Parameters & Unpublished Paper Details Omitted)

---

## Abstract

ODEv2 is an engineered stable sampling solution for Flow Matching, extending the early LowStep/SA-ODE Stable philosophy of mathematical simplicity and stability through "low-step linear+shift scheduling, first-order ODE backbone, adaptive multi-step extrapolation, and end-phase stabilization" to ensure convergence stability. For complex scenarios, it introduces ODE+SDE hybrid with edge/frequency awareness and "Wuxing" (Five Elements) dynamic modulation to achieve superior performance in video temporal consistency and detail preservation.

**Application Scenarios**: Fast, stable sampling for Flow Matching models (including low-step and video temporal scenarios)

---

## Table of Contents

1. [Development Timeline](#development-timeline)
2. [Core Technical Evolution](#core-technical-evolution)
3. [Key Features](#key-features)
4. [Architecture Design](#architecture-design)
5. [Ecosystem & Citations](#ecosystem--citations)
6. [Confidentiality Statement](#confidentiality-statement)
7. [References](#references)

---

## Development Timeline

### Phase 1: Early Conception (LowStep / SA-ODE Stable)

**Objective**: Achieve high-quality sampling within 4-8 steps through mathematical simplicity and stable convergence

**Core Concepts**:
- Linear sigma scheduling (`σ = 1 - t`) minimizes accumulated error in low-step scenarios
- Shift transformation (`σ' = shift·σ / (1 + (shift-1)·σ)`) optimizes distribution
- First-order Euler ODE stepping provides exact solutions along optimal transport paths
- Low-step stability strategies avoid numerical instabilities from higher-order methods

**Theoretical Foundation**:
- Optimal transport approximation in Flow Matching's linear interpolation structure
- Exactness proof for first-order methods in linear velocity fields
- Complexity minimization principle (elimination of unnecessary computational overhead)

**Reference**: [ode-t1 Repository README](https://github.com/eddyhhlure1Eddy/ode-t1)

---

### Phase 2: Wan Integration & Stabilization Implementation

**Objective**: Implement a production-ready ODE stable sampler in ComfyUI-WanVideoWrapper

**Key Technologies**:
- **Adaptive Order**: Adams-Bashforth 2/3-order multi-step extrapolation
  - High σ region: Order reduction for stability (2nd or 1st order)
  - Mid-range region: High order for accuracy (3rd order)
  - Convergence region: Order reduction for stability (1st or 2nd order)

- **Convergence Phase Optimization**:
  - EMA velocity smoothing (Exponential Moving Average)
  - dt step size damping (progressive reduction in low σ region)
  - End-phase stabilization fusion (historical average velocity weighted blending)

- **Implementation Form**: Pure ODE path without stochastic noise injection

**Technical Characteristics**:
- Segmented heuristic strategies with different solver configurations per σ region
- Historical velocity buffer mechanism supporting multi-step extrapolation and smoothing
- Low-step specialization (≤8 steps use conservative strategies)

**Reference Implementation**: `FlowMatchSAODEStableScheduler`

---

### Phase 3: Community Recognition & Citation

**Description**: Open-source maintainers integrated the stable ODE solution into their repositories with explicit source attribution in file headers

**Citation Example**:
```python
"""
SA-ODE Stable - SA-Solver ODE version optimized for convergence stability
Based on successful sa_solver/ode, further improving convergence stability
from https://github.com/eddyhhlure1Eddy/ode-ComfyUI-WanVideoWrapper
"""
```
dz/dt = g ⊙ z ⊙ ( 1 + c * (G @ z) - c * (I @ z) )

**Reference File**: `kijai/ComfyUI-WanVideoWrapper/wanvideo/schedulers/fm_sa_ode.py`

---

### Phase 4: ODEv2 Engineering Extension (IChing/Wuxing Scheduler)

**Objective**: Enhance quality and consistency in complex scenarios (video edges/temporal) while maintaining stability

**Core Innovations**:

1. **Wuxing (Five Elements) Dynamics-Driven Modulation**
   - Wood: Solver order bias
   - Fire: Velocity smoothing strength
   - Earth: Step size damping modulation
   - Metal: End-phase fusion strength
   - Water: Threshold shift

2. **ODE+SDE Hybrid Architecture**
   - ODE backbone: Deterministic velocity integration
   - SDE injection: Controlled stochastic exploration
   - Adaptive mixing: Dynamic ratio adjustment based on σ and Wuxing state

3. **Edge Awareness & Frequency Shaping**
   - Gradient-based edge detection (2D/3D)
   - SDE noise suppression in edge regions
   - ODE weight boost in edge regions
   - High-pass noise shaping (reduces low-frequency blockiness artifacts)

4. **Low-Step Detail Stability**
   - Color reference capture and restoration
   - Light smoothing in non-edge regions
   - High-frequency detail jitter suppression (hair/eyes, etc.)

5. **Enhanced End-Phase Convergence**
   - Conservative step construction with historical average velocity
   - Wuxing-modulated fusion strength and trigger threshold
   - Trailing/ghosting artifact risk control

**Reference Implementation**: `wanvideo/schedulers/iching_wuxing_scheduler.py`

---

## Core Technical Evolution

### 1. Scheduling Strategy

| Phase | Strategy | Characteristics |
|-------|----------|-----------------|
| Early | Linear + shift transform | Low-step stability, minimal error |
| ODEv2 | Low-step linear + normal-step cosine + shift | Balance stability and smoothness |

**Mathematical Expression**:
```
Low-step (≤10): σ = 1 - t
Normal-step (>10): σ = 0.5(1 + cos(πt))
Unified transform: σ' = shift·σ / (1 + (shift-1)·σ), shift=3.0
```

---

### 2. Integration & Extrapolation

**First-Order Euler (Base)**:
```
x_{t+1} = x_t + v_t · dt
```

**Adams-Bashforth Multi-Step Extrapolation**:
- **Third-order**: `v = (23/12)v_n - (16/12)v_{n-1} + (5/12)v_{n-2}`
- **Second-order**: `v = 1.5v_n - 0.5v_{n-1}`
- **First-order**: `v = v_n`

**Adaptive Order Selection**:
- Early stage (σ > 0.7): Low order for stability
- Mid-stage (0.15 < σ ≤ 0.7): High order for accuracy
- Convergence stage (σ ≤ 0.15): Reduced order for stability
- Low-step mode (≤8 steps): Low order throughout

---

### 3. Stability Governance

**Convergence Phase Smoothing**:
```
v_smooth = α · v_smooth_prev + (1-α) · v_current
```
- α modulated by Wuxing "Fire" element
- Activated only when σ < threshold

**Step Size Damping**:
```
dt' = dt · damping,  damping = 0.5 + 0.5(σ/threshold)
```
- threshold modulated by Wuxing "Earth" element

**End-Phase Stabilization Fusion**:
```
v_avg = mean(v_{-3}, v_{-2}, v_{-1})
x_stable = x + v_avg · dt
x_final = β·x_pred + (1-β)·x_stable
```
- β jointly modulated by Wuxing "Metal" and "Water" elements

---

### 4. ODE+SDE Hybrid Dynamics

**ODE Path** (Deterministic):
```
x_ode = x + v · dt
```

**SDE Path** (Stochastic):
```
x_sde = x_ode + σ_noise · ε
```
- ε is high-pass shaped noise
- σ_noise modulated by σ and Wuxing "Fire" element

**Mixing Weight**:
```
w_ode = base_weight + earth_bias + water_bias + ghost_suppress
x_final = w_ode · x_ode + (1-w_ode) · x_sde
```

**Edge-Aware Adjustment**:
```
edge_mask = detect_gradient(x)
w_ode_map = w_ode + edge_ode_boost · edge_mask
x_final = w_ode_map · x_ode + (1-w_ode_map) · x_sde
```

---

### 5. Frequency & Edge Processing

**High-Pass Noise Shaping**:
```
ε_hp = ε - blur(ε)
ε_shaped = (1-α)·ε + α·ε_hp
```
- Reduces low-frequency blockiness artifacts
- Improves temporal consistency

**Edge Detection & Suppression**:
```
grad_x = |x[:,:,:,1:] - x[:,:,:,:-1]|
grad_y = |x[:,:,1:,:] - x[:,:,:-1,:]|
edge_soft = normalize(grad_x + grad_y)
ε_final = ε · (1 - suppress · edge_soft)
```

---

## Key Features

### ✓ Low-Step Stability
- Linear+shift scheduling
- First-order ODE backbone
- Order reduction strategy
- dt damping

### ✓ Multi-Step Extrapolation
- Adams-Bashforth 2/3-order
- Adaptive order selection
- Historical velocity buffering

### ✓ End-Phase Stability
- Historical average velocity fusion
- Trailing artifact risk control
- Wuxing-modulated fusion strength

### ✓ Hybrid Awareness
- ODE+SDE adaptive mixing
- Edge-aware weight adjustment
- High-pass noise shaping

### ✓ Detail Preservation
- Color consistency restoration
- Non-edge smoothing
- High-frequency jitter suppression

### ✓ Dynamic Adaptation
- Wuxing continuous modulation
- State-driven strategy switching
- Real-time parameter optimization

---

## Architecture Design

### Component Hierarchy

```
IChingWuxingScheduler
├── WuxingDynamics (Five Elements Dynamics Engine)
│   ├── State Update (RK4)
│   ├── Generation/Inhibition Matrices
│   └── Parameter Mapping
│
├── Time Scheduling (set_timesteps)
│   ├── Low-step Linear
│   ├── Normal-step Cosine
│   └── Shift Transform
│
├── Main Step (step)
│   ├── Velocity Extrapolation
│   ├── Velocity Smoothing
│   ├── dt Damping
│   ├── ODE+SDE Hybrid
│   ├── Edge Awareness
│   ├── Smoothing & Deblocking
│   └── End-Phase Stabilization
│
└── Auxiliary Functions
    ├── Detail Stability
    ├── Noise Injection
    └── Numerical Robustness
```

### Data Flow

```
Input: model_output, timestep, sample
  ↓
Wuxing State Update (RK4)
  ↓
Velocity Buffering & Extrapolation (Adams-Bashforth)
  ↓
Velocity Smoothing (EMA, Fire-modulated)
  ↓
Step Size Damping (Earth-modulated)
  ↓
ODE Path Computation
  ↓
SDE Noise Generation & Shaping
  ↓
Edge Detection & Awareness
  ↓
ODE+SDE Mixing (Earth/Water-modulated)
  ↓
Smoothing & Deblocking
  ↓
Detail Stabilization (Low-step)
  ↓
End-Phase Stabilization Fusion (Metal-modulated)
  ↓
Output: prev_sample
```

---

## Ecosystem & Citations

### File & Implementation References

1. **Early Conception**
   - Repository: [ode-t1](https://github.com/eddyhhlure1Eddy/ode-t1)
   - Content: LowStep / SA-ODE Stable overview and theoretical foundation

2. **Community Citation**
   - File: `kijai/ComfyUI-WanVideoWrapper/wanvideo/schedulers/fm_sa_ode.py`
   - Description: File header comments reference original source

3. **Current Implementation**
   - File: `wanvideo/schedulers/iching_wuxing_scheduler.py`
   - Content: Complete ODEv2 implementation (IChing/Wuxing Scheduler)

### Technology Stack

- **Core Framework**: PyTorch
- **Diffusion Library**: diffusers
- **Application Platform**: ComfyUI-WanVideoWrapper
- **Compute Backend**: CUDA (optional)

---

## Confidentiality Statement

### Public Content

✓ Algorithm principles and design philosophy  
✓ Major technical roadmap and evolution  
✓ Engineering strategies and feature descriptions  
✓ References and citation relationships  

### Non-Disclosed Content

✗ Internal tuning parameters and coefficient tables  
✗ Precise threshold functions and curves  
✗ Specific coefficients of Wuxing dynamics equations  
✗ Kernel morphology details for frequency processing  
✗ Formula derivations and proofs from unpublished papers  
✗ Internal testing matrices and benchmark data  
✗ Normalization method details for edge detection  

---

## Application Scenarios & Limitations

### Application Scenarios

- Diffusion models with Flow Matching architecture
- Low-step (4-12 steps) fast sampling
- Video generation with temporal consistency requirements
- Scenarios requiring edge/detail stability

### Current Limitations

- Primarily optimized for Flow Matching; other architectures require adaptation
- Step count upper bound influenced by model training characteristics
- Wuxing parameters need tuning for specific scenarios
- Some features have limited effectiveness at extremely low steps (<4)

### Future Directions

- Content-based adaptive step count prediction
- Cross-architecture optimization (DDPM/DDIM variants)
- Real-time interactive generation support
- More efficient CUDA kernel implementations

---

## References


### Implementation References

- [ode-t1 Repository](https://github.com/eddyhhlure1Eddy/ode-t1) - LowStep Original Conception
- [ode-ComfyUI-WanVideoWrapper](https://github.com/eddyhhlure1Eddy/ode-ComfyUI-WanVideoWrapper) - SA-ODE Stable Implementation
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - Community Integration

---

## Version History


- **v2.1** (2025-Q4): Edge Awareness & Frequency Optimization Enhancement (Current)

---

## License & Acknowledgments

**Author**: eddy  
**License**: Apache 2.0  
**Acknowledgments**: Thanks to the open-source community for recognizing and contributing to stable sampling technology

---

## Contact

For technical questions or collaboration inquiries, please contact via:

- **GitHub**: [eddyhhlure1Eddy](https://github.com/eddyhhlure1Eddy)
- **Project Homepage**: [ode-t1](https://github.com/eddyhhlure1Eddy/ode-t1)

---

*This document is a public technical overview. Internal parameters and unpublished paper details are not disclosed.*  
*Last Updated: October 2025*

