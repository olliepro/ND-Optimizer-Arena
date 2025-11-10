# ND Optimizer Arena

This project renders a procedurally generated loss field and lets you poke it with multiple optimizers (SGD, Momentum, RMSProp, Adam, AdamW, Muon) across 2–15 dimensions. The source lives entirely in `index.html`, so this README highlights the key pieces you are likely to look for when modifying or extending the arena.

## Loss-Landscape Building Blocks

The function `rebuildFullLossND` stitches together the ingredients below to produce the final `loss_nd(x)` used everywhere (plotting, gradients, optimizer updates). Each component is exposed as a separate helper so you can tweak the recipe in isolation.

| Component | Function | Notes |
| --- | --- | --- |
| Quadratic bowl + anisotropy | `base_loss_nd` | Starts with an anisotropic quadratic well aligned to the slice center, with per-dimension stretch factors in `ANIS`. Hidden dimensions are randomly widened or squeezed when you change `ND`, so each reroll feels different even if the visible slice looks similar.
| Fourier texture | `fourier_noise_nd` | Adds multi-frequency sinusoidal noise sampled from `fbasis`, giving the terrain a fractal look. The basis vectors are reseeded whenever the domain/seed changes, so the texture moves with the landscape seed slider.
| Ripples & radial bumps | `base_loss_nd` | Low-amplitude sinusoids tied to `X0`, `X1`, and the overall radius inject medium-scale undulations on top of the quadratic bowl.
| Finite Gaussian spikes | `finite_spikes_nd` | Adds bump-like wells and peaks with gaussian falloff. `regenerateSpikesForDomainND` scatters their centers uniformly through all dimensions; widths are multiplied by `HIDDEN_WIDTH_BOOST` for hidden axes and softened by `HIDDEN_SOFTEN` in evaluation so slices still intersect them without spawning an exponential number of spikes.
| Reciprocal ("infinite") spikes | `infinite_spikes_nd` | Adds asymptotic spikes of the form `scale / (\sum |x-c|^p + eps)`. As with the Gaussian bumps, widths widen in hidden dimensions, and `spikeGain()` scales their amplitude like `(max(1, ND/2))^0.7` to keep them visible as the space grows.
| Log dent + radial fallback | `rebuildFullLossND` | The log dent slowly pulls everything toward the star's neighborhood, while a small fraction (`INTERP_BETA = 0.2`) of the pure radial term keeps the energy bounded when all other terms cancel out.

These components run on the **base** coordinates (not the noisy observation space) so the same function drives ground-truth comparisons, the contour plot, and the optimiser feedback loop.

## N-D Visualization Strategy

Plotly can only show a 2D surface, so the arena renders a live slice through the N-D field:

- `setDomain` and `fitDomainToContainer` keep the visible window aligned with the canvas aspect ratio and rebuild `DOM_MIN`/`DOM_MAX` for all dimensions whenever you pan, zoom, or regenerate.
- For `d >= 2`, the current slice is frozen at `hiddenStart[d-2]`. `refreshHiddenStart` initializes (or clamps) those coordinates so they remain inside the ND domain as it shifts. When you drag the 2D start marker, only the first two coordinates move; the hidden coordinates stay fixed unless you regenerate or change `ND`.
- `startVector` copies the visible cursor plus the stored hidden coordinates so every optimizer starts from the same full N-D position despite only showing an XY projection.
- `effectiveGridSizes`, `linspace`, and the adaptive sampler downscale the contour grid for larger `ND` to compensate for heavier gradient calls, then `rebuildColourStats` applies the log-spectrum coloring (legend: “log-spectrum colouring” in the header) so different orders of magnitude remain distinguishable.

In practice this means you can explore how optimizers behave in a 10-D landscape while still dragging points around a 2D canvas—the hidden coordinates simply define which slice you see.

## Gradient Estimation & Observation Noise

- `grad_nd` performs central finite differences with a reusable scratch buffer (`gradScratchPlus/minus`) so it does not reallocate every frame. The step size `GRAD_H` is shared by all optimizers, keeping updates consistent.
- `grad_full` / `grad_base` choose between the full procedural field and the “true” base field used for star placement.
- `grad_observed_nd` adds zero-mean Gaussian noise (scaled by `effectiveNoiseStd()` and your batch-size slider) before calling `grad_full`. This simulates minibatch gradients: optimizers run on the noisy signal while the star/minima logic still queries the exact field. Values are clipped with `clampMagVec` inside each optimizer to avoid exploding steps when the noise spikes.

## Minima Estimation Pipeline

The white ★ marker is not hard-coded; it is estimated every time you change the seed, dimension, or domain:

1. **Global sampling (`globalGuessND`)** – draw `GLOBAL_GUESS_SAMPLES` random points inside `DOM_MIN/DOM_MAX` and keep the best base-loss value.
2. **Annealed local search (`refineFromND`)** – repeatedly sample random directions inside a shrinking radius (see `radii` sequence in `refineStarWithGlobal`). This ropes in nearby wells that the coarse global pass might have missed.
3. **Gradient polish (`gradientPolishND`)** – run a cosine-annealed learning rate schedule on `grad_base` with norm clipping (`CLIP_NORM`). This locks onto the nearest smooth minimum without being distracted by the noisy spikes added later.
4. **Alignment & bookkeeping** – `alignLandscapeCenterToStar` recenters the domain, `rebuildFullLossND` folds the spike/texture terms back in, and `updateLegendLosses` propagates the final scalar so the sidebar labels are always in sync.

Because the estimator works in the clean base loss, the ★ stays stable across noisy minibatch draws, yet you still see how the noisy optimizers chase it in the rendered slice.

## Hidden-Dimension Spike Widening

Higher-dimensional slices would rarely intersect a thin spike placed somewhere in the unseen coordinates. Two mechanisms counter this:

- **Generation-time widening** – in `regenerateSpikesForDomainND`, every width entry `a[d]` receives a factor of `HIDDEN_WIDTH_BOOST` when `d >= 2`. This physically makes the Gaussian bumps and reciprocal spikes wider along the hidden axes, so slices are more likely to cut through them.
- **Evaluation-time softening** – when computing each spike’s contribution (`finite_spikes_nd`, `infinite_spikes_nd`), the code divides the hidden-axis deltas by `HIDDEN_SOFTEN`, effectively inflating their standard deviation again. Together with the dimension-aware `spikeGain` and `spikeDensityBoost`, this keeps the landscape interesting without requiring exponentially more spikes as `ND` grows.

Feel free to customize those constants if you want either sharper hidden spikes (harder optimization problems) or gentler ones (clearer visualizations).
