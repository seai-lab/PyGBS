import numpy as np

from gbs import compute_presence_nonpresence_ssi, compute_relative_performance_ssi

center = np.array([0.3 * 0.5 * np.pi, -0.4 * np.pi])
radius = 0.1
presence_points = center + 0.8 * radius * (2 * (np.random.rand(100, 2) - 0.5))
presence_values = np.random.choice([-1, 1], presence_points.shape[0], p=[0.25, 0.75])

# for density in [8000, 12000, 16000, 20000]:
for density in np.array([8000, 12000, 16000, 20000]):
    pnp_ssi = compute_presence_nonpresence_ssi(presence_points, center, radius, density, k=4)
    rp_ssi = compute_relative_performance_ssi(presence_points, presence_values, center, radius, density, k=4)
    print("Presence v.s. Non-Presence SSI Score: ", pnp_ssi, "Relative Performance SSI Score: ", rp_ssi)