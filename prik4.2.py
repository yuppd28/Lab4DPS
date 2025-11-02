
import numpy as np
import matplotlib.pyplot as plt

def phase_linear(a, b, c, d,
                 xlim=(-3, 3), ylim=(-3, 3),
                 density=1.3, title=None,
                 init_points=None, t_end=12, h=0.02):

    A = np.array([[a, b], [c, d]], dtype=float)

    def f(t, z):
        x, y = z
        return np.array([a*x + b*y, c*x + d*y], dtype=float)


    def rk4(f, y0, t_span, h=0.01):
        t0, t1 = t_span
        t = t0
        y = np.array(y0, dtype=float)
        T = [t]; Y = [y.copy()]
        while t < t1 - 1e-12:
            h_eff = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + 0.5*h_eff, y + 0.5*h_eff*k1)
            k3 = f(t + 0.5*h_eff, y + 0.5*h_eff*k2)
            k4 = f(t + h_eff, y + h_eff*k3)
            y += (h_eff/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            t += h_eff
            T.append(t); Y.append(y.copy())
        return np.array(T), np.array(Y)


    Y, X = np.mgrid[ylim[0]:ylim[1]:100j, xlim[0]:xlim[1]:100j]
    U = a*X + b*Y
    V = c*X + d*Y

    plt.figure(figsize=(6,6))
    plt.streamplot(X, Y, U, V, density=density, arrowsize=1)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)


    if init_points is None:
        init_points = [(-2, -1), (2, 0.5), (-1.5, 2), (1.5, -2)]
    for p in init_points:
        _, Z = rk4(f, p, (0, t_end), h=h)
        plt.plot(Z[:,0], Z[:,1], lw=1.8, label=f'IC {p}')

    eigvals = np.linalg.eigvals(A)
    plt.title((title or "Лінійна: x' = ax+by, y' = cx+dy")
              + f"\nВласні значення: {np.round(eigvals, 3)}")
    plt.xlabel('x'); plt.ylabel('y'); plt.grid(True); plt.legend(loc='best')
    plt.xlim(xlim); plt.ylim(ylim); plt.tight_layout()


phase_linear(a=0, b=1, c=-1, d=0,
             title="Центр: x' = y, y' = -x",
             density=1.5, t_end=25, h=0.02)


phase_linear(a=1, b=0, c=0, d=-1,
             title="Сідло: x' = x, y' = -y",
             density=1.4, t_end=12, h=0.02)

plt.show()
