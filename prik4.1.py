import numpy as np
import matplotlib.pyplot as plt

def make_PQ_homogeneous(m, a_coeffs, b_coeffs):

    a_coeffs = np.asarray(a_coeffs, dtype=float)
    b_coeffs = np.asarray(b_coeffs, dtype=float)
    assert len(a_coeffs) == m+1 and len(b_coeffs) == m+1

    def P(x, y):
        val = 0.0
        for k in range(m+1):
            val += b_coeffs[k]*(x**(m-k))*(y**k)
        return val

    def Q(x, y):
        val = 0.0
        for k in range(m+1):
            val += a_coeffs[k]*(x**(m-k))*(y**k)
        return val

    return P, Q


def build_A_B_from_coeffs(m, a_coeffs, b_coeffs, phi):

    phi = np.asarray(phi)
    c = np.cos(phi); s = np.sin(phi)
    A = np.zeros_like(phi, dtype=float)
    B = np.zeros_like(phi, dtype=float)
    for k in range(m+1):
        A += a_coeffs[k]*(c**(m-k))*(s**k)
        B += b_coeffs[k]*(c**(m-k))*(s**k)
    return A, B

def plot_homogeneous_phase(m, a_coeffs, b_coeffs,
                           xlim=(-2.5,2.5), ylim=(-2.5,2.5),
                           density=1.4, title=None,
                           show_polar_diagnostics=True):

    P, Q = make_PQ_homogeneous(m, a_coeffs, b_coeffs)


    Y, X = np.mgrid[ylim[0]:ylim[1]:200j, xlim[0]:xlim[1]:200j]
    U = P(X, Y); V = Q(X, Y)

    plt.figure(figsize=(6,6))
    plt.streamplot(X, Y, U, V, density=density, arrowsize=1)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
    plt.title(title or f"Однорідна система (степінь m={m})")
    plt.xlabel('x'); plt.ylabel('y'); plt.grid(True)
    plt.xlim(xlim); plt.ylim(ylim); plt.tight_layout()

    if show_polar_diagnostics:
        phi = np.linspace(0, 2*np.pi, 2000, endpoint=False)
        A, B = build_A_B_from_coeffs(m, a_coeffs, b_coeffs, phi)
        N = A*np.cos(phi) - B*np.sin(phi)
        Z = A*np.sin(phi) + B*np.cos(phi)
        eps = 1e-9
        R = np.where(np.abs(N) < eps, np.nan, Z/N)

        plt.figure(figsize=(7,3.6))
        plt.plot(phi, R, lw=1.5)

        zc_idx = np.where(np.diff(np.signbit(N)))[0]
        plt.scatter(phi[zc_idx], np.zeros_like(zc_idx, dtype=float),
                    s=20, marker='o', label='N(φ)=0')
        plt.title("R(φ) = Z(φ) / N(φ)  (діагностика у куті φ)")
        plt.xlabel("φ"); plt.ylabel("Z/N")
        plt.grid(True); plt.legend(loc='best'); plt.tight_layout()


m = 2
a_coeffs = [0, 2, 0]
b_coeffs = [1, 0, -1]

plot_homogeneous_phase(m, a_coeffs, b_coeffs,
                       title="Однорідна m=2: P=x^2 - y^2, Q=2xy",
                       show_polar_diagnostics=True)

plt.show()
