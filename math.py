import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_solution(a, b):
    # Ensure b > a
    if b < a:
        a, b = b, a

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Blue square at origin with edge length b
    blue_square = patches.Rectangle((0, 0), b, b, facecolor='blue', alpha=0.5, edgecolor='black')
    ax.add_patch(blue_square)

    # Red square anchored at (-a,0) with edge length a
    red_square = patches.Rectangle((-a, 0), a, a, facecolor='red', alpha=0.5, edgecolor='black')
    ax.add_patch(red_square)

    # Green rectangle with width b and height a
    rect = patches.Rectangle((-a, 0), b, a, facecolor='green', alpha=0.3, edgecolor='black')
    ax.add_patch(rect)

    # --- First L-shaped polygon ---
    l_shape_coords = [
        (-a + b, 0),
        (b, 0),
        (b, 2*a),
        (a, 2*a),
        (a, a),
        (b-a, a)
    ]
    l_shape = patches.Polygon(l_shape_coords, closed=True, facecolor='orange', alpha=0.5, edgecolor='black')
    ax.add_patch(l_shape)

    # --- Second polygon on top ---
    second_poly_coords = [
        (0, a),
        (a, a),
        (a, 2*a),
        (b-a, 2*a),
        (b-a, 3*a),
        (0, 3*a)
    ]
    second_poly = patches.Polygon(second_poly_coords, closed=True, facecolor='purple', alpha=0.5, edgecolor='black')
    ax.add_patch(second_poly)

    # ---- Keep original plot limits ----
    margin = max(a, b) * 0.3
    ax.set_xlim(-a - margin, b + margin)
    ax.set_ylim(0 - margin, b + margin)  # original upper limit

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Solution visualization with rectangles and polygons: a={a}, b={b}')

    plt.grid(True)
    plt.show()

def check_identity_ac_minus_b(N):
    holds = []
    fails = []

    for a in range(1, N + 1):
        for b in range(1, N + 1):
            num = a*a + b*b
            den = a*b + 1

            if num % den != 0:
                continue

            c = num // den
            if c > N:
                continue

            # Only check when b > a
            if b > a:
                if (a*c - b)**2 == c:
                    holds.append((a, b, c))
                else:
                    fails.append((a, b, c))

    # ---- Output ----
    print(f"Total solutions with a,b,c ≤ {N} and b > a: {len(holds) + len(fails)}")

    print("\nSolutions where (a*c - b)^2 = c HOLDS:")
    for s in holds:
        print(s)

    print("\nChecking for counterexamples:")
    if not fails:
        print("✔ No counterexamples found — identity holds for ALL solutions with b > a")
    else:
        print("✘ Counterexamples found:")
        for f in fails:
            print(f)



# Example run
N = 10000
check_identity_ac_minus_b(N)


def check_a_divides_b_implies_ratio_equals_c(N):
    holds = []
    fails = []

    for a in range(1, N + 1):
        for b in range(1, N + 1):
            num = a*a + b*b
            den = a*b + 1

            if num % den != 0:
                continue

            c = num // den
            if c > N:
                continue

            # Only check when a divides b
            if b % a == 0:
                if b // a == c:
                    holds.append((a, b, c))
                else:
                    fails.append((a, b, c))

    # ---- Output ----
    print(f"Total solutions with a,b,c ≤ {N} and a | b: {len(holds) + len(fails)}")

    print("\nSolutions where a | b AND b/a = c (HOLDS):")
    for s in holds:
        print(s)

    print("\nChecking for counterexamples:")
    if not fails:
        print("✔ No counterexamples found — whenever a | b, we have b/a = c")
    else:
        print("✘ Counterexamples found:")
        for f in fails:
            print(f)


# Example run

check_a_divides_b_implies_ratio_equals_c(10000)


def check_non_divisible_case(N):
    holds = []
    fails = []

    for a in range(1, N + 1):
        for b in range(1, N + 1):
            num = a*a + b*b
            den = a*b + 1

            if num % den != 0:
                continue

            c = num // den
            if c > N:
                continue

            # Restrict to b > a and a does NOT divide b
            if b > a and b % a != 0:
                if (a*c - b)**2 == c:
                    holds.append((a, b, c))
                else:
                    fails.append((a, b, c))

    # ---- Output ----
    print(f"Total solutions with b > a and a ∤ b: {len(holds) + len(fails)}")

    print("\nSolutions where (a*c - b)^2 = c HOLDS:")
    for s in holds:
        print(s)

    print("\nChecking for counterexamples:")
    if not fails:
        print("✔ No counterexamples found — identity holds for ALL solutions with b > a and a ∤ b")
    else:
        print("✘ Counterexamples found:")
        for f in fails:
            print(f)


# Example run
N = 10000
check_non_divisible_case(N)