import streamlit as st
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Session 3 — LP Production Planning",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Session 3: Linear Programming for Production and Sales Planning")
st.caption("FH Aachen · Optimization and Machine Learning · SS 2026 · Group 7")

# ── Fixed machine consumption coefficients (given in the problem) ─────────────
A1 = np.array([2.0, 3.0, 1.0])  # Machine 1 hours per unit (A, B, C)
A2 = np.array([1.0, 1.0, 2.0])  # Machine 2 hours per unit (A, B, C)
PRODUCTS = ["A", "B", "C"]
COLORS = ["#1976D2", "#388E3C", "#F57C00"]


# ── Shared LP solver ──────────────────────────────────────────────────────────
def solve_lp(profit, c1, c2, d_max):
    n = len(profit)
    x = cp.Variable(n, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(np.array(profit, dtype=float) @ x),
        [
            A1 @ x <= float(c1),
            A2 @ x <= float(c2),
            x <= np.array(d_max, dtype=float),
        ],
    )
    prob.solve(solver=cp.CLARABEL)
    if prob.status == "optimal":
        return prob.value, x.value
    return None, None


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab3, tab4, tab5 = st.tabs(
    [
        "📦 Tasks 1 & 2 — Basic Model",
        "🔍 Task 3 — Scenario Analysis",
        "📅 Task 4 — Multi-Period",
        "🧪 Task 5 — Experiment Framework",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 & 2
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed parameters (given in the problem statement)
PROFIT  = np.array([20.0, 25.0, 18.0])
D_MAX   = np.array([40.0, 30.0, 50.0])
C1, C2  = 100.0, 80.0

val_t1, x_t1 = solve_lp(PROFIT, C1, C2, D_MAX)

with tab1:
    st.header("Tasks 1 & 2 — Basic Production Planning & Verification")
    st.markdown(
        "A company produces three products **A, B, C**. "
        "We formulate the profit maximisation as a Linear Program and verify the solution."
    )
    st.divider()

    # ── Problem data table ────────────────────────────────────────────────────
    st.subheader("Problem Data")
    df_data = pd.DataFrame(
        {
            "Profit per unit (€)":    PROFIT,
            "Machine 1 (h/unit)":     A1,
            "Machine 2 (h/unit)":     A2,
            "Max. demand (units)":    D_MAX.astype(int),
        },
        index=PRODUCTS,
    )
    st.dataframe(df_data, use_container_width=True)
    st.caption("Machine capacities: M1 = 100 h · M2 = 80 h")
    st.divider()

    # ── Step 1: Define production variables ───────────────────────────────────
    st.subheader("Step 1 — Define the production variables")
    st.markdown(
        "We introduce one decision variable per product — the quantity to produce. "
        "The solver will find the values that maximise profit. "
        "All quantities must be non-negative (you can't produce a negative amount)."
    )
    st.latex(
        r"\mathbf{x} = \begin{pmatrix} x_A \\ x_B \\ x_C \end{pmatrix} \geq 0"
    )
    with st.expander("📄 Show code"):
        st.code(
            "import numpy as np\n"
            "import cvxpy as cp\n"
            "\n"
            "products = [\"A\", \"B\", \"C\"]\n"
            "x = cp.Variable(3, nonneg=True)   # x_A, x_B, x_C ≥ 0",
            language="python",
        )
    st.divider()

    # ── Step 2: Objective function ────────────────────────────────────────────
    st.subheader("Step 2 — Maximize total profit")
    st.markdown(
        "The objective is the dot product of profit vector and production vector — "
        "profit per unit times quantity, summed over all products."
    )
    st.latex(r"\max\; 20\,x_A + 25\,x_B + 18\,x_C \;=\; \max\; \mathbf{p}^\top \mathbf{x}")
    with st.expander("📄 Show code"):
        st.code(
            "profit = np.array([20., 25., 18.])\n"
            "objective = cp.Maximize(profit @ x)",
            language="python",
        )
    st.divider()

    # ── Step 3: Machine capacity constraints ──────────────────────────────────
    st.subheader("Step 3 — Enforce machine capacity constraints")
    st.markdown(
        "Each product consumes machine hours. The total hours used across all products "
        "must not exceed each machine's available capacity."
    )
    st.latex(
        r"\underbrace{2x_A + 3x_B + x_C}_{\text{hours used on M1}} \leq 100"
        r"\qquad"
        r"\underbrace{x_A + x_B + 2x_C}_{\text{hours used on M2}} \leq 80"
    )
    with st.expander("📄 Show code"):
        st.code(
            "a1 = np.array([2., 3., 1.])   # M1 hours per unit\n"
            "a2 = np.array([1., 1., 2.])   # M2 hours per unit\n"
            "c1, c2 = 100., 80.             # machine capacities\n"
            "\n"
            "constraints = [\n"
            "    a1 @ x <= c1,   # Machine 1 capacity\n"
            "    a2 @ x <= c2,   # Machine 2 capacity\n"
            "]",
            language="python",
        )
    st.divider()

    # ── Step 4: Demand upper bounds ───────────────────────────────────────────
    st.subheader("Step 4 — Enforce demand upper bounds")
    st.markdown(
        "The market can only absorb a limited amount of each product. "
        "Producing more than the demand makes no sense — it would just sit unsold."
    )
    st.latex(r"x_A \leq 40 \qquad x_B \leq 30 \qquad x_C \leq 50")
    with st.expander("📄 Show code"):
        st.code(
            "d_max = np.array([40., 30., 50.])\n"
            "constraints.append(x <= d_max)",
            language="python",
        )
    st.divider()

    # ── Step 5: Solve ─────────────────────────────────────────────────────────
    st.subheader("Step 5 — Solve the model and print the optimal plan")
    st.markdown(
        "We pass objective and constraints to cvxpy and call `.solve()`. "
        "The solver (CLARABEL) finds the globally optimal solution in milliseconds."
    )
    with st.expander("📄 Show code"):
        st.code(
            "prob = cp.Problem(cp.Maximize(profit @ x), constraints)\n"
            "prob.solve(solver=cp.CLARABEL)\n"
            "\n"
            "print(f'Status:  {prob.status}')\n"
            "print(f'Profit:  {prob.value:.2f} €')\n"
            "print(f'Plan:    A={x.value[0]:.1f}  B={x.value[1]:.1f}  C={x.value[2]:.1f}')",
            language="python",
        )

    if x_t1 is not None:
        st.metric("Maximum Total Profit", f"{val_t1:.2f} €")
        col_chart, col_nums = st.columns([2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            bars = ax.bar(PRODUCTS, x_t1, color=COLORS, edgecolor="black", width=0.5)
            for bar, v in zip(bars, x_t1):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                )
            ax.set_xlabel("Product")
            ax.set_ylabel("Production quantity (units)")
            ax.set_title("Optimal Production Plan")
            ax.grid(axis="y", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col_nums:
            st.markdown("**Optimal quantities**")
            for prod, val in zip(PRODUCTS, x_t1):
                st.metric(f"x_{prod}", f"{val:.1f} units")
    else:
        st.error("Solver returned infeasible.")

    st.divider()

    # ── Task 2 ────────────────────────────────────────────────────────────────
    st.header("Task 2 — Structured Output & Verification")
    st.markdown(
        "We inspect the solution in detail: production quantities, machine utilisation, "
        "active demand bounds, and a formal plausibility check."
    )

    if x_t1 is not None:
        tol = 1e-6
        used_m1 = A1 @ x_t1
        used_m2 = A2 @ x_t1

        col_t2a, col_t2b = st.columns(2)

        with col_t2a:
            st.markdown("**Optimal production quantities & active demand bounds**")
            st.caption("Active = demand constraint is binding (no slack left)")
            df_prod = pd.DataFrame(
                {
                    "Quantity (units)": x_t1.round(2),
                    "Max Demand":       D_MAX.astype(int),
                    "Unused demand":    (D_MAX - x_t1).round(2),
                    "Active?":          ["✓" if abs(x_t1[i] - D_MAX[i]) < tol else "—" for i in range(3)],
                },
                index=PRODUCTS,
            )
            st.dataframe(df_prod, use_container_width=True)

        with col_t2b:
            st.markdown("**Machine utilisation — used vs. unused capacity**")
            st.caption("Active = machine is running at full capacity")
            df_mach = pd.DataFrame(
                {
                    "Used (h)":     [round(used_m1, 2), round(used_m2, 2)],
                    "Capacity (h)": [int(C1), int(C2)],
                    "Unused (h)":   [round(C1 - used_m1, 2), round(C2 - used_m2, 2)],
                    "Active?":      [
                        "✓" if C1 - used_m1 < tol else "—",
                        "✓" if C2 - used_m2 < tol else "—",
                    ],
                },
                index=["M1", "M2"],
            )
            st.dataframe(df_mach, use_container_width=True)

        all_ok = (
            used_m1 <= C1 + tol
            and used_m2 <= C2 + tol
            and np.all(x_t1 <= D_MAX + tol)
            and np.all(x_t1 >= -tol)
        )
        if all_ok:
            st.success("✓ All constraints satisfied (numerical tolerance 1e-6)")
        else:
            st.error("✗ Constraint violation detected!")

        with st.expander("📄 Show verification code"):
            st.code(
                "# Numerical tolerance: solver results are never exactly 100.0,\n"
                "# but e.g. 99.9999998 — we allow a tiny gap of 1e-6\n"
                "tol = 1e-6\n"
                "\n"
                "# a1 = [2, 3, 1]: machine hours per unit on M1\n"
                "# a1 @ x.value = 2*x_A + 3*x_B + 1*x_C  →  total hours used on M1\n"
                "used_m1  = a1 @ x.value\n"
                "used_m2  = a2 @ x.value\n"
                "\n"
                "# How many hours are left unused on M1\n"
                "# slack = 0 means the machine is running at full capacity (constraint is active/binding)\n"
                "slack_m1 = c1 - used_m1\n"
                "slack_m2 = c2 - used_m2\n"
                "\n"
                "# For each product: is the produced quantity (almost) equal to max demand?\n"
                "# True → demand constraint is active (no slack left in the market)\n"
                "active_demand = np.abs(x.value - d_max) < tol\n"
                "\n"
                "# Final plausibility check: verify all four constraint groups hold\n"
                "all_ok = (\n"
                "    used_m1 <= c1 + tol           # M1 not overloaded\n"
                "    and used_m2 <= c2 + tol        # M2 not overloaded\n"
                "    and np.all(x.value <= d_max + tol)  # no product exceeds its demand limit\n"
                "    and np.all(x.value >= -tol)    # no negative production (sanity check)\n"
                ")",
                language="python",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Task 3 — Scenario Analysis")
    st.markdown(
        "Effect of changing **one parameter at a time**. "
        "Baseline uses the default values from the problem statement."
    )

    BASE_PROFIT = np.array([20.0, 25.0, 18.0])
    BASE_DMAX   = np.array([40.0, 30.0, 50.0])
    BASE_C1, BASE_C2 = 100.0, 80.0

    scenarios = {
        "Baseline":            (BASE_PROFIT.copy(), BASE_C1, BASE_C2, BASE_DMAX.copy()),
        "Higher Profit for C":   (np.array([20.0, 25.0, 30.0]), BASE_C1, BASE_C2, BASE_DMAX.copy()),
        "More Capacity on Machine 2":   (BASE_PROFIT.copy(), BASE_C1, 110.0, BASE_DMAX.copy()),
        "Lower Demand for B":   (BASE_PROFIT.copy(), BASE_C1, BASE_C2, np.array([40.0, 10.0, 50.0])),
    }

    records = []
    for name, (p, c1_s, c2_s, d) in scenarios.items():
        val, x = solve_lp(p, c1_s, c2_s, d)
        records.append(
            {
                "Scenario": name,
                "x_A": round(x[0], 2) if x is not None else None,
                "x_B": round(x[1], 2) if x is not None else None,
                "x_C": round(x[2], 2) if x is not None else None,
                "Total Profit (€)": round(val, 2) if val is not None else None,
                "Δ vs Baseline (€)": None,
            }
        )

    base_val = records[0]["Total Profit (€)"]
    for r in records:
        if r["Total Profit (€)"] is not None:
            r["Δ vs Baseline (€)"] = round(r["Total Profit (€)"] - base_val, 2)

    df_scen = pd.DataFrame(records).set_index("Scenario")
    st.dataframe(df_scen, use_container_width=True)

    # Comparison charts
    fig3, (ax_profit, ax_delta) = plt.subplots(1, 2, figsize=(11, 4))
    names    = list(scenarios.keys())
    profits  = [r["Total Profit (€)"] for r in records]
    deltas   = [r["Δ vs Baseline (€)"] for r in records]
    bar_cols = ["#607D8B", "#1976D2", "#388E3C", "#F57C00"]

    ax_profit.bar(names, profits, color=bar_cols, edgecolor="black")
    ax_profit.axhline(base_val, color="red", linestyle="--", linewidth=1.5,
                      label=f"Baseline = {base_val:.0f} €")
    ax_profit.set_ylabel("Total Profit (€)")
    ax_profit.set_title("Profit by Scenario")
    ax_profit.tick_params(axis="x", rotation=18)
    ax_profit.legend()
    ax_profit.grid(axis="y", alpha=0.4)

    delta_cols = ["#607D8B"] + [
        "#388E3C" if d > 0 else "#F44336" for d in deltas[1:]
    ]
    ax_delta.bar(names, deltas, color=delta_cols, edgecolor="black")
    ax_delta.axhline(0, color="black", linewidth=0.8)
    ax_delta.set_ylabel("Δ Profit vs Baseline (€)")
    ax_delta.set_title("Profit Change vs Baseline")
    ax_delta.tick_params(axis="x", rotation=18)
    ax_delta.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    best = max(records[1:], key=lambda r: r["Δ vs Baseline (€)"] or 0)
    st.info(
        f"**Strongest effect:** {best['Scenario']}  →  "
        f"+{best['Δ vs Baseline (€)']:.2f} € compared to baseline"
    )

    with st.expander("📄 Show Python code"):
        st.code(
            """\
baseline_value = prob.value

scenarios = {
    "Baseline":           (profit,                   c1,   c2,   d_max),
    "Higher Profit for C":  (np.array([20.,25.,30.]), c1,   c2,   d_max),
    "More Capacity on Machine 2":   (profit,                   c1,   110., d_max),
    "Lower Demand for B":  (profit,                   c1,   c2,   np.array([40.,10.,50.])),
}

for name, (p, c1_s, c2_s, d) in scenarios.items():
    x = cp.Variable(3, nonneg=True)
    prob = cp.Problem(cp.Maximize(p @ x),
                      [a1 @ x <= c1_s, a2 @ x <= c2_s, x <= d])
    prob.solve()
    delta = prob.value - baseline_value
""",
            language="python",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Task 4 — Multi-Period Production Planning with Inventory")
    st.markdown(
        "The single-period LP is extended to two periods. Production uses machine capacity, "
        "sales are bounded by period-specific maximum demand, and unsold production can be "
        "stored after period 1 and sold in period 2."
    )
    st.divider()

    d_t1 = np.array([20.0, 15.0, 30.0])
    d_t2 = np.array([25.0, 20.0, 35.0])
    profit_t4 = np.array([20.0, 25.0, 18.0])

    col_setup, col_model = st.columns([1.05, 1.15], gap="large")

    with col_setup:
        st.subheader("Fixed Task Data")
        df_t4_data = pd.DataFrame(
            {
                "Profit (€/unit)": profit_t4,
                "M1 (h/unit)": A1,
                "M2 (h/unit)": A2,
                "Max demand P1": d_t1.astype(int),
                "Max demand P2": d_t2.astype(int),
            },
            index=PRODUCTS,
        )
        st.dataframe(df_t4_data, use_container_width=True)
        st.caption("Machine capacities in each period: M1 = 100 h, M2 = 80 h")

        hc = st.slider(
            "Inventory holding cost h (€/unit/period)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.25,
        )

    with col_model:
        st.subheader("Model")
        st.markdown(
            "**Decision variables:** production $x$, sales $y$, and inventory $s$ "
            "for all products and both periods."
        )
        st.latex(r"x_t, y_t, s_t \in \mathbb{R}_{\geq 0}^{3}, \qquad t \in \{1,2\}")
        st.latex(r"0 \leq y_1 \leq d^{(1)}, \qquad 0 \leq y_2 \leq d^{(2)}")
        st.latex(r"s_1 = x_1 - y_1")
        st.latex(r"s_2 = s_1 + x_2 - y_2")
        st.latex(
            r"\max\; p_{\mathrm{profit}}^\top y_1 + p_{\mathrm{profit}}^\top y_2 "
            r"- h\,\mathbf{1}^\top s_1 - h\,\mathbf{1}^\top s_2"
        )
        st.caption(
            "This demo additionally imposes s₂ = 0. Therefore the final-inventory cost term is "
            "included in the notation but evaluates to zero in the solved instance."
        )

    st.divider()

    st.subheader("Optimal Two-Period Plan")

    col_results = st.container()

    x_var = cp.Variable((3, 2), nonneg=True)   # production x_i,t
    y_var = cp.Variable((3, 2), nonneg=True)   # sales y_i,t
    s_var = cp.Variable((3, 2), nonneg=True)   # inventory s_i,t

    obj4 = cp.Maximize(
        profit_t4 @ y_var[:, 0]
        + profit_t4 @ y_var[:, 1]
        - hc * cp.sum(s_var[:, 0])
        - hc * cp.sum(s_var[:, 1])
    )
    cons4 = [
        A1 @ x_var[:, 0] <= 100.0, A2 @ x_var[:, 0] <= 80.0,   # machine P1
        A1 @ x_var[:, 1] <= 100.0, A2 @ x_var[:, 1] <= 80.0,   # machine P2
        y_var[:, 0] <= d_t1,                                   # demand upper bound P1
        y_var[:, 1] <= d_t2,                                   # demand upper bound P2
        s_var[:, 0] == x_var[:, 0] - y_var[:, 0],              # inventory balance P1
        s_var[:, 1] == s_var[:, 0] + x_var[:, 1] - y_var[:, 1], # inventory balance P2
        s_var[:, 1] == 0,                                      # no unsold stock after horizon
    ]
    prob4 = cp.Problem(obj4, cons4)
    prob4.solve(solver=cp.CLARABEL)

    with col_results:
        if prob4.status == "optimal":
            # Clip numerical noise from interior-point solver (values < 1e-4 are effectively 0)
            x4 = np.where(np.abs(x_var.value) < 1e-4, 0.0, x_var.value)
            y4 = np.where(np.abs(y_var.value) < 1e-4, 0.0, y_var.value)
            s4 = np.where(np.abs(s_var.value) < 1e-4, 0.0, s_var.value)

            total_revenue = float(profit_t4 @ y4[:, 0] + profit_t4 @ y4[:, 1])
            total_holding = float(hc * np.sum(s4[:, 0]))
            carryover_units = float(np.sum(s4[:, 0]))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Objective", f"{prob4.value:.2f} €")
            m2.metric("Sales revenue", f"{total_revenue:.2f} €")
            m3.metric("Holding cost", f"{total_holding:.2f} €")
            m4.metric("Carryover s₁", f"{carryover_units:.2f} units")

            fig4, axes = plt.subplots(1, 3, figsize=(12, 3.8))
            idx = np.arange(3)
            w = 0.35

            axes[0].bar(idx - w/2, x4[:, 0], w, label="Period 1", color="#1976D2", edgecolor="black")
            axes[0].bar(idx + w/2, x4[:, 1], w, label="Period 2", color="#90CAF9", edgecolor="black")
            axes[0].set_xticks(idx); axes[0].set_xticklabels(PRODUCTS)
            axes[0].set_ylabel("Units"); axes[0].set_title("Production x")
            axes[0].legend(); axes[0].grid(axis="y", alpha=0.4)

            axes[1].bar(idx - w/2, y4[:, 0], w, label="Period 1", color="#388E3C", edgecolor="black")
            axes[1].bar(idx + w/2, y4[:, 1], w, label="Period 2", color="#A5D6A7", edgecolor="black")
            axes[1].set_xticks(idx); axes[1].set_xticklabels(PRODUCTS)
            axes[1].set_ylabel("Units"); axes[1].set_title("Sales y")
            axes[1].legend(); axes[1].grid(axis="y", alpha=0.4)

            axes[2].bar(idx - w/2, s4[:, 0], w, label="After P1", color="#F57C00", edgecolor="black")
            axes[2].bar(idx + w/2, s4[:, 1], w, label="After P2", color="#FFCC80", edgecolor="black")
            axes[2].set_xticks(idx); axes[2].set_xticklabels(PRODUCTS)
            axes[2].set_ylabel("Units"); axes[2].set_title("Inventory s")
            axes[2].legend(); axes[2].grid(axis="y", alpha=0.4)
            if s4.max() < 0.01:
                axes[2].set_ylim(0, 1)  # show empty chart clearly when no inventory

            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close()

            c4a, c4b, c4c = st.columns(3)
            with c4a:
                st.markdown("**Production x**")
                st.dataframe(pd.DataFrame(x4.round(2), index=PRODUCTS, columns=["P1", "P2"]), use_container_width=True)
            with c4b:
                st.markdown("**Sales y**")
                st.dataframe(pd.DataFrame(y4.round(2), index=PRODUCTS, columns=["P1", "P2"]), use_container_width=True)
            with c4c:
                st.markdown("**Inventory s**")
                st.dataframe(pd.DataFrame(s4.round(2), index=PRODUCTS, columns=["After P1", "After P2"]), use_container_width=True)

            cap_rows = []
            for period_idx, label in enumerate(["P1", "P2"]):
                used_m1 = float(A1 @ x4[:, period_idx])
                used_m2 = float(A2 @ x4[:, period_idx])
                cap_rows.extend(
                    [
                        {
                            "Period": label,
                            "Machine": "M1",
                            "Used (h)": round(used_m1, 2),
                            "Capacity (h)": 100.0,
                            "Unused (h)": round(100.0 - used_m1, 2),
                        },
                        {
                            "Period": label,
                            "Machine": "M2",
                            "Used (h)": round(used_m2, 2),
                            "Capacity (h)": 80.0,
                            "Unused (h)": round(80.0 - used_m2, 2),
                        },
                    ]
                )
            st.markdown("**Capacity check**")
            st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)

            if carryover_units < 1e-4:
                st.info(
                    "No carryover is used in this scenario. That is a valid optimum: "
                    "with these parameters, selling from each period's own production is better "
                    "than producing early and paying holding costs."
                )
            elif hc == 0:
                st.info(
                    f"Carryover is used: {carryover_units:.2f} units are produced in period 1 "
                    "and sold in period 2. With zero holding cost, several optimal plans can have "
                    "the same objective value."
                )
            else:
                st.info(
                    f"Carryover is used: {carryover_units:.2f} units are produced in period 1 "
                    "and sold in period 2."
                )

            tol4 = 1e-4
            bal_ok = (
                np.allclose(s4[:, 0], x4[:, 0] - y4[:, 0], atol=tol4)
                and np.allclose(s4[:, 1], s4[:, 0] + x4[:, 1] - y4[:, 1], atol=tol4)
                and np.allclose(s4[:, 1], 0.0, atol=tol4)
            )
            if bal_ok:
                st.success("✓ Inventory balance equations satisfied and final inventory is zero")
            else:
                st.warning("⚠ Inventory balance check failed")

            with st.expander("📄 Show full solve code"):
                st.code(
                    "d_t1 = np.array([20., 15., 30.])  # max demand period 1 (upper bound)\n"
                    "d_t2 = np.array([25., 20., 35.])  # max demand period 2 (upper bound)\n"
                    "holding_cost = 1.0\n"
                    "\n"
                    "x = cp.Variable((3, 2), nonneg=True)  # x[i,t]: units produced\n"
                    "y = cp.Variable((3, 2), nonneg=True)  # y[i,t]: units sold (≤ demand)\n"
                    "s = cp.Variable((3, 2), nonneg=True)  # s[i,t]: inventory after period t\n"
                    "\n"
                    "prob = cp.Problem(\n"
                    "    cp.Maximize(\n"
                    "        profit @ y[:, 0] + profit @ y[:, 1]  # revenue from sales\n"
                    "        - holding_cost * cp.sum(s[:, 0])      # inventory cost after P1\n"
                    "        - holding_cost * cp.sum(s[:, 1])      # inventory cost after P2\n"
                    "    ),\n"
                    "    [\n"
                    "        a1 @ x[:, 0] <= 100,  a2 @ x[:, 0] <= 80,  # machine P1\n"
                    "        a1 @ x[:, 1] <= 100,  a2 @ x[:, 1] <= 80,  # machine P2\n"
                    "        y[:, 0] <= d_t1,           # sales upper bound P1\n"
                    "        y[:, 1] <= d_t2,           # sales upper bound P2\n"
                    "        s[:, 0] == x[:, 0] - y[:, 0],             # balance P1\n"
                    "        s[:, 1] == s[:, 0] + x[:, 1] - y[:, 1],  # balance P2\n"
                    "        s[:, 1] == 0,                            # no unsold stock after horizon\n"
                    "    ]\n"
                    ")\n"
                    "prob.solve()\n"
                    "\n"
                    "print('Production x:', x.value.round(2))\n"
                    "print('Sales y:     ', y.value.round(2))\n"
                    "print('Inventory plan s:')\n"
                    "print(s.value.round(2))",
                    language="python",
                )
        else:
            st.error(
                f"Solver status: **{prob4.status}**. Check the parameter values or try another solver."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Task 5 — Automatic Experiment Framework")
    st.markdown(
        "Define any parameter set, run it, and results accumulate in the table. "
        "Or load the 6 default experiments to get started instantly."
    )

    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    # ── Pre-load defaults ─────────────────────────────────────────────────────
    if st.button("⚡ Load 6 default experiments"):
        defaults = [
            ("Baseline",         [20, 25, 18], 100, 80,  [40, 30, 50]),
            ("High profit C",    [20, 25, 30], 100, 80,  [40, 30, 50]),
            ("Tight M1 (50h)",   [20, 25, 18],  50, 80,  [40, 30, 50]),
            ("Demand doubled",   [20, 25, 18], 100, 80,  [80, 60, 100]),
            ("Only A matters",   [20,  1,  1], 100, 80,  [40, 30, 50]),
            ("High capacity",    [20, 25, 18], 220, 170, [40, 30, 50]),
        ]
        for name, p, c1_d, c2_d, d in defaults:
            val, x = solve_lp(
                np.array(p, dtype=float), c1_d, c2_d, np.array(d, dtype=float)
            )
            st.session_state.experiments.append(
                {
                    "Name": name,
                    "Profit A/B/C": "/".join(map(str, p)),
                    "Cap M1/M2": f"{c1_d}/{c2_d}",
                    "Demand A/B/C": "/".join(map(str, d)),
                    "x_A": round(x[0], 2) if x is not None else None,
                    "x_B": round(x[1], 2) if x is not None else None,
                    "x_C": round(x[2], 2) if x is not None else None,
                    "Total Profit (€)": round(val, 2) if val is not None else None,
                    "Status": "optimal" if val is not None else "infeasible",
                }
            )

    # ── Custom experiment form ────────────────────────────────────────────────
    with st.expander("➕ Add custom experiment"):
        with st.form("exp_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("**Profit (€/unit)**")
                e_pA = st.number_input("Profit A", value=20.0, step=1.0)
                e_pB = st.number_input("Profit B", value=25.0, step=1.0)
                e_pC = st.number_input("Profit C", value=18.0, step=1.0)
            with f2:
                st.markdown("**Machine capacity (h)**")
                e_c1 = st.number_input("Machine 1", value=100.0, step=10.0)
                e_c2 = st.number_input("Machine 2", value=80.0, step=10.0)
            with f3:
                st.markdown("**Max. demand**")
                e_dA = st.number_input("Demand A", value=40.0, step=5.0)
                e_dB = st.number_input("Demand B", value=30.0, step=5.0)
                e_dC = st.number_input("Demand C", value=50.0, step=5.0)

            e_name = st.text_input(
                "Experiment name",
                value=f"Exp {len(st.session_state.experiments) + 1}",
            )
            if st.form_submit_button("▶ Run & add to table"):
                val, x = solve_lp(
                    np.array([e_pA, e_pB, e_pC]),
                    e_c1, e_c2,
                    np.array([e_dA, e_dB, e_dC]),
                )
                st.session_state.experiments.append(
                    {
                        "Name": e_name,
                        "Profit A/B/C": f"{e_pA:.0f}/{e_pB:.0f}/{e_pC:.0f}",
                        "Cap M1/M2": f"{e_c1:.0f}/{e_c2:.0f}",
                        "Demand A/B/C": f"{e_dA:.0f}/{e_dB:.0f}/{e_dC:.0f}",
                        "x_A": round(x[0], 2) if x is not None else None,
                        "x_B": round(x[1], 2) if x is not None else None,
                        "x_C": round(x[2], 2) if x is not None else None,
                        "Total Profit (€)": round(val, 2) if val is not None else None,
                        "Status": "optimal" if val is not None else "infeasible",
                    }
                )
                if val is not None:
                    st.success(f"Added '{e_name}' → profit = {val:.2f} €")
                else:
                    st.warning(f"Added '{e_name}' → infeasible")

    # ── Results table & chart ─────────────────────────────────────────────────
    if st.session_state.experiments:
        df_exp = pd.DataFrame(st.session_state.experiments)
        st.subheader(f"Results — {len(df_exp)} experiments")
        st.dataframe(df_exp, use_container_width=True)

        # Summary bar chart
        valid = df_exp[df_exp["Total Profit (€)"].notna()]
        if len(valid) >= 2:
            fig5, ax5 = plt.subplots(figsize=(max(6, len(valid) * 1.3), 4))
            bars5 = ax5.bar(
                valid["Name"], valid["Total Profit (€)"],
                color="steelblue", edgecolor="black",
            )
            for bar, v in zip(bars5, valid["Total Profit (€)"]):
                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9,
                )
            ax5.set_ylabel("Total Profit (€)")
            ax5.set_title("Experiment Results Summary")
            ax5.tick_params(axis="x", rotation=30)
            ax5.grid(axis="y", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close()

        if st.button("🗑 Clear all experiments"):
            st.session_state.experiments = []
            st.rerun()
    else:
        st.info(
            "No experiments yet — click **Load 6 default experiments** "
            "or add your own via the form above."
        )

    with st.expander("📄 Show Python code"):
        st.code(
            """\
def run_experiment(name, profit, c1, c2, d_max):
    x = cp.Variable(3, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(np.array(profit) @ x),
        [a1 @ x <= c1, a2 @ x <= c2, x <= np.array(d_max)]
    )
    prob.solve()
    return {
        "name":   name,
        "status": prob.status,
        "profit": prob.value,
        "plan":   x.value,
    }

# Run 6 scenarios
results = [
    run_experiment("Baseline",        [20, 25, 18], 100, 80,  [40, 30, 50]),
    run_experiment("High profit C",   [20, 25, 30], 100, 80,  [40, 30, 50]),
    run_experiment("Tight M1 (50h)",  [20, 25, 18],  50, 80,  [40, 30, 50]),
    run_experiment("Demand doubled",  [20, 25, 18], 100, 80,  [80, 60, 100]),
    run_experiment("Only A matters",  [20,  1,  1], 100, 80,  [40, 30, 50]),
    run_experiment("High capacity",   [20, 25, 18], 220, 170, [40, 30, 50]),
]

for result in results:
    print(result["name"], result["status"], result["profit"], result["plan"])
""",
            language="python",
        )
