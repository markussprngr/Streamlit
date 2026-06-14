import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import streamlit as st

from perceptron_core import accuracy, make_synthetic_data, predict, train_perceptron


st.set_page_config(
    page_title="Session 6 | Perceptron Lab",
    page_icon="◩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17211d;
        --muted: #5c665f;
        --paper: #f4f0e7;
        --surface: #fffdf8;
        --surface-2: #eee9de;
        --mint: #ccebdc;
        --mint-dark: #145a46;
        --coral: #e95d45;
        --coral-soft: #f9d5ce;
        --amber: #f6dfaa;
        --amber-dark: #684700;
        --blue-soft: #dceaf3;
        --blue-dark: #214e68;
        --line: #c9c1b2;
    }
    .stApp {
        background:
            linear-gradient(rgba(23,33,29,.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23,33,29,.035) 1px, transparent 1px),
            var(--paper);
        background-size: 32px 32px;
        color: var(--ink);
        font-family: "Avenir Next", "Helvetica Neue", sans-serif;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: var(--ink);
    }
    h1, h2, h3 {
        color: var(--ink) !important;
        font-family: "Avenir Next", "Helvetica Neue", sans-serif !important;
        letter-spacing: -.035em;
    }
    h2 { margin-top: .4rem !important; }
    code, .stCode { font-family: "Menlo", "SFMono-Regular", monospace !important; }
    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 5px 14px rgba(23,33,29,.07);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: .78rem !important;
        font-weight: 700 !important;
        letter-spacing: .035em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }
    .hero {
        border: 2px solid var(--ink);
        padding: 2.2rem 2.4rem;
        margin: .3rem 0 1.7rem;
        background: linear-gradient(115deg, rgba(185,245,216,.92) 0 62%, rgba(255,107,82,.9) 62%);
        box-shadow: 9px 9px 0 var(--ink);
    }
    .hero-kicker {
        color: var(--ink);
        font-family: "Menlo", "SFMono-Regular", monospace;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .78rem;
    }
    .hero-title { font-size: clamp(2.4rem, 5vw, 4.7rem); line-height: .94; font-weight: 800; letter-spacing: -.065em; max-width: 850px; }
    .hero-sub { max-width: 680px; font-size: 1.05rem; margin-top: 1.1rem; }
    .formula-card {
        border-left: 7px solid var(--coral);
        background: var(--surface);
        padding: .9rem 1.2rem;
        margin: .6rem 0 1rem;
        box-shadow: 0 4px 12px rgba(23,33,29,.05);
    }
    .formula-label {
        color: var(--muted);
        font-family: "Menlo", "SFMono-Regular", monospace;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .06em;
        margin-bottom: -.3rem;
        text-transform: uppercase;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--line) !important;
        border-left: 7px solid var(--coral) !important;
        border-radius: 0 10px 10px 0 !important;
        box-shadow: 0 4px 12px rgba(23,33,29,.05);
    }
    .status-good, .status-warn {
        border-radius: 10px;
        padding: .9rem 1rem;
        margin-top: 1rem;
        font-size: .95rem;
    }
    .status-good {
        color: var(--mint-dark) !important;
        background: var(--mint);
        border: 1px solid #75b89c;
    }
    .status-warn {
        color: #752b20 !important;
        background: var(--coral-soft);
        border: 1px solid #de9387;
    }
    .status-good *, .status-warn * {
        color: inherit !important;
    }

    /* Tab navigation */
    div[data-baseweb="tab-list"] {
        display: grid !important;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .55rem;
        margin-bottom: 1.7rem;
        overflow: visible !important;
    }
    button[data-baseweb="tab"] {
        width: 100%;
        min-width: 0;
        min-height: 58px;
        height: auto;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(255,253,248,.82);
        color: var(--ink) !important;
        padding: .65rem .55rem;
        box-shadow: 0 3px 8px rgba(23,33,29,.04);
    }
    button[data-baseweb="tab"]:hover {
        border-color: var(--ink);
        background: var(--surface);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-color: var(--ink);
        background: var(--ink);
        box-shadow: 4px 4px 0 var(--coral);
    }
    button[data-baseweb="tab"] p {
        color: var(--ink) !important;
        font-size: clamp(.72rem, 1.05vw, .9rem) !important;
        font-weight: 700;
        line-height: 1.18;
        text-align: center;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #fffdf8 !important;
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Inputs and controls */
    [data-testid="stWidgetLabel"] p, label p {
        color: var(--ink) !important;
        font-weight: 700 !important;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input {
        color: var(--ink) !important;
        background: var(--surface) !important;
        border-color: var(--line) !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="input"] input {
        color: var(--ink) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: var(--coral) !important;
        border-color: var(--surface) !important;
    }
    [data-testid="stSlider"] > div > div > div > div {
        color: var(--muted) !important;
    }

    /* Tables: force a light neutral shell around Streamlit's canvas grid */
    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--surface);
        box-shadow: 0 5px 14px rgba(23,33,29,.06);
    }
    [data-testid="stDataFrame"] > div {
        background: var(--surface) !important;
    }
    [data-testid="stDataFrame"] button {
        color: var(--ink) !important;
        background: var(--surface) !important;
    }

    /* Alerts and expandable code */
    [data-testid="stAlertContainer"] {
        border-radius: 10px;
        box-shadow: none;
        border-width: 1px !important;
        border-style: solid !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
        color: var(--blue-dark) !important;
        background: #d9eaf4 !important;
        border-color: #8fb8ce !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
        color: var(--amber-dark) !important;
        background: #f8e3ad !important;
        border-color: #d2aa45 !important;
    }
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
        color: #0c4f3b !important;
        background: #bfe4d1 !important;
        border-color: #65a98b !important;
    }
    [data-testid="stAlertContainer"] p,
    [data-testid="stAlertContainer"] span,
    [data-testid="stAlertContainer"] svg {
        color: inherit !important;
        fill: currentColor !important;
    }
    [data-testid="stAlertContainer"] p {
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: rgba(255,253,248,.72);
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: var(--ink) !important;
    }
    [data-testid="stCode"] {
        border-radius: 8px;
    }
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    @media (max-width: 900px) {
        div[data-baseweb="tab-list"] {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .hero { padding: 1.6rem; }
    }
    @media (max-width: 600px) {
        div[data-baseweb="tab-list"] {
            grid-template-columns: 1fr;
        }
        button[data-baseweb="tab"] { min-height: 46px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NAMES = np.array(["Müller", "Kroos", "Reus", "Gomez"])
X_TRAIN = np.array([[10.0, 0.1], [2.0, 0.7], [6.0, 0.6], [8.0, 0.1]])
Y_TRAIN = np.array([1, -1, 1, -1])
GOETZE = np.array([[8.0, 0.4]])
COLORS = {-1: "#ff6b52", 1: "#177e68"}


def show_notebook_code(title, code):
    """Show a compact Jupyter-style solution without Streamlit-specific code."""
    with st.expander(f"Notebook code · {title}"):
        st.caption(
            "Compact presentation version matching the calculation shown above. "
            "Interactive Streamlit controls are intentionally omitted."
        )
        st.code(code.strip(), language="python")


def boundary_plot(
    X,
    y,
    weights=None,
    bias=0.0,
    names=None,
    new_point=None,
    title="Feature plane",
    show_regions=False,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#fffdf7")

    x_pad = max(0.8, 0.12 * np.ptp(X[:, 0]))
    y_pad = max(0.08, 0.18 * np.ptp(X[:, 1]))
    x_limits = (X[:, 0].min() - x_pad, X[:, 0].max() + x_pad)
    y_limits = (X[:, 1].min() - y_pad, X[:, 1].max() + y_pad)

    if weights is not None and show_regions:
        xx, yy = np.meshgrid(
            np.linspace(*x_limits, 220),
            np.linspace(*y_limits, 220),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = predict(grid, weights, bias).reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[-1.5, 0, 1.5],
            cmap=ListedColormap(["#ffd6cf", "#c8f4df"]),
            alpha=0.55,
        )

    for label, marker, class_name in [(-1, "s", "Class −1"), (1, "o", "Class +1")]:
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=COLORS[label],
            marker=marker,
            s=115,
            edgecolor="#17211d",
            linewidth=1.2,
            label=class_name,
            zorder=4,
        )

    if names is not None:
        for point, name in zip(X, names):
            ax.annotate(name, point, xytext=(7, 7), textcoords="offset points", fontsize=9)

    if weights is not None:
        if abs(weights[1]) > 1e-12:
            x_line = np.linspace(*x_limits, 200)
            y_line = -(weights[0] * x_line + bias) / weights[1]
            ax.plot(x_line, y_line, color="#17211d", linewidth=2.3, label="Decision boundary")
        elif abs(weights[0]) > 1e-12:
            ax.axvline(-bias / weights[0], color="#17211d", linewidth=2.3, label="Decision boundary")

    if new_point is not None:
        ax.scatter(
            new_point[0, 0],
            new_point[0, 1],
            marker="*",
            s=260,
            c="#f0b429",
            edgecolor="#17211d",
            linewidth=1.2,
            label="Götze (unknown)",
            zorder=5,
        )
        ax.annotate("Götze", new_point[0], xytext=(7, 7), textcoords="offset points")

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_xlabel("Goals")
    ax.set_ylabel("PCR value")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True, facecolor="#fffdf7")
    fig.tight_layout()
    return fig


def error_plot(errors, title="Classification errors after each epoch"):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#fffdf7")
    epochs = np.arange(1, len(errors) + 1)
    ax.step(epochs, errors, where="mid", color="#ff6b52", linewidth=2.2)
    ax.scatter(epochs, errors, color="#17211d", s=24, zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassified samples E")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def training_history_plot(result, n_samples, title="Training history"):
    """Show both remaining errors and the updates made during each epoch."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#fffdf7")

    epochs = np.arange(0, len(result.errors) + 1)
    errors = np.r_[n_samples, result.errors]
    ax.plot(
        epochs,
        errors,
        color="#e95d45",
        marker="o",
        linewidth=2.2,
        label="Errors after epoch",
        zorder=3,
    )
    ax.bar(
        epochs[1:],
        result.updates_per_epoch,
        color="#177e68",
        alpha=0.45,
        label="Updates during epoch",
    )
    ax.set_xlabel("Epoch (0 = before training)")
    ax.set_ylabel("Number of samples")
    ax.set_title(title, loc="left", fontweight="bold")
    tick_step = max(1, int(np.ceil(len(result.errors) / 10)))
    ticks = np.arange(0, len(result.errors) + 1, tick_step)
    if ticks[-1] != len(result.errors):
        ticks = np.r_[ticks, len(result.errors)]
    ax.set_xticks(ticks)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True, facecolor="#fffdf7")
    fig.tight_layout()
    return fig


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">FH Aachen · OML · Practical Session 06</div>
      <div class="hero-title">Learning a line from mistakes.</div>
      <div class="hero-sub">
        Build the perceptron from scratch, watch the decision boundary move,
        and test exactly where linear classification stops working.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_data, tab_train, tab_experiments, tab_limits, tab_synthetic = st.tabs(
    [
        "01 · Data geometry",
        "02 · Train & predict",
        "03 · Sensitivity",
        "04 · Practical limits",
        "05 · Synthetic lab",
    ]
)

with tab_data:
    st.header("Task 1 · Data preparation and visualization")
    left, right = st.columns([0.9, 1.5], gap="large")
    with left:
        data = pd.DataFrame(
            {
                "Name": ["Müller", "Kroos", "Reus", "Gomez", "Götze"],
                "Goals": [10, 2, 6, 8, 8],
                "PCR value": [0.1, 0.7, 0.6, 0.1, 0.4],
                "Label": ["+1", "−1", "+1", "−1", "?"],
            }
        )
        st.dataframe(data, width="stretch", hide_index=True)
        st.markdown(
            '<div class="formula-card"><b>Score:</b> s(x) = wᵀx + b<br>'
            '<b>Prediction:</b> sign(s(x))</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "The first four samples form the training set. Götze is held back and classified only "
            "after training. Labels are encoded as `{−1, +1}`."
        )
    with right:
        fig = boundary_plot(X_TRAIN, Y_TRAIN, names=NAMES, new_point=GOETZE)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    show_notebook_code(
        "Task 1 · Prepare and visualize the data",
        """
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[10.0, 0.1], [2.0, 0.7],
              [ 6.0, 0.6], [8.0, 0.1]])
y = np.array([+1, -1, +1, -1])
x_goetze = np.array([[8.0, 0.4]])  # not used for training

for label, marker in [(-1, "s"), (+1, "o")]:
    mask = y == label
    plt.scatter(X[mask, 0], X[mask, 1],
                marker=marker, label=f"Class {label:+d}")

plt.scatter(*x_goetze[0], marker="*", s=180, label="Götze (?)")
plt.xlabel("Goals")
plt.ylabel("PCR value")
plt.legend()
plt.show()
""",
    )

with tab_train:
    st.header("Tasks 2 & 3 · Perceptron training, prediction, and boundary")
    result = train_perceptron(
        X_TRAIN,
        Y_TRAIN,
        learning_rate=1.0,
        initial_weights=np.zeros(2),
        initial_bias=0.0,
        max_epochs=1000,
        shuffle=False,
    )
    goetze_score = float((GOETZE @ result.weights + result.bias)[0])
    goetze_prediction = int(predict(GOETZE, result.weights, result.bias)[0])

    formulas, results = st.columns([0.95, 1.55], gap="large")
    with formulas:
        with st.container(border=True):
            st.markdown('<div class="formula-label">1 · Perceptron update</div>', unsafe_allow_html=True)
            st.latex(r"y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\leq 0")
            st.latex(
                r"\mathbf{w}\leftarrow\mathbf{w}+\alpha y_i\mathbf{x}_i,"
                r"\qquad b\leftarrow b+\alpha y_i"
            )
        with st.container(border=True):
            st.markdown('<div class="formula-label">2 · Augmented vectors</div>', unsafe_allow_html=True)
            st.latex(
                r"\tilde{\mathbf{x}}_i="
                r"\begin{pmatrix}\mathbf{x}_i\\1\end{pmatrix},"
                r"\qquad"
                r"\tilde{\mathbf{w}}="
                r"\begin{pmatrix}\mathbf{w}\\b\end{pmatrix}"
            )
        with st.container(border=True):
            st.markdown('<div class="formula-label">3 · Hyperplane</div>', unsafe_allow_html=True)
            st.latex(r"\mathbf{w}^{\top}\mathbf{x}+b=0")
            st.latex(
                r"\tilde{\mathbf{w}}\leftarrow"
                r"\tilde{\mathbf{w}}+\alpha y_i\tilde{\mathbf{x}}_i"
            )

        status_class = "status-good" if result.converged else "status-warn"
        status_text = "Converged" if result.converged else "Stopped at epoch limit"
        st.markdown(
            f'<div class="{status_class}"><b>{status_text}</b><br>'
            f'Fixed setup: α = 1, zero initialization, fixed order.<br>'
            f'{result.updates} updates across {result.epochs} epochs.</div>',
            unsafe_allow_html=True,
        )

    with results:
        m1, m2, m3 = st.columns(3)
        m1.metric("Updates", result.updates)
        m2.metric("Götze score", f"{goetze_score:.3f}")
        m3.metric("Götze prediction", f"{goetze_prediction:+d}")
        st.markdown("**Final augmented weight vector**")
        st.latex(
            rf"\tilde{{\mathbf{{w}}}}^*="
            rf"\begin{{pmatrix}}"
            rf"{result.augmented_weights[0]:g}\\"
            rf"{result.augmented_weights[1]:g}\\"
            rf"{result.augmented_weights[2]:g}"
            rf"\end{{pmatrix}}"
            rf"="
            rf"\begin{{pmatrix}}w_1\\w_2\\b\end{{pmatrix}}"
        )

        fig = boundary_plot(
            X_TRAIN,
            Y_TRAIN,
            result.weights,
            result.bias,
            names=NAMES,
            new_point=GOETZE,
            title="Learned linear classifier",
            show_regions=True,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    prediction_table = pd.DataFrame(
        {
            "Sample": NAMES,
            "True label": Y_TRAIN,
            "Score wᵀx+b": (X_TRAIN @ result.weights + result.bias).round(4),
            "Prediction": predict(X_TRAIN, result.weights, result.bias),
            "Correct?": predict(X_TRAIN, result.weights, result.bias) == Y_TRAIN,
        }
    )
    st.dataframe(prediction_table, width="stretch", hide_index=True)
    st.success(
        "The training data is linearly separable: the perceptron converges to a hyperplane that "
        "classifies every training sample correctly. The separating hyperplane is not unique."
    )

    show_notebook_code(
        "Tasks 2 & 3 · Train, predict, and plot the boundary",
        """
def augment_with_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((X, ones))


def perceptron_train(X, y, alpha=1.0, w0=None, max_epochs=1000, order=None):
    X_aug = augment_with_bias(X)
    w = np.zeros(X_aug.shape[1]) if w0 is None else np.array(w0, dtype=float).copy()
    update_history = []

    for epoch in range(max_epochs):
        errors = 0
        indices = np.arange(len(y)) if order is None else np.array(order)
        for idx in indices:
            margin = y[idx] * np.dot(w, X_aug[idx])
            if margin <= 0:
                w += alpha * y[idx] * X_aug[idx]
                errors += 1

        predictions = np.where(X_aug @ w >= 0, +1, -1)
        error_rate = np.mean(predictions != y)
        update_history.append(
            {"epoch": epoch + 1, "updates": errors,
             "error_rate": error_rate, "w": w.copy()}
        )
        if errors == 0:
            return w, True, pd.DataFrame(update_history)

    return w, False, pd.DataFrame(update_history)


w_star, converged, history_df = perceptron_train(X, y)
X_aug = augment_with_bias(X)
train_prediction = np.where(X_aug @ w_star >= 0, +1, -1)
goetze_aug = np.r_[x_goetze[0], 1.0]
goetze_prediction = int(np.sign(goetze_aug @ w_star) or 1)

x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
y_line = -(w_star[0] * x_line + w_star[2]) / w_star[1]
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.scatter(*x_goetze[0], marker="*", s=180, label="Götze")
plt.plot(x_line, y_line, "k-", label="Decision boundary")
plt.xlabel("Goals")
plt.ylabel("PCR value")
plt.legend()
plt.show()

print("Final augmented w =", w_star)
print("Training predictions:", train_prediction)
print("Götze:", goetze_prediction)
""",
    )

with tab_experiments:
    st.header("Task 4 · Influence of learning rate, initialization, and order")
    st.markdown(
        "Each row changes one or more algorithm settings. All runs use the same four training samples."
    )
    configurations = [
        ("Baseline", 1.0, [0.0, 0.0], 0.0, False, 0),
        ("Small α", 0.1, [0.0, 0.0], 0.0, False, 0),
        ("α = 0.5", 0.5, [0.0, 0.0], 0.0, False, 0),
        ("Positive init", 1.0, [0.1, 0.1], 0.1, False, 0),
        ("Tilted init", 1.0, [-0.5, 1.0], 0.0, False, 0),
        ("Shuffled · seed 3", 1.0, [0.0, 0.0], 0.0, True, 3),
        ("Shuffled · seed 17", 1.0, [0.0, 0.0], 0.0, True, 17),
    ]
    rows = []
    experiment_results = {}
    for name, lr, w0, b0, shuffled, random_seed in configurations:
        run = train_perceptron(
            X_TRAIN,
            Y_TRAIN,
            learning_rate=lr,
            initial_weights=w0,
            initial_bias=b0,
            max_epochs=600,
            shuffle=shuffled,
            random_state=random_seed,
        )
        experiment_results[name] = run
        rows.append(
            {
                "Configuration": name,
                "α": lr,
                "Initial w": str(w0),
                "Shuffled": shuffled,
                "Updates": run.updates,
                "Epochs": run.epochs,
                "Final w": np.array2string(run.weights, precision=3),
                "Final b": round(run.bias, 3),
                "Converged": run.converged,
                "Götze": int(predict(GOETZE, run.weights, run.bias)[0]),
            }
        )
    summary = pd.DataFrame(rows)
    st.dataframe(summary, width="stretch", hide_index=True)

    selected_run = st.selectbox("Inspect configuration", list(experiment_results))
    chosen = experiment_results[selected_run]
    chart_a, chart_b = st.columns(2)
    with chart_a:
        fig = boundary_plot(
            X_TRAIN,
            Y_TRAIN,
            chosen.weights,
            chosen.bias,
            names=NAMES,
            new_point=GOETZE,
            title=selected_run,
            show_regions=True,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with chart_b:
        fig = error_plot(chosen.errors, f"Training trajectory · {selected_run}")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.info(
            "The learning rate mainly scales the parameter vector when initialization is zero. "
            "Initialization and ordering can lead to genuinely different separating lines because "
            "the perceptron does not maximize a unique objective."
        )

    st.subheader("Recommended practical check · feature scaling")
    scaling_runs = []
    mean = X_TRAIN.mean(axis=0)
    std = X_TRAIN.std(axis=0)
    minimum = X_TRAIN.min(axis=0)
    span = X_TRAIN.max(axis=0) - minimum
    scaled_datasets = {
        "Raw features": (X_TRAIN, GOETZE),
        "Z-score standardization": ((X_TRAIN - mean) / std, (GOETZE - mean) / std),
        "Min-max scaling": ((X_TRAIN - minimum) / span, (GOETZE - minimum) / span),
    }
    for scaling_name, (X_scaled, goetze_scaled) in scaled_datasets.items():
        scaled_run = train_perceptron(X_scaled, Y_TRAIN, max_epochs=1000)
        scaling_runs.append(
            {
                "Representation": scaling_name,
                "Converged": scaled_run.converged,
                "Epochs": scaled_run.epochs,
                "Updates": scaled_run.updates,
                "Training accuracy": f"{100 * accuracy(X_scaled, Y_TRAIN, scaled_run.weights, scaled_run.bias):.0f}%",
                "Götze": int(predict(goetze_scaled, scaled_run.weights, scaled_run.bias)[0]),
            }
        )
    st.dataframe(pd.DataFrame(scaling_runs), width="stretch", hide_index=True)
    st.caption(
        "The lecture notes explicitly recommend comparable feature scales. Here, z-score "
        "standardization reduces the fixed-order run from hundreds of updates to only a few, "
        "while preserving the training result and Götze prediction."
    )
    show_notebook_code(
        "Task 4 · Compare learning settings",
        """
configs = [
    ("Baseline", 1.0, [0.0, 0.0], False, 0),
    ("Small alpha", 0.1, [0.0, 0.0], False, 0),
    ("Positive init", 1.0, [0.1, 0.1], False, 0),
    ("Shuffled", 1.0, [0.0, 0.0], True, 3),
]

rows = []
for name, alpha, w0, shuffle, seed in configs:
    run = train_perceptron(
        X, y, learning_rate=alpha, initial_weights=w0,
        max_epochs=600, shuffle=shuffle, random_state=seed
    )
    rows.append({
        "Configuration": name,
        "Updates": run.updates,
        "Final w": np.round(run.weights, 3),
        "Converged": run.converged,
    })

pd.DataFrame(rows)

# Optional scaling check recommended in the lecture notes
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
scaled_run = train_perceptron(X_standardized, y, max_epochs=1000)
print("Updates after standardization:", scaled_run.updates)
""",
    )

with tab_limits:
    st.header("Task 5 · Nonseparable data and practical limits")
    st.markdown(
        "Choose one of three single-feature modifications. Each example makes perfect linear "
        "separation impossible for a different geometric reason."
    )

    impossible_examples = {
        "1 · Contradictory duplicate": {
            "description": (
                "**Change:** Gomez's goals: `8 → 10`. Müller and Gomez then have identical "
                "features but opposite labels. The same point cannot receive both predictions."
            ),
            "change": (3, 0, 10.0),
            "title": "Identical coordinates with opposite labels",
        },
        "2 · Alternating points on one line": {
            "description": (
                "**Change:** Reus's PCR value: `0.6 → 0.1`. Reus, Gomez, and Müller lie on the "
                "horizontal line `PCR = 0.1` in the order `+1, −1, +1`. A straight boundary "
                "cannot separate this alternating pattern."
            ),
            "change": (2, 1, 0.1),
            "title": "Alternating labels on PCR = 0.1",
        },
        "3 · Crossing class geometry": {
            "description": (
                "**Change:** Gomez's PCR value: `0.1 → 0.4`. The line segment connecting the "
                "two positive samples crosses the segment connecting the two negative samples. "
                "Their convex hulls overlap, so no separating line exists."
            ),
            "change": (3, 1, 0.4),
            "title": "Overlapping class geometry",
        },
    }
    selected_example = st.selectbox(
        "Nonseparable modification",
        list(impossible_examples),
    )
    example = impossible_examples[selected_example]
    st.markdown(example["description"])

    X_IMPOSSIBLE = X_TRAIN.copy()
    row, column, new_value = example["change"]
    X_IMPOSSIBLE[row, column] = new_value
    impossible = train_perceptron(
        X_IMPOSSIBLE,
        Y_TRAIN,
        learning_rate=1.0,
        max_epochs=60,
        shuffle=False,
    )
    original = train_perceptron(X_TRAIN, Y_TRAIN, max_epochs=500)

    modified_data = pd.DataFrame(
        {
            "Name": NAMES,
            "Goals": X_IMPOSSIBLE[:, 0],
            "PCR value": X_IMPOSSIBLE[:, 1],
            "Label": Y_TRAIN,
        }
    )
    st.dataframe(modified_data, width="stretch", hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original converged", "Yes" if original.converged else "No")
    m2.metric("Modified converged", "Yes" if impossible.converged else "No")
    m3.metric("Updates after 60 epochs", impossible.updates)
    m4.metric(
        "Final modified accuracy",
        f"{100 * accuracy(X_IMPOSSIBLE, Y_TRAIN, impossible.weights, impossible.bias):.0f}%",
    )

    left, right = st.columns(2)
    with left:
        fig = boundary_plot(
            X_IMPOSSIBLE,
            Y_TRAIN,
            impossible.weights,
            impossible.bias,
            names=NAMES,
            title=example["title"],
            show_regions=True,
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        fig = error_plot(impossible.errors, "Errors do not settle at zero")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.warning(
            "The epoch limit is essential. Without it, the classic perceptron can update forever "
            "on nonseparable data. A pocket variant, logistic regression, or a soft-margin SVM is "
            "more appropriate in that setting."
        )
    show_notebook_code(
        "Task 5 · Try three nonseparable modifications",
        """
examples = {
    "duplicate": (3, 0, 10.0),   # Gomez goals: 8 -> 10
    "collinear": (2, 1, 0.1),    # Reus PCR: 0.6 -> 0.1
    "overlap": (3, 1, 0.4),      # Gomez PCR: 0.1 -> 0.4
}

row, column, new_value = examples["duplicate"]
X_impossible = X.copy()
X_impossible[row, column] = new_value

run = train_perceptron(
    X_impossible, y, learning_rate=1.0,
    max_epochs=60, shuffle=False
)

plt.step(range(1, len(run.errors) + 1), run.errors, where="mid")
plt.xlabel("Epoch")
plt.ylabel("Misclassified samples")
plt.show()

print("Converged:", run.converged)
print("Final accuracy:", accuracy(
    X_impossible, y, run.weights, run.bias
))
""",
    )

with tab_synthetic:
    st.header("Task 6 · Random synthetic data")
    st.markdown(
        "The nonseparable sample contains one pair of identical feature vectors with opposite "
        "labels. This guarantees nonseparability for every seed and sample-size setting."
    )
    seed_syn = st.slider("Synthetic random seed", 0, 100, 12)
    samples = st.slider("Samples per class", 10, 80, 30, 5)
    max_epochs_syn = st.slider("Maximum synthetic epochs", 10, 300, 100, 10)

    synthetic_rows = []
    syn_cols = st.columns(2)
    for column, kind, heading in zip(
        syn_cols,
        ["separable", "nonseparable"],
        ["Clearly separable", "Guaranteed nonseparable"],
    ):
        X_syn, y_syn = make_synthetic_data(kind, samples, seed_syn)
        run_syn = train_perceptron(
            X_syn,
            y_syn,
            learning_rate=0.5,
            max_epochs=max_epochs_syn,
            shuffle=True,
            random_state=seed_syn,
        )
        acc_syn = accuracy(X_syn, y_syn, run_syn.weights, run_syn.bias)
        synthetic_rows.append(
            {
                "Dataset": heading,
                "Converged": run_syn.converged,
                "Epochs": run_syn.epochs,
                "Updates": run_syn.updates,
                "Training accuracy": f"{100 * acc_syn:.1f}%",
                "Final w": np.array2string(run_syn.weights, precision=3),
                "Final b": round(run_syn.bias, 3),
            }
        )
        with column:
            st.subheader(heading)
            fig = boundary_plot(
                X_syn,
                y_syn,
                run_syn.weights,
                run_syn.bias,
                title=f"{heading} · accuracy {100 * acc_syn:.1f}%",
                show_regions=True,
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            fig = training_history_plot(
                run_syn,
                len(X_syn),
                f"Training history · {heading}",
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            if kind == "separable":
                st.caption(
                    "At epoch 0 all scores equal zero, so no sample is strictly classified. "
                    "The first training pass needs one update and already leaves zero errors; "
                    "the second pass confirms convergence with zero further updates."
                )

    st.dataframe(pd.DataFrame(synthetic_rows), width="stretch", hide_index=True)
    st.success(
        "Plausibility check: the separated Gaussian clusters converge to 100% training accuracy; "
        "the contradictory sample reaches the epoch limit and retains errors."
    )
    show_notebook_code(
        "Task 6 · Compare synthetic data sets",
        """
results = []
for kind in ["separable", "nonseparable"]:
    X_syn, y_syn = make_synthetic_data(
        kind, n_per_class=30, random_state=12
    )
    run = train_perceptron(
        X_syn, y_syn, learning_rate=0.5,
        max_epochs=100, shuffle=True, random_state=12
    )
    results.append({
        "Dataset": kind,
        "Converged": run.converged,
        "Updates": run.updates,
        "Accuracy": accuracy(
            X_syn, y_syn, run.weights, run.bias
        ),
    })

pd.DataFrame(results)
""",
    )
