import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional for partial plot
import statsmodels.api as sm

IN_PATH = "data/processed/analysis_ready_2019.csv"
OUT_DIR = "outputs/figures"


def style_ax(ax):
    ax.grid(True, which="major", linewidth=0.8, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_axisbelow(True)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def scatter_with_fit(x, y, xlabel, ylabel, title, out_path, jitter_x=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if jitter_x > 0:
        rng = np.random.default_rng(42)
        x_plot = x + rng.normal(0, jitter_x, size=x.shape)
    else:
        x_plot = x

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()

    ax.scatter(x_plot, y, alpha=0.75, s=55)

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, m * xx + b, linewidth=2.0)

        ax.text(
            0.02, 0.95,
            f"Slope: {m:.3f} pp activity per 1 pp SBA",
            transform=ax.transAxes,
            fontsize=11,
            va="top"
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, pad=12)

    style_ax(ax)
    savefig(fig, out_path)


def binned_means_plot(df, xcol, ycol, bins, xlabel, ylabel, title, out_path):
    df2 = df[[xcol, ycol]].dropna().copy()
    df2["bin"] = pd.cut(df2[xcol], bins=bins, include_lowest=True)

    grp = df2.groupby("bin")[ycol]
    means = grp.mean()
    counts = grp.size()

    fig = plt.figure(figsize=(8.5, 5.2))
    ax = plt.gca()

    ax.plot(range(len(means)), means.values, marker="o", linewidth=2.0)

    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([str(b) for b in means.index], rotation=25, ha="right", fontsize=10)

    for i, (m, n) in enumerate(zip(means.values, counts.values)):
        ax.text(i, m, f" n={int(n)}", fontsize=10, va="bottom")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, pad=12)

    style_ax(ax)
    savefig(fig, out_path)


def partial_relationship_plot(df, y, x, controls, xlabel, ylabel, title, out_path):
    """
    Residualize y and x on controls, then plot residuals with a fit line.
    """
    cols = [y, x] + controls
    d = df[cols].dropna().copy()

    Y = d[y].astype(float)
    X = d[x].astype(float)
    C = sm.add_constant(d[controls].astype(float))

    y_resid = sm.OLS(Y, C).fit().resid
    x_resid = sm.OLS(X, C).fit().resid

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()

    ax.scatter(x_resid, y_resid, alpha=0.75, s=55)

    if len(x_resid) >= 2:
        m, b = np.polyfit(x_resid, y_resid, 1)
        xx = np.linspace(x_resid.min(), x_resid.max(), 200)
        ax.plot(xx, m * xx + b, linewidth=2.0)
        ax.text(
            0.02, 0.95,
            f"Partial slope: {m:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            va="top"
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, pad=12)

    style_ax(ax)
    savefig(fig, out_path)


def main():
    df = pd.read_csv(IN_PATH)

    # --- FIG 1: Scatter (SBA vs Female activity) with jitter to reduce stacking
    scatter_with_fit(
        x=df["sba_pct"],
        y=df["female_activity_rate_pct"],
        xlabel="Births attended by skilled health personnel (%)",
        ylabel="Female physical activity rate (%)",
        title="Healthcare access and female physical activity (2019)",
        out_path=f"{OUT_DIR}/fig1_scatter_sba_activity_pretty.png",
        jitter_x=0.15,  # small jitter (in percentage points)
    )

    # --- FIG 2: Same relationship but without jitter (clean raw), with fit
    scatter_with_fit(
        x=df["sba_pct"],
        y=df["female_activity_rate_pct"],
        xlabel="Births attended by skilled health personnel (%)",
        ylabel="Female physical activity rate (%)",
        title="Linear association (2019)",
        out_path=f"{OUT_DIR}/fig2_scatter_line_pretty.png",
        jitter_x=0.0,
    )

    # --- FIG 3: GDP vs activity
    scatter_with_fit(
        x=df["log_gdp_pc"],
        y=df["female_activity_rate_pct"],
        xlabel="Log GDP per capita",
        ylabel="Female physical activity rate (%)",
        title="Economic development and female physical activity (2019)",
        out_path=f"{OUT_DIR}/fig3_scatter_gdp_activity_pretty.png",
        jitter_x=0.0,
    )

    # --- FIG 4: Binned means (recommended because SBA is clumped near 100)
    binned_means_plot(
        df=df,
        xcol="sba_pct",
        ycol="female_activity_rate_pct",
        bins=[0, 70, 80, 90, 95, 98, 100],
        xlabel="Skilled birth attendance (%) bins",
        ylabel="Mean female physical activity rate (%)",
        title="Average female activity by healthcare access level (2019)",
        out_path=f"{OUT_DIR}/fig4_binned_means_pretty.png",
    )

    # --- FIG 5: Partial relationship (controls for GDP)
    partial_relationship_plot(
        df=df,
        y="female_activity_rate_pct",
        x="sba_pct",
        controls=["log_gdp_pc"],
        xlabel="Healthcare access residual (net of GDP)",
        ylabel="Female activity residual (net of GDP)",
        title="Partial association controlling for GDP (2019)",
        out_path=f"{OUT_DIR}/fig5_partial_sba_gdp_pretty.png",
    )

    print("Saved pretty figures to outputs/figures/")


if __name__ == "__main__":
    main()
