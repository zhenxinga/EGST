import numpy as np
import functions as fn


def compute_acf(series, max_lag):
    series = np.nan_to_num(series.astype(np.float64))
    series = series - np.nanmean(series)
    var = np.nanvar(series)
    acf_vals = np.empty(max_lag, dtype=np.float64)

    for lag in range(1, max_lag + 1):
        valid = (
            ~np.isnan(series[:-lag])
            &
            ~np.isnan(series[lag:])
        )

        if valid.sum() == 0 or var == 0:
            acf_vals[lag - 1] = np.nan
        else:
            acf_vals[lag - 1] = (
                (
                    series[:-lag][valid]
                    *
                    series[lag:][valid]
                ).sum()
                /
                (valid.sum() * var)
            )

    return np.nan_to_num(acf_vals)


def compute_power_spectrum(series, interval_seconds=5 * 60):
    series = (
        np.nan_to_num(series.astype(np.float64))
        -
        np.nanmean(series)
    )

    fft_vals = np.fft.rfft(series)
    power = np.abs(fft_vals) ** 2

    freqs = np.fft.rfftfreq(
        len(series),
        d=interval_seconds
    )

    idx = freqs > 0
    freqs = freqs[idx]
    power = power[idx]

    periods = 1.0 / freqs / 3600.0

    mask = (
        np.isfinite(power)
        &
        np.isfinite(periods)
    )

    return periods[mask], power[mask]


def main():
    occ, prc, adj, col, dis, cap, time_idx, inf = fn.read_dataset()

    T, N = occ.shape

    print(f"Loaded occupancy: T={T}, N={N}")

    max_lag = 288 * 7

    for node in range(N):
        series = occ[:, node]

        if (
            np.all(np.isnan(series))
            or
            np.all(series == 0)
        ):
            print(f"Station {node}: 全0/NaN，跳过")
            continue

        acf_vals = compute_acf(
            series=series,
            max_lag=max_lag
        )

        periods, power = compute_power_spectrum(
            series=series
        )

        print(f"Station {node}: ACF and power spectrum computed")

    print("\n" + "=" * 60)
    print("所有站点计算完成！")
    print(f"站点数量: {N}")
    print("=" * 60)


if __name__ == "__main__":
    main()
