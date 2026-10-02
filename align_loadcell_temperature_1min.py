r"""按固定时间窗计算 LoadCell 与 RTDD2 温度的均值并对齐时间戳。

输入日期后，脚本按日期自动生成源文件和输出文件名。例如日期为
2026-09-23 时，默认使用：
    D:\CrystaLiZe\LoadCell\Data\2026-09-23_LoadCells.csv
    D:\CrystaLiZe\LoadCell\Temperature\2026-09-23_LakeShore.csv
    D:\CrystaLiZe\LoadCell\2026-09-23_LoadCell_RTDD5_1min.csv
2026-09-28 后，默认使用：
    D:\CrystaLiZe\LoadCell\2026-09-23_LoadCell_RTDD2_1min.csv

时间窗采用左闭右开区间，例如：
    00:00:00 <= timestamp < 00:01:00  ->  00:00:00
    00:01:00 <= timestamp < 00:02:00  ->  00:01:00

若某一分钟的某个数据源完全没有采样，对应输出单元格保持为空，
不会用 0 或插值值代替原始缺测。

固定筛选规则：
    LC1 < 0 的点不参与均值计算；
    LC2 < 2,000,000 的点不参与均值计算。

可选参数 --filter-3sigma 会在每个一分钟窗口内，分别对每个测量列
计算均值和总体标准差，并在计算窗口均值前剔除超出 mean +/- 3 sigma 的点。

命令行调用示例：
    python align_loadcell_temperature_1min.py --date 2026-09-23
    python align_loadcell_temperature_1min.py --date 2026-09-23 \
        --start-time 08:30 --end-time 12:00 --filter-3sigma

在 Jupyter Notebook 中导入后调用：
    main(["--date", "2026-09-23"])
    main(["--date", "2026-09-23", "--start-time", "08:30",
          "--end-time", "12:00", "--filter-3sigma"])

开始时间包含在输出中，结束时间不包含。例如 08:30 至 12:00 会输出
08:30、08:31、...、11:59 共 210 个一分钟窗口。结束时间允许写成 24:00。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = Path(r"D:\CrystaLiZe\LoadCell")

LOADCELL_TIME_COLUMN = "Timestamp"
LOADCELL_VALUE_COLUMNS = ["Raw LC1 (Mean)", "Raw LC2 (Mean)"]
TEMPERATURE_TIME_COLUMN = "Time"
TEMPERATURE_VALUE_COLUMN = "RTDD2 Temperature(K)"
LOADCELL_MINIMUMS = {
    "Raw LC1 (Mean)": 1500000,# 0.0,
    "Raw LC2 (Mean)": 2_000_000.0,
}


def parse_date(value: str) -> str:
    """解析严格的 YYYY-MM-DD 日期格式。"""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"日期必须使用 YYYY-MM-DD 格式，例如 2026-09-23；收到：{value}"
        ) from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise argparse.ArgumentTypeError(
            f"日期必须使用 YYYY-MM-DD 格式，例如 2026-09-23；收到：{value}"
        )
    return value


def parse_clock_time(value: str) -> str:
    """解析严格的 HH:MM 时间格式；额外允许 24:00 作为日终。"""
    if value == "24:00":
        return value
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"时间必须使用 HH:MM 格式，例如 08:30；收到：{value}"
        ) from exc
    if parsed.strftime("%H:%M") != value:
        raise argparse.ArgumentTypeError(
            f"时间必须使用 HH:MM 格式，例如 08:30；收到：{value}"
        )
    return value


def clock_to_minutes(value: str) -> int:
    if value == "24:00":
        return 24 * 60
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按一分钟窗口平均并对齐两道 LoadCell 数据与 RTDD2 温度数据。"
    )
    parser.add_argument(
        "--date",
        type=parse_date,
        required=True,
        help="待处理日期，格式为 YYYY-MM-DD，例如 2026-09-23。",
    )
    parser.add_argument(
        "--start-time",
        type=parse_clock_time,
        default="00:00",
        help="开始时间（包含），格式为 HH:MM；默认 00:00。",
    )
    parser.add_argument(
        "--end-time",
        type=parse_clock_time,
        default="24:00",
        help="结束时间（不包含），格式为 HH:MM；默认 24:00。",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"数据根目录（默认：{DEFAULT_BASE_DIR}）。",
    )
    parser.add_argument(
        "--loadcell",
        type=Path,
        default=None,
        help="可选：覆盖由日期自动生成的 LoadCell CSV 路径。",
    )
    parser.add_argument(
        "--temperature",
        type=Path,
        default=None,
        help="可选：覆盖由日期自动生成的 LakeShore CSV 路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：覆盖自动生成的输出 CSV 路径。",
    )
    parser.add_argument(
        "--filter-3sigma",
        action="store_true",
        help="启用每分钟窗口内、逐列的 3σ 异常点筛选（默认关闭）。",
    )
    if argv is None and "ipykernel" in sys.modules:
        # Jupyter 会自动添加形如 "-f <kernel.json>" 的内核参数。
        # 解析脚本认识的参数，并忽略这些由内核附加的参数。
        arguments, _unknown = parser.parse_known_args()
    else:
        arguments = parser.parse_args(argv)

    start_minutes = clock_to_minutes(arguments.start_time)
    end_minutes = clock_to_minutes(arguments.end_time)
    if start_minutes >= 24 * 60:
        parser.error("--start-time 必须早于 24:00。")
    if end_minutes <= start_minutes:
        parser.error("--end-time 必须晚于 --start-time；暂不支持跨日期时间段。")

    return arguments


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """根据日期和目录生成输入、输出路径，显式路径参数优先。"""
    loadcell_path = args.loadcell or (
        args.base_dir / "Data" / f"{args.date}_LoadCells.csv"
    )
    temperature_path = args.temperature or (
        args.base_dir / "Temperature" / f"{args.date}_LakeShore.csv"
    )

    if args.output is not None:
        output_path = args.output
    else:
        is_full_day = args.start_time == "00:00" and args.end_time == "24:00"
        time_suffix = ""
        if not is_full_day:
            start_label = args.start_time.replace(":", "")
            end_label = args.end_time.replace(":", "")
            time_suffix = f"_{start_label}-{end_label}"
        output_path = (
            args.base_dir
            / f"{args.date}_LoadCell_RTDD2_1min{time_suffix}.csv"
        )

    return loadcell_path, temperature_path, output_path


def build_time_bounds(date: str, start_time: str, end_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """把日期和分钟级首尾时间转为左闭右开的时间边界。"""
    day_start = pd.Timestamp(date)
    start = day_start + pd.Timedelta(minutes=clock_to_minutes(start_time))
    end = day_start + pd.Timedelta(minutes=clock_to_minutes(end_time))
    return start, end


def check_columns(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} 缺少必要列：{missing}")


def read_loadcell(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [LOADCELL_TIME_COLUMN, *LOADCELL_VALUE_COLUMNS]
    check_columns(frame, required, path)

    frame = frame[required].copy()
    frame[LOADCELL_TIME_COLUMN] = pd.to_datetime(
        frame[LOADCELL_TIME_COLUMN],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )
    for column in LOADCELL_VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame.set_index(LOADCELL_TIME_COLUMN).sort_index()


def read_temperature(path: Path) -> pd.DataFrame:
    required = [TEMPERATURE_TIME_COLUMN, TEMPERATURE_VALUE_COLUMN]
    frame = pd.read_csv(path, usecols=required)
    check_columns(frame, required, path)

    frame[TEMPERATURE_TIME_COLUMN] = pd.to_datetime(
        frame[TEMPERATURE_TIME_COLUMN],
        format="%Y-%m-%d %H_%M_%S",
        errors="raise",
    )
    frame[TEMPERATURE_VALUE_COLUMN] = pd.to_numeric(
        frame[TEMPERATURE_VALUE_COLUMN], errors="raise"
    )

    return frame.set_index(TEMPERATURE_TIME_COLUMN).sort_index()


def filter_loadcell_minimums(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """剔除低于各 LoadCell 固定下限的点，并返回逐列剔除数量。"""
    filtered = frame.copy()
    removed_counts: dict[str, int] = {}

    for column, minimum in LOADCELL_MINIMUMS.items():
        invalid = filtered[column] < minimum
        removed_counts[column] = int(invalid.sum())
        filtered.loc[invalid, column] = pd.NA

    return filtered, removed_counts


def filter_3sigma_per_minute(
    frame: pd.DataFrame,
    sigma: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """在每分钟窗口内逐列剔除超出 mean +/- sigma * std 的数据点。

    标准差使用该分钟窗口内有效数据点的总体标准差（ddof=0）。
    这是单次筛选，不进行迭代；标准差为 0 或窗口内仅有一个有效点时不剔除。
    """
    if sigma <= 0:
        raise ValueError("sigma 必须大于 0。")
    if frame.empty:
        return frame.copy(), {column: 0 for column in frame.columns}

    minute_keys = frame.index.floor("1min")
    grouped = frame.groupby(minute_keys)
    minute_means = grouped.transform("mean")
    minute_stds = grouped.transform(lambda values: values.std(ddof=0))

    outliers = frame.sub(minute_means).abs().gt(minute_stds.mul(sigma))
    outliers &= frame.notna() & minute_stds.gt(0)

    removed_counts = {
        column: int(outliers[column].sum()) for column in frame.columns
    }
    return frame.mask(outliers), removed_counts


def minute_mean(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name=frame.index.name),
            columns=frame.columns,
            dtype=float,
        )
    return frame.resample("1min", label="left", closed="left").mean()


def align_to_requested_timeline(
    loadcell_minute: pd.DataFrame,
    temperature_minute: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    # 严格按用户指定范围建立完整分钟时间轴。源数据缺测时保留空值。
    timeline = pd.date_range(start=start, end=end, freq="1min", inclusive="left")
    aligned = loadcell_minute.reindex(timeline).join(
        temperature_minute.reindex(timeline), how="left"
    )
    aligned.index.name = "Timestamp"
    return aligned


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    loadcell_path, temperature_path, output_path = resolve_paths(args)
    start, end = build_time_bounds(args.date, args.start_time, args.end_time)

    loadcell = read_loadcell(loadcell_path)
    temperature = read_temperature(temperature_path)

    loadcell = loadcell.loc[(loadcell.index >= start) & (loadcell.index < end)].copy()
    temperature = temperature.loc[
        (temperature.index >= start) & (temperature.index < end)
    ].copy()

    loadcell, minimum_removed = filter_loadcell_minimums(loadcell)

    sigma_removed = {
        **{column: 0 for column in LOADCELL_VALUE_COLUMNS},
        TEMPERATURE_VALUE_COLUMN: 0,
    }
    if args.filter_3sigma:
        loadcell, loadcell_sigma_removed = filter_3sigma_per_minute(loadcell)
        temperature, temperature_sigma_removed = filter_3sigma_per_minute(temperature)
        sigma_removed.update(loadcell_sigma_removed)
        sigma_removed.update(temperature_sigma_removed)

    aligned = align_to_requested_timeline(
        minute_mean(loadcell),
        minute_mean(temperature),
        start,
        end,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(
        output_path,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.15g",
        na_rep="",
    )

    complete_rows = int(aligned.notna().all(axis=1).sum())
    print(f"处理日期：{args.date}")
    print(f"时间范围：[{args.start_time}, {args.end_time})")
    print(f"LoadCell 文件：{loadcell_path}")
    print(f"温度文件：{temperature_path}")
    print(f"输出文件：{output_path}")
    print(f"分钟数：{len(aligned)}")
    print(f"三列数据均完整的分钟数：{complete_rows}")
    print("固定下限筛选剔除点数：")
    for column, count in minimum_removed.items():
        print(f"  {column}: {count}")
    print(f"3σ 筛选：{'已启用' if args.filter_3sigma else '未启用'}")
    if args.filter_3sigma:
        print("3σ 筛选剔除点数：")
        for column, count in sigma_removed.items():
            print(f"  {column}: {count}")
    print("各列缺测分钟数：")
    for column, count in aligned.isna().sum().items():
        print(f"  {column}: {int(count)}")


if __name__ == "__main__":
    main()
