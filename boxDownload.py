from .config import config
from .readfile import adjust_path, invPolyFitXe

from box_sdk_gen import BoxClient, JWTConfig, BoxJWTAuth
import importlib.resources as resources
import tempfile
import os
import pandas as pd
import datetime
import re

import requests
import time
from tqdm import tqdm

# --- Box client setup ---
box_json_path = resources.files("crystalize_slowcontrolsdata") / "box_private_config.json"

jwt_config = JWTConfig.from_config_file(box_json_path)
auth = BoxJWTAuth(config=jwt_config)
client = BoxClient(auth=auth)

def list_csv_files_recursive(folder_id):
    """
    Recursively search a Box folder and its subfolders for CSV files.
    Returns [(datetime, file_item)] sorted by timestamp extracted from filename.
    """
    files_info = []
    
    # FIX: Matches YYYY-MM-DD, then optionally matches an exact HH_MM_SS timestamp block.
    # It allows any trailing characters (like _MFC) safely before the end of the file.
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[ ](\d{2}_\d{2}_\d{2}))?")

    def walk(folder_id):
        marker = None

        while True:
            response = client.folders.get_folder_items(
                folder_id=folder_id,
                limit=1000,
                marker=marker
            )

            for item in response.entries:

                if item.type == "folder":
                    walk(item.id)

                elif item.type == "file" and item.name.endswith(".csv"):
                    match = pattern.match(item.name)
                    if match:
                        date_part = match.group(1)
                        time_part = match.group(2) # Will successfully extract '21_25_34' or None
                        
                        try:
                            if time_part:
                                file_dt = datetime.datetime.strptime(
                                    f"{date_part} {time_part}",
                                    "%Y-%m-%d %H_%M_%S"
                                )
                            else:
                                file_dt = datetime.datetime.strptime(date_part, "%Y-%m-%d")
                                
                            files_info.append((file_dt, item))
                        except ValueError:
                            continue

            if response.next_marker:
                marker = response.next_marker
            else:
                break

    walk(folder_id)

    files_info.sort(key=lambda x: x[0])
    return files_info

def download_file(client, file_id, local_path, session=None, chunk_size=8*1024*1024, max_retries=3):
    """
    Download a single file from Box with retries and progress bar.
    """
    session = session or requests.Session()
    url = f"https://api.box.com/2.0/files/{file_id}/content"

    def get_headers():
        token = client.auth.retrieve_token().access_token
        return {"Authorization": f"Bearer {token}"}

    for attempt in range(max_retries):
        try:
            with session.get(url, headers=get_headers(), stream=True) as r:
                if r.status_code == 401 and attempt < max_retries - 1:
                    continue
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0)) or None
                with open(local_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(local_path),
                    smoothing=0.1,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return local_path
        except requests.RequestException as e:
            print(f"Retry {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(1)

    raise RuntimeError(f"Failed to download file {file_id} after {max_retries} attempts")

def readBoxCSV(
    key,
    timeCol,
    start_time,
    end_time,
    download=False,
    return_df=True
):
    """
    Read CSV files from a Box folder (recursively) between two timestamps.
    """
    # 1. Normalize start_time and end_time to datetime objects if passed as tuples
    if isinstance(start_time, (tuple, list)):
        start_time = datetime.datetime(*start_time)
    if isinstance(end_time, (tuple, list)):
        end_time = datetime.datetime(*end_time)

    folder_id = config['Box'][key]
    download_dir = adjust_path(config['Local'][key])
    
    if download and download_dir is None:
        raise ValueError("download_dir must be specified if download=True")

    files_info = list_csv_files_recursive(folder_id)

    if not files_info:
        return pd.DataFrame() if return_df else []

    # Find file immediately before start_time
    prev_file = None
    for dt, item in reversed(files_info):
        if dt < start_time:
            prev_file = (dt, item)
            break

    in_range_files = [
        (dt, item) for dt, item in files_info
        if start_time <= dt <= end_time
    ]

    files_to_use = []
    if prev_file:
        files_to_use.append(prev_file)
    files_to_use.extend(in_range_files)

    if not files_to_use:
        return pd.DataFrame() if return_df else []

    if download:
        os.makedirs(download_dir, exist_ok=True)

    dfs = []
    downloaded_paths = []

    temp_ctx = tempfile.TemporaryDirectory() if (return_df and not download) else None
    tmpdir = temp_ctx.name if temp_ctx else None

    for _, file_item in files_to_use:
        if download:
            local_path = os.path.join(download_dir, file_item.name)
        else:
            local_path = os.path.join(tmpdir, file_item.name)

        download_file(client, file_item.id, local_path)
        downloaded_paths.append(local_path)

        if return_df:
            try:
                df = pd.read_csv(local_path)
                
                # Standardize time column name immediately if missing
                if timeCol not in df.columns:
                    possible_time_cols = ["Time", "Time (s)", "time", "time (s)", "Timestamp"]
                    for candidate in possible_time_cols:
                        if candidate in df.columns:
                            df.rename(columns={candidate: timeCol}, inplace=True)
                            break
                            
                dfs.append(df)
            except Exception as e:
                print(f"Failed reading {file_item.name}: {e}")

    if temp_ctx:
        temp_ctx.cleanup()

    if not return_df:
        return downloaded_paths

    if not dfs:
        return pd.DataFrame()

    # Concatenate all downloaded dataframes
    df = pd.concat(dfs, ignore_index=True)

    if timeCol not in df.columns:
        print(f"Time column '{timeCol}' not found in downloaded data.")
        return pd.DataFrame()

# ==================== DEBUG START ====================

#     # 1. Remove original null values
#     df = df.dropna(subset=[timeCol])

#     print("\n========== Time Parsing Debug ==========")
#     print("Rows before parsing:", len(df))
#     print("Time dtype before parsing:", df[timeCol].dtype)

#     print("\nFirst 10 raw timestamps:")
#     for x in df[timeCol].head(10):
#         print(repr(x))


#     # 2. Convert Time column to string
#     time_series = (
#         df[timeCol]
#         .astype(str)
#         .str.strip()
#         .str.replace("_", ":", regex=False)
#     )

#     print("\nFirst 10 timestamps after replacing '_':")
#     for x in time_series.head(10):
#         print(repr(x))


#     # 3. Parse timestamps
#     df[timeCol] = pd.to_datetime(
#         time_series,
#         format="%Y-%m-%d %H:%M:%S",
#         errors="coerce"
#     )

#     # 4. Check parsing results
#     print("\nFirst 10 parsed timestamps:")
#     print(df[timeCol].head(10))

#     print("\nNaT count:")
#     print(df[timeCol].isna().sum(), "/", len(df))


#     # 5. Remove failed timestamps
#     df = df.dropna(subset=[timeCol])

#     print("\nRows remaining after removing NaT:")
#     print(len(df))


#     # 6. Check actual data time range
#     print("\nActual data time range:")
#     print("min =", df[timeCol].min())
#     print("max =", df[timeCol].max())


#     # 7. Check requested time range
#     print("\nRequested time range:")
#     print("start =", start_time)
#     print("end   =", end_time)

#     print("\nTypes:")
#     print("start_time:", type(start_time))
#     print("end_time:  ", type(end_time))
#     print("Time dtype:", df[timeCol].dtype)


#     # 8. Test the two filtering conditions separately
#     mask_start = df[timeCol] >= start_time
#     mask_end = df[timeCol] <= end_time
#     mask_both = mask_start & mask_end

#     print("\nFiltering results:")
#     print("Total rows:       ", len(df))
#     print("Rows >= start:    ", mask_start.sum())
#     print("Rows <= end:      ", mask_end.sum())
#     print("Rows inside range:", mask_both.sum())


#     # 9. Apply final filter
#     df = df.loc[mask_both].copy()
#     df = df.sort_values(by=timeCol)

#     print("\nFinal dataframe shape:")
#     print(df.shape)

#     print("========================================")
# ===================== DEBUG END =====================
    # Clean nulls in the time column
    df = df.dropna(subset=[timeCol])
    
    # Keep string and numeric temporal entries
    df = df[df[timeCol].apply(lambda x: isinstance(x, (str, int, float)))]

    # Robust parsing: Handles "YYYY-MM-DD HH_MM_SS" and standard formats
    try:
#         time_series = df[timeCol].astype(str).str.replace('_', ':')
#         df[timeCol] = pd.to_datetime(time_series, errors='coerce', format='mixed')
        df[timeCol] = pd.to_datetime(
                df[timeCol].astype(str),
                format="%Y-%m-%d %H_%M_%S",
                errors="coerce"
            )
    
    except Exception as e:
        print(f"Time parsing failed: {e}")
        return pd.DataFrame()

    # Drop any rows where time conversion failed
    df = df.dropna(subset=[timeCol])

    # Sort and filter inside target window
    df = df.sort_values(by=timeCol)
    df = df[(df[timeCol] >= start_time) & (df[timeCol] <= end_time)]

    # Specific calculation for Transducers key
    if key == "Transducers":
        if "ICV (bar)" in df.columns:
            df["ICV (K)"] = invPolyFitXe(df["ICV (bar)"])

    return df
