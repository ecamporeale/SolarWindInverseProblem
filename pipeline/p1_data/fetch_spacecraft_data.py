#!/usr/bin/env python3
"""
Multi-spacecraft data fetcher for SW_inverse.

Supports: ACE, STEREO-A, Parker Solar Probe, Solar Orbiter
Outputs standardized hourly CSV with all spacecraft merged.

Usage:
    python fetch_spacecraft_data.py --start 2021-01-01 --end 2021-12-31 --spacecraft ACE STEREO_A PSP
    
Requirements:
    pip install astropy sunpy pandas numpy
    Optional: pip install cdas (for PSP CDAWeb access)
"""

import argparse
import re
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import tempfile
import urllib.request
import urllib.error
from contextlib import contextmanager

STEREO_A_FILENAME_RE = re.compile(r"STA_L2_PLA_1DMax_1hr_(\d{8})_V(\d+)\.cdf")

AU_IN_KM = 149597870.7

try:
    import cdflib
except ImportError:
    cdflib = None

try:
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    from sunpy.coordinates import frames, get_horizons_coord
except ImportError as e:
    print(f"Warning: astropy/sunpy not available: {e}")
    print("Install with: pip install astropy sunpy")


class SpacecraftDataFetcher:
    """Unified interface for fetching multi-spacecraft solar wind data.
    
    NOTE: This fetcher retrieves REAL observational data only. 
    No synthetic data is generated as fallback.
    """
    
    def __init__(self, data_dir: Path = Path("data"), cache_raw: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_raw = cache_raw
        self._stereo_a_year_listing_cache: Dict[int, Dict[str, str]] = {}
        print(f"Data directory: {self.data_dir}")

    def _stereo_a_year_listing(self, year: int, base_url: str) -> Dict[str, str]:
        """date_str -> best (highest-version) filename for one year, fetched once and
        cached. The archive's version tag drifts by year (V09 for 2011-2014, V10/V11
        mixed for 2015-2017, V11/V12 later) -- guessing a fixed version per day, as
        this used to do, silently 404s on every day for any year that doesn't match
        the guess. Listing the real directory once avoids that entirely."""
        if year in self._stereo_a_year_listing_cache:
            return self._stereo_a_year_listing_cache[year]

        best: Dict[str, str] = {}
        try:
            with urllib.request.urlopen(f"{base_url}{year}/", timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
            for date_str, version_str in STEREO_A_FILENAME_RE.findall(html):
                filename = f"STA_L2_PLA_1DMax_1hr_{date_str}_V{version_str}.cdf"
                if date_str not in best or int(version_str) > int(
                    STEREO_A_FILENAME_RE.match(best[date_str]).group(2)
                ):
                    best[date_str] = filename
        except Exception as exc:
            print(f"  WARNING: could not list STEREO-A directory for {year}: {exc}")

        self._stereo_a_year_listing_cache[year] = best
        return best

    def _raw_spacecraft_dir(self, spacecraft_id: str) -> Path:
        raw_dir = self.data_dir / spacecraft_id.lower()
        raw_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir

    def _raw_file_path(self, spacecraft_id: str, filename: str) -> Path:
        return self._raw_spacecraft_dir(spacecraft_id) / filename

    @contextmanager
    def _disable_requests_ssl_verification(self):
        """SOAR and Horizons requests need SSL verification disabled in this environment."""
        import requests

        requests.packages.urllib3.disable_warnings()
        original_request = requests.Session.request

        def patched_request(session, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return original_request(session, method, url, **kwargs)

        requests.Session.request = patched_request
        try:
            yield requests
        finally:
            requests.Session.request = original_request

    def _download_solo_pas_file(self, current_day: datetime) -> Optional[Path]:
        """Download one Solar Orbiter SWA/PAS ground-moments CDF from SOAR."""
        date_str = current_day.strftime("%Y%m%d")
        data_item_id = f"solo_L2_swa-pas-grnd-mom_{date_str}"
        url = (
            "https://soar.esac.esa.int/soar-sl-tap/data"
            f"?retrieval_type=PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE"
        )

        with self._disable_requests_ssl_verification() as requests:
            response = requests.get(url, timeout=60)

        if response.status_code != 200 or not response.content:
            return None

        disposition = response.headers.get("Content-Disposition", "")
        filename = None
        if "filename=" in disposition:
            filename = disposition.split("filename=", 1)[1].strip().strip('"')
        if not filename:
            filename = f"solo_L2_swa-pas-grnd-mom_{date_str}.cdf"

        filepath = self._raw_file_path("SOLO", filename)
        filepath.write_bytes(response.content)

        if filepath.stat().st_size <= 1000:
            filepath.unlink(missing_ok=True)
            return None

        # SOAR returns a tar bundle when several versions of the day's file exist (e.g. V01-V03; seen 2026-10-22):
        # keep the highest version's CDF and drop the tar
        import tarfile
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath) as tf:
                cdfs = [m for m in tf.getmembers() if m.name.endswith('.cdf')]
                if not cdfs:
                    filepath.unlink(missing_ok=True)
                    return None
                best = max(cdfs, key=lambda m: int(re.search(r'_V(\d+)\.cdf$', m.name).group(1)) if re.search(r'_V(\d+)\.cdf$', m.name) else 0)
                out = self._raw_file_path("SOLO", Path(best.name).name)
                out.write_bytes(tf.extractfile(best).read())
            filepath.unlink(missing_ok=True)
            return out

        return filepath

    def _get_solo_carrington_coords(self, times: pd.Series) -> pd.DataFrame:
        """Query Solar Orbiter ephemeris from JPL Horizons and convert to Carrington."""
        horizons_times = Time(times.to_numpy())
        # JPL Horizons accepts at most 10,000 time values per request ("For more than 10,000 time values, use dictionary
        # input"): query in chunks (seen 2026-10-23 with five years of hourly data)
        lon, lat, rad = [], [], []
        step = 9000
        for i in range(0, len(horizons_times), step):
            tt = horizons_times[i:i + step]
            with self._disable_requests_ssl_verification():
                coord = get_horizons_coord("Solar Orbiter", tt)
            c = coord.transform_to(frames.HeliographicCarrington(observer='self', obstime=tt))
            lon.append(c.lon.to_value(u.deg)); lat.append(c.lat.to_value(u.deg)); rad.append(c.radius.to_value(u.au))

        class _C:                                        # same attributes as a transformed coordinate, concatenated
            pass
        carrington = _C()
        carrington.lon = np.concatenate(lon) * u.deg
        carrington.lat = np.concatenate(lat) * u.deg
        carrington.radius = np.concatenate(rad) * u.au
        return pd.DataFrame({
            "Time": pd.to_datetime(times).reset_index(drop=True),
            "HC_lon": carrington.lon.to_value(u.deg) % 360.0,
            "HC_lat": carrington.lat.to_value(u.deg),
            "HC_radius": carrington.radius.to_value(u.au),
        })
    
    # =====================================================================
    # STEREO-A Data Fetcher
    # =====================================================================
    
    def fetch_stereo_a(
        self, 
        start: datetime, 
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch STEREO-A PLASTIC data from SSC archive.
        
        Data is fetched from STEREO Science Center.
        Format: Daily CDF files with 1-hour cadence.
        
        Args:
            start: Start date
            end: End date
        
        Returns:
            DataFrame with columns: Time, Speed, HC_lon, HC_lat, HC_radius, Spacecraft_ID
        """
        print(f"\n{'='*60}")
        print(f"Fetching STEREO-A data ({start.date()} to {end.date()})")
        print(f"{'='*60}")

        if cdflib is None:
            raise RuntimeError("cdflib is required to read STEREO-A CDF files. Install with: pip install cdflib")
        
        dfs = []
        current_day = start

        base_url = "https://stereo-ssc.nascom.nasa.gov/data/ins_data/plastic/level2/Protons/Derived_from_1D_Maxwellian/ahead/1hr/"

        while current_day <= end:
            year = current_day.year
            date_str = current_day.strftime("%Y%m%d")

            # First check the cache under any version tag already on disk, so a
            # prior run's downloads (under whichever version was current then)
            # are reused rather than re-fetched.
            cached_match = next(
                (p for p in self._raw_spacecraft_dir("STEREO_A").glob(f"STA_L2_PLA_1DMax_1hr_{date_str}_V*.cdf")
                 if p.stat().st_size > 1000),
                None,
            )
            if cached_match is not None:
                filepath = cached_match
                print(f"  Using cached: {filepath.name}")
            else:
                year_listing = self._stereo_a_year_listing(year, base_url)
                filename = year_listing.get(date_str)
                if filename is None:
                    print(f"  Missing archive file for {date_str}")
                    current_day += timedelta(days=1)
                    continue

                filepath = self._raw_file_path("STEREO_A", filename)
                url = f"{base_url}{year}/{filename}"
                try:
                    print(f"  Downloading {filename}...", end="", flush=True)
                    with urllib.request.urlopen(url, timeout=20) as response, open(filepath, "wb") as out_file:
                        out_file.write(response.read())
                    print(" ✓")
                except Exception as exc:
                    print(f" FAILED ({exc})")
                    current_day += timedelta(days=1)
                    continue

            try:
                cdf = cdflib.CDF(str(filepath))
                epoch = cdflib.cdfepoch.to_datetime(cdf.varget("epoch_1hr"))
                speed = np.asarray(cdf.varget("proton_bulk_speed_1hr"), dtype=float)
                lon_hee = np.asarray(cdf.varget("spcrft_lon_hee_1hr"), dtype=float)
                lat_hee = np.asarray(cdf.varget("spcrft_lat_hee_1hr"), dtype=float)
                distance_raw = np.asarray(cdf.varget("heliocentric_dist_1hr"), dtype=float)
                distance_au = np.where(distance_raw > 10.0, distance_raw / AU_IN_KM, distance_raw)

                df_parsed = pd.DataFrame({
                    "Time": pd.to_datetime(epoch),
                    "Speed": speed,
                    "Lon_HEE": lon_hee,
                    "Lat_HEE": lat_hee,
                    "Distance": distance_au,
                })

                df_parsed = df_parsed.replace([-1e31, -9.99e30, -9999.0], np.nan)
                df_parsed = df_parsed[np.isfinite(df_parsed["Speed"])]
                df_parsed = df_parsed[(df_parsed["Speed"] > 100.0) & (df_parsed["Speed"] < 1500.0)]

                if not df_parsed.empty:
                    dfs.append(df_parsed)
                    print(f"    Parsed {len(df_parsed)} hourly records")
            except Exception as e:
                print(f"    Parse error for {filepath.name}: {e}")

            current_day += timedelta(days=1)
        
        if not dfs:
            print("  WARNING: No STEREO-A data parsed")
            return pd.DataFrame()
        
        # Combine and standardize
        data = pd.concat(dfs, ignore_index=True).sort_values('Time').reset_index(drop=True)

        # Daily files are already hourly; just de-duplicate overlapping timestamps.
        data = data.drop_duplicates(subset=['Time']).reset_index(drop=True)

        # Convert to Carrington coordinates
        data = self._convert_to_carrington(data, "STEREO_A")

        print(f"  Final: {len(data)} hourly samples from {data['Time'].min()} to {data['Time'].max()}")
        return data
    
    # =====================================================================
    # ACE Data Fetcher (existing)
    # =====================================================================
    
    def fetch_ace(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch ACE data from existing file or generate dummy.
        
        Uses existing ACE_speed_lon_lat_hr.txt if available.
        """
        ace_file = self.data_dir / "ACE_speed_lon_lat_hr.txt"
        
        if ace_file.exists():
            print(f"\n{'='*60}")
            print(f"Loading ACE data from {ace_file}")
            print(f"{'='*60}")
            df = pd.read_csv(ace_file, parse_dates=['Time'])
            df['Spacecraft_ID'] = 'ACE'
            
            # Filter to requested date range
            df = df[(df['Time'] >= start) & (df['Time'] <= end)]
            
            # Standardize column names
            df = df.rename(columns={'HC_lon': 'HC_lon', 'HC_lat': 'HC_lat', 'HC_radius': 'HC_radius'})
            if 'Speed' not in df.columns:
                df['Speed'] = df['speed']

            df['Speed'] = pd.to_numeric(df['Speed'], errors='coerce')
            df = df.replace([-1e31, -9.99e30, -9999.0, -9999.9], np.nan)
            df = df.dropna(subset=['Time', 'Speed', 'HC_lon', 'HC_lat', 'HC_radius'])
            df = df[(df['Speed'] > 0.0) & (df['Speed'] < 2000.0)].reset_index(drop=True)
            
            print(f"  Loaded {len(df)} samples from {df['Time'].min()} to {df['Time'].max()}")
            return df
        else:
            print(f"\nWarning: ACE file not found at {ace_file}")
            print("  ACE data can be fetched using: python preprocess_ACE.py")
            return pd.DataFrame()
    
    # =====================================================================
    # Parker Solar Probe Data Fetcher (Stub)
    # =====================================================================
    
    def fetch_parker_solar_probe(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch Parker Solar Probe SWEAP/SPC level-3 moments from the SWEAP archive.
        
        Uses the public archive at:
        https://sweap.cfa.harvard.edu/pub/data/sci/sweap/spc/L3/YYYY/MM/
        
        Args:
            start: Start date
            end: End date
        
        Returns:
            DataFrame with standardized columns
        """
        print(f"\n{'='*60}")
        print(f"Fetching Parker Solar Probe data ({start.date()} to {end.date()})")
        print(f"{'='*60}")

        if cdflib is None:
            raise RuntimeError("cdflib is required to read PSP CDF files. Install with: pip install cdflib")

        dfs = []
        current_day = start

        while current_day <= end:
            year = current_day.strftime("%Y")
            month = current_day.strftime("%m")
            date_str = current_day.strftime("%Y%m%d")
            filename = f"psp_swp_spc_l3i_{date_str}_v02.cdf"
            base_url = f"https://sweap.cfa.harvard.edu/pub/data/sci/sweap/spc/L3/{year}/{month}/"
            filepath = self._raw_file_path("PSP", filename)

            if filepath.exists() and filepath.stat().st_size > 1000:
                print(f"  Using cached: {filepath.name}")
            else:
                downloaded = False
                candidate_names = [
                    filename,
                    f"psp_swp_spc_l3i_{date_str}_v03.cdf",
                    f"psp_swp_spc_l3i_{date_str}_v01.cdf",
                ]
                for candidate_name in candidate_names:
                    candidate_url = base_url + candidate_name
                    candidate_path = self._raw_file_path("PSP", candidate_name)
                    try:
                        print(f"  Downloading {candidate_name}...", end="", flush=True)
                        with urllib.request.urlopen(candidate_url, timeout=20) as response, open(candidate_path, "wb") as out_file:
                            out_file.write(response.read())
                        print(" ✓")
                        filepath = candidate_path
                        downloaded = True
                        break
                    except Exception:
                        pass

                if not downloaded:
                    print(f"  Missing archive file for {date_str}")
                    current_day += timedelta(days=1)
                    continue

            try:
                cdf = cdflib.CDF(str(filepath))
                epoch = pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget("Epoch")))
                vp_fit_rtn = np.asarray(cdf.varget("vp_fit_RTN"), dtype=float)
                carr_lon = np.asarray(cdf.varget("carr_longitude"), dtype=float)
                carr_lat = np.asarray(cdf.varget("carr_latitude"), dtype=float)
                sc_pos_hci = np.asarray(cdf.varget("sc_pos_HCI"), dtype=float)

                # Use bulk speed magnitude from RTN components.
                speed = np.linalg.norm(vp_fit_rtn, axis=1)

                # Spacecraft position is stored in km; convert to AU.
                radius_au = np.linalg.norm(sc_pos_hci, axis=1) / AU_IN_KM

                df_parsed = pd.DataFrame({
                    "Time": epoch,
                    "Speed": speed,
                    "HC_lon": np.mod(carr_lon, 360.0),
                    "HC_lat": carr_lat,
                    "HC_radius": radius_au,
                })

                df_parsed = df_parsed.replace([-1e31, -9.99e30, -9999.0], np.nan)
                df_parsed = df_parsed[np.isfinite(df_parsed["Speed"])]
                df_parsed = df_parsed[(df_parsed["Speed"] > 100.0) & (df_parsed["Speed"] < 2000.0)]

                if not df_parsed.empty:
                    dfs.append(df_parsed)
                    print(f"    Parsed {len(df_parsed)} PSP records")
            except Exception as e:
                print(f"    Parse error for {filepath.name}: {e}")

            current_day += timedelta(days=1)

        if not dfs:
            print("  WARNING: No PSP data parsed")
            return pd.DataFrame()

        data = pd.concat(dfs, ignore_index=True).sort_values('Time').reset_index(drop=True)
        data['Time_rounded'] = data['Time'].dt.floor('h')
        data_hourly = data.groupby('Time_rounded').agg({
            'Speed': 'mean',
            'HC_lon': 'mean',
            'HC_lat': 'mean',
            'HC_radius': 'mean',
        }).reset_index()
        data_hourly = data_hourly.rename(columns={'Time_rounded': 'Time'})
        data_hourly['Spacecraft_ID'] = 'PSP'

        print(f"  Final: {len(data_hourly)} hourly PSP samples from {data_hourly['Time'].min()} to {data_hourly['Time'].max()}")
        return data_hourly
    
    # =====================================================================
    # Solar Orbiter Data Fetcher (Stub)
    # =====================================================================
    
    def fetch_solar_orbiter(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch Solar Orbiter SWA/PAS ground-moments data from ESA SOAR.
        
        Speed comes from the public SWA/PAS moments product.
        Spacecraft Carrington longitude/latitude/radius comes from JPL Horizons.
        
        Args:
            start: Start date
            end: End date
        
        Returns:
            DataFrame with standardized columns
        """
        print(f"\n{'='*60}")
        print(f"Fetching Solar Orbiter data ({start.date()} to {end.date()})")
        print(f"{'='*60}")

        if cdflib is None:
            raise RuntimeError("cdflib is required to read Solar Orbiter CDF files. Install with: pip install cdflib")

        dfs = []
        current_day = start

        while current_day <= end:
            date_str = current_day.strftime("%Y%m%d")
            solo_dir = self._raw_spacecraft_dir("SOLO")
            cached_matches = sorted(solo_dir.glob(f"solo_L2_swa-pas-grnd-mom_{date_str}_V*.cdf"))
            filepath = cached_matches[-1] if cached_matches else None

            if filepath is not None and filepath.stat().st_size > 1000:
                print(f"  Using cached: {filepath.name}")
            else:
                print(f"  Downloading SWA/PAS moments for {date_str}...", end="", flush=True)
                filepath = self._download_solo_pas_file(current_day)
                if filepath is None:
                    print(" missing")
                    current_day += timedelta(days=1)
                    continue
                print(f" ✓ ({filepath.name})")

            try:
                cdf = cdflib.CDF(str(filepath))
                epoch = pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget("Epoch")))
                velocity_rtn = np.asarray(cdf.varget("V_RTN"), dtype=float)
                quality_factor = np.asarray(cdf.varget("quality_factor"), dtype=float).reshape(-1)

                speed = np.linalg.norm(velocity_rtn, axis=1)
                df_parsed = pd.DataFrame({
                    "Time": epoch,
                    "Speed": speed,
                    "quality_factor": quality_factor,
                })

                df_parsed = df_parsed.replace([-1e31, -9.99e30, -9999.0], np.nan)
                df_parsed = df_parsed[np.isfinite(df_parsed["Speed"])]
                df_parsed = df_parsed[(df_parsed["Speed"] > 100.0) & (df_parsed["Speed"] < 2000.0)]
                df_parsed = df_parsed[np.isfinite(df_parsed["quality_factor"])]
                df_parsed = df_parsed[df_parsed["quality_factor"] >= 0.0]

                if not df_parsed.empty:
                    dfs.append(df_parsed[["Time", "Speed"]])
                    print(f"    Parsed {len(df_parsed)} Solar Orbiter records")
            except Exception as e:
                print(f"    Parse error for {filepath.name}: {e}")

            current_day += timedelta(days=1)

        if not dfs:
            print("  WARNING: No Solar Orbiter data parsed")
            return pd.DataFrame()

        data = pd.concat(dfs, ignore_index=True).sort_values('Time').reset_index(drop=True)
        data['Time_rounded'] = data['Time'].dt.floor('h')
        data_hourly = data.groupby('Time_rounded').agg({'Speed': 'mean'}).reset_index()
        data_hourly = data_hourly.rename(columns={'Time_rounded': 'Time'})

        solo_coords = self._get_solo_carrington_coords(data_hourly['Time'])
        data_hourly = data_hourly.merge(solo_coords, on='Time', how='inner')
        data_hourly['Spacecraft_ID'] = 'SOLO'

        canonical = ['Time', 'Speed', 'HC_lon', 'HC_lat', 'HC_radius', 'Spacecraft_ID']
        data_hourly = data_hourly[canonical].dropna().reset_index(drop=True)

        print(f"  Final: {len(data_hourly)} hourly Solar Orbiter samples from {data_hourly['Time'].min()} to {data_hourly['Time'].max()}")
        return data_hourly
    
    # =====================================================================
    # Coordinate Transformation
    # =====================================================================
    
    def _convert_to_carrington(self, df: pd.DataFrame, spacecraft_id: str) -> pd.DataFrame:
        """
        Convert spacecraft coordinates to Heliographic Carrington.
        
        Handles: HEE or HCI spherical coordinates → Heliographic Carrington
        """
        print(f"  Converting {spacecraft_id} coordinates to Heliographic Carrington...", end="", flush=True)
        
        try:
            times = Time(df['Time'].values)

            if 'Lon_HEE' in df.columns and 'Lat_HEE' in df.columns:
                coord = SkyCoord(
                    lon=df['Lon_HEE'].values * u.deg,
                    lat=df['Lat_HEE'].values * u.deg,
                    distance=df['Distance'].values * u.au,
                    frame=frames.HeliocentricEarthEcliptic(obstime=times),
                    representation_type='spherical',
                )
            else:
                coord = SkyCoord(
                    lon=df['Lon_HCI'].values * u.deg,
                    lat=df['Lat_HCI'].values * u.deg,
                    distance=df['Distance'].values * u.au,
                    frame=frames.HeliographicStonyhurst(obstime=times),
                    representation_type='spherical',
                )
            
            # Transform to Heliographic Carrington
            carrington = coord.transform_to(frames.HeliographicCarrington(observer='self', obstime=times))
            
            df['HC_lon'] = carrington.lon.deg % 360
            df['HC_lat'] = carrington.lat.deg
            
        except Exception as e:
            print(f" (warning: {e})")
            lon_col = 'Lon_HEE' if 'Lon_HEE' in df.columns else 'Lon_HCI'
            lat_col = 'Lat_HEE' if 'Lat_HEE' in df.columns else 'Lat_HCI'
            df['HC_lon'] = df[lon_col] % 360
            df['HC_lat'] = df[lat_col]
        
        # Ensure radius in AU
        if 'Distance' in df.columns:
            df['HC_radius'] = df['Distance']
        else:
            df['HC_radius'] = 1.0
        
        # Standardize
        df['Spacecraft_ID'] = spacecraft_id
        
        canonical = ['Time', 'Speed', 'HC_lon', 'HC_lat', 'HC_radius', 'Spacecraft_ID']
        result = df[canonical].dropna().reset_index(drop=True)
        
        print(f" ✓ ({len(result)} valid samples)")
        return result
    
    # =====================================================================
    # Merge & Export
    # =====================================================================
    
    def merge_all_spacecraft(
        self,
        dataframes: Dict[str, pd.DataFrame],
        output_path: Path = None,
    ) -> pd.DataFrame:
        """
        Merge multiple spacecraft DataFrames into single observation set.
        
        Handles:
        - Different cadences (resample to common hourly grid)
        - Gaps and NaNs
        - Time-zone alignment
        """
        print(f"\n{'='*60}")
        print(f"Merging spacecraft data")
        print(f"{'='*60}")
        
        if not dataframes:
            print("  ERROR: No spacecraft data provided")
            return pd.DataFrame()
        
        # Remove empty dataframes
        dataframes = {k: v for k, v in dataframes.items() if len(v) > 0}
        
        if not dataframes:
            print("  ERROR: All spacecraft dataframes are empty")
            return pd.DataFrame()
        
        # Concatenate all while preserving spacecraft identity per row.
        all_data = pd.concat(dataframes.values(), ignore_index=True)
        all_data = all_data.sort_values(['Time', 'Spacecraft_ID']).reset_index(drop=True)
        all_data = all_data.drop_duplicates(subset=['Time', 'Spacecraft_ID']).reset_index(drop=True)
        
        print(f"  Combined data from {len(dataframes)} spacecraft:")
        for sc_id in dataframes.keys():
            n = len(dataframes[sc_id])
            t_min = dataframes[sc_id]['Time'].min()
            t_max = dataframes[sc_id]['Time'].max()
            print(f"    {sc_id:15s}: {n:6d} samples ({t_min.date()} to {t_max.date()})")
        
        unique_times = all_data['Time'].nunique()
        print(f"\n  Combined rows: {len(all_data)} (across {unique_times} unique timestamps)")
        print(f"  Date range: {all_data['Time'].min()} to {all_data['Time'].max()}")
        print(f"  Speed range: {all_data['Speed'].min():.1f} to {all_data['Speed'].max():.1f} km/s")
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_data.to_csv(output_path, index=False)
            print(f"\n  Saved to: {output_path}")
        
        return all_data


# =========================================================================
# CLI Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch multi-spacecraft solar wind data for SW_inverse (REAL DATA ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This fetcher retrieves real observational data from public archives.
No synthetic data is generated as fallback. 
If data cannot be retrieved, the script will fail with a clear error message.

Examples:
  # Fetch ACE only (existing file)
  python fetch_spacecraft_data.py --start 2021-01-01 --end 2021-12-31 --spacecraft ACE
  
  # Fetch STEREO-A (requires internet access to STEREO Science Center)
  python fetch_spacecraft_data.py --start 2021-06-01 --end 2021-12-31 \\
      --spacecraft STEREO_A --output stereo_a_2021.csv
  
  # Merge with ACE
  python fetch_spacecraft_data.py --start 2021-06-01 --end 2021-12-31 \\
      --spacecraft ACE STEREO_A --output merged_2021.csv
        """
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--spacecraft", nargs="+", default=["ACE"],
                        choices=["ACE", "STEREO_A", "PSP", "SOLO"],
                        help="Spacecraft to fetch")
    parser.add_argument("--data-dir", default="data", help="Output data directory")
    parser.add_argument("--output", default="multi_spacecraft_obs.csv", help="Output filename")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of raw files")
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    
    if start > end:
        print("ERROR: start date must be <= end date")
        return 1
    
    fetcher = SpacecraftDataFetcher(
        data_dir=Path(args.data_dir),
        cache_raw=not args.no_cache
    )
    
    all_dfs = {}
    
    # Fetch requested spacecraft
    if "ACE" in args.spacecraft:
        ace_df = fetcher.fetch_ace(start, end)
        if len(ace_df) > 0:
            all_dfs["ACE"] = ace_df
    
    if "STEREO_A" in args.spacecraft:
        try:
            stereo_df = fetcher.fetch_stereo_a(start, end)
            if len(stereo_df) > 0:
                all_dfs["STEREO_A"] = stereo_df
        except Exception as e:
            print(f"ERROR: Failed to fetch STEREO-A: {e}")
    
    if "PSP" in args.spacecraft:
        try:
            psp_df = fetcher.fetch_parker_solar_probe(start, end)
            if len(psp_df) > 0:
                all_dfs["PSP"] = psp_df
        except Exception as e:
            print(f"ERROR: Failed to fetch PSP: {e}")
    
    if "SOLO" in args.spacecraft:
        try:
            so_df = fetcher.fetch_solar_orbiter(start, end)
            if len(so_df) > 0:
                all_dfs["SOLO"] = so_df
        except Exception as e:
            print(f"ERROR: Failed to fetch Solar Orbiter: {e}")
    
    # Check if we have any data
    if not all_dfs:
        print("\nERROR: No spacecraft data successfully fetched")
        return 1
    
    # Merge and save
    merged = fetcher.merge_all_spacecraft(
        all_dfs,
        output_path=Path(args.data_dir) / args.output
    )
    
    if len(merged) == 0:
        print("\nERROR: Merged dataset is empty")
        return 1
    
    print("\n" + "="*60)
    print("SUCCESS")
    print("="*60)
    print(f"Output file: {args.data_dir}/{args.output}")
    print(f"Rows: {len(merged)}")
    print(f"Time span: {merged['Time'].min()} to {merged['Time'].max()}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
