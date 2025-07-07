import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema


def calculate_bilotta_afi(ds,temp_var,mask, units='F',
                         start_month=8, end_month=7):
    """
    Calculate Air Freezing Index (AFI) using the Bilotta et al. (2015) methodology.

    This methodology calculates the AFI as the difference between the highest and lowest
    extrema points in the cumulative freezing degree-days curve over a cold season.

    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset containing temperature data; coordinates name
        should be "time","lat","lon"
    temp_var : str, optional
        Name of temperature variable if ds is a Dataset (default: 'temperature')
    start_month : int, optional
        Start month of the cold season (default: 8 for August)
    end_month : int, optional
        End month of the cold season (default: 7 for July)

    Returns:
    --------
    xarray.DataArray
        Seasonal AFI values
    """
    if units=='F':
      threshold=32
    else:
      print('error units /n convert to F')

    # Select the temperature data
    if isinstance(ds, xr.Dataset):
        temp_data = ds[temp_var]
    else:
        temp_data = ds

    # Calculate daily departures (freezing degree-days)
    # Note: In the paper they use (Tavg - 32°F), but here we generalize with a threshold parameter
    fdd_daily = temp_data - threshold

    # Create a seasonal year coordinate for grouping
    # For August-July cold season, if month >= 8, use the current year, else use previous year
    month = temp_data['time'].dt.month
    Tyear = temp_data['time'].dt.year

    # Create a cold season year identifier (the year in which the cold season ends)
    cold_season = xr.zeros_like(Tyear)*np.nan
    for iy in range(Tyear[0].data,Tyear[-1].data):
      idt = np.where(( temp_data['time'] >= np.datetime64(f'{iy}-{start_month:02d}-01')) & \
                     ( temp_data['time'] < np.datetime64(f'{iy+1}-{end_month+1:02d}-01')))[0]
      cold_season[idt] = iy

    # Group data by cold season year and compute cumulative FDD
    fdd_daily = fdd_daily.assign_coords({'cold_season':cold_season})
    grouped = fdd_daily.groupby(cold_season)

    # Create DataArray to store results
    Tlon=ds.lon.data
    Tlat=ds.lat.data
    Tafiyear=np.arange(Tyear[0].data,Tyear[-1].data,1)
    afi_da=np.zeros((Tyear[-1].data-Tyear[0].data,len(Tlat),len(Tlon)))*np.nan

    for year, group in grouped:
      if year >0:
        it=np.where(Tafiyear==year)[0][0]
        # Calculate cumulative sum for this cold season
        cumulative_fdd = group.cumsum(dim='time')
        idlat,idlon=np.where(mask==1)
        for ij in range(0,len(idlat)):
          jj=idlat[ij]
          ii=idlon[ij]
          # Find critical points for AFI calculation
          cumulative_fddij=cumulative_fdd.isel(lon=ii,lat=jj)
          maxima = argrelextrema(cumulative_fddij.data, np.greater)[0]
          minima = argrelextrema(cumulative_fddij.data, np.less)[0]

          if (len(maxima)>0) & (len(minima) >0):

            # Define Min FDD as minimal extrema:
            min_idx = np.where(cumulative_fddij[minima]==cumulative_fddij[minima].min())[0][0]
            min_idx = minima[min_idx]
            min_fdd = cumulative_fddij.isel({'time': min_idx}).values
            min_date = cumulative_fddij['time'].isel({'time': min_idx}).values

            # Define Max FDD as maximal extrema happening before Min FDD:
            maxima=maxima[np.where(maxima<=min_idx)[0]]
            if len(maxima)>0:
              max_idx = np.where(cumulative_fddij[maxima]==cumulative_fddij[maxima].max())[0][0]
              max_idx = maxima[max_idx]
              max_fdd = cumulative_fddij.isel({'time': max_idx}).values
              max_date = cumulative_fddij['time'].isel({'time': max_idx}).values
            else:
              max_idx=min_idx
              max_fdd=min_fdd
              max_date=min_date

            afi_da[it,jj,ii]=max_fdd - min_fdd
          else:
            afi_da[it,jj,ii]=0.

    afi_da=xr.DataArray(afi_da,\
                        dims=['time','lat','lon'],\
                        coords={'time':Tafiyear,'lat':ds.lat,'lon':ds.lon})
    afi_da.name='air_freezing_index'
    afi_da.attrs['units']=f'degree-days ("°{units}")'
    afi_da.attrs['long_name']='Air Freezing Index (Bilotta et al. method)'
    afi_da.attrs['description']='Difference between highest and lowest extrema points of cumulative FDD'

    return afi_da



def calculate_bilotta_afi_method2(ds,temp_var,mask, units='F',
                         start_month=8, end_month=7):
    """
    Calculate Air Freezing Index (AFI) using the Bilotta et al. (2015) methodology.

    This methodology calculates the AFI as the difference between the highest and lowest
    extrema points in the cumulative freezing degree-days curve over a cold season.

    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset containing temperature data; coordinates name
        should be "time","lat","lon"
    temp_var : str, optional
        Name of temperature variable if ds is a Dataset (default: 'temperature')
    start_month : int, optional
        Start month of the cold season (default: 8 for August)
    end_month : int, optional
        End month of the cold season (default: 7 for July)

    Returns:
    --------
    xarray.DataArray
        Seasonal AFI values
    """
    if units=='F':
      threshold=32
    else:
      print('error units /n convert to F')

    # Select the temperature data
    if isinstance(ds, xr.Dataset):
        temp_data = ds[temp_var]
    else:
        temp_data = ds

    # Calculate daily departures (freezing degree-days)
    # Note: In the paper they use (Tavg - 32°F), but here we generalize with a threshold parameter
    fdd_daily = temp_data - threshold

    # Create a seasonal year coordinate for grouping
    # For August-July cold season, if month >= 8, use the current year, else use previous year
    month = temp_data['time'].dt.month
    Tyear = temp_data['time'].dt.year

    # Create a cold season year identifier (the year in which the cold season ends)
    cold_season = xr.zeros_like(Tyear)*np.nan
    for iy in range(Tyear[0].data,Tyear[-1].data):
      idt = np.where(( temp_data['time'] >= np.datetime64(f'{iy}-{start_month:02d}-01')) & \
                     ( temp_data['time'] < np.datetime64(f'{iy+1}-{end_month+1:02d}-01')))[0]
      cold_season[idt] = iy

    # Group data by cold season year and compute cumulative FDD
    fdd_daily = fdd_daily.assign_coords({'cold_season':cold_season})
    grouped = fdd_daily.groupby(cold_season)

    # Create DataArray to store results
    Tlon=ds.lon.data
    Tlat=ds.lat.data
    Tafiyear=np.arange(Tyear[0].data,Tyear[-1].data,1)
    afi_da=np.zeros((Tyear[-1].data-Tyear[0].data,len(Tlat),len(Tlon)))*np.nan

    for year, group in grouped:
      if year >0:
        it=np.where(Tafiyear==year)[0][0]
        # Calculate cumulative sum for this cold season
        cumulative_fdd = group.cumsum(dim='time')
        idlat,idlon=np.where(mask==1)
        for ij in range(0,len(idlat)):
          jj=idlat[ij]
          ii=idlon[ij]
          # Find critical points for AFI calculation
          cumulative_fddij=cumulative_fdd.isel(lon=ii,lat=jj)
          maxima = argrelextrema(cumulative_fddij.data, np.greater)[0]
          minima = argrelextrema(cumulative_fddij.data, np.less)[0]

          if (len(maxima)>0) & (len(minima) >0):

            # Define Min FDD as minimal extrema:
            max_idx = np.where(cumulative_fddij[maxima]==cumulative_fddij[maxima].max())[0][0]
            max_idx = maxima[max_idx]
            max_fdd = cumulative_fddij.isel({'time': max_idx}).values
            max_date = cumulative_fddij['time'].isel({'time': max_idx}).values

            # Define Max FDD as maximal extrema happening before Min FDD:
            minima=minima[np.where(minima>max_idx)[0]]
            if len(minima)>0:
              min_idx = np.where(cumulative_fddij[minima]==cumulative_fddij[minima].min())[0][0]
              min_idx = minima[min_idx]
              min_fdd = cumulative_fddij.isel({'time': min_idx}).values
              min_date = cumulative_fddij['time'].isel({'time': min_idx}).values
            else:
              min_idx=max_idx
              min_fdd=max_fdd
              min_date=max_date

            afi_da[it,jj,ii]=max_fdd - min_fdd
          else:
            afi_da[it,jj,ii]=0.

    afi_da=xr.DataArray(afi_da,\
                        dims=['time','lat','lon'],\
                        coords={'time':Tafiyear,'lat':ds.lat,'lon':ds.lon})
    afi_da.name='air_freezing_index'
    afi_da.attrs['units']=f'degree-days ("°{units}")'
    afi_da.attrs['long_name']='Air Freezing Index (Bilotta et al. method)'
    afi_da.attrs['description']='Difference between highest and lowest extrema points of cumulative FDD'

    return afi_da

def calculate_bilotta_afi_method3(ds,temp_var,mask, units='F',
                         start_month=8, end_month=7):
    """
    Calculate Air Freezing Index (AFI) using the Bilotta et al. (2015) methodology.

    This methodology calculates the AFI as the difference between the highest and lowest
    extrema points in the cumulative freezing degree-days curve over a cold season.

    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset containing temperature data; coordinates name
        should be "time","lat","lon"
    temp_var : str, optional
        Name of temperature variable if ds is a Dataset (default: 'temperature')
    start_month : int, optional
        Start month of the cold season (default: 8 for August)
    end_month : int, optional
        End month of the cold season (default: 7 for July)

    Returns:
    --------
    xarray.DataArray
        Seasonal AFI values
    """
    if units=='F':
      threshold=32
    else:
      print('error units /n convert to F')

    # Select the temperature data
    if isinstance(ds, xr.Dataset):
        temp_data = ds[temp_var]
    else:
        temp_data = ds

    # Calculate daily departures (freezing degree-days)
    # Note: In the paper they use (Tavg - 32°F), but here we generalize with a threshold parameter
    fdd_daily = temp_data - threshold

    # Create a seasonal year coordinate for grouping
    # For August-July cold season, if month >= 8, use the current year, else use previous year
    month = temp_data['time'].dt.month
    Tyear = temp_data['time'].dt.year

    # Create a cold season year identifier (the year in which the cold season ends)
    cold_season = xr.zeros_like(Tyear)*np.nan
    for iy in range(Tyear[0].data,Tyear[-1].data):
      idt = np.where(( temp_data['time'] >= np.datetime64(f'{iy}-{start_month:02d}-01')) & \
                     ( temp_data['time'] < np.datetime64(f'{iy+1}-{end_month+1:02d}-01')))[0]
      cold_season[idt] = iy

    # Group data by cold season year and compute cumulative FDD
    fdd_daily = fdd_daily.assign_coords({'cold_season':cold_season})
    grouped = fdd_daily.groupby(cold_season)
    # Create DataArray to store results
    Tlon=ds.lon.data
    Tlat=ds.lat.data
    Tafiyear=np.arange(Tyear[0].data,Tyear[-1].data,1)
    afi_da=np.zeros((Tyear[-1].data-Tyear[0].data,len(Tlat),len(Tlon)))*np.nan

    for year, group in grouped:
      if year >0:
        it=np.where(Tafiyear==year)[0][0]
        # Calculate cumulative sum for this cold season
        cumulative_fdd = group.cumsum(dim='time')
        idlat,idlon=np.where(mask==1)
        for ij in range(0,len(idlat)):
          jj=idlat[ij]
          ii=idlon[ij]
          # Find critical points for AFI calculation
          cumulative_fddij=cumulative_fdd.isel(lon=ii,lat=jj)
          maxima = argrelextrema(cumulative_fddij.data, np.greater)[0]
          minima = argrelextrema(cumulative_fddij.data, np.less)[0]
          afi=0
          if (len(maxima)>0) & (len(minima) >0):
            for k in range(0,min(5,len(maxima))):
              minima = argrelextrema(cumulative_fddij.data, np.less)[0]
              # Define Min FDD as minimal extrema:
              if k==0:
                max_idx = \
                        np.where(cumulative_fddij[maxima]==cumulative_fddij[maxima].max())[0][0]
                max_idx = maxima[max_idx]
              else:
                maxima = np.delete(maxima, np.where(maxima==max_idx)[0])
                max_idx =\
                        np.where(cumulative_fddij[maxima]==cumulative_fddij[maxima].max())[0][0]
                max_idx = maxima[max_idx]

              max_fdd = cumulative_fddij.isel({'time': max_idx}).values
              max_date = cumulative_fddij['time'].isel({'time': max_idx}).values

              # Define Max FDD as maximal extrema happening before Min FDD:
              minima=minima[np.where(minima>max_idx)[0]]

              if len(minima)>0:
                min_idx =\
                        np.where(cumulative_fddij[minima]==cumulative_fddij[minima].min())[0][0]
                min_idx = minima[min_idx]
                min_fdd = cumulative_fddij.isel({'time': min_idx}).values
                min_date = cumulative_fddij['time'].isel({'time': min_idx}).values
              else:
                min_idx=max_idx
                min_fdd=max_fdd
                min_date=max_date

              afi_k = max(afi,max_fdd - min_fdd)
              afi = afi_k
            afi_da[it,jj,ii]=afi

          else:
            afi_da[it,jj,ii]=0.

    afi_da=xr.DataArray(afi_da,\
                        dims=['time','lat','lon'],\
                        coords={'time':Tafiyear,'lat':ds.lat,'lon':ds.lon})
    afi_da.name='air_freezing_index'
    afi_da.attrs['units']=f'degree-days ("°{units}")'
    afi_da.attrs['long_name']='Air Freezing Index (Bilotta et al. method)'
    afi_da.attrs['description']='Difference between highest and lowest extrema points of cumulative FDD'

    return afi_da



def calculate_bilotta_afi_return_periods_normal(afi_da,afi_limit=0, return_periods=[2, 5, 10, 25, 50, 100], method='gev'):
    """
    Calculate AFI return periods using GEV or empirical distribution as in Bilotta et al.

    Parameters:
    -----------
    afi_da : xarray.DataArray
        DataArray containing seasonal AFI values
    return_periods : list, optional
        List of return periods in years to calculate (default: [2, 5, 10, 25, 50, 100])
    method : str, optional
        Method to use for return period calculation: 'gev' for Generalized Extreme Value
        or 'empirical' for non-parametric approximation (default: 'gev')

    Returns:
    --------
    dict
        Dictionary mapping return periods to AFI values
    """
    try:
        import scipy.stats as stats
    except ImportError:
        raise ImportError("SciPy is required for return period calculations.")

    # Extract AFI values
    afi_values = afi_da.values

    # Remove zeros if any (as in Bilotta et al.)
    nonzero_afis = afi_values[afi_values > afi_limit]

    # If insufficient non-zero values, return None
    if len(nonzero_afis) < 15:
        return None,0

    result = {}

    if method == 'gev':
        imethod=1
        try:
            # Calculate return period values
            for rp in return_periods:
                # Convert return period to probability (1 - 1/RP)
                p = 1 - 1/rp
                # Get the AFI value for this return period
                afi_rp = stats.norm.ppf(p,loc=nonzero_afis.mean(), scale=nonzero_afis.std())
                result[rp] = afi_rp

            # Perform goodness-of-fit test
            _, p_value =\
            stats.kstest(nonzero_afis,stats.norm.cdf,args=(nonzero_afis.mean(),nonzero_afis.std()),N=9999)

            # If poor fit, fall back to empirical method
            if p_value < 0.05:
                method = 'empirical'
                result = {}  # Reset result dictionary
        except:
            # If GEV fitting fails, fall back to empirical method
            method = 'empirical'

    if method == 'empirical':
        imethod=2
        # Sort AFI values in descending order
        sorted_afis = np.sort(nonzero_afis)[::-1]
        n = len(sorted_afis)

        # Calculate empirical return periods using Weibull formula
        for rp in return_periods:
            # Find the rank corresponding to this return period
            rank = max(1, min(n, round(n / rp)))
            # Get the AFI value for this rank
            afi_rp = sorted_afis[rank - 1]
            result[rp] = afi_rp

    return result,imethod


def calculate_bilotta_afi_return_periods(afi_da,afi_limit=0, return_periods=[2, 5, 10, 25, 50, 100], method='gev'):
    """
    Calculate AFI return periods using GEV or empirical distribution as in Bilotta et al.

    Parameters:
    -----------
    afi_da : xarray.DataArray
        DataArray containing seasonal AFI values
    return_periods : list, optional
        List of return periods in years to calculate (default: [2, 5, 10, 25, 50, 100])
    method : str, optional
        Method to use for return period calculation: 'gev' for Generalized Extreme Value
        or 'empirical' for non-parametric approximation (default: 'gev')

    Returns:
    --------
    dict
        Dictionary mapping return periods to AFI values
    """
    try:
        import scipy.stats as stats
    except ImportError:
        raise ImportError("SciPy is required for return period calculations.")

    # Extract AFI values
    afi_values = afi_da.values

    # Remove zeros if any (as in Bilotta et al.)
    nonzero_afis = afi_values[afi_values > afi_limit]

    # If insufficient non-zero values, return None
    if len(nonzero_afis) < 15:
        return None,0

    result = {}

    if method == 'gev':
        imethod=1
        try:
            # Fit GEV distribution
            shape, loc, scale = stats.genextreme.fit(nonzero_afis,\
                    loc=nonzero_afis.mean(), scale=nonzero_afis.std(), method="MLE")

            # Calculate return period values
            for rp in return_periods:
                # Convert return period to probability (1 - 1/RP)
                p = 1 - 1/rp
                # Get the AFI value for this return period
                afi_rp = stats.genextreme.ppf(p, shape, loc, scale)
                result[rp] = afi_rp

            # Perform goodness-of-fit test
            _, p_value = stats.kstest(nonzero_afis, 'genextreme', args=(shape, loc, scale),N=9999)

            # If poor fit, fall back to empirical method
            if p_value < 0.05:
                method = 'empirical'
                result = {}  # Reset result dictionary
        except:
            # If GEV fitting fails, fall back to empirical method
            method = 'empirical'

    if method == 'empirical':
        imethod=2
        # Sort AFI values in descending order
        sorted_afis = np.sort(nonzero_afis)[::-1]
        n = len(sorted_afis)

        # Calculate empirical return periods using Weibull formula
        for rp in return_periods:
            # Find the rank corresponding to this return period
            rank = max(1, min(n, round(n / rp)))
            # Get the AFI value for this rank
            afi_rp = sorted_afis[rank - 1]
            result[rp] = afi_rp

    return result,imethod

def calculate_bilotta_afi_return_periods_weibull(afi_da,afi_limit=0,\
        return_periods=[2, 5, 10, 25, 50, 100], method='gev'):
    """
    Calculate AFI return periods using GEV or empirical distribution as in Bilotta et al.

    Parameters:
    -----------
    afi_da : xarray.DataArray
        DataArray containing seasonal AFI values
    return_periods : list, optional
        List of return periods in years to calculate (default: [2, 5, 10, 25, 50, 100])
    method : str, optional
        Method to use for return period calculation: 'gev' for Generalized Extreme Value
        or 'empirical' for non-parametric approximation (default: 'gev')

    Returns:
    --------
    dict
        Dictionary mapping return periods to AFI values
    """
    try:
        import scipy.stats as stats
    except ImportError:
        raise ImportError("SciPy is required for return period calculations.")

    # Extract AFI values
    afi_values = afi_da.values

    # Remove zeros if any (as in Bilotta et al.)
    nonzero_afis = afi_values[afi_values > afi_limit]

    # If insufficient non-zero values, return None
    if len(nonzero_afis) < 15:
        return [],0

    result = {}

    if method == 'gev':
        imethod=1
        try:
            # Fit GEV distribution
            params = stats.weibull_min.fit(nonzero_afis, floc=0,\
                    method="MLE") # floc=0 : fixeslocation parameter at 0
            c_est, loc_est, scale_est = params

            # Calculate return period values
            for rp in return_periods:
                # Convert return period to probability (1 - 1/RP)
                p = 1 - 1/rp
                # Get the AFI value for this return period
                afi_rp = stats.weibull_min(c_est, loc_est,scale_est).ppf(p)
                result[rp] = afi_rp

            # Perform goodness-of-fit test
            _, p_value = stats.kstest(nonzero_afis, stats.weibull_min.name, params,N=9999)   # return p-value
            # If poor fit, fall back to empirical method
            if p_value < 0.05:
                method = 'empirical'
                result = {}  # Reset result dictionary
        except:
            # If GEV fitting fails, fall back to empirical method
            method = 'empirical'

    if method == 'empirical':
        imethod=2
        # Sort AFI values in descending order
        sorted_afis = np.sort(nonzero_afis)[::-1]
        n = len(sorted_afis)

        # Calculate empirical return periods using Weibull formula
        for rp in return_periods:
            # Find the rank corresponding to this return period
            rank = max(1, min(n, round(n / rp)))
            # Get the AFI value for this rank
            afi_rp = sorted_afis[rank - 1]
            result[rp] = afi_rp

    return result,imethod


def calculate_bilotta_afi_return_periods_gumbel(afi_da,afi_limit=0,\
        return_periods=[2, 5, 10, 25, 50, 100], method='gev'):
    """
    Calculate AFI return periods using GEV or empirical distribution as in Bilotta et al.

    Parameters:
    -----------
    afi_da : xarray.DataArray
        DataArray containing seasonal AFI values
    return_periods : list, optional
        List of return periods in years to calculate (default: [2, 5, 10, 25, 50, 100])
    method : str, optional
        Method to use for return period calculation: 'gev' for Generalized Extreme Value
        or 'empirical' for non-parametric approximation (default: 'gev')

    Returns:
    --------
    dict
        Dictionary mapping return periods to AFI values
    """
    try:
        import scipy.stats as stats
    except ImportError:
        raise ImportError("SciPy is required for return period calculations.")

    # Extract AFI values
    afi_values = afi_da.values

    # Remove zeros if any (as in Bilotta et al.)
    nonzero_afis = afi_values[afi_values > afi_limit]

    # If insufficient non-zero values, return None
    if len(nonzero_afis) < 15:
        return [],0

    result = {}

    if method == 'gev':
        imethod=1
        try:
            # Fit GEV distribution
            shape, loc, scale = stats.genextreme.fit(nonzero_afis,fc=0, \
            method="MLE") # fc=0 fixes the shape parameter at 0

            # Calculate return period values
            for rp in return_periods:
                # Convert return period to probability (1 - 1/RP)
                p = 1 - 1/rp
                # Get the AFI value for this return period
                afi_rp = stats.genextreme.ppf(p, shape, loc, scale)
                result[rp] = afi_rp

            # Perform goodness-of-fit test
            _, p_value = stats.kstest(nonzero_afis, 'genextreme', args=(shape, loc, scale),N=9999)

            # If poor fit, fall back to empirical method
            if p_value < 0.05:
                method = 'empirical'
                result = {}  # Reset result dictionary
        except:
            # If GEV fitting fails, fall back to empirical method
            method = 'empirical'

    if method == 'empirical':
        imethod=2
        # Sort AFI values in descending order
        sorted_afis = np.sort(nonzero_afis)[::-1]
        n = len(sorted_afis)

        # Calculate empirical return periods using Weibull formula
        for rp in return_periods:
            # Find the rank corresponding to this return period
            rank = max(1, min(n, round(n / rp)))
            # Get the AFI value for this rank
            afi_rp = sorted_afis[rank - 1]
            result[rp] = afi_rp

    return result,imethod

def calculate_frost_depth(afi_100yr):
    """
    Calculate estimated frost depth from 100-year AFI using Brown's formula as in Bilotta et al.

    Parameters:
    -----------
    afi_100yr : float
        100-year return period AFI value in °C-days

    Returns:
    --------
    float
        Estimated frost depth in meters
    """
    # Convert from °C to °F if needed
    # Note: This assumes afi_100yr is in °C-days. If in °F-days, no conversion needed.
    # afi_100yr_f = afi_100yr * 1.8

    # Brown's formula as used in Bilotta et al.
    # Note: The coefficient 0.0174 already accounts for °F units in the original formula
    # If using °C-days, this coefficient would need adjustment
    frost_depth = 0.0174 * (afi_100yr ** 0.67)

    return frost_depth

def plot_cumulative_fdd(ds, ilon,ilat,temp_var='temperature', units='F',year=None,
                              start_month=8, end_month=7):
    """
    Plot cumulative FDD curve for a specific cold season to visualize the Bilotta AFI calculation.

    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset containing temperature data coordinates name
        should be "time","lat","lon"
    temp_var : str, optional
        Name of temperature variable if ds is a Dataset (default: 'temperature')
    threshold : float, optional
        Freezing threshold temperature (default: 0°C)
    year : int, optional
        The year representing the end of the cold season to plot (default: None, uses the first full season)
    start_month : int, optional
        Start month of the cold season (default: 8 for August)
    end_month : int, optional
        End month of the cold season (default: 7 for July)
    """
    if units=='F':
      threshold=32
    else:      print('error units /n convert to F')

    # Select the temperature data
    temp_data = ds[temp_var].sel(lon=ilon,lat=ilat).load()
    Tyear = temp_data['time'].dt.year
    # Calculate daily departures (freezing degree-days)
    # Note: In the paper they use (Tavg - 32°F), but here we generalize with a threshold parameter
    fdd_daily = temp_data - threshold

    # Create a cold season year identifier (the year in which the cold season ends)
    cold_season = xr.zeros_like(Tyear)
    for iy in range(Tyear[0].data,Tyear[-1].data):
      idt = np.where(( temp_data['time'] >= np.datetime64(f'{iy}-{start_month:02d}-01')) & \
                     ( temp_data['time'] < np.datetime64(f'{iy+1}-{end_month+1:02d}-01')))[0]
      cold_season[idt] = iy

    # If no specific year is provided, use te first full cold season year
    if year is None:
        year = temp_data['time'].dt.year.data[0]

    # Group data by cold season year and compute cumulative FDD
    fdd_daily = fdd_daily.assign_coords({'cold_season':cold_season})

    # Filter data for the selected cold season
    season_data = fdd_daily.where(fdd_daily.cold_season==year,drop=True)

    # Calculate cumulative FDD
    cumulative_fdd = season_data.cumsum(dim='time')

    # Find critical points for AFI calculation
    maxima = argrelextrema(cumulative_fdd.data, np.greater)[0]
    minima = argrelextrema(cumulative_fdd.data, np.less)[0]
    afi=0
    if (len(maxima)>0) & (len(minima) >0):
      # Define Min FDD as minimal extrema:
      min_idx = np.where(cumulative_fdd[minima]==cumulative_fdd[minima].min())[0][0]
      min_idx = minima[min_idx]
      min_fdd = cumulative_fdd.isel({'time': min_idx}).values
      min_date = cumulative_fdd['time'].isel({'time': min_idx}).values

      # Define Max FDD as maximal extrema happening before Min FDD:
      maxima=maxima[np.where(maxima<=min_idx)[0]]

      if len(maxima)>0:
        max_idx = np.where(cumulative_fdd[maxima]==cumulative_fdd[maxima].max())[0][0]
        max_idx = maxima[max_idx]
        max_fdd = cumulative_fdd.isel({'time': max_idx}).values
        max_date = cumulative_fdd['time'].isel({'time': max_idx}).values
      else:
        min_idx=max_idx
        max_fdd=min_fdd
        max_date=min_date

      afi = max_fdd - min_fdd

    # Create the plot
    fig, axs =plt.subplots(nrows=2,ncols=1,figsize=(14,10),)
    axs=axs.flatten()
    season_data.plot(ax=axs[0])
    axs[0].axhline(y=0, color='darkgrey', linestyle='--', alpha=0.75)
    axs[0].grid(True)
    axs[0].set_ylabel(f'Freezing Degree-Days ("°F")')
    max_idx = np.where(cumulative_fdd[maxima]==cumulative_fdd[maxima].max())[0][0]
    max_idx = maxima[max_idx]
    max_fdd = cumulative_fdd.isel({'time': max_idx}).values
    max_date = cumulative_fdd['time'].isel({'time': max_idx}).values

    min_idx = np.where(cumulative_fdd[minima]==cumulative_fdd[minima].min())[0][0]
    min_idx = minima[min_idx]
    min_fdd = cumulative_fdd.isel({'time': min_idx}).values
    min_date = cumulative_fdd['time'].isel({'time': min_idx}).values

    afi = max_fdd - min_fdd

    # Create the plot
    fig, axs =plt.subplots(nrows=2,ncols=1,figsize=(14,10),)
    axs=axs.flatten()
    season_data.plot(ax=axs[0])
    axs[0].axhline(y=0, color='darkgrey', linestyle='--', alpha=0.75)
    axs[0].grid(True)
    axs[0].set_ylabel(f'Freezing Degree-Days ("°F")')
    axs[0].set_title(f'Freezing Degree-Days for Cold Season \
        {year}-{year+1} at {ilon}$\degree$E;{ilat}$\degree$N')
    cumulative_fdd.plot(ax=axs[1],label='Cumulative FDD')

    # Mark the extrema points
    axs[1].scatter(max_date, max_fdd, color='red', marker='o', s=100, label=f'Max FDD: {max_fdd:.1f}')
    axs[1].scatter(min_date, min_fdd, color='blue', marker='o', s=100, label=f'Min FDD: {min_fdd:.1f}')

    # Add annotation for AFI
    axs[1].axhline(y=max_fdd, color='gray', linestyle='--', alpha=0.5)
    axs[1].axhline(y=min_fdd, color='gray', linestyle='--', alpha=0.5)
    axs[1].annotate(f'AFI = {afi:.1f}',
            xy=(season_data.time[round((max_idx + min_idx)/2)], (max_fdd + min_fdd)/2),
            xytext=(0, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->'), ha='center')

    axs[1].set_title(f'Cumulative Freezing Degree-Days for Cold Season \
        {year}-{year+1} at {ilon}$\degree$E;{ilat}$\degree$N')
    axs[1].set_ylabel(f'Freezing Degree-Days ("°F")')
    axs[1].legend()
    axs[1].grid(True)
    plt.show()

def plot_cumulative_fdd_v1(ds, ilon,ilat,temp_var='temperature', units='F',year=None,
                              start_month=8, end_month=7):
    """
    Plot cumulative FDD curve for a specific cold season to visualize the Bilotta AFI calculation.

    Parameters:
    -----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset containing temperature data coordinates name
        should be "time","lat","lon"
    temp_var : str, optional
        Name of temperature variable if ds is a Dataset (default: 'temperature')
    threshold : float, optional
        Freezing threshold temperature (default: 0°C)
    year : int, optional
        The year representing the end of the cold season to plot (default: None, uses the first full season)
    start_month : int, optional
        Start month of the cold season (default: 8 for August)
    end_month : int, optional
        End month of the cold season (default: 7 for July)
    """
    if units=='F':
      threshold=32
    else:      print('error units /n convert to F')

    # Select the temperature data
    temp_data = ds[temp_var].sel(lon=ilon,lat=ilat).load()
    Tyear = temp_data['time'].dt.year
    # Calculate daily departures (freezing degree-days)
    # Note: In the paper they use (Tavg - 32°F), but here we generalize with a threshold parameter
    fdd_daily = temp_data - threshold

    # Create a cold season year identifier (the year in which the cold season ends)
    cold_season = xr.zeros_like(Tyear)
    for iy in range(Tyear[0].data,Tyear[-1].data):
      idt = np.where(( temp_data['time'] >= np.datetime64(f'{iy}-{start_month:02d}-01')) & \
                     ( temp_data['time'] < np.datetime64(f'{iy+1}-{end_month+1:02d}-01')))[0]
      cold_season[idt] = iy

    # If no specific year is provided, use te first full cold season year
    if year is None:
        year = temp_data['time'].dt.year.data[0]

    # Group data by cold season year and compute cumulative FDD
    fdd_daily = fdd_daily.assign_coords({'cold_season':cold_season})

    # Filter data for the selected cold season
    season_data = fdd_daily.where(fdd_daily.cold_season==year,drop=True)

    # Calculate cumulative FDD
    cumulative_fdd = season_data.cumsum(dim='time')
    maxima = argrelextrema(cumulative_fdd.data, np.greater)[0]
    minima = argrelextrema(cumulative_fdd.data, np.less)[0]
    # Find critical points for AFI calculation
    max_idx = np.where(cumulative_fdd[maxima]==cumulative_fdd[maxima].max())[0][0]
    max_idx = maxima[max_idx]
    max_fdd = cumulative_fdd.isel({'time': max_idx}).values
    max_date = cumulative_fdd['time'].isel({'time': max_idx}).values

    min_idx = np.where(cumulative_fdd[minima]==cumulative_fdd[minima].min())[0][0]
    min_idx = minima[min_idx]
    min_fdd = cumulative_fdd.isel({'time': min_idx}).values
    min_date = cumulative_fdd['time'].isel({'time': min_idx}).values

    afi = max_fdd - min_fdd

    # Create the plot
    fig, axs =plt.subplots(nrows=2,ncols=1,figsize=(14,10),)
    axs=axs.flatten()
    season_data.plot(ax=axs[0])
    axs[0].axhline(y=0, color='darkgrey', linestyle='--', alpha=0.75)
    axs[0].grid(True)
    axs[0].set_ylabel(f'Freezing Degree-Days ("°F")')
    axs[0].set_title(f'Freezing Degree-Days for Cold Season \
        {year}-{year+1} at {ilon}$\degree$E;{ilat}$\degree$N')
    cumulative_fdd.plot(ax=axs[1],label='Cumulative FDD')

    # Mark the extrema points
    axs[1].scatter(max_date, max_fdd, color='red', marker='o', s=100, label=f'Max FDD: {max_fdd:.1f}')
    axs[1].scatter(min_date, min_fdd, color='blue', marker='o', s=100, label=f'Min FDD: {min_fdd:.1f}')

    # Add annotation for AFI
    axs[1].axhline(y=max_fdd, color='gray', linestyle='--', alpha=0.5)
    axs[1].axhline(y=min_fdd, color='gray', linestyle='--', alpha=0.5)
    axs[1].annotate(f'AFI = {afi:.1f}',
            xy=(season_data.time[round((max_idx + min_idx)/2)], (max_fdd + min_fdd)/2),
            xytext=(0, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->'), ha='center')

    axs[1].set_title(f'Cumulative Freezing Degree-Days for Cold Season \
        {year}-{year+1} at {ilon}$\degree$E;{ilat}$\degree$N')
    axs[1].set_ylabel(f'Freezing Degree-Days ("°F")')
    axs[1].legend()
    axs[1].grid(True)
    plt.show()


# Example usage with xarray data
def example_usage():
    """
    Example showing how to use the Bilotta AFI calculation functions with xarray data.
    """
    # Create sample temperature data for multiple years
    # This is just for demonstration - you'll use your actual xarray dataset
    dates = pd.date_range('2017-01-01', '2023-12-31', freq='D')
    ndays = len(dates)

    # Create synthetic temperatures with seasonal cycle
    t = np.arange(ndays) * 2 * np.pi / 365.25
    temp = 10 - 15 * np.cos(t) + np.random.normal(0, 3, ndays)

    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            'temperature': (['time'], temp)
        },
        coords={
            'time': dates
        }
    )
    ds.temperature.attrs['units'] = '°C'

    # Calculate Bilotta AFI for each cold season
    afi_values = calculate_bilotta_afi(ds, temp_var='temperature', threshold=0)

    # Calculate return periods
    return_periods = calculate_bilotta_afi_return_periods(afi_values)

    # Calculate frost depth for 100-year return period
    if return_periods is not None and 100 in return_periods:
        frost_depth = calculate_frost_depth(return_periods[100])
        print(f"100-year AFI: {return_periods[100]:.1f} °C-days")
        print(f"Estimated frost depth: {frost_depth:.2f} m")

    # Plot results
    plt.figure(figsize=(10, 6))
    afi_values.plot.bar()
    plt.title('Air Freezing Index (Bilotta et al. method)')
    plt.ylabel('AFI (°C-days)')
    plt.grid(True, axis='y')
    plt.show()

    # Plot cumulative FDD curve for a specific cold season
    plot_cumulative_fdd(ds, year=2023)

# If you want to run the example
# example_usage()
