import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

def make_df(file_name):
    '''Creates a Dataframe that drops any empty columns and those without an AMBULANCE entry'''
    df = pd.read_csv(file_name)
    df = df.dropna(subset=['TYP_DESC', 'INCIDENT_DATE', 'INCIDENT_TIME', 'Latitude', 'Longitude'])
    df = df[df['TYP_DESC'].str.contains("AMBULANCE")]
    return df

def add_date_time_features(df):
    '''Adds an additional column WEEK_DAY corresponding to the date in INCIDENT_DATE,
    and another column INCIDENT_MIN, taking the time from INCIDENT_TIME and stores it as minutes
    since midnight
    '''
    df['WEEK_DAY'] = -1
    df['INCIDENT_MIN'] = 0.0
    for idx, row in df.iterrows():
        datat = pd.to_datetime(row['INCIDENT_DATE'])
        df.at[idx, 'WEEK_DAY'] = datat.dayofweek
        tim = pd.to_datetime(row['INCIDENT_TIME'], format='%H:%M:%S').time()
        time_since_mid = tim.hour * 60 + tim.minute  + tim.second / 60.0
        df.at[idx, 'INCIDENT_MIN'] = time_since_mid
    return df

def filter_by_time(df, days=None, start_min = 0, end_min = 1439):
    '''Returns a DataFrame with entries restricted to weekdays in days and within start and
    end incident times'''
    rdrop = []
    # Not Sure if end_min is a digit or will pass in 'am' or 'pm'
    if isinstance(end_min,str):
        if 'am' in end_min:
            end_min = 60 * 11 + 59
        elif 'pm' in end_min:
            end_min = 1439
    for  idx, row in df.iterrows():
        incident = pd.to_numeric(row['INCIDENT_MIN'], errors='coerce')
        day = row['WEEK_DAY']
        if (incident < start_min or incident > end_min) or (days is not None and day not in days):
            rdrop.append(idx)
    df = df.drop(rdrop)
    return df

def compute_kmeans(df, num_clusters = 8, n_init = 'auto', random_state = 1870):
    '''Compute kmean clusters using sklearn function'''
    if n_init == 'auto':
        n_init = 10
    columns = df[['Latitude','Longitude']]
    kclusters = KMeans(n_clusters=num_clusters, n_init=n_init,
                       random_state=random_state).fit(columns)
    return kclusters.cluster_centers_, kclusters.labels_

def compute_gmm(df, num_clusters = 8, random_state = 1870):
    '''Compute kmean clusters using sklearn function'''
    columns = df[['Latitude','Longitude']]
    gmixture = GaussianMixture(n_components=num_clusters,random_state=random_state).fit(columns)
    return gmixture.predict(columns)

def compute_agglom(df, num_clusters=8, linkage='ward'):
    '''Rund the Agglomerative model using sklearn function'''
    columns = df[['Latitude','Longitude']]
    agglom = AgglomerativeClustering(n_clusters=num_clusters,linkage=linkage).fit_predict(columns)
    return agglom

def compute_spectral(df, num_clusters = 8, affinity='rbf',random_state=1870):
    '''Runs the Spectral Clustering model using sklearn function'''
    columns = df[['Latitude','Longitude']]
    spectral = SpectralClustering(n_clusters=num_clusters,affinity=affinity,
                                  random_state=random_state).fit_predict(columns)
    return spectral

def compute_explained_variance(df, k_vals = None, random_state = 55):
    '''Returns a list of squared distances of samples to their closest cluster center for each K'''
    #If k_vals is None
    columns = df[['Latitude','Longitude']]
    df = list(columns.itertuples(index=False, name=None))
    result = []
    if k_vals is None:
        k_vals = [1,2,3,4,5]
    for itera in k_vals:
        kcluster = KMeans(n_clusters=itera, random_state=random_state).fit(columns)
        momentum = 0
        for key in df:
            middle = kcluster.cluster_centers_[kcluster.predict([key])[0]]
            sqd = 0
            for ele, val in enumerate(key):
                sqd += (val - middle[ele]) ** 2
            momentum += sqd
        result.append(momentum)
    return result

def test_add_date_time_features():
    '''Test function for add_time_features function'''
    columns = {'INCIDENT_DATE':['2023-11-17', ['2023-11-10'],['2023-11-03']],
               'INCIDENT_TIME': ['10:21:00', '01:11:00', '06:06:00']}
    df = add_date_time_features(pd.DataFrame(columns))
    assert df['INCIDENT_MIN'].tolist() == [621,71,366]
    assert 'INCIDENT_MIN' in df.columns
    assert 'WEEK_DAY' in df.columns
    assert df['WEEK_DAY'].tolist() == [4,4,4]

def test_filter_by_time():
    '''Test function for filter_by_time function'''
    columns = {'INCIDENT_MIN':[100,200,300,1000],
               'WEEK_DAY':[3,4,5,6]}
    df = filter_by_time(pd.DataFrame(columns))
    df = filter_by_time(df, days=[3,4,5], start_min=100, end_min=300)
    assert df['INCIDENT_MIN'].tolist() == [100,200,300]
    assert df['WEEK_DAY'].tolist() == [3,4,5]
