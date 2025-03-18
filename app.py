#MODULES
from opensearchpy.connection.http_requests import RequestsHttpConnection
from opensearchpy.exceptions import OpenSearchException
from aws_requests_auth.aws_auth import AWSRequestsAuth
from opensearchpy.client import OpenSearch
import plotly.express as px
import streamlit as st
import pandas as pd
import datetime
import calendar
import warnings
import requests
import json
import config


#FUNCTION DEFFINITIONS
@st.cache_data
def create_client(hosts, _es_auth_prod):

    client = OpenSearch(
                        hosts= hosts,
                        port= 443,
                        http_auth=_es_auth_prod,
                        connection_class=RequestsHttpConnection
                        )

    return client

# downloads data from an index on Elastic-Search
def download_data(_client, index, query):
    
    data = []
    scroll_id = None
    try:

        response  = _client.search(index= index, body= query, scroll='2m', size= 1000 )

        # this scroll ID is used to fetch subsequent batches
        scroll_id = response['_scroll_id']
        # The actual data from the initial query is stored here
        hits = response['hits']['hits']
        while len(hits) > 0:
            data.extend(hits)
            response  = _client.scroll(scroll_id= scroll_id, scroll= '2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
    
    except OpenSearchException as e:
        print(f'an ERROR occured,\n {e}')
    
    finally:
        try:
            if scroll_id != None:
                _client.clear_scroll(scroll_id= scroll_id)
        except OpenSearchException as e:
            print(f'FAILED to terminate the scroll {e}')
    
    return data

# creates the query and downloads the data
def get_index_util(_client, index, start_date, end_date, schedular, columns):
    
    index = index + str(end_date.year)

    query = {
        'query': {
            'bool': {
                'must': [
                    {'term' :
                        { 'schedular.keyword' : {'value' : schedular}}},
                    {'range' :
                        {'date' : {'gte' : start_date.isoformat(),
                                   'lt' :    end_date.isoformat()}}}
                ]
            }
        },
        '_source' : columns,
    }
    
    data = download_data(_client, index, query)

    # 2 PREPROCESS
    lst = []
    for datum in data:
        lst.append( datum['_source'] )
    
    df = pd.DataFrame(lst)
    
    return df

def get_index(_client, index, columns, start_date, end_date, schedular):
     # 0 SPECIFY
    #columns = ['date', 'isin', 'schedular', 'mean_directionality', 'root_mean_squared_error'] #, 'actual_monthly_returns', 'monthly_predictions'
    #index = 'eq_etf_model_metrics_'
    
    # 1 GET
    if start_date.year != end_date.year:
        
        last  = datetime.datetime(year= start_date.year, month= 12, day= 31).date()
        first = datetime.datetime(year=   end_date.year, month= 1,  day= 1).date()
        
        df1= get_index_util(_client, index, start_date, last, schedular, columns)
        df2= get_index_util(_client, index, first, end_date, schedular, columns)

        df = pd.concat([df1, df2], ignore_index= True)
    else:
        df = get_index_util(_client, index, start_date, end_date, schedular, columns)

    if df.empty:
        st.error('Data Unavailable')
        st.stop()

    # <-- If I include 'tic', then some ETFs have NaN in tic AND I cannot exclude those
    tmp = pd.Series([True]*len(df))
    for col in df.columns:
        tmp &= df[col].notna()
    df = df.loc[tmp, :]

    df['date'] = df['date'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).dt.date
    df['Date'] = df['date'].apply(func= lambda x: x.strftime('%b %d, %Y'))
    df.sort_values(by= ['date'], inplace= True)
    df.reset_index(inplace= True, drop= True)
    
    mapper = {'root_mean_squared_error':'rmse', 'mean_directionality':'mean_dir', 'avg_confidence_score':'mean_conf', 'monthly_predictions':'predicted_er'}
    df.rename(mapper= mapper, axis= 1, inplace= True)
    
    return df

def get_tags(tags):
    
    url = config.cred_1['url']
    dfs = []
    public_etfs = pd.DataFrame()
    
    for k in tags.keys():
        response = requests.get(url+k)
    
        if 200 == response.status_code:
            text = response.text
        else:
            raise Exception(f'FAILED to get data for {k}')
    
        js = json.loads(text)
        df = pd.DataFrame(js['data']['masteractivefirms_' + k])
        df = df.loc[: , ['isin', 'tic', 'exchange']]
        if k == 'goetf' or k == 'sectors':
            tmp = df[df['exchange'] != 'private'].loc[:, ['isin', 'exchange']]
            public_etfs = pd.concat([public_etfs, tmp], ignore_index= True)
        df.drop(['exchange'], axis= 1, inplace= True)    
        df[tags[k]] = 1
        dfs.append( df )
    
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = pd.merge(left = df, right= dfs[i], on= ['isin', 'tic'], how= 'outer')
    
    public_etfs['public'] = 1
    public_etfs.drop(['exchange'], axis= 1, inplace= True)
    df = pd.merge(left = df, right= public_etfs, on= ['isin'], how= 'left')
    
    df.fillna(value= 0, inplace= True)
    return df

@st.cache_data(ttl = 1800)
def get_data(index, columns, start_date, end_date, schedular= 'Monthly'):

    cred= config.cred_2 
    es_auth_cred =  AWSRequestsAuth(
                            aws_access_key= cred['aws_access_key'],
                            aws_secret_access_key= cred['aws_secret_access_key'],
                            aws_host= cred['aws_host'],
                            aws_region= cred['aws_region'],
                            aws_service= cred['aws_service'])
       
    hosts = [cred['host']]
    
    client = create_client(hosts, es_auth_cred)
    
    df = get_index(client, index, columns, start_date, end_date, schedular)

    if index == 'eq_etf_model_metrics_':
        tags = {'goetf': 'AIGO', 'bnpetf': 'BNP', 'dbetf': 'DB', 'sectors': 'AISRT'}
        
        i2t = get_tags(tags)
        dfs = {}
        for product in ['AIGO', 'BNP', 'DB', 'AISRT']:
            dfs[product] = pd.merge(left= df, right= i2t[i2t[product] > 0].loc[:, ['isin', 'tic', 'public']], on= ['isin'], how= 'inner')
        
        return dfs
    else:
        return df

# Returns the no. of days in a month (for a specific year)
@st.cache_data
def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]
    
def plot_directionality(df):

    fig = px.histogram(df, x= 'mean_dir', color= 'Date', barmode='group', color_discrete_sequence= color_palette, nbins= 11, range_x= [-9, 109])
    fig.update_traces(
        xbins=dict(
            start=0,  # Start of the first bin
            end=100,  # End of the last bin
            size=10   # Bin width (10 units)
        )
    )
    fig.update_layout(xaxis_title='date', yaxis_title='count', xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 10), bargap= 0)
    return fig

def isins_with_the_worst_rmse(df, n, flag= False):
    
    grouped = df.groupby('isin')
    
    a = grouped['rmse'].unique()
    a = pd.DataFrame({'isin': a.index, 'rmse': a.apply(lambda x: sum(x)/len(x))}).nlargest(n= n, columns= 'rmse')
    
    a.reset_index(drop= True, inplace= True)
    cols = []
    if flag:
        tmp = df.loc[:, ['isin', 'company', 'ticker']]
        tmp.drop_duplicates(inplace= True)
        a = pd.merge(left= a, right= tmp, how= 'left', on= 'isin')
        cols = ['isin', 'ticker', 'company', 'rmse']
    else:
        tmp = df.loc[:, ['isin', 'tic']]
        tmp.drop_duplicates(inplace= True)
        a = pd.merge(left= a, right= tmp, how= 'left', on= 'isin')
        a.rename({'tic' : 'ticker'}, axis= 1, inplace= True)
        cols = ['isin', 'ticker', 'rmse']
    a.index += 1
    a = a.loc[:, cols]
    
    return a, grouped

def isins_with_the_worst_directionality(df_grouped, n, flag= False):
    
    #grouped = df.groupby('isin')
    
    a = df_grouped['mean_dir'].unique()
    
    # Create a new DataFrame with the unique values and their corresponding counts
    a = pd.DataFrame({'isin': a.index, 'mean_directionality': a.apply(lambda x: sum(x)/len(x))}).nsmallest(n= n, columns= 'mean_directionality')
    a['mean_directionality'] = a['mean_directionality'].apply(round)
    a.reset_index(drop= True, inplace= True)
    cols = []
    if flag:
        tmp = df.loc[:, ['isin', 'company', 'ticker']]
        tmp.drop_duplicates(inplace= True)
        a = pd.merge(left= a, right= tmp, how= 'left', on= 'isin')
        cols = ['isin', 'ticker', 'company', 'mean_directionality']
    else:
        tmp = df.loc[:, ['isin', 'tic']]
        tmp.drop_duplicates(inplace= True)
        a = pd.merge(left= a, right= tmp, how= 'left', on= 'isin')
        a.rename({'tic' : 'ticker'}, axis= 1, inplace= True)
        cols = ['isin', 'ticker', 'mean_directionality']
        
    a.index += 1
    a = a.loc[:, cols]

    return a


def create_pane_directionality(dfs, k):
        st.markdown(f"<h6 style='text-align: center;'>{k}</h6>", unsafe_allow_html=True)
        fig = plot_directionality(dfs[k])
        st.plotly_chart(fig)

def determine_ymax(dfs):
    _max_ = -1
    for k in dfs.keys():
        tmp = round(dfs[k].loc[:, ['rmse']].max().iloc[0]) + 1
        if tmp > _max_:
            _max_ = tmp
    return _max_
        
def create_pane_rmse(dfs, k, y_max):
    st.markdown(f"<h6 style='text-align: center;'>{k}</h6>", unsafe_allow_html=True)
    fig = px.histogram(dfs[k], x= "rmse", marginal= 'violin', opacity= 0.5, range_x= [0, y_max], nbins= 10)
    st.plotly_chart(fig)
    
st.set_page_config(layout= "wide")

#warnings.simplefilter("ignore", UserWarning)

color_palette = px.colors.qualitative.Plotly

# GLOBAL VARIABLES
cred= config.cred_2   
es_auth_cred = AWSRequestsAuth(
                            aws_access_key= cred['aws_access_key'],
                            aws_secret_access_key= cred['aws_secret_access_key'],
                            aws_host= cred['aws_host'],
                            aws_region= cred['aws_region'],
                            aws_service= cred['aws_service'])
   
hosts = [cred['host']]

years = pd.Series(reversed(range(2008, 2026)))
months = pd.Series((range(1, 13, 1)))
months = pd.to_datetime(months, format='%m').dt.month_name().str.slice(stop= 3)

# GET today's date
today = datetime.datetime.now().date()
past  = today - datetime.timedelta(weeks= 1)

#MAIN

# creates 3 columns of the specified widths
start, _, end = st.columns([4, 1, 4])

# Accepts two inputs (start & end dates)
with start:
    # Creates a list of days, months & years, for the user to select from
    st.markdown("<h6 style='text-align: center;'>Start date</h6>", unsafe_allow_html=True)
    day, month, year = st.columns(3)

    dd = past.day
    mm = past.strftime("%b")
    yy = past.year
    with year:
        y1 = int( st.selectbox('Select a year', years, key= '1', index= int(years[years == yy].index[0])) )
    with month:
        m1 = st.selectbox('Select a month', months, key= '2', index= int(months[months == mm].index[0]))
        m1 = months[months == m1].index[0] + 1
    with day:
        num_days = get_days_in_month(y1, m1)
        d1 = st.selectbox('Select Day', pd.Series(range(1, num_days + 1)), key= '3', index= dd-1)

with end:
    st.markdown("<h6 style='text-align: center;'>End date</h6>", unsafe_allow_html=True)
    day, month, year = st.columns(3)
    dd = today.day
    mm = today.strftime("%b")
    yy = today.year
    with year:
        y2 = int(st.selectbox('Select a year', years, key= '4', index= int(years[years == yy].index[0])))
    with month:
        m2 = st.selectbox('Select a month', months, key= '5', index= int(months[months == mm].index[0]))
        m2 = months[months == m2].index[0] + 1
    with day:
        num_days = get_days_in_month(y2, m2)
        d2 = st.selectbox('Select Day', pd.Series(range(1, num_days + 1)), key= '6', index= dd-1)

# converts input string to a datetime object
start_date = datetime.datetime(year= y1, month= m1, day= d1).date() 
end_date   = datetime.datetime(year= y2, month= m2, day= d2).date()

# error check
if end_date <= start_date:
    st.stop()


st.divider()
st.markdown("<h4 style='text-align: center;'>Level 1: Product view</h4>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Directionality count</h5>", unsafe_allow_html=True)
columns = ['date', 'isin', 'schedular', 'mean_directionality', 'root_mean_squared_error'] #, 'actual_monthly_returns', 'monthly_predictions'
index = 'eq_etf_model_metrics_'
dfs = get_data(index, columns, start_date, end_date, schedular= 'Monthly')
l1, r1 = st.columns(2)
l2, r2 = st.columns(2)
    
with l1:    
    k = 'AIGO'
    create_pane_directionality(dfs, k)
with r1:    
    k = 'AISRT'
    create_pane_directionality(dfs, k)

with l2:    
    k = 'BNP'
    create_pane_directionality(dfs, k)

with r2:    
    k = 'DB'
    create_pane_directionality(dfs, k)


st.markdown("<h5 style='text-align: center;'>RMSE distribution</h5>", unsafe_allow_html=True)
l1, r1 = st.columns(2)
l2, r2 = st.columns(2)

y_max = determine_ymax(dfs)
with l1:
    k = 'AIGO'
    create_pane_rmse(dfs, k, y_max)
with r1:
    k = 'AISRT'
    create_pane_rmse(dfs, k, y_max)

with l2:
    k = 'BNP'
    create_pane_rmse(dfs, k, y_max)

with r2:
    k = 'DB'
    create_pane_rmse(dfs, k, y_max)

st.text('')
st.text('')
st.divider()
st.text("")
st.text("")
st.markdown("<h4 style='text-align: center;'>Level 2: ETF view</h4>", unsafe_allow_html=True)
df =pd.DataFrame()

if 'product' not in st.session_state:
    st.session_state['product'] = None
if 'etfs' not in st.session_state:
    st.session_state['etfs'] = []
if 'flag' not in st.session_state:
    st.session_state['flag'] = True
if 'product_2' not in st.session_state:
    st.session_state['product_2'] = None

products = list(dfs.keys())
products.sort()


st.text("")
st.text("")

_, l, c, r, _ = st.columns([1, 4, 4, 4, 1])
with l:
    if not st.checkbox(label= 'Select Product', key= 'check_1'):
        # Ensures that If 'check_1' is not selected, then 'check_2' cannot be selected
        st.session_state['check_2'] = False 
    else:
        
        # In case of a date-change, the previous choice is reloaded
        if  st.session_state['product'] != None:
            idx = products.index(st.session_state['product'])
        else:
            idx = None

        # accepts an input from the user
        product = st.selectbox(label= "Choose a product ETF", options= products, index= idx, label_visibility= 'collapsed')

        # The previous choice is stored (to be able to reload)
        if product != None:
            st.session_state['product'] = product
            df = dfs[product]
            st.session_state['check_2'] = True     

with c:
    if st.checkbox(label= 'Select ETFs', key= 'check_2'):
        if st.session_state['product'] != None:

            etfs = st.multiselect(label= 'Select ETFs', options= sorted(list(df['tic'].unique())), default= [], max_selections= 4, label_visibility= 'collapsed')
            
            df = df[df['tic'].isin(etfs)]
            
if df.empty or len(etfs) == 0: #or len(st.session_state['etfs']) == 0:
    st.stop()

n = len(etfs)
isins = {}

st.markdown("<h5 style='text-align: center;'>Directionality count</h5>", unsafe_allow_html=True)
l1, r1 = st.columns(2)
l2, r2 = st.columns(2)

with l1:    
    if n > 0:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[0]}</h6>", unsafe_allow_html=True)
        isins['one'] = df[df['tic'].isin([etfs[0]])]
        fig = plot_directionality(isins['one'])
        st.plotly_chart(fig)
with r1:    
    if n > 1:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[1]}</h6>", unsafe_allow_html=True)
        isins['two'] = df[df['tic'].isin([etfs[1]])]
        fig = plot_directionality(isins['two'])
        st.plotly_chart(fig)

with l2:    
    if n > 2:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[2]}</h6>", unsafe_allow_html=True)
        isins['three'] = df[df['tic'].isin([etfs[2]])]
        fig = plot_directionality(isins['three'])
        st.plotly_chart(fig)

with r2:    
    if n > 3:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[3]}</h6>", unsafe_allow_html=True)
        isins['four'] = df[df['tic'].isin([etfs[3]])]
        fig = plot_directionality(isins['four'])
        st.plotly_chart(fig)
        
st.text("")
st.text("")

st.markdown("<h5 style='text-align: center;'>RMSE distribution</h5>", unsafe_allow_html=True)
l1, r1 = st.columns(2)
l2, r2 = st.columns(2)

with l1:
    st.markdown(f"<h6 style='text-align: center;'>{etfs[0]}</h6>", unsafe_allow_html=True)
    fig = px.histogram(isins['one'], x= "rmse", marginal= 'violin', opacity= 0.5, range_x= [0, round(df.loc[:, ['rmse']].max()).iloc[0]+1], nbins= 10)
    #fig.update_layout(bargap=0)  # Set to 0 to remove gaps between bars
    st.plotly_chart(fig)

with r1:
    if 'two' in isins:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[1]}</h6>", unsafe_allow_html=True)
        fig = px.histogram(isins['two'], x= "rmse", marginal= 'violin', opacity= 0.5, range_x= [0, round(df.loc[:, ['rmse']].max()).iloc[0]+1], nbins= 10)

        st.plotly_chart(fig)

with l2:
    if 'three' in isins:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[2]}</h6>", unsafe_allow_html=True)
        fig = px.histogram(isins['three'], x= "rmse", marginal= 'violin', opacity= 0.5, range_x= [0, round(df.loc[:, ['rmse']].max()).iloc[0]+1], nbins= 10)

        st.plotly_chart(fig)

with r2:
    if 'four' in isins:
        st.markdown(f"<h6 style='text-align: center;'>{etfs[3]}</h6>", unsafe_allow_html=True)
        fig = px.histogram(isins['four'], x= "rmse", marginal= 'violin', opacity= 0.5, range_x= [0, round(df.loc[:, ['rmse']].max()).iloc[0]+1], nbins= 10)

        st.plotly_chart(fig)

st.text("")
st.text("")

st.markdown("<h5 style='text-align: center;'>Worst RMSE</h5>", unsafe_allow_html=True)
bar, _, _ = st.columns([1, 2, 2])
with bar:
    n = bar.number_input('Select the no. of ISINs to display', min_value= 0, max_value= len(df), value= 4, key='bar1')
_, c, _ = st.columns([1, 1.5 ,1])
with c:
    a, df_grouped = isins_with_the_worst_rmse(df, n)
    st.table(a)
    st.download_button(label="Download as CSV", data= a.to_csv(index=True), file_name='ETF_worst_rmse.csv', mime='text/csv')

st.text("")
st.text("")

st.markdown("<h5 style='text-align: center;'>Worst Directionality</h5>", unsafe_allow_html=True)
bar2, _, _ = st.columns([1, 2, 2])
with bar2:
    n = bar2.number_input('Select the no. of ISINs to display', min_value= 0, max_value= len(df), value= 4, key= 'bar2')
_, c, _ = st.columns([1, 1.5 ,1])
with c:
    b = isins_with_the_worst_directionality(df_grouped, n)
    st.table(b) #st.table(a.style.map(highlight_common)) #.style.map(highlight_common))
    st.download_button(label="Download as CSV", data= b.to_csv(index=True), file_name='ETF_worst_dir.csv', mime='text/csv')

st.text("")
st.text("")
st.divider()
st.text("")
st.text("")
st.markdown("<h4 style='text-align: center;'>Level 3: ISIN view</h4>", unsafe_allow_html=True)

pe = pd.read_csv('public_etfs.csv')
mapper = {'NAME' : 'company', 'Ticker' : 'ticker'}
pe.rename(mapper, axis= 1, inplace= True)
public_etfs = set(pe['etf'].unique())
etf_3 = [None]

index      = 'eq_best_model_metrics_'
columns    = ['date','isin', 'schedular', 'rmse', 'mean_dir']
best_metrics = get_data(index, columns, start_date, end_date, schedular= 'Monthly')

l, c, r = st.columns(3)
df = pd.DataFrame()
with l:
    info = 'This feature is available only for public ETFs'
    if not st.checkbox(label= 'Select Product', help= info, key= 'check_3'):
        st.session_state['check_4'] = False
        st.session_state['check_5'] = False
    else:
        # In case of a date-change, the previous choice is reloaded
        if  st.session_state['product_2'] != None:
            idx = products.index(st.session_state['product_2'])
        else:
            idx = None
        
        products = ['AIGO', 'AISRT']
        # accepts an input from the user
        prod = st.selectbox(label= "Choose a product", options= products, index= idx, label_visibility= 'collapsed')

        # The previous choice is stored (to be able to reload)
        if prod != None:
            st.session_state['product_2'] = prod
            #df = dfs[prod]
            st.session_state['check_4'] = True     

with c:
    #info = 'Information for some ETFs is unavailable' # i.e., The xlsx file does not exist on S3
    if not st.checkbox(label= 'Select ETFs', key= 'check_4'): #, help = info
        st.session_state['check_5'] = False
    else:
        if st.session_state['check_3']:
            if st.session_state['product_2'] != None:
                df_tmp = dfs[st.session_state['product_2']]
                lst = sorted(list(set(df_tmp['tic'].unique()) & public_etfs))
                etf_3 = [st.selectbox(label= 'Select ETFs', options= lst, index= None, label_visibility= 'collapsed')]
                st.session_state['check_5'] = True 

with r:
    info = 'By default, all ISINs are selected'
    if st.checkbox(label= 'Select ISINs', key= 'check_5', help= info):
            #if st.session_state['check_3'] and st.session_state['check_4']:

        if etf_3[0] != None:
            # Find the ISINs which belong to this ETF
            isins_2 = pe[pe['etf'].isin(etf_3)]['ISIN']
            # Selct from the best-model's metrics
            cond = best_metrics['isin'].isin(isins_2)
            df = best_metrics[cond]
            df = pd.merge(left= df, right= pe, how='inner', left_on= 'isin', right_on= 'ISIN')
            df.drop(['ISIN'], axis= 1, inplace= True)
            df['t-n'] = df['ticker'] + '-' + df['company']
            col = 't-n'
            options = sorted(list(df[col].unique()))
            isins = st.multiselect(label= 'Select ISINs', options= options, default= [], label_visibility= 'collapsed')
            if len(isins) > 0:
                df = df[df[col].isin(isins)]
                
if not st.session_state['check_3'] or df.empty:
    st.stop()

st.markdown(f"<h6 style='text-align: center;'>Directionality count</h6>", unsafe_allow_html=True)
st.plotly_chart(plot_directionality(df))
st.markdown(f"<h6 style='text-align: center;'>rmse</h6>", unsafe_allow_html=True)
fig = px.histogram(dfs[k], x= "rmse", marginal= 'violin', opacity= 0.5, range_x= [0, y_max], nbins= 10)
st.plotly_chart(fig, key= 'qwerty')

st.text("")
st.text("")

st.markdown("<h5 style='text-align: center;'>Worst RMSE<get_i/h5>", unsafe_allow_html=True)
bar3, _, _ = st.columns([1, 2, 2])
with bar3:
    n = bar3.number_input('Select the no. of ISINs to display', min_value= 0, max_value= len(df), value= min(5, len(df)), key='bar3')
#_, c, _ = st.columns([1, 1.5 ,1])
#with c:
a, df_grouped = isins_with_the_worst_rmse(df, n, flag= True)
st.table(a)
st.download_button(label="Download as CSV", data= a.to_csv(index=True), file_name= st.session_state['product_2'] + '_ISINs_worst_rmse.csv', mime='text/csv')

st.text("")
st.text("")

st.markdown("<h5 style='text-align: center;'>Worst Directionality</h5>", unsafe_allow_html=True)
bar4, _, _ = st.columns([1, 2, 2])
with bar4:
    n = bar4.number_input('Select the no. of ISINs to display', min_value= 0, max_value= len(df), value= min(5, len(df)), key= 'bar4')
#_, c, _ = st.columns([1, 1.5 ,1])
#with c:
b = isins_with_the_worst_directionality(df_grouped, n, flag= True)
st.table(b) #st.table(a.style.map(highlight_common)) #.style.map(highlight_common))
st.download_button(label="Download as CSV", data= b.to_csv(index=True), file_name=st.session_state['product_2'] + '_ISINs_worst_dir.csv', mime='text/csv')
