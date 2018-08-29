#!/usr/bin/python

from utils_libs import *

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# --- parameters ---

def para_parser(para_path):
    
    with open(para_path, "r") as ins:
        array = []
        para_dict = {}
    
        for line in ins:
            newline = line.strip('\n')
            tmpline = newline.split(',')
            
            if tmpline[1] =='int':
                para_dict.update( {tmpline[0]:int(tmpline[2])} )
            elif tmpline[1] =='bool':
                para_dict.update( {tmpline[0]: False if tmpline[2] == 'False' else True} )
        
        return para_dict


# --- utilities ---
def parse_date_time_minute(x):
    tmp = datetime.datetime.fromtimestamp(x/1000.0)
    return str(tmp.year) +'-'+ str(tmp.month)+'-'+str(tmp.day)+' '+ str(tmp.hour) + '-' + str(tmp.minute)

def parse_date_time_hour(x):
    tmp = datetime.datetime.fromtimestamp(x/1000.0)
    return str(tmp.hour)

def parse_date_time_month(x):
    tmp = datetime.datetime.fromtimestamp(x/1000.0)
    return str(tmp.month)


def multivariate_ts_plot( dta_df, title_str ):
    
    matplotlib.rcParams.update({'font.size': 15})
    figure_size = (15.4,7)
    legend_font = 8.5
    fig = plt.figure()
    fig.set_size_inches( figure_size )
    
    tmpt = range(dta_df.shape[0])
    for i in dta_df.columns:
        
        tmpx = list(dta_df[i])    
        plt.plot( tmpt, tmpx, label= i )

    plt.title( title_str )
    plt.ylabel('Value')
    plt.xlabel('Time')
    # plt.legend( loc='upper left',fontsize=12 )
    plt.legend(loc='upper left')
    #     bbox_to_anchor=(0., 1.0, 1., .10),
    #            loc=0,
    #            ncol=5, mode="expand", borderaxespad=0., fontsize= legend_font , numpoints=1 )
    
def plot_features( test_ask, test_bid ):
    # price
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches((15,7))
    ax1.plot( [i[0][0] for i in test_ask], label = 'ask' )
    ax1.plot( [i[0][0] for i in test_bid], label = 'bid' )
    plt.legend()
    plt.title( "Price mean" )
    
    # amount
    fig2, ax2 = plt.subplots() 
    fig2.set_size_inches((15,7))
    ax2.plot( [i[0][1] for i in test_ask] , label = 'ask' )
    ax2.plot( [i[0][1] for i in test_bid] , label = 'bid' )
    plt.legend()
    plt.title( "Amount mean" )
    
    # price variance
    fig3, ax3 = plt.subplots() 
    fig3.set_size_inches((15,7))
    ax3.plot( [i[1][0][0] for i in test_ask] , label = 'ask' )
    ax3.plot( [i[1][0][0] for i in test_bid] , label = 'bid' )
    plt.legend()
    plt.title( "Price variance" )
    
    # amount variance 
    fig4, ax4 = plt.subplots() 
    fig4.set_size_inches((15,7))
    ax4.plot( [i[1][1][1] for i in test_ask] , label = 'ask' )
    ax4.plot( [i[1][1][1] for i in test_bid] , label = 'bid' )
    plt.legend()
    plt.title( "Amount variance" )
    
    # price and amount covariance
    fig5, ax5 = plt.subplots() 
    fig5.set_size_inches((15,7))
    ax5.plot( [i[1][0][1] for i in test_ask] , label = 'ask' )
    ax5.plot( [i[1][0][1] for i in test_bid] , label = 'bid' )
    plt.legend()
    plt.title( "Price and Amount covariance" )


# --- prepare training and testing data ---

# feature gropus: auto-regressive volatility,  order book 

def selection_on_minute_features(x):
    
    ipca = PCA(n_components=4)
    ipca.fit(x)
    return ipca.transform(x), sum(ipca.explained_variance_ratio_)

def prepare_feature_target(features_minu, vol_hour, all_loc_hour, \
                           order_minu, order_hour, bool_feature_selection, step_gap, point_wise):
    
    tmpcnt = len(vol_hour)
    y = []
    x = []
    
    var_explained = []
    
    for i in range( order_hour + step_gap, tmpcnt ):
        
        if all_loc_hour[i - order_hour - step_gap] - order_minu < 0:
            continue
        
        y.append( vol_hour[i] )
        x.append( [ vol_hour[i - order_hour - step_gap : i - step_gap] ] )
        
        if len(features_minu)!=0:
            
            tmp_minu_idx = all_loc_hour[i - step_gap] 
            
            if tmp_minu_idx - order_minu < 0:
                print(" ----- Order_minute ?")
                
            if bool_feature_selection == True:
                
                tmpfeatures = features_minu[tmp_minu_idx-order_minu : tmp_minu_idx]
                tmpft, tmpvar = selection_on_minute_features(tmpfeatures)
                
                var_explained.append( tmpvar )
                x[-1].append( tmpft )
                
            else:
                
                if point_wise == False:
                    
                    x[-1].append(features_minu[tmp_minu_idx-order_minu : tmp_minu_idx])
                
                else:
                    
                    tmpx = []
                    for k in range(i - order_hour - step_gap, i - step_gap):
                        
                        minu_idx = all_loc_hour[k] 
                        tmpx.append(features_minu[minu_idx-order_minu : minu_idx])
                
                    x[-1].append( tmpx )
                    
    return x,y, var_explained


def conti_normalization_train_dta(dta):
    
    original_shape = np.shape(dta)
    
    if len(original_shape)>=3:
        tmp_dta = np.reshape(dta, [original_shape[0], -1] )
        normed_dta = preprocessing.scale(tmp_dta) 
        
        return np.reshape(normed_dta, original_shape)
    else:
        return preprocessing.scale(dta)

def conti_normalization_test_dta(dta, ref_data):
    
    shape_ref_data = np.shape(ref_data)
    shape_dta   = np.shape(dta)
    
    if len(shape_ref_data)>=3:
        tmp_ref_data = np.reshape(ref_data, [shape_ref_data[0], -1])
        tmp_dta = np.reshape(dta,   [shape_dta[0], -1])
    else:
        tmp_ref_data = ref_data
        tmp_dta = dta
    
    mean_dim = np.mean(tmp_ref_data, axis=0)
    std_dim = np.std(tmp_ref_data, axis=0)
    
#    print '--test--', mean_dim, std_dim
    
    df = pd.DataFrame()
    dta_df = pd.DataFrame(tmp_dta)   
    cols = range(np.shape(tmp_dta)[1])
    
#    print '--test--', cols
    
    for i in cols:
        df[i] = (dta_df[i]- mean_dim[i])*1.0/std_dim[i]
    
    if len(shape_ref_data)>=3:
        return np.reshape(df.as_matrix(), shape_dta)
    else:
        return df.as_matrix()
    
# order_hour: remove the front order_hour data to align with regression approaches
def training_testing_statistic(features_minu, vol_hour, all_loc_hour, order_minu, order_hour, \
                               train_split_ratio, bool_feature_selection):
    tmpcnt = len(vol_hour)
    ex = []
    var_explained = []
    
    for i in range(1, tmpcnt):
        
        if len(features_minu)!=0:
            
            tmp_minu_idx = all_loc_hour[i]
            
            if tmp_minu_idx - order_minu < 0:
                print("Order_minute ?")
            
            if bool_feature_selection == True:
                
                tmpfeatures = features_minu[tmp_minu_idx - order_minu : tmp_minu_idx]
                tmpft, tmpvar = selection_on_minute_features( tmpfeatures )
                var_explained.append( tmpvar )
                
                ex.append( tmpft.flatten() )
            
            else:
                ex.append( np.asarray(features_minu[tmp_minu_idx-order_minu : tmp_minu_idx]).flatten() )
        
    tmp_split = int(train_split_ratio*(tmpcnt-order_hour-1)) + order_hour
            
    # xtrain, extrain, xtest, extest
    return vol_hour[1:tmp_split+1], ex[:tmp_split], vol_hour[tmp_split+1:], ex[tmp_split:]

# prepare minute level return series 
def training_testing_garch(vol_hour, all_loc_hour, order_hour, train_split_ratio, price_minu):
    
    tmpcnt = len(vol_hour)
    tmp_split = int(train_split_ratio*(tmpcnt - order_hour - 1)) + order_hour
            
    return_hour = []
    for i in range(1, len(all_loc_hour)):
        tmp = price_minu[ all_loc_hour[i-1]:all_loc_hour[i] ]
        tmp_return =[]
        
        if len(tmp)==1:
            return_hour.append(0.0)
        else:
            for j in range(1, len(tmp)):
                tmp_return.append((tmp[j]-tmp[j-1])/(tmp[j-1]+1e-5)*100)
        
            return_hour.append( mean(tmp_return) )
        
        # check for single return 
        #if np.isnan(return_hour[-1]):
        #    print all_loc_hour[i-1], all_loc_hour[i]
            
        
    tmp = price_minu[ all_loc_hour[i]: ]
    tmp_return =[]
    for j in range(1, len(tmp)):
        tmp_return.append( (tmp[j]-tmp[j-1])/(tmp[j-1]+1e-5)*100 )
    
    return_hour.append( mean(tmp_return) )
    
    # vol train, return train, vol test, return test
    return vol_hour[1:tmp_split+1], return_hour[1:tmp_split+1], vol_hour[tmp_split+1:], return_hour[tmp_split+1:]


def training_testing_mixture_rnn(x, y, train_split_ratio):

    tmp_split = int(train_split_ratio*len(y))
    
    return x[:tmp_split], y[:tmp_split], x[tmp_split:], y[tmp_split:]


def training_testing_mixture_mlp(x, y, train_split_ratio):
    
    for i in range(len(x)):
        ins = x[i]
        
        tmp = []
        for j in ins:
            tmp.append( list(np.asarray(j).flatten()) )
        
        x[i] = tmp

    tmp_split = int(train_split_ratio*len(y))
    
    return x[:tmp_split], y[:tmp_split], x[tmp_split:], y[tmp_split:]


def training_testing_plain_regression(x, y, train_split_ratio): 
    
    for i in range(len(x)):
        ins = x[i]
        tmp = []
        for j in ins:
            tmp += list(np.asarray(j).flatten())        
        x[i] = tmp
    
    tmp_split = int(train_split_ratio*len(y))
    
    xtrain = x[:tmp_split]
    xtest  = x[tmp_split:]
    
    # feature normalization 
    xtest = conti_normalization_test_dta( xtest, xtrain )
    xtrain= conti_normalization_train_dta( xtrain )
    
    return xtrain, y[:tmp_split], xtest, y[tmp_split:]


# --- calculate metrics in order book data ---

# price, volumn w.r.t. minute

def cal_price_req_minu(data_minu):
    
    price_minu =[]
    req_minu   =[]
    
    for i in range(len(data_minu)):
        
        if len(data_minu[i][0])==0:
            #print "\n at minute ", i, " ask "
            price_minu.append( max([j[0] for j in data_minu[i][1]]) )
    
        elif len(data_minu[i][1])==0:
            #print "\n at minute ", i, " bid "
            price_minu.append( min([j[0] for j in data_minu[i][0]]) )
    
        else:
            tmpmin = min([j[0] for j in data_minu[i][0]])
            tmpmax = max([j[0] for j in data_minu[i][1]])
        
            price_minu.append( (tmpmin + tmpmax)/2.0 )
    
        req_minu.append( [len(data_minu[i][0]), len(data_minu[i][1])] )
        
        
    return price_minu, req_minu
    
# price volatility w.r.t. hour
def cal_price_volatility_hour( loc_hour, price_minu ):
    pvol_hour = []
    
    for i in range(1, len(loc_hour)):
        pvol_hour.append( sqrt(var(price_minu[ loc_hour[i-1]:loc_hour[i] ])) )
        
    pvol_hour.append( sqrt(var(price_minu[ loc_hour[i]: ])) )
    
    return pvol_hour


# return volatility w.r.t. hour
def cal_return_volatility_hour( loc_hour, price_minu, return_type ):
    rvol_hour = []
    return_minu = []
    
    #print 'Begin'
    
    for i in range(1, len(loc_hour)):
        tmp = price_minu[ loc_hour[i-1]:loc_hour[i] ]
        
        if len(tmp)<=1:
            rvol_hour.append( 0.0 )
            continue
        
        tmp_return =[]
        for j in range(1, len(tmp)):
            
            if return_type == 'per':
                # percent change return
                tmp_return.append( (tmp[j]-tmp[j-1])/(tmp[j-1]+1e-5)*100 )
            elif return_type == 'log':
                # log return
                tmp_return.append(log(tmp[j]*1.0/(tmp[j-1]+1e-5)+1e-5))
    
        return_minu += tmp_return    
        rvol_hour.append( np.std(tmp_return) )
        
    tmp = price_minu[ loc_hour[i]: ]
    tmp_return =[]
    for j in range(1, len(tmp)):
        
        if return_type == 'per':
            tmp_return.append( (tmp[j]-tmp[j-1])/(tmp[j-1]+1e-5)*100 )
        elif return_type == 'log':
            tmp_return.append(log(tmp[j]*1.0/(tmp[j-1]+1e-5)+1e-5))
    
    rvol_hour.append( np.std(tmp_return) )
    return_minu += tmp_return
    
    print('Done')
    
    return return_minu, rvol_hour

# --- Load order book data files ---
# organize data into minute-wise format 
def load_raw_order_book_files(file_addr, bool_dump):
    
    files = sorted(glob.glob(file_addr))

    all_dta_minu = []
    all_loc_hour = []
    
    all_loc_month = []
    pre_month = 0

    for i in range( len(files) ):
        dta_df = pd.read_csv( files[i] ,sep=',')
        print("Current : " + files[i], dta_df.shape)
    
        all_df = dta_df
    
        all_df['date_time'] = all_df['date'].map( parse_date_time_minute )
        all_df['hour']      = all_df['date'].map( parse_date_time_hour )
        cur_month = parse_date_time_month( all_df['date'].iloc[0] )

        
        if cur_month != pre_month:
            all_loc_month.append( len(all_dta_minu) )
            pre_month = cur_month
        
        
        
        minute_tick = list(all_df['date_time'].unique())
        print("   ", len(minute_tick), minute_tick[-1])
    
        dta_minu = [] 
        tmp_hour = []
    
        for i in range(len(minute_tick)):
            tmp_df = all_df[ all_df['date_time']==minute_tick[i] ]
    
            tmp_df_a = np.asarray( tmp_df[tmp_df['type']=='a'][['price','amount']] )
            tmp_df_b = np.asarray( tmp_df[tmp_df['type']=='b'][['price','amount']] )
    
            dta_minu.append( [tmp_df_a, tmp_df_b] )    
            tmp_hour.append( tmp_df['hour'].iloc[0] )
    
        pre_hour = 0 
        loc_hour = []
    
        offset = len(all_dta_minu)
        loc_hour.append( offset )
    
        for i in range(len(minute_tick)):
            if i==0:
                pre_hour = tmp_hour[i]
            else:
                if tmp_hour[i] != pre_hour:
                    loc_hour.append(i+offset)
                    pre_hour = tmp_hour[i]
        
        
        all_dta_minu += dta_minu
        all_loc_hour += loc_hour
    
        print("   ", len(all_dta_minu), len(all_loc_hour))
    
    if bool_dump == True:
        np.asarray(all_dta_minu).dump("../dataset/bitcoin/dta_minu.dat")
        np.asarray(all_loc_hour).dump("../dataset/bitcoin/loc_hour.dat")
        np.asarray(all_loc_month).dump("../dataset/bitcoin/loc_month.dat") 
    
    return all_dta_minu, all_loc_hour, all_loc_month
    

# --- extract features from asking and biding sides in order book ---
# TO DO: quantile features

# distributional features

# analytical posterior: sampling by enumerating and calculating density
# approximate posterior: sampling via MCMC
    
# pymc?
def poterior_sample_norm_2d(x, n_samples):
    return 1
    
def poterior_sample_log_norm_2d(x, n_samples):
    return 1
    
def map_log_norm_2d(x):
    
    if len(x) == 0:
        return [0.0, 0.0], [0.0, 0.0, 0.0]
    
    elif len(x) == 1:
        return [ x[0][0], x[0][1] ], [0.0, 0.0, 0.0]
    
    else:
        
        tmpx = [ [i[0]+1e-5, i[1]+1e-5] for i in x]
        logx = log(tmpx)
        
        post_log_mu, post_log_cov = map_norm_2d(logx)
        
        post_mu = [ exp(post_log_mu[i] + post_log_cov[i]/2.0) for i in range(2) ]
        
#         post_cov = [ [0.0, 0.0] for i in range(2) ]
        
#         for i in range(2):
#             for j in range(2):
#                 tmp = post_log_cov[2] if i!=j else post_log_cov[i]  
#                 post_cov[i][j] = exp( post_log_mu[i]+post_log_mu[j]+0.5*(post_log_cov[i]+post_log_cov[j]) )*\
#                 ( exp(tmp)-1.0 )
                
        var0 = exp( post_log_mu[0]+post_log_mu[0]+0.5*(post_log_cov[0]+post_log_cov[0]) )*\
        ( exp(post_log_cov[0])-1.0 )
        var1 = exp( post_log_mu[1]+post_log_mu[1]+0.5*(post_log_cov[1]+post_log_cov[1]) )*\
        ( exp(post_log_cov[1])-1.0 )
        cov = exp( post_log_mu[0]+post_log_mu[1]+0.5*(post_log_cov[0]+post_log_cov[1]) )*\
        ( exp(post_log_cov[2])-1.0 )
        
        return list(post_mu), [var0, var1, cov], list(post_mu), \
               [post_log_cov[0][0], post_log_cov[1][1], post_log_cov[0][1]] 

def map_norm_2d( x ):
    
    if len(x) == 0:
        return [0.0, 0.0], [0.0, 0.0, 0.0]
    
    elif len(x) == 1:
        return [ x[0][0], x[0][1] ], [0.0, 0.0, 0.0]
    
    else:
        mle_mu  = np.mean(x, axis=0) 
        mle_cov = np.cov(x, rowvar=0)
    
        m_0 = mle_mu
        k0  = 0.01
        v0 = 2.0 + 2.0
        S_0 = np.diag(np.diag(mle_cov))*1.0/len(x)
    
        x_ba = mle_mu
    
        #S = np.zeros((2, 2))
        #for i in x:
        #    S = np.add(S, np.outer(i, i))
            
        S = np.matmul( np.asmatrix(x).transpose() , np.asmatrix(x) )
        
        N = len(x)
        m_N = k0*1.0/(k0+N)*m_0 + N*1.0/(k0+N)*x_ba 
    
        vN = v0 + N    
        kN = k0 + N
        
        S_N = S_0 + S + k0*np.outer(m_0, m_0) - kN*np.outer(m_N, m_N)
        
        cov_mode = S_N*1.0/(vN+2.0+2.0)
        
        return list(m_N), [ cov_mode.item((0, 0)), cov_mode.item((1, 1)), cov_mode.item((1, 0)) ] 
    
def mle_norm_2d( x ):
    
    if len(x) == 0:
        return [0.0, 0.0], [0.0, 0.0]
    elif len(x) == 1:
        return [x[0][0], x[0][1]], [0.0, 0.0]
    else:
        tmp = np.cov(x, rowvar=0)
        return list(np.mean(x, axis=0)), [tmp[0][0], tmp[1][1]] 
    
def skewness(x):
    
    if len(x) == 0:
        return [0.0, 0.0]

    elif len(x) == 1:
        return [0.0, 0.0]
    
    else:
        return list(sp.stats.skew(x,0))
    
def loglk_norm( x, mu, cov ):
    
    var = multivariate_normal(mean=mu, cov=[[cov[0], cov[2]], [cov[2],cov[1]]])
    return sum( var.logpdf(x) )

def likelihood_ratio_test(llmin, llmax, df):
    return sp.stats.chisqprob(-2.0*(llmin-llmax), df)


# ---- financial features added by Nino ----

def bid_ask_spread(all_dta_minu, tmp_idx):
    x_a = all_dta_minu[tmp_idx][0]
    x_b = all_dta_minu[tmp_idx][1]
    
    #if ask side is empty --- find last ask side and use it
    if (market_depth_a_volume(x_a)==0):
        return abs(find_last_ask_price(all_dta_minu, tmp_idx)-x_b[0][0])
    
    #if bid side is empty -- find last bid side and use it
    if (market_depth_b_volume(x_b)==0):
        return abs(x_a[0][0] - find_last_bid_price(all_dta_minu, tmp_idx))
    
    #calucate difference -- spread
    return abs(x_a[0][0]-x_b[0][0])

def bid_ask_spread_weighted(all_dta_minu, tmp_idx):
    x_a = all_dta_minu[tmp_idx][0]
    x_b = all_dta_minu[tmp_idx][1]
    
    #either bid or ask side is empty -- call just bid_ask_spread function
    if ((market_depth_b_volume(x_b)==0)|(market_depth_a_volume(x_a)==0)):
        return bid_ask_spread(all_dta_minu, tmp_idx)
    
    
    # calculate avg bid on first 10 % of orders
    idx = np.shape(x_b)[0]/10
    
    if (idx==0): #smaller than 10
        idx = np.shape(x_b)[0]
    
    #cumulative price of 10% of bid volume
    cum_bid = 0.0
    for i in range ( idx ):
        cum_bid+=x_b[i][0]
    cum_bid = cum_bid/idx
    
    # calculate avg ask on first 10 % of orders
    idx = np.shape(x_a)[0]/10
    
    if (idx==0): #smaller than 10
        idx = np.shape(x_a)[0]
    
    #cumulative price of 10% of bid volume
    cum_ask = 0.0
    for i in range ( idx ):
        cum_ask+=x_a[i][0]
    cum_ask = cum_ask/idx
    
    return abs(cum_ask-cum_bid)

def market_depth_a_volume(x_a):
    #number of orders
    return (np.shape(x_a)[0])

def market_depth_b_volume(x_b):
    #number of orders
    return (np.shape(x_b)[0])

def market_depth_a_btc(x_a):
    #sum of btc in ask side
    btc_sum = 0.0
    for i in range ( np.shape(x_a)[0] ):
        btc_sum+=x_a[i][1]
    return btc_sum

def market_depth_b_btc(x_b):
    #sum of btc in bid side
    btc_sum = 0.0
    for i in range ( np.shape(x_b)[0] ):
        btc_sum+=x_b[i][1]
    return btc_sum

def find_last_bid_price(all_dta_minu, idx):
    #search for last available bid
    tmp_b = all_dta_minu[idx][1]
    while(market_depth_b_volume(tmp_b)==0):
        idx=idx-1;
        tmp_b = all_dta_minu[idx][1]
        
    return tmp_b[0][0]

def find_last_ask_price(all_dta_minu, idx):
    #search for last available ask
    tmp_a = all_dta_minu[idx][0]
    while(market_depth_a_volume(tmp_a)==0):
        idx=idx-1;
        tmp_a = all_dta_minu[idx][0]
        
    return tmp_a[0][0]

def bid_ask_slope(all_dta_minu, tmp_idx):
    #calculated the volume in the tail that belongs closest to the current price
    #essentially sum until some delta price -- is estimated from data -> from first 10 % of orders
    x_a = all_dta_minu[tmp_idx][0]
    x_b = all_dta_minu[tmp_idx][1]
    
    if (market_depth_b_volume(x_b)==0): #bid is empty
        cum_bid = 0.0
        idx = np.shape(x_a)[0]/10 #use 10% on ask sid
        delta = abs(x_a[idx][0] - x_a[0][0]) #critical value for summation
    else:
        #find delta valule for price for the first 10% of orders on bid side and use it also for ask side
        idx = np.shape(x_b)[0]/10
        delta = abs(x_b[idx][0] - x_b[0][0]) #critical value for summation
    
        #cumulative volume of orders on bid side until the delta price
        cum_bid = 0.0
        for i in range ( idx ):
            cum_bid+=x_b[i][1]
        
    if (market_depth_a_volume(x_a)==0): #ask is empty
        cum_ask = 0
    else:    
        #cumulative volume of orders on ask side until the delta price
        cum_ask = 0.0
        for i in range ( np.shape(x_a)[0] ):
            if ( x_a[i][0]  <= (delta+x_a[0][0]) ):
                cum_ask+=x_a[i][1]
    
    
    return cum_bid, cum_ask
    

def orderbook_stat_features(all_dta_minu, tmp_idx):
    tmp_a = all_dta_minu[tmp_idx][0]
    tmp_b = all_dta_minu[tmp_idx][1]
 
    f = []
    f.append(bid_ask_spread(all_dta_minu, tmp_idx))
    f.append(bid_ask_spread_weighted(all_dta_minu, tmp_idx))
    
    f.append(market_depth_a_volume(tmp_a))
    f.append(market_depth_b_volume(tmp_b))
    # absolute value ?
    f.append(market_depth_a_volume(tmp_a)-market_depth_b_volume(tmp_b))
    
    f.append(market_depth_a_btc(tmp_a))
    f.append(market_depth_b_btc(tmp_b))
    # absolute value ?
    f.append(market_depth_a_btc(tmp_a)-market_depth_b_btc(tmp_b))
    
    cum_bid, cum_ask = bid_ask_slope(all_dta_minu, tmp_idx)
    f.append(cum_bid)
    f.append(cum_ask)
    return f



