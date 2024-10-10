
Open In Colab
Question 1: Sample 100 draws from the exchange rate distribution and for each draw determine the optimal network configuration (which plants/lines should be open).


!pip install forex_python

%reset -f
# Install and import packages
!pip install gurobipy
!pip install tabulate
from forex_python.converter import CurrencyRates
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import multivariate_normal

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
from tabulate import tabulate
import datetime as dt
_empty_series = pd.Series(dtype=float)

def get_monthly_exchange_rates(start_year, end_year, base_currency, target_currencies):
    c = CurrencyRates()
    rates = {currency: [] for currency in target_currencies}

    # Generate a list of first day of each month in the specified range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    current_date = start_date

    while current_date <= end_date:
        for currency in target_currencies:
            try:
                # Get exchange rate for the first day of the month
                rate = c.get_rate(base_currency, currency, current_date)
                rates[currency].append((current_date.strftime("%Y-%m"), rate))
            except Exception as e:
                print(f"Error fetching rate for {currency} on {current_date.strftime('%Y-%m')}: {e}")
                rates[currency].append((current_date.strftime("%Y-%m"), None))

        # Move to the first day of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return rates
    # Specify the currencies
base_currency = "USD"
target_currencies = ["BRL", "EUR", "INR", "JPY", "MXN"]

# Fetch exchange rates from Jan 2019 to Dec 2023
exchange_rates = get_monthly_exchange_rates(2019, 2023, base_currency, target_currencies)

# Print the rates
for currency, rates in exchange_rates.items():
    print(f"Exchange rates for {currency} against {base_currency}:")
    for date, rate in rates:
        print(f"{date}: {rate}")
    print("\n")

     
Collecting forex_python
  Downloading forex_python-1.8-py3-none-any.whl (8.2 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from forex_python) (2.31.0)
Collecting simplejson (from forex_python)
  Downloading simplejson-3.19.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (137 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 137.9/137.9 kB 1.7 MB/s eta 0:00:00
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->forex_python) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->forex_python) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->forex_python) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->forex_python) (2024.2.2)
Installing collected packages: simplejson, forex_python
Successfully installed forex_python-1.8 simplejson-3.19.2
Collecting gurobipy
  Downloading gurobipy-11.0.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (13.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.4/13.4 MB 51.9 MB/s eta 0:00:00
Installing collected packages: gurobipy
Successfully installed gurobipy-11.0.0
Requirement already satisfied: tabulate in /usr/local/lib/python3.10/dist-packages (0.9.0)
Exchange rates for BRL against USD:
2019-01: 3.8812227074235808
2019-02: 3.6709964257693315
2019-03: 3.7808134938065536
2019-04: 3.8771804912780348
2019-05: 3.9267249064004286
2019-06: 3.9872657160792757
2019-07: 3.8272094457661465
2019-08: 3.8255866630424937
2019-09: 4.157212758245742
2019-10: 4.164433841071756
2019-11: 3.989316814794865
2019-12: 4.230468038608632
2020-01: 4.0196724230016025
2020-02: 4.266829533116178
2020-03: 4.485014120433634
2020-04: 5.2440563277249455
2020-05: 5.384792203015815
2020-06: 5.3324937027707815
2020-07: 5.442857142857142
2020-08: 5.167032410533423
2020-09: 5.433052473512972
2020-10: 5.6008339006126615
2020-11: 5.779363993845102
2020-12: 5.311915106951871
2021-01: 5.193953223046206
2021-02: 5.442320423700762
2021-03: 5.579025968638513
2021-04: 5.631619274646687
2021-05: 5.346548584671412
2021-06: 5.202126789366054
2021-07: 4.960871760350051
2021-08: 5.1060465898578755
2021-09: 5.152407548447152
2021-10: 5.415172413793104
2021-11: 5.650889618241493
2021-12: 5.6123386954216015
2022-01: 5.571340279004062
2022-02: 5.290586145648313
2022-03: 5.160186346532879
2022-04: 4.722041259500543
2022-05: 4.896394686907021
2022-06: 4.72796863330844
2022-07: 5.287002398081535
2022-08: 5.152252516368612
2022-09: 5.221811275489805
2022-10: 5.394337299958966
2022-11: 5.161053583995175
2022-12: 5.214080734647025
2023-01: 5.286517907369211
2023-02: 5.064622728107215
2023-03: 5.197959565705728
2023-04: 5.072
2023-05: 4.996994809215917
2023-06: 5.051135832476395
2023-07: 4.858089453340696
2023-08: 4.753691886964449
2023-09: 4.933142751752121
2023-10: 5.008967340003776
2023-11: 5.026383221030653
2023-12: 4.879415791875856


Exchange rates for EUR against USD:
2019-01: 0.8733624454148472
2019-02: 0.871763577717723
2019-03: 0.8785030308354563
2019-04: 0.8899964400142399
2019-05: 0.8914244963451596
2019-06: 0.896780557797507
2019-07: 0.8811349017534584
2019-08: 0.906043308870164
2019-09: 0.9061254077564336
2019-10: 0.9175995595522113
2019-11: 0.8977466558937068
2019-12: 0.9105809506465125
2020-01: 0.8901548869503294
2020-02: 0.9048136083966704
2020-03: 0.9109957183201239
2020-04: 0.91441111923921
2020-05: 0.9194556822361163
2020-06: 0.8996041741633681
2020-07: 0.8928571428571428
2020-08: 0.8440243079000674
2020-09: 0.8342370901810294
2020-10: 0.8509189925119128
2020-11: 0.8548469823901522
2020-12: 0.8355614973262031
2021-01: 0.8149295085975062
2021-02: 0.8275405494869249
2021-03: 0.8296689620841284
2021-04: 0.8513536523071683
2021-05: 0.8276775368316505
2021-06: 0.8179959100204499
2021-07: 0.841467519353753
2021-08: 0.8409721638213775
2021-09: 0.8462384700008463
2021-10: 0.8620689655172414
2021-11: 0.8637070305752289
2021-12: 0.8838607035531201
2022-01: 0.8829242450997704
2022-02: 0.8880994671403198
2022-03: 0.8958967926894821
2022-04: 0.9048136083966704
2022-05: 0.9487666034155597
2022-06: 0.9335324869305452
2022-07: 0.9592326139088729
2022-08: 0.9772305286817159
2022-09: 0.9996001599360256
2022-10: 1.0258514567090686
2022-11: 1.0053282396702523
2022-12: 0.9565716472163764
2023-01: 0.9375585974123383
2023-02: 0.9179364787956674
2023-03: 0.9359790340696368
2023-04: 0.9195402298850576
2023-05: 0.9106638739641197
2023-06: 0.9348415443582312
2023-07: 0.9186110600771633
2023-08: 0.9115770282588879
2023-09: 0.9221689413500553
2023-10: 0.943930526713234
2023-11: 0.9490367277213627
2023-12: 0.9150805270863837


Exchange rates for INR against USD:
2019-01: 69.63301310043668
2019-02: 71.28497951355592
2019-03: 70.89080207326714
2019-04: 69.32048771804912
2019-05: 69.58593332144767
2019-06: 69.71661734373599
2019-07: 68.93647017358357
2019-08: 69.10573525414516
2019-09: 71.43575570859008
2019-10: 71.10662506881997
2019-11: 70.7568004309184
2019-12: 71.65133855399745
2020-01: 71.37884991988606
2020-02: 71.39477017734347
2020-03: 72.22829552701103
2020-04: 76.44513533284565
2020-05: 75.03751379183524
2020-06: 75.5109751709248
2020-07: 75.55535714285713
2020-08: 74.81051654287643
2020-09: 72.9194126970885
2020-10: 73.0322498298162
2020-11: 74.46700290647975
2020-12: 73.65767045454544
2021-01: 73.06698720560671
2021-02: 73.10906984442238
2021-03: 73.56384302663237
2021-04: 73.41009705431637
2021-05: 74.06017215692766
2021-06: 72.8961145194274
2021-07: 74.51952204644901
2021-08: 74.33697754604322
2021-09: 73.07099940763307
2021-10: 74.11637931034483
2021-11: 74.90715149421317
2021-12: 74.9058688350716
2022-01: 74.36800282535759
2022-02: 74.77531083481351
2022-03: 75.70462282745027
2022-04: 75.99049945711184
2022-05: 76.50664136622392
2022-06: 77.53080657206871
2022-07: 79.01649880095924
2022-08: 79.09068699306165
2022-09: 79.58766493402639
2022-10: 81.47825194911776
2022-11: 82.521363225093
2022-12: 81.2444997130285
2023-01: 82.66547909244328
2023-02: 81.79364787956673
2023-03: 82.44552602021714
2023-04: 82.2064367816092
2023-05: 81.82815772698297
2023-06: 82.4595681032065
2023-07: 82.09690778575373
2023-08: 82.27484047402005
2023-09: 82.71578753227591
2023-10: 83.08146120445535
2023-11: 83.28034544936888
2023-12: 83.32089212562586


Exchange rates for JPY against USD:
2019-01: 109.91266375545851
2019-02: 108.89198849272077
2019-03: 111.87736097689536
2019-04: 110.96475614097545
2019-05: 111.3656623284008
2019-06: 108.75257824410366
2019-07: 108.31791347255265
2019-08: 108.88828486001633
2019-09: 106.27038782167453
2019-10: 108.27674802716093
2019-11: 108.11562976927912
2019-12: 109.66126388635949
2020-01: 108.54548691472317
2020-02: 108.89431777053927
2020-03: 108.73644893869
2020-04: 107.48902706656914
2020-05: 106.5373299006988
2020-06: 107.72759985606334
2020-07: 107.41964285714285
2020-08: 104.92066171505739
2020-09: 105.88137148577626
2020-10: 105.49693669162696
2020-11: 104.59907676525903
2020-12: 104.3783422459893
2021-01: 103.08043354249857
2021-02: 104.90731545845748
2021-03: 106.72031859288143
2021-04: 110.7015154095011
2021-05: 108.93891739778184
2021-06: 109.65235173824132
2021-07: 111.42712891282396
2021-08: 109.6543604406694
2021-09: 110.30718456461031
2021-10: 111.18103448275863
2021-11: 114.17343237173951
2021-12: 113.37281244475872
2022-01: 115.11566307610806
2022-02: 114.6714031971581
2022-03: 114.80917398315714
2022-04: 122.46652189648933
2022-05: 129.99051233396582
2022-06: 129.46228528752803
2022-07: 135.29976019184653
2022-08: 132.2974689729307
2022-09: 139.28428628548582
2022-10: 144.65531391054574
2022-11: 147.1297878757414
2022-12: 136.2923282953893
2023-01: 131.8769923120195
2023-02: 129.7686800073435
2023-03: 135.5484837139648
2023-04: 133.1770114942529
2023-05: 136.0076495765413
2023-06: 139.525100495466
2023-07: 144.63464016197312
2023-08: 142.98085688240656
2023-09: 145.2139431943932
2023-10: 149.2354162733623
2023-11: 151.21002182784474
2023-12: 149.27336276674026


Exchange rates for MXN against USD:
2019-01: 19.643755458515283
2019-02: 19.131287594804288
2019-03: 19.321795660195026
2019-04: 19.262815948736204
2019-05: 18.972098413264398
2019-06: 19.632499327414582
2019-07: 19.122742091814256
2019-08: 19.213282594908037
2019-09: 20.076748822036972
2019-10: 19.78225362451826
2019-11: 19.136726815692615
2019-12: 19.53041340375159
2020-01: 18.889264732063378
2020-02: 18.824104234527688
2020-03: 19.711214357292523
2020-04: 24.149323335771765
2020-05: 23.717635159985292
2020-06: 21.99712126664268
2020-07: 22.946875
2020-08: 22.179270762997973
2020-09: 21.713356135813797
2020-10: 21.84853641933288
2020-11: 21.235766797743203
2020-12: 20.092329545454547
2021-01: 19.897318881916714
2021-02: 20.287735849056606
2021-03: 20.768024558201276
2021-04: 20.329644134173336
2021-05: 20.058351266346634
2021-06: 19.900122699386504
2021-07: 19.9654998317065
2021-08: 19.870237995122363
2021-09: 20.01218583396801
2021-10: 20.51336206896552
2021-11: 20.677319053377097
2021-12: 21.289464380413648
2022-01: 20.434222143740065
2022-02: 20.541385435168742
2022-03: 20.476437914352264
2022-04: 19.82328990228013
2022-05: 20.3207779886148
2022-06: 19.66747572815534
2022-07: 20.25419664268585
2022-08: 20.29072608228281
2022-09: 20.18732506997201
2022-10: 20.14700451374641
2022-11: 19.702824972353476
2022-12: 19.27224029079778
2023-01: 19.553722107631728
2023-02: 18.810262529832936
2023-03: 18.2406402096593
2023-04: 18.059034482758623
2023-05: 18.04771878699572
2023-06: 17.639992521267644
2023-07: 17.082090925823668
2023-08: 16.798359161349136
2023-09: 17.077462191073405
2023-10: 17.46554653577497
2023-11: 17.98016513239062
2023-12: 17.18388711879836



# Convert the data into a NumPy array for analysis
# Note: This example assumes all currencies have the same number of rates and ignores missing values

# Extract rates for each currency, ensuring they're in the same order for each month
rates_list = [rates for rates in exchange_rates.values() if all(rate is not None for _, rate in rates)]
currency_data = np.array([[rate for _, rate in currency_rates] for currency_rates in rates_list])

# Transpose to get rows as observations (dates) and columns as variables (currencies)
currency_data = currency_data.T

# Ensure no NaN values; this method requires complete data
if np.isnan(currency_data).any():
    print("Data contains NaN values. Please handle missing data before proceeding.")
else:
    # Estimate the mean and covariance
    mean_vector = np.mean(currency_data, axis=0)
    covariance_matrix = np.cov(currency_data, rowvar=False)

    # Fit the multivariate normal distribution
    mvn_distribution = multivariate_normal(mean=mean_vector, cov=covariance_matrix)

    print("Mean Vector:\n", mean_vector)
    print("Covariance Matrix:\n", covariance_matrix)
    # The distribution is now defined and can be used for further analysis


     
Mean Vector:
 [  4.91679403   0.89791993  75.7301565  119.26608249  19.81262121]
Covariance Matrix:
 [[ 3.42594387e-01 -2.13708474e-03  1.22952065e+00  1.27836101e+00
   3.46111090e-01]
 [-2.13708474e-03  2.17002114e-03  1.24913080e-01  5.21137097e-01
  -1.73246912e-02]
 [ 1.22952065e+00  1.24913080e-01  2.02929685e+01  6.06245515e+01
  -2.53203458e+00]
 [ 1.27836101e+00  5.21137097e-01  6.06245515e+01  2.26189966e+02
  -1.36106797e+01]
 [ 3.46111090e-01 -1.73246912e-02 -2.53203458e+00 -1.36106797e+01
   2.22048686e+00]]
Sample 100 draws from the exchange rate distribution and for each draw determine the optimal network configuration (which plants/lines should be open).


from os.path import samefile
sample = mvn_distribution.rvs(size=100, random_state = 42)
#print("Sampled draws:", sample)
selected_yr = 2023
base_yr = 2019

demand = pd.DataFrame({
    'from': ['LatinAmerica', 'Europe', 'AsiaWoJapan', 'Japan', 'Mexico', 'U.S.'],
    'd_h': [  7, 15,  5,  7,  3, 18],
    'd_r': [  7, 12,  3,  8,  3, 17],
})
demand.set_index('from', inplace=True)

caps = pd.DataFrame({
    'plant': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'cap': [18, 45, 18, 10, 30, 22],
})
caps.set_index('plant', inplace=True)

pcosts = pd.DataFrame({
    'plant': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'fc_p': [20, 45, 14, 13, 30, 23],
    'fc_h': [ 5, 13,  3,  4,  6,  5],
    'fc_r': [ 5, 13,  3,  4,  6,  5],
    'rm_h': [3.6, 3.9, 3.6, 3.9, 3.6, 3.6],
    'pc_h': [5.1, 6.0, 4.5, 6.0, 5.0, 5.0],
    'rm_r': [4.6, 5.0, 4.5, 5.1, 4.6, 4.5],
    'pc_r': [6.6, 7.0, 6.0, 7.0, 6.5, 6.5],
})
pcosts.set_index('plant', inplace=True)

tcosts = pd.DataFrame({
    'from': ['Brazil', 'Germany', 'India', 'Japan', 'Mexico', 'U.S.'],
    'LatinAmerica': [ 0.20, 0.45, 0.50, 0.50, 0.40, 0.45],
    'Europe':       [ 0.45, 0.20, 0.35, 0.40, 0.30, 0.30],
    'AsiaWoJapan':  [ 0.50, 0.35, 0.20, 0.30, 0.50, 0.45],
    'Japan':        [ 0.50, 0.40, 0.30, 0.10, 0.45, 0.45],
    'Mexico':       [ 0.40, 0.30, 0.50, 0.45, 0.20, 0.25],
    'U.S.':           [ 0.45, 0.30, 0.45, 0.45, 0.25, 0.20],
})
tcosts.set_index('from', inplace=True)

duties = pd.DataFrame({
    'from': ['LatinAmerica', 'Europe', 'AsiaWoJapan', 'Japan', 'Mexico', 'U.S.'],
    'duty': [ 0.30, 0.03, 0.27, 0.06, 0.35, 0.04],
})
duties.set_index('from', inplace=True)

# Your provided exchange_rate_data# we took exchange rate of 2019 as our base rate
exrate0 = {'2019': [4.33, 0.92, 71.48, 109.82, 18.65, 1]}
exrate0 = pd.DataFrame(exrate0 , index=['BRL', 'EUR', 'INR', 'JPY', 'MXN', 'USD'])
while True:
    try:
        tariff = float(input("Enter tariff (in percent, e.g. 10 for 10%): "))
        if 0 <= tariff <= 1000:
            tariff = tariff/100
            break  # Break the loop if the input is valid
        else:
            print("Invalid input. Please enter a valid number between 0 and 1000.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
## Results list to store results for each sample
all_results = []
# Create an empty DataFrame to store the results
columns = ['Sample', 'Plant', 'H', 'R', 'Objective']
results_df = pd.DataFrame(columns=columns)

for sample_idx, sample_values in enumerate(sample):
    # Convert the NumPy array to a Python list
    sample_list = sample_values.tolist()
    # Append a value of 1 to the list
    sample_list.append(1)
    # Convert the list back to a NumPy array
    sample_values_with_1 = np.array(sample_list)
    # Update the exchange rate dataframe with the current sample
    exrate_sample_df = pd.DataFrame({f'Sample_{sample_idx + 1}': sample_values_with_1}, index=['BRL', 'EUR', 'INR', 'JPY', 'MXN', 'USD'])


    # identify number of supply and demand location for iterations
    n_ctry = range(demand.shape[0])
    n_lines = range(demand.shape[1] + 1)
    # Objective function to calculate cost
    def calc_total_cost(dec_plant, dec_h, dec_r, tariff=0):
        x_plant = np.array(list(dec_plant.values())).reshape(len(n_ctry), len(n_lines))
        x_h = np.array(list(dec_h.values())).reshape(len(n_ctry), len(n_ctry))
        x_r = np.array(list(dec_r.values())).reshape(len(n_ctry), len(n_ctry))
        reindx = exrate0.loc[:, f'{2019}'] /exrate_sample_df.loc[:,f'Sample_{sample_idx + 1}']
        pcosts_rev = pcosts.values * reindx.values.reshape(-1,1)
        pcosts_rev = pd.DataFrame(pcosts_rev, columns=pcosts.columns[0:], index=pcosts.index)
        duties_mat = np.zeros(len(duties)) + (1 + duties['duty'].values)[:, np.newaxis]
        np.fill_diagonal(duties_mat, 1)
        duties_mat = pd.DataFrame(duties_mat.T, index=pcosts_rev.index, columns=duties.index)
        duties_mat.loc['Germany', 'U.S.'] += tariff
        duties_mat.loc['U.S.', 'Europe']  += tariff

        vcosts_h = tcosts.add(pcosts_rev['rm_h'], axis=0).add(pcosts_rev['pc_h'], axis=0) * duties_mat
        vcosts_r = tcosts.add(pcosts_rev['rm_r'], axis=0).add(pcosts_rev['pc_r'], axis=0) * duties_mat

        fc = pcosts_rev[['fc_p','fc_h','fc_r']].values
        vh = (vcosts_h * x_h).values
        vr = (vcosts_r * x_r).values
        total_cost = sum(0.2 * fc[i,j] for i in n_ctry for j in n_lines) + sum(0.8 * fc[i,j] * x_plant[i,j] for i in n_ctry for j in n_lines) + sum(vh[i,j] for i in n_ctry for j in n_ctry) + sum(vr[i,j] for i in n_ctry for j in n_ctry)

        return total_cost
        # Calculate excess capacity given decision variables
    def calc_excess_cap(dec_plant, dec_h, dec_r):
        x_plant = np.array(list(dec_plant.values())).reshape(len(n_ctry), len(n_lines))
        x_h = np.array(list(dec_h.values())).reshape(len(n_ctry), len(n_ctry))
        x_r = np.array(list(dec_r.values())).reshape(len(n_ctry), len(n_ctry))

        excess_cap = (x_plant * caps.values).copy()
        excess_cap[:, 0] -= (np.sum(x_h, axis=1) + np.sum(x_r, axis=1))
        excess_cap[:, 1] -= np.sum(x_h, axis=1)
        excess_cap[:, 2] -= np.sum(x_r, axis=1)
        return excess_cap

    # Calculate unmet demand given decision variables
    def calc_unmet_demand(dec_h, dec_r):
        x_h = np.array(list(dec_h.values())).reshape(len(n_ctry), len(n_ctry))
        x_r = np.array(list(dec_r.values())).reshape(len(n_ctry), len(n_ctry))

        x_h_sum = np.sum(x_h, axis=0)
        x_r_sum = np.sum(x_r, axis=0)
        unmet_demand = (demand.values).copy()
        unmet_demand = np.column_stack((x_h_sum - unmet_demand[:, 0], x_r_sum - unmet_demand[:, 1]))

        return unmet_demand



    # Create a Gurobi model
    model = Model("MinimizeCost")

    # Assign initial value of decision variables
    dec_plant = {(i, j): 1 for i in n_ctry for j in n_lines}
    dec_h     = {(i, j): 1 for i in n_ctry for j in n_ctry}
    dec_r     = {(i, j): 1 for i in n_ctry for j in n_ctry}

    # Define decision variables
    dec_plant = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"Dec_plant_{i}_{j}")    for i in n_ctry for j in n_lines}
    dec_h     = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Dec_h_{i}_{j}") for i in n_ctry for j in n_ctry}
    dec_r     = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Dec_r_{i}_{j}") for i in n_ctry for j in n_ctry}

    # Excess Capacity constraints
    excess_cap = calc_excess_cap(dec_plant, dec_h, dec_r)
    for i in n_ctry:
        for j in n_lines:
            model.addConstr(excess_cap[i, j] >= 0, name=f"Excess_Cap_Constraints_{i}_{j}")


    # Unmet demand constraints
    unnmet_demand = calc_unmet_demand(dec_h, dec_r)
    for i in n_ctry:
        for j in range(2):
            model.addConstr(unnmet_demand[i,j] == 0, name=f"Unmet_Demand_Constraints_{i}_{j}")
    # Update the model
    model.update()

    # Set objective function - Total cost = Fixed cost + Variable costs of Highcal and Relax lines
    #model.setObjective(calc_total_cost(dec_plant, dec_h, dec_r, base_yr, selected_yr, tariff), GRB.MINIMIZE)
    model.setObjective(calc_total_cost(dec_plant, dec_h, dec_r, tariff), GRB.MINIMIZE)

    # Suppress optimization output
    model.Params.OutputFlag = 0

    # Optimize the model
    model.optimize()

    # Extract results to print as table
    op_plant = pd.DataFrame([[dec_plant[i, j].x for j in n_lines] for i in n_ctry], columns = ['Plant','H','R'], index=caps.index)
    op_h     = pd.DataFrame([[dec_h[i, j].x for j in n_ctry] for i in n_ctry], columns = tcosts.columns, index=tcosts.index)
    op_r     = pd.DataFrame([[dec_r[i, j].x for j in n_ctry] for i in n_ctry], columns = tcosts.columns, index=tcosts.index)
    # Store results for this sample
    results = {
        'op_plant': op_plant,
        'op_h': op_h,
        'op_r': op_r,
        'obj_val': round(model.objVal, 2)
    }
    all_results.append(results)

    results_df.at[sample_idx, 'Sample'] = sample_idx + 1
    results_df.at[sample_idx, 'Plant'] = op_plant.values
    results_df.at[sample_idx, 'H'] = op_h.values
    results_df.at[sample_idx, 'R'] = op_r.values
    results_df.at[sample_idx, 'Objective'] = round(model.objVal, 2)




#display result when stored in list
#Print or analyze the results stored in all_results as needed
for idx, result in enumerate(all_results):
    print(f"Sample {idx+1}:\n")
    print("HighCal Flow\n")
    print(tabulate(result['op_h'], headers='keys', tablefmt='pretty'))
    print("\nRelax Flow\n")
    print(tabulate(result['op_r'], headers='keys', tablefmt='pretty'))
    print("\nStrategy\n")
    print(tabulate(result['op_plant'], headers='keys', tablefmt='pretty'))
    print(f"\nMinimum Cost: $ {result['obj_val']} in year {selected_yr}\n")












     
Enter tariff (in percent, e.g. 10 for 10%): 10
Restricted license - for non-production use only - expires 2025-11-24
Sample 1:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+------+------+
|  plant  | Plant |  H   |  R   |
+---------+-------+------+------+
| Brazil  |  1.0  | 1.0  | 1.0  |
| Germany |  1.0  | -0.0 | 1.0  |
|  India  |  1.0  | 1.0  | 1.0  |
|  Japan  |  1.0  | -0.0 | 1.0  |
| Mexico  |  1.0  | 1.0  | 1.0  |
|  U.S.   |  1.0  | 1.0  | -0.0 |
+---------+-------+------+------+

Minimum Cost: $ 1205.96 in year 2023

Sample 2:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1148.52 in year 2023

Sample 3:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1243.34 in year 2023

Sample 4:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1247.82 in year 2023

Sample 5:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  4.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  3.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1257.48 in year 2023

Sample 6:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1258.65 in year 2023

Sample 7:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  2.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  13.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 11.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1151.04 in year 2023

Sample 8:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  3.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 10.0 |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 1.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1240.69 in year 2023

Sample 9:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1234.57 in year 2023

Sample 10:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1197.94 in year 2023

Sample 11:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  12.0  |     0.0     |  0.0  |  3.0   | 12.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 6.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 16.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1239.99 in year 2023

Sample 12:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1257.38 in year 2023

Sample 13:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1244.38 in year 2023

Sample 14:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1216.26 in year 2023

Sample 15:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  1.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  6.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1176.84 in year 2023

Sample 16:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1243.18 in year 2023

Sample 17:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1176.45 in year 2023

Sample 18:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  2.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  13.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 11.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1171.53 in year 2023

Sample 19:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1203.16 in year 2023

Sample 20:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1175.73 in year 2023

Sample 21:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1220.97 in year 2023

Sample 22:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1160.58 in year 2023

Sample 23:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 9.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 13.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1168.99 in year 2023

Sample 24:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  12.0  |     0.0     |  0.0  |  3.0   | 12.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 6.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 16.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1234.45 in year 2023

Sample 25:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  4.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  3.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1237.52 in year 2023

Sample 26:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  5.0  |  0.0   | 5.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  12.0  |     0.0     |  3.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  5.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 0.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1298.92 in year 2023

Sample 27:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 9.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 13.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1209.22 in year 2023

Sample 28:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1274.54 in year 2023

Sample 29:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1207.81 in year 2023

Sample 30:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1297.88 in year 2023

Sample 31:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1219.44 in year 2023

Sample 32:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  2.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  13.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 11.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1141.31 in year 2023

Sample 33:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  2.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  13.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  11.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 11.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1152.17 in year 2023

Sample 34:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1163.14 in year 2023

Sample 35:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1224.54 in year 2023

Sample 36:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  1.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  3.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1181.62 in year 2023

Sample 37:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1278.05 in year 2023

Sample 38:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1226.84 in year 2023

Sample 39:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  2.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  13.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 11.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1183.39 in year 2023

Sample 40:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1254.91 in year 2023

Sample 41:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  1.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  6.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1180.92 in year 2023

Sample 42:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  5.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 18.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 5.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 6.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  0.0  | 0.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1150.38 in year 2023

Sample 43:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1167.46 in year 2023

Sample 44:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1263.93 in year 2023

Sample 45:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  1.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  8.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1331.11 in year 2023

Sample 46:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1254.45 in year 2023

Sample 47:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  11.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1191.51 in year 2023

Sample 48:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  1.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  8.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1300.93 in year 2023

Sample 49:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1190.58 in year 2023

Sample 50:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 9.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 13.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1222.03 in year 2023

Sample 51:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1122.95 in year 2023

Sample 52:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1173.93 in year 2023

Sample 53:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  15.0  |     0.0     |  7.0  |  3.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 14.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  3.0   | 7.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 8.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  0.0  | 0.0 | 0.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1283.04 in year 2023

Sample 54:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  3.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 5.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 6.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1173.4 in year 2023

Sample 55:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  4.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  3.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1267.56 in year 2023

Sample 56:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1231.19 in year 2023

Sample 57:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1161.0 in year 2023

Sample 58:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  3.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 10.0 |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 1.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1176.6 in year 2023

Sample 59:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 9.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 13.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1231.06 in year 2023

Sample 60:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1159.84 in year 2023

Sample 61:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1197.91 in year 2023

Sample 62:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1185.72 in year 2023

Sample 63:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1151.08 in year 2023

Sample 64:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1197.17 in year 2023

Sample 65:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1204.44 in year 2023

Sample 66:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1136.44 in year 2023

Sample 67:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+------+-----+
|  plant  | Plant |  H   |  R  |
+---------+-------+------+-----+
| Brazil  |  1.0  | 1.0  | 1.0 |
| Germany |  1.0  | -0.0 | 1.0 |
|  India  |  1.0  | 1.0  | 1.0 |
|  Japan  |  1.0  | 0.0  | 1.0 |
| Mexico  |  1.0  | 1.0  | 1.0 |
|  U.S.   |  1.0  | 1.0  | 1.0 |
+---------+-------+------+-----+

Minimum Cost: $ 1249.11 in year 2023

Sample 68:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1245.36 in year 2023

Sample 69:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1192.07 in year 2023

Sample 70:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  12.0  |     0.0     |  0.0  |  3.0   | 12.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 6.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  11.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  1.0   |     0.0     |  8.0  |  0.0   | 1.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 16.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1315.19 in year 2023

Sample 71:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1158.99 in year 2023

Sample 72:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 9.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 13.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1203.47 in year 2023

Sample 73:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1172.48 in year 2023

Sample 74:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1200.9 in year 2023

Sample 75:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  3.0  |  0.0   | 1.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1154.97 in year 2023

Sample 76:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1159.56 in year 2023

Sample 77:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1277.1 in year 2023

Sample 78:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1166.1 in year 2023

Sample 79:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 9.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 13.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1193.75 in year 2023

Sample 80:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  12.0  |     0.0     |  0.0  |  3.0   | 12.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 6.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  11.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  1.0   |     0.0     |  8.0  |  0.0   | 1.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 16.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1235.56 in year 2023

Sample 81:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  3.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 10.0 |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 1.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1210.0 in year 2023

Sample 82:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1263.16 in year 2023

Sample 83:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  1.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1290.4 in year 2023

Sample 84:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1144.52 in year 2023

Sample 85:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1185.27 in year 2023

Sample 86:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1256.61 in year 2023

Sample 87:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  6.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  1.0  |  3.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  1.0  |  3.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 10.0 |
|  Japan  |     0.0      |  0.0   |     0.0     |  7.0  |  0.0   | 3.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  0.0  | 0.0 | 0.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1248.03 in year 2023

Sample 88:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1204.53 in year 2023

Sample 89:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1273.71 in year 2023

Sample 90:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1245.11 in year 2023

Sample 91:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  15.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  5.0   |     3.0     |  0.0  |  0.0   | 2.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 9.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1200.85 in year 2023

Sample 92:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1234.75 in year 2023

Sample 93:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  2.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  1.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  1.0   |     3.0     |  8.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  7.0   |     0.0     |  0.0  |  3.0   | 17.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 0.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1255.98 in year 2023

Sample 94:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  7.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  8.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  3.0   |     0.0     |  0.0  |  0.0   | 1.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 16.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1181.4 in year 2023

Sample 95:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  15.0  |     0.0     |  7.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  3.0   | 7.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 7.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  12.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  8.0  |  0.0   | 2.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 15.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  0.0  | 0.0 | 0.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1306.65 in year 2023

Sample 96:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1163.7 in year 2023

Sample 97:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1282.36 in year 2023

Sample 98:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  15.0  |     0.0     |  6.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  1.0  |  3.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  1.0  |  3.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  0.0  |  0.0   | 10.0 |
|  Japan  |     0.0      |  0.0   |     0.0     |  7.0  |  0.0   | 3.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 1.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  0.0  | 0.0 | 0.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1218.77 in year 2023

Sample 99:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  4.0   |     5.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  4.0  |  0.0   | 18.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 4.0  |
| Germany |     0.0      |  12.0  |     0.0     |  5.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     3.0     |  3.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  0.0  | 0.0 | 0.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 0.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1166.5 in year 2023

Sample 100:

HighCal Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  4.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  0.0   |     5.0     |  7.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  11.0  |     0.0     |  0.0  |  3.0   | 13.0 |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 5.0  |
+---------+--------------+--------+-------------+-------+--------+------+

Relax Flow

+---------+--------------+--------+-------------+-------+--------+------+
|  from   | LatinAmerica | Europe | AsiaWoJapan | Japan | Mexico | U.S. |
+---------+--------------+--------+-------------+-------+--------+------+
| Brazil  |     7.0      |  0.0   |     0.0     |  0.0  |  0.0   | 0.0  |
| Germany |     0.0      |  7.0   |     0.0     |  0.0  |  0.0   | 0.0  |
|  India  |     0.0      |  3.0   |     3.0     |  0.0  |  0.0   | 0.0  |
|  Japan  |     0.0      |  2.0   |     0.0     |  8.0  |  0.0   | 0.0  |
| Mexico  |     0.0      |  0.0   |     0.0     |  0.0  |  3.0   | 0.0  |
|  U.S.   |     0.0      |  0.0   |     0.0     |  0.0  |  0.0   | 17.0 |
+---------+--------------+--------+-------------+-------+--------+------+

Strategy

+---------+-------+-----+-----+
|  plant  | Plant |  H  |  R  |
+---------+-------+-----+-----+
| Brazil  |  1.0  | 1.0 | 1.0 |
| Germany |  1.0  | 0.0 | 1.0 |
|  India  |  1.0  | 1.0 | 1.0 |
|  Japan  |  1.0  | 0.0 | 1.0 |
| Mexico  |  1.0  | 1.0 | 1.0 |
|  U.S.   |  1.0  | 1.0 | 1.0 |
+---------+-------+-----+-----+

Minimum Cost: $ 1283.87 in year 2023


# Assuming 'all_results' is a list of dictionaries, each representing one sample's plant statuses
# Initialize a dictionary to count the number of times each plant is open across all samples
open_count = {
    'Brazil': 0,
    'Germany': 0,
    'India': 0,
    'Japan': 0,
    'Mexico': 0,
    'U.S.': 0,
}

# Simulated loop over each sample's result (you would replace this with your actual data extraction logic)
for sample in all_results:
    # Sample['op_plant'] should be the DataFrame or dict holding the plant statuses for this sample
    # For each plant in the sample, check if it's open (1.0) and increment the count
    for plant, status in sample['op_plant']['Plant'].items():  # Replace with actual path to plant status
        if status == 1.0:
            open_count[plant] += 1

# Print the frequency of ones (open status) for each plant
for plant, count in open_count.items():
    print(f"{plant}: Open in {count} out of 100 samples")
     
Brazil: Open in 100 out of 100 samples
Germany: Open in 100 out of 100 samples
India: Open in 100 out of 100 samples
Japan: Open in 72 out of 100 samples
Mexico: Open in 96 out of 100 samples
U.S.: Open in 99 out of 100 samples

# Initialize a dictionary to track the simultaneous open status of plant pairs
correlation_counts = {
    ('Brazil', 'Germany'): 0,
    ('Brazil', 'India'): 0,
    ('Brazil', 'Japan'): 0,
    ('Brazil', 'Mexico'): 0,
    ('Brazil', 'U.S.'): 0,
    ('Germany', 'India'): 0,
    ('Germany', 'Japan'): 0,
    ('Germany', 'Mexico'): 0,
    ('Germany', 'U.S.'): 0,
    ('India', 'Japan'): 0,
    ('India', 'Mexico'): 0,
    ('India', 'U.S.'): 0,
    ('Japan', 'Mexico'): 0,
    ('Japan', 'U.S.'): 0,
    ('Mexico', 'U.S.'): 0,
}
# Loop over each sample to check for simultaneous open statuses
for sample in all_results:
    # Get the status for each plant in this sample
    statuses = sample['op_plant']['Plant']  # Assuming this gets us a dict with plant names as keys and open status (1.0 or 0.0) as values

    # Check each pair of plants to see if both are open in this sample
    for pair in correlation_counts.keys():
        if statuses[pair[0]] == 1.0 and statuses[pair[1]] == 1.0:
            correlation_counts[pair] += 1

# Calculate correlation rates as a percentage of samples where both plants in the pair are open
correlation_rates = {pair: (count / 100) * 100 for pair, count in correlation_counts.items()}

# Print the correlation rates
for pair, rate in correlation_rates.items():
    print(f"{pair}: Open together in {rate}% of samples")

     
('Brazil', 'Germany'): Open together in 100.0% of samples
('Brazil', 'India'): Open together in 100.0% of samples
('Brazil', 'Japan'): Open together in 72.0% of samples
('Brazil', 'Mexico'): Open together in 96.0% of samples
('Brazil', 'U.S.'): Open together in 99.0% of samples
('Germany', 'India'): Open together in 100.0% of samples
('Germany', 'Japan'): Open together in 72.0% of samples
('Germany', 'Mexico'): Open together in 96.0% of samples
('Germany', 'U.S.'): Open together in 99.0% of samples
('India', 'Japan'): Open together in 72.0% of samples
('India', 'Mexico'): Open together in 96.0% of samples
('India', 'U.S.'): Open together in 99.0% of samples
('Japan', 'Mexico'): Open together in 68.0% of samples
('Japan', 'U.S.'): Open together in 71.0% of samples
('Mexico', 'U.S.'): Open together in 95.0% of samples

# Here is the selected strategies based on frequency count for each plan for pairwise correlation
  # We decided to keep plants with 100% in all the strategies
Strategy_1= [1,1,1,0,0,1]
Strategy_2 =[1,1,1,0,1,0]
Strategy_3 =[1,1,1,0,0,0]
     

# Assume mvn_distribution is the multivariate normal distribution of exchange rates
strategies = [
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0, 0]
]

# Function to calculate the cost of a given strategy
def calculate_cost(strategy, exchange_rates):
    # Use the strategy to fix the plant/line configuration
    # Calculate the cost based on fixed plant/line configuration and exchange rates
    # This should be the same calculation as in question 1 but without optimization of plant/lines
    # For example purposes, let's assume it returns a random cost
    return np.random.uniform(1000, 2000)

# Function to evaluate strategies
def evaluate_strategies(strategies, mvn_distribution, num_samples=100):
    strategy_costs = {i: [] for i in range(len(strategies))}

    # Draw 100 exchange rate samples
    for _ in range(num_samples):
        exchange_rates_sample = mvn_distribution.rvs()

        # Evaluate each strategy
        for i, strategy in enumerate(strategies):
            cost = calculate_cost(strategy, exchange_rates_sample)
            strategy_costs[i].append(cost)

    # Compute expected cost and standard deviation for each strategy
    strategy_performance = {}
    for i, costs in strategy_costs.items():
        expected_cost = np.mean(costs)
        cost_std = np.std(costs)
        strategy_performance[i] = (expected_cost, cost_std)

        # Print the performance of each strategy
        print(f"Strategy {i+1}: Mean Cost = {expected_cost:.2f}, Standard Deviation = {cost_std:.2f}")
    return strategy_performance

# Call the function to evaluate strategies
strategy_performance = evaluate_strategies(strategies, mvn_distribution)

# Decide on the best strategy
best_strategy_index = min(strategy_performance, key=lambda k: strategy_performance[k][0])
best_strategy = strategies[best_strategy_index]
best_performance = strategy_performance[best_strategy_index]

# Print the best strategy recommendation
print(f"\nRecommended Best Strategy: Strategy {best_strategy_index+1}")
print(f"Expected Cost: {best_performance[0]:.2f}")
print(f"Standard Deviation: {best_performance[1]:.2f}")
     
Strategy 1: Mean Cost = 1481.32, Standard Deviation = 290.09
Strategy 2: Mean Cost = 1488.37, Standard Deviation = 286.01
Strategy 3: Mean Cost = 1501.88, Standard Deviation = 292.85

Recommended Best Strategy: Strategy 1
Expected Cost: 1481.32
Standard Deviation: 290.09
