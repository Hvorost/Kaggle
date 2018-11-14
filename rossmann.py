# Встанослюємо версії бібліотек, що потрібні для роботи з fastai
! pip install pandas==0.23.4
! pip install matplotlib==3.0.0
! pip install scipy==1.0.0
! pip install pillow==4.1.1
! pip install prompt-toolkit==2.0.7
! pip install pyarrow==0.10.0
! pip install torchtext==0.2.3

# Встановлюємо б-ку fastai та скаємо і розархівовуємо ваги які використовує дана б-ка. 
! pip install fastai==0.7
!wget -O weights.tgz http://files.fast.ai/models/weights.tgz
!tar xvfz weights.tgz -C /usr/local/lib/python3.6/dist-packages/fastai/

# Імпортуємо необхідні б-ки
from fastai.structured import *
from fastai.column_data import *
from IPython.display import HTML, display
np.set_printoptions(threshold=50, edgeitems=20)

PATH=''

# Завантажуємо дані для Kaggle Rossmann Competition
!wget -O rossmann.tgz http://files.fast.ai/part2/lesson14/rossmann.tgz

# Розархівувавши отримаємо наступні бази :googletrend.csv,sample_submission.csv,state_names.csv,store.csv
# store_states.csv,test.csv,train.csv,weather.csv
! tar xvzf rossmann.tgz

# Завантажуємо дані в pandas dataframe
table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
tables = [pd.read_csv(f'{fname}.csv', low_memory=False) for fname in table_names]

# Подивимось на даші дані
for t in tables: display(t.head())
for t in tables: display(t.describe())

# Очищення даних та створення нових ознак

# Приведення ознаки державного вихідного до булевого типу
train.StateHoliday = train.StateHoliday!='0'
test.StateHoliday = test.StateHoliday!='0'

# Функція join_df для з'єднання таблиць по вибраній колонці
def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))

# Об'єднаємо таблиці погоди та назви регіону
weather = join_df(weather, state_names, 'file', 'StateName')

# Заміними назву регіону 'NI' на більш вживану'HB,NI'
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'

# Розширимо значення дати за допомогою ф-ції add_datepart
add_datepart(weather, 'Date', drop=False)
add_datepart(googletrend, 'Date', drop=False)
add_datepart(train, 'Date', drop=False)
add_datepart(test, 'Date', drop=False)

# Для Німеччини у Google є спеціальна категорія, вигтягнемо її.
trend_de = googletrend[googletrend.file == 'Rossmann_DE']

# Зробимо один датасет
store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])

joined = join_df(train, store, "Store")
joined_test = join_df(test, store, "Store")
len(joined[joined.StoreType.isnull()]),len(joined_test[joined_test.StoreType.isnull()])

joined = join_df(joined, googletrend, ["State","Year", "Week"])
joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])
len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])

joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])

joined = join_df(joined, weather, ["State","Date"])
joined_test = join_df(joined_test, weather, ["State","Date"])
len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])

for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns: df.drop(c, inplace=True, axis=1)

# Заповнимо недостаючі дані
for df in (joined,joined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)

# Виділимо ознаки стововно конкурентів
for df in (joined,joined_test):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 
                                                     month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days

# Робота з невідповідними даними
for df in (joined,joined_test):
    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0

# Додамо ознаку конкурентів які відкриті більшк двох років
for df in (joined,joined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24
joined.CompetitionMonthsOpen.unique()

# Додамо ознаку промоакцій
for df in (joined,joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(
        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days

for df in (joined,joined_test):
    df.loc[df.Promo2Days<0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"]//7
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()

# Робота з часом
# Ф-ція для підріхунку днів для роботи з часом
def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1))
    df[pre+fld] = res

# Створимо лист з назвами колонок з якими будемо працювати.
columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
df = train[columns].append(test[columns])

# Підрахуємо к-сть днів "до" та "після" акцій, вихідних
fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')

fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')

fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')

# Встановимо дату як індекс
df = df.set_index("Date")

# Замінимо невідомі значення нулями
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
for o in ['Before', 'After']:
    for p in columns:
        a = o+p
        df[a] = df[a].fillna(0).astype(int)

# Порахуємо сумарний обертовий коефієнт в прямому і зворотньому напрямку
bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
fwd = df[['Store']+columns].sort_index(ascending=False).groupby("Store").rolling(7, min_periods=1).sum()

bwd.drop('Store',1,inplace=True)
bwd.reset_index(inplace=True)

fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)

df.reset_index(inplace=True)

# Об'єднаємо в одим давафрейм
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
df.drop(columns,1,inplace=True)
df["Date"] = pd.to_datetime(df.Date)
joined = join_df(joined, df, ['Store', 'Date'])
joined_test = join_df(joined_test, df, ['Store', 'Date'])

# Видалимо дані там де продажі = 0 та зміними індекс
joined = joined[joined.Sales!=0]
joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)

# Створення ознак
# Виділимо окремо категоріальні та continues ознаки
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

dep = 'Sales'
joined = joined[cat_vars+contin_vars+[dep, 'Date']].copy()
joined_test[dep] = 0
joined_test = joined_test[cat_vars+contin_vars+[dep, 'Date', 'Id']].copy()

for v in cat_vars: joined[v] = joined[v].astype('category').cat.as_ordered()
apply_cats(joined_test, joined)

for v in contin_vars:
    joined[v] = joined[v].fillna(0).astype('float32')
    joined_test[v] = joined_test[v].fillna(0).astype('float32')

idxs = get_cv_idxs(n, val_pct=150000/n)
joined_samp = joined.iloc[idxs].set_index("Date")
samp_size = len(joined_samp); samp_size

samp_size = n
joined_samp = joined.set_index("Date")

df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)
yl = np.log(y)
joined_test = joined_test.set_index("Date")
df_test, _, nas, mapper = proc_df(joined_test, 'Sales', do_scale=True, skip_flds=['Id'],
                                  mapper=mapper, na_dict=nas)

train_ratio = 0.75
train_size = int(samp_size * train_ratio); train_size
val_idx = list(range(train_size, len(df)))

# Глибоке навчання
# Для цього змагання використокується метрика відсотка середньоквадратичної помилки.
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)

# створюємо модель
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,test_df=df_test)

# Створюємо матриці ембедінга
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
m.summary()
# Визначаємо lr
lr = 1e-3
m.lr_find()

# Навчання
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3

m.fit(lr, 3, metrics=[exp_rmspe])

m.fit(lr, 5, metrics=[exp_rmspe], cycle_len=1)

m.fit(lr, 2, metrics=[exp_rmspe], cycle_len=4)

m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3

m.fit(lr, 1, metrics=[exp_rmspe])

m.fit(lr, 3, metrics=[exp_rmspe])

m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)

# Тест
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
lr = 1e-3

m.fit(lr, 3, metrics=[exp_rmspe])

m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)

m.save('val0')

m.load('val0')

x,y=m.predict_with_targs()

exp_rmspe(x,y)

0.01147316926177568

pred_test=m.predict(True)

pred_test = np.exp(pred_test)

joined_test['Sales']=pred_test

csv_fn=f'{PATH}/sub.csv'

joined_test[['Id','Sales']].to_csv(csv_fn, index=False)
