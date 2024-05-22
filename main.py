import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("Dataset/diabetes.csv")
    data["Insulin"] = data["Insulin"].replace(0, np.nan)
    data["Glucose"] = data["Glucose"].replace(0, np.nan)
    return data


df = load()
print(df.head())

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df["New_Is_Insulin_Missing"] = df["Insulin"].notnull().astype('int')
print(df.groupby("New_Is_Insulin_Missing").agg({"Outcome": "mean"}))
print(df.head())

test_stat, pvalue = proportions_ztest(count=[df.loc[df["New_Is_Insulin_Missing"] == 1, "Outcome"].sum(),
                                             df.loc[df["New_Is_Insulin_Missing"] == 0, "Outcome"].sum()], #Outcome olanların sayısı
                                      nobs=[df.loc[df["New_Is_Insulin_Missing"] == 1, "Outcome"].shape[0],
                                            df.loc[df["New_Is_Insulin_Missing"] == 0, "Outcome"].shape[0]])#Gözlem sayısı

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Aykırı Değer Problemini Çözme
#############################################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"] #değişken tipi object ise
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(cat_cols)
print(num_cols)
print(cat_but_car)

num_cols = [col for col in num_cols]
print(df.shape)

for col in num_cols:
    print(col, check_outlier(df, col))
    replace_with_thresholds(df, col)
    print(col, check_outlier(df, col))

#############################################
# Missing Values (Eksik Değerler)
#############################################

# eksik gozlem var mı yok mu sorgusu
print(df.isnull().values.any())

# degiskenlerdeki eksik deger sayisi
print(df.isnull().sum().sort_values(ascending=False))

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

print(na_cols)

#############################################
# Eksik Değer Problemini Çözme
#############################################

#############################################
# Medyan ile değiştirme
#############################################


def replace_with_mean(dataframe, variable):
    variable_mean = dataframe[variable].mean()
    dataframe[variable] = dataframe[variable].replace(np.nan, variable_mean)


for col in na_cols:
    replace_with_mean(df, col)

# eksik gozlem var mı yok mu sorgusu
print(df.isnull().values.any())

#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# One-Hot Encoding
#############################################


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 15 >= df[col].nunique() > 2]

print(ohe_cols)

print(one_hot_encoder(df, ohe_cols).head())

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################


def standart_scaler(dataframe, variable):
    ss = StandardScaler()
    dataframe[variable + '_standart_scaled'] = ss.fit_transform(dataframe[[variable]])


for col in num_cols:
    standart_scaler(df, col)

print(df.head())

#############################################
# 8. Model
#############################################

y = df["Outcome"]
X = df.drop(["New_Is_Insulin_Missing", "Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(accuracy_score(y_pred, y_test))

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
y = dff["Outcome"]
X = dff.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(accuracy_score(y_pred, y_test))