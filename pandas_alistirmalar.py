

##################################################
# Pandas Alıştırmalar
##################################################


import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df=sns.load_dataset("titanic")
df.head()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
df.sex.value_counts(dropna=False)
# male      577
# female    314

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
for col in df.columns:
    print(f"Unique Numbers of {col}: {df[col].nunique()}")

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################
df['pclass'].nunique(dropna=False)
# 3


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
df[['pclass','parch']].nunique(dropna=False)
# pclass    3
# parch     7

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################
print(df.embarked.dtype)
df['embarked']=df['embarked'].astype('category')
print(df.embarked.dtype)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################
df[df['embarked']=='C']


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################
df[df['embarked']!='S']
df[~(df['embarked']=='S')]


#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################
df[(df['age']<30)&(df['sex']=='female')]


#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################
df[(df['fare'] > 500) | (df['age'] > 70)]


#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################
df.isnull().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
df.drop(['who'], axis=1, inplace=True)
df.pop("who")

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
df.deck.mode()[0]
df.deck.fillna(df.deck.mode()[0], inplace=True)
df.loc[df['deck'].isnull(),['deck']]=df.deck.mode()[0]
# df.loc[df['deck'].isnull(),['deck']]='C'


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
df['age'].median()
df['age'].fillna(df['age'].median(),inplace=True)
df.age.isnull().sum()

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################
df.groupby(['pclass','sex'])['survived'].agg(['sum','count','mean'])
df.groupby(['pclass','sex'])[['survived']].agg(['sum','count','mean'])
df.groupby(['pclass','sex']).agg({'survived':['sum', 'count', 'mean']})


#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
df['age_flag'] = df['age'].apply(lambda x: 1 if x < 30 else 0)

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
df=sns.load_dataset("tips")
df.head()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df.groupby('time')[['total_bill']].agg(['sum','min','max','mean'])
df.groupby('time').agg({'total_bill':['sum','min','max','mean']})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df.groupby(['day','time']).agg({'total_bill':['sum','min','max','mean']})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################
df[(df['sex']=='Female')&(df['time']=='Lunch')].groupby(['day']).agg({'total_bill':['sum','min','max','mean'], 'tip':['sum','min','max','mean']})
df[(df['sex']=='Female')&(df['time']=='Lunch')].groupby(['day'])[['total_bill','tip']].agg(['sum','min','max','mean'])

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################
round(df[(df['size']<3) & (df['total_bill']>10)]['total_bill'].mean(),2)
# 17.18

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################
df['total_bill_tip_sum']=df['total_bill']+df['tip']
df.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################
df_first30=df.sort_values('total_bill_tip_sum',ascending=False).head(30) #[:30]
df_first30


#########################################
# Algo 19.01
# Bir yarışma için katılımcıların puanları göz önüne alındığında,
# ikincinin puanı bulmanız istenmektedir. Puanlar 2 3 6 6 5 olarak verilmiştir.
# İlk olarak kaç katılımcı olduğunu kontrol edin.
# Sonrasında puanları alın.
# Ve ikinciyi bulun.
#
# Örnek girdi:
# 5
# 2 3 6 6 5
# Örnek çıktı:
# 5
#########################################

def find_second():
    participants = int(input(
        "Lütfen katılımcı sayısını giriniz:"
    ))
    points = input(
        "Lütfen katılımcıların puanlarını giriniz:"
    )
    points_list = [int(point) for point in points.split()]
    print(f'{participants} katılımcı var ve ikincinin puanı: {sorted(list(set(points_list)))[-2]}')

find_second()
