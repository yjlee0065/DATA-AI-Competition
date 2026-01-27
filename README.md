
# 대전 중구 지역경제 활성화를 위한 성심당 방문객 행선지 추천 시스템
소비패턴 분석과 크롤링을 바탕으로

## 1. 분석 배경 설명 및 분석 결과
### 1.1 분석 배경
- 성심당은 대전의 방문객 유입을 늘리며 대전 내 압도적인 인기 관광지로 자리잡음
- 상생 프로젝트 등을 통해 지역 경제 활성화에도 기여하고 있음
- 대전세종포럼(2024 가을 통권 제90호, 대전세종연구원)에 따르면 대전 관광객들의 **목적지의 약 60%가 성심당**이며, 관광유형은 **대부분이 당일치기 여행**임
- 현재 성심당의 인기는 대전 중구 전체 상권 및 관광 산업 활성화로 이어지지 않고 있음.

  ➡️ 이용 고객이 타 관광지도 방문하고 체류형 관광으로 발전하기 위해서는 **데이터 분석이 필요** 

### 1.2 성심당 방문객 소비 패턴 분석 결과  
(상세한 분석 과정 및 결과는 ppt에서 확인 가능)
- 성심당과 식당, 카페 등의 소비는 많이 이루어지지만,
**숙박이나 관광지 관련 소비는 매우 적음**
- 점심과 오후에 가장 이용 건수 높았다가 **18-22 시간대에는 감소**
- 대부분이 **당일치기 여행**이며, **성심당과 식사 외의 관광 이루어지지 않음**

## 2. Pain Point & Solution

### 2.1 Pain Point
- 당일치기 여행 증가로 대전에 머무는 시간 감소
- 성심당과 식사, 카페 등 요식업에 집중된 소비
- 고객들의 대전 관광지 인지 정도 낮음
  
### 2.2 Solution

**고객 유형과 이동 패턴에 맞춰 성심당 방문객들을 대전의 다양한 관광 명소로 유도하는 서비스 구축 필요**  
1. 관광지 데이터를 수집한 후 크롤링  
2. 전처리를 거쳐 tf-idf 행렬 생성 
3. 생성된 tf-idf 행렬로 관관지들을 군집화  
4. 사용자의 유동 패턴을 고려하여 관광지 우선 순위 추천  

❓관광지가 이미 역사관광, 문화관광 등으로 구분되어 있는데 군집화를 하고자 하는 이유 :

기존의 구분은 관광지의 성격만 나타낼 뿐 관광객들이 느끼는 감정과 감성을 반영하고 있지 않음  
➡️ 크롤링을 통해 관광객들이 생각하는 관광지의 감상과 느낌 추출 & 더 세부적인 테마나 감성적인 카테고리 도출 가능
<br>

## 3. 관광지 추천 시스템 구현
### 3.1 관광지 데이터 수집
대전 중구에 위치한 관광지들을 파악하기 위해 한국관광 데이터랩 사이트에서 대전 중구의 중심 관광지명들을 파일로 수집 (대전_관광지_수정.csv)

출처: https://datalab.visitkorea.or.kr/datalab/portal/loc/getAreaDataForm.do#  

<br>

### 3.2 블로그 크롤링

- 2020년 이후의 블로그 글만 크롤링
- 하나의 블로그에 여러 관광지들에 대한 리뷰가 같이 있는 경우가 많은 ‘여행’ 과 ‘코스’ 키워드를 제외
- 부동산 관련 홍보 블로그가 많이 검색되는 ‘부동산’, ‘월세’, ‘분양’, ‘주택’ 키워드를 제외
- 그 외에 불필요하게 같이 검색되었던 ‘공주’와 ‘주차장’ 키워드도 제외
- 관광지들 중 다른 지역에 같은 이름의 관광지가 존재하는 것이 있어 ‘대전’ 키워드를 반드시 포함해 검색
  
**⇒ 34개 관광지에 대해 50개씩, 총 1700개의 블로그 크롤링 (blog.csv)**  
<br>

### 3.3 블로그 글 전처리

- 블로그의 글을 가지고 군집화를 하기 전 단어의 출현빈도나 관계를 파악하기 위해 토큰화 진행  
- 불필요한 단어들을 불용어 사전에 넣어 제거 
<br>

### 3.4 군집분석

```ruby
vectorizer = TfidfVectorizer(max_features=300)
tfidf_matrix = vectorizer.fit_transform(blog_df['token'])
# 결과 출력
pd.DataFrame(tfidf_matrix.toarray())
word = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=word, index=blog_df['place'])

# 표준화
scaler = MinMaxScaler()
tfidf_scaled = scaler.fit_transform(tfidf_df)
```
- 파이썬의 TfidfVectorizer함수를 이용해 전처리된 관광지별 블로그글을 가지고 **tf-idf행렬**을 생성
- 생성된 tf-idf행렬을 이용해 **K-Means 군집분석**을 수행  
<br> <br>

```ruby
# KMeans Inertia 계산
ks = range(1, 10)
inertias = []
for k in ks:
    model = KMeans(n_clusters=k, random_state=3)
    model.fit(tfidf_df)
    inertias.append(model.inertia_)

# Silhouette Score 계산
silhouette_scores = []
cluster_range = range(2, 10)
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=3)
    labels = kmeans.fit_predict(tfidf_df)
    silhouette_avg = silhouette_score(tfidf_df, labels)
    silhouette_scores.append(silhouette_avg)

# 실루엣 스코어를 배열로 변환
scores_array = np.array(silhouette_scores).reshape(-1, 1)

# 플롯 설정 (2개의 서브플롯을 하나의 화면에 배치)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 첫 번째 플롯: ks vs inertias
axes[0].plot(ks, inertias, '-o', color='chocolate')
axes[0].set_xlabel('number of clusters, k')
axes[0].set_ylabel('inertia')
axes[0].set_title('Inertia vs Number of Clusters')
axes[0].set_xticks(ks)

# 두 번째 플롯: Silhouette Scores 히트맵
sns.heatmap(scores_array, annot=True, cmap="YlOrBr", yticklabels=cluster_range, ax=axes[1])
axes[1].set_ylabel('n_clusters')
axes[1].set_xlabel('silhouette_score')
axes[1].set_title('Silhouette Scores for different number of clusters')

# 플롯 간 간격 자동 조정
plt.tight_layout()
plt.show()
```

<img src="https://github.com/user-attachments/assets/1aa8daae-5355-4dab-86ab-19a49deccad1" width="800"/>

- 적절한 군집 수를 결정하기  위해 **Elbow Method**로 그래프 생성
- 군집 간 거리의 합을 나타내는 inertia 값이 급격히 떨어지며 꺾이는 5를 군집의 숫자로 결정  
<br> <br>

```ruby
k = 5
kmeans = KMeans(n_clusters=k, random_state=3)
kmeans.fit(tfidf_df)

blog_df['cluster'] = kmeans.labels_
tfidf_cluster = tfidf_df.copy()
tfidf_cluster['cluster'] = kmeans.labels_
tfidf_scaled['cluster'] = kmeans.labels_

# 2D PCA 수행
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(tfidf_scaled)

# 3D PCA 수행
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(tfidf_scaled)

# 플롯 설정
fig = plt.figure(figsize=(16, 7))

# 2D PCA 서브플롯
ax1 = fig.add_subplot(121)  # 1행 2열 첫 번째 플롯
unique_clusters = set(tfidf_cluster['cluster'])
for cluster in unique_clusters:
    cluster_points = pca_result_2d[tfidf_cluster['cluster'] == cluster]
    ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=50, alpha=0.7)

# 2D 플롯 설정
ax1.set_title("2D PCA Plot of TF-IDF Data by Cluster")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.legend()
ax1.grid(True)

# 3D PCA 서브플롯
ax2 = fig.add_subplot(122, projection='3d')  # 1행 2열 두 번째 플롯 (3D)
for cluster in unique_clusters:
    cluster_points = pca_result_3d[tfidf_cluster['cluster'] == cluster]
    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster}', s=50, alpha=0.7)

# 3D 플롯 설정
ax2.set_title("3D PCA Plot of TF-IDF Data by Cluster")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_zlabel("Principal Component 3")
ax2.legend()

# 전체 레이아웃 조정
plt.tight_layout()
plt.show()
```

<img src="https://github.com/user-attachments/assets/03752f2d-4293-44cf-9690-6a59d353e580" width="800"/>

- 결정된 군집의 수를 토대로 K=5인 K-Means 군집화를 수행
- tf-idf행렬을 PCA(주성분분석)을 통해 2차원과 3차원으로 차원축소를 한 후 군집들을 시각화  
<br> <br>

**각 군집에 포함된 관광지 확인**

```ruby
cluster_places = blog_df.groupby('cluster')['place'].apply(list).reset_index()
pd.set_option('display.max_colwidth', None)
cluster_places[['place']]
```

<img src="https://github.com/user-attachments/assets/d410b518-ea51-46ec-a2c7-984febe24cec" width="1000"/>  

<br> <br>

**각 군집의 속성을 파악하기 위해 군집별로 빈도수가 많은 단어들 확인**

```ruby
# 클러스터별로 많이 나온 단어 추출
def extract_top_words_by_cluster(df, num_words=100):
    cluster_top_words = {}

    for cluster, group in df.groupby('cluster'):
        all_adjectives = ' '.join(group['token'].tolist())
        words = all_adjectives.split() 

        # 단어 빈도 계산
        word_counts = Counter(words)
        # 가장 많이 나온 단어 num_words개 추출
        top_words = word_counts.most_common(num_words)
        # 클러스터별로 저장
        cluster_top_words[cluster] = top_words

    return cluster_top_words

# 결과 추출
top_words_by_cluster = extract_top_words_by_cluster(blog_df)

# 클러스터별 상위 10개 단어 출력
for cluster, top_words in top_words_by_cluster.items():
    print(f"Cluster {cluster}: {top_words}")
```

<img src="https://github.com/user-attachments/assets/cf914dd9-3a2b-4bbf-aff5-04c7983ffc5c" width="1000"/>  

<br>

**⇒ 군집별 빈도수가 높은 단어를 기준으로 군집의 이름 결정**
<br> <br>

### 3.5 관광지별 유동인구 데이터 생성

```ruby
# pop_data에서 'pop202307'부터 'pop202406'까지의 데이터프레임 이름 가져오기
pop_dfs = [f"pop20230{i}" for i in range(7, 10)] + [f"pop20240{i}" for i in range(1, 7)]

# 각 데이터프레임에서 열을 가져오고, 셀번호 중복 행을 제거한 후 합치기
combined_df = pd.DataFrame()

for df_name in pop_dfs:
    # pop_data에서 데이터프레임을 가져오기
    df = pop_data[df_name]
    
    # 열 선택 및 중복된 셀번호 행 제거
    df_filtered = df[['셀번호', 'x좌표', 'y좌표', "행정동코드"]].drop_duplicates(subset='셀번호')
    
    # 합치기
    combined_df = pd.concat([combined_df, df_filtered], ignore_index=True)

# 최종 합친 데이터프레임에서 셀번호 중복 행 다시 제거
final_df = combined_df.drop_duplicates(subset='셀번호')
final_df = final_df.merge(hdong[['행정동코드', "읍면동명"]], how="left", left_on="행정동코드", right_on="행정동코드")
final_df = final_df.drop_duplicates(subset='셀번호').reset_index()
final_df.to_csv("cell.csv", index=False)
```
- 유동인구 데이터 내에 존재하는 모든 cell들의 x좌표와 y좌표를 구하기 위해 데이터를 합침
- cell 정보 데이터는 'cell.csv'로 저장
 
<br>

```ruby
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import folium
from tqdm import tqdm
from pyproj import Proj, Transformer

tqdm.pandas()  # pandas의 tqdm 확장 활성화

# KATEC 좌표계와 WGS84 좌표계 정의
WGS84 = {'proj': 'latlong', 'datum': 'WGS84', 'ellps': 'WGS84'}
KATEC = {'proj': 'tmerc', 'lat_0': '38N', 'lon_0': '128E', 
         'ellps': 'bessel', 'x_0': 400000, 'y_0': 600000,
         'k': 0.9999, 'units': 'm',
         'towgs84': '-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43'}

# KATEC -> WGS84 변환 함수
def KATEC_to_wgs84(x, y):
    transformer = Transformer.from_proj(Proj(**KATEC), Proj(**WGS84), always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon

# 좌표 변환 적용 (progress_apply를 통해 진행 바 표시)
final_df[['latitude', 'longitude']] = final_df.progress_apply(lambda row: KATEC_to_wgs84(row['x좌표'], row['y좌표']), axis=1, result_type='expand')

```

- 각 관광지들과의 거리를 계산하기 위해 유동인구 데이터의 각 셀에 대해 좌표 변환
  <br>
  **KATEC → WG584**
- 변환 후 해당 셀들과 대전 각 셀들과 관광지 반경 100m를 folium을 이용하여 시각화 

<br>

![image](https://github.com/user-attachments/assets/b5a08936-8089-4097-97c3-e69d581a6eea)
<br>
```ruby
# 변환된 좌표를 지도에 표시
m = folium.Map(location=[final_df['latitude'].mean(), final_df['longitude'].mean()], zoom_start=15)

# 각 좌표에 검은 점 추가 (팝업 제거, 점 크기 조정)
for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=1,  # 점의 크기
        color='black',  # 점의 색깔
        fill=True,
        fill_opacity=1
    ).add_to(m)

# 관광지명과 반경 100미터 표시 및 점의 개수 계산
tourist_counts = []
tourist_points = []  # 반경 내 점들의 셀번호를 저장할 리스트

for _, row in 관광지.iterrows():
    # 관광지명 마커 추가
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['관광지명']
    ).add_to(m)
    
    lat = row['Latitude']
    lon = row['Longitude']

    # 반경 100미터 원 추가
    folium.Circle(
        location=[lat, lon],
        radius=100,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.3,
        weight=1,
    ).add_to(m)

    # 관광지와 셀 간 거리 계산하여 반경 100미터 내 셀번호 저장
    within_radius_cell = []
    for _, cell_row in final_df.iterrows():
        distance = geodesic((lat, lon), (cell_row['latitude'], cell_row['longitude'])).meters
        if distance <= 100:
            within_radius_cell.append(cell_row['셀번호'])  # 100m 이내의 셀번호 저장
    
    count = len(within_radius_cell)  # 반경 내 점의 개수

    # 결과 저장
    tourist_counts.append({'관광지명': row['관광지명'], '점 개수': count})
    tourist_points.append({'관광지명': row['관광지명'], '셀번호': within_radius_cell})
  
  # 점 개수와 셀번호 데이터프레임으로 정리
tourist_counts_df = pd.DataFrame(tourist_counts)
tourist_points_df = pd.DataFrame(tourist_points)
# tourist_counts_df, tourist_points_df, 관광지 세 개의 데이터프레임을 '관광지명'을 기준으로 병합
tour = pd.merge(tourist_counts_df, tourist_points_df, on='관광지명', how='inner')
tour = pd.merge(tour, 관광지, on='관광지명', how='inner')
```
- 이후 파이썬의 geopy.distance 라이브러리의 geodesic 함수를 사용해 관광지와 셀 간의 거리를 계산하여 반경 100내 셀 번호의 정보를 저장
- 각 관광지 반경 100m 내 셀 번호와 셀 개수를 합쳐 'tour.csv'로 저장
  
<br>

```ruby
# pop_data 안의 각 데이터프레임에 적용할 함수
def group_age_columns(df):
    # 남성 연령대 그룹화
    df['남성_10대'] = df[['남성10~14', '남성15~19']].sum(axis=1)
    df['남성_20대'] = df[['남성20~24', '남성25~29']].sum(axis=1)
    df['남성_30대'] = df[['남성30~34', '남성35~39']].sum(axis=1)
    df['남성_40대'] = df[['남성40~44', '남성45~49']].sum(axis=1)
    df['남성_50대'] = df[['남성50~54', '남성55~59']].sum(axis=1)
    df['남성_60대'] = df[['남성60~64', '남성65~69']].sum(axis=1)
    df['남성_70대이상'] = df[['남성70세 이상']].sum(axis=1)

    # 여성 연령대 그룹화
    df['여성_10대'] = df[['여성10~14', '여성15~19']].sum(axis=1)
    df['여성_20대'] = df[['여성20~24', '여성25~29']].sum(axis=1)
    df['여성_30대'] = df[['여성30~34', '여성35~39']].sum(axis=1)
    df['여성_40대'] = df[['여성40~44', '여성45~49']].sum(axis=1)
    df['여성_50대'] = df[['여성50~54', '여성55~59']].sum(axis=1)
    df['여성_60대'] = df[['여성60~64', '여성65~69']].sum(axis=1)
    df['여성_70대이상'] = df[['여성70세 이상']].sum(axis=1)

    # 불필요한 원본 컬럼 삭제
    columns_to_drop = [
        '남성10세미만', '남성10~14', '남성15~19', '남성20~24', '남성25~29', '남성30~34', '남성35~39', 
        '남성40~44', '남성45~49', '남성50~54', '남성55~59', '남성60~64', '남성65~69', '남성70세 이상', 
        '여성10세미만', '여성10~14', '여성15~19', '여성20~24', '여성25~29', '여성30~34', '여성35~39', 
        '여성40~44', '여성45~49', '여성50~54', '여성55~59', '여성60~64', '여성65~69', '여성70세 이상'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

# pop_data 안의 모든 데이터프레임에 연령대 그룹화를 적용
pop_data_keys = ['pop202307', 'pop202308', 'pop202309', 'pop202310', 'pop202311', 'pop202312', 
                 'pop202401', 'pop202402', 'pop202403', 'pop202404', 'pop202405', 'pop202406']

for key in pop_data_keys:
    group_age_columns(pop_data[key])

# 결과 확인 (예시로 pop202307 데이터프레임 확인)
#print(pop_data['pop202307'].head())
```
- 유동인구 데이터의 연령대를 카드 데이터와 맞추기 위해 위와 같이 재범주화

<br>

```ruby
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# 데이터 합치기
popdata_combined_spring = pd.concat([pop_data['pop202403'], pop_data['pop202404'], pop_data['pop202405']], ignore_index=True)

# 날짜 변수(일자)를 datetime 타입으로 변환
popdata_combined_spring['일자'] = pd.to_datetime(popdata_combined_spring['일자'], format='%Y%m%d')

# 공휴일 계산 (예시로 미국 공휴일을 사용, 한국 공휴일은 따로 처리해야 함)
cal = calendar()
holidays = cal.holidays(start=popdata_combined_spring['일자'].min(), end=popdata_combined_spring['일자'].max())

# '휴일' 변수 생성: 토, 일요일 또는 공휴일이면 '휴일', 그렇지 않으면 '평일'
popdata_combined_spring['holiday'] = popdata_combined_spring['일자'].apply(lambda x: '휴일' if x.weekday() >= 5 or x in holidays else '평일')

# 'time_period' 변수 생성: 시간대를 오전, 점심, 오후, 저녁, 심야로 분류
def time_to_period(hour):
    if 6 <= hour < 11:
        return '오전'
    elif 11 <= hour < 15:
        return '점심'
    elif 15 <= hour < 18:
        return '오후'
    elif 18 <= hour < 22:
        return '저녁'
    else:
        return '심야'

# '시간대' 변수를 시간대 범주로 변환
popdata_combined_spring['time_period'] = popdata_combined_spring['시간대'].apply(time_to_period)
```
**(해당 예시는 봄(2023년 3월, 2024년 4월, 2024년 5월) 관광지별 반경 100m 유동인구 데이터)**
- 유동인구 데이터의 일자 변수를 이용하여 평일이면 '평일', 주말 또는 공휴일이면 '휴일'로 구분하는 holiday 변수 생성
- 카드 데이터와 시간대를 맞추기 위해 'time_period' 변수 새롭게 생성
  <br>**오전:6시-10시, 점심:11시-14시, 오후:15-17시, 저녁:18-21시, 심야:22시-5시**


<br>

```ruby
# 필요한 남성 및 여성 연령대 컬럼
columns_to_median = [
    '남성_10대', '남성_20대', '남성_30대', '남성_40대', '남성_50대', '남성_60대', '남성_70대이상',
    '여성_10대', '여성_20대', '여성_30대', '여성_40대', '여성_50대', '여성_60대', '여성_70대이상'
]

# 관광지명별로 데이터를 그룹핑하고 중앙값 계산
result_list = []

for _, row in tour.iterrows():
    # 각 관광지명에 해당하는 셀번호 리스트
    cell_numbers = row['셀번호']
    
    # popdata_combined에서 해당 셀번호들에 해당하는 데이터 필터링
    filtered_data = popdata_combined_spring[popdata_combined_spring['셀번호'].isin(cell_numbers)]
    
    # 'holiday', 'time_period'별로 그룹핑하여 남녀 연령대별 인구의 중앙값 계산
    grouped_data = filtered_data.groupby(['holiday', 'time_period'])[columns_to_median].median().reset_index()
    
    # 관광지명 열 추가
    grouped_data['관광지명'] = row['관광지명']
    
    # 결과를 리스트에 저장
    result_list.append(grouped_data)

# 결과를 하나의 데이터프레임으로 병합
final_result_spring = pd.concat(result_list, ignore_index=True)
```
- 휴일 여부, 시간대, 성별 및 나이별로 관광지 반경 100m 내의 셀에 속하는 유동인구의 median값을 구함

<br>
  
> **median 값을 사용하는 이유**
> ![image](https://github.com/user-attachments/assets/3e0d290a-6579-400b-8dd3-22ee8687e95d)

> → 전체 유동인구 데이터의 시간대별 데이터가 어떻게 분포되어 있는지 확인 
> <br> ⇒ skewed, 평일에는 0값이 많음.
> <br>⇒ 여러 cell들의 대푯값으로 mean을 사용하면 분포 왜곡이 있을 것이라 판단하여 median 사용

<br> <br>

```ruby
# 관광지명별로 그룹핑된 데이터를 가공하여 열 이름을 결합한 형태로 변환하는 코드

# 필요한 남성 및 여성 연령대 컬럼
columns_to_median = [
    '남성_10대', '남성_20대', '남성_30대', '남성_40대', '남성_50대', '남성_60대', '남성_70대이상',
    '여성_10대', '여성_20대', '여성_30대', '여성_40대', '여성_50대', '여성_60대', '여성_70대이상'
]

# 그룹핑된 데이터프레임을 넓은 형식으로 변환하는 함수
def create_wide_format(df):
    wide_format = df.pivot_table(
        index='관광지명',
        columns=['holiday', 'time_period'],
        values=columns_to_median
    )
    
    # MultiIndex를 단일 컬럼으로 변환 (열 이름을 "holiday_time_period_컬럼명" 형식으로 변환)
    wide_format.columns = [f'{h}_{t}_{col}' for h, t, col in wide_format.columns]
    
    # 인덱스를 초기화하여 관광지명을 열로 변환
    wide_format.reset_index(inplace=True)
    
    return wide_format

# 주어진 데이터로 넓은 형식의 데이터프레임 생성
final_result_wide_spring = create_wide_format(final_result_spring)
final_result_wide_spring.rename(columns=lambda x: '봄_' + x if x != '관광지명' else x, inplace=True)
```
- 변수명을 '계절_휴일여부_시간대_성별_나이' 로 변경
- 봄, 여름(2023년 7-8월, 2024년 6월), 가을(2023년 9-11월), 겨울(2023년 12월, 2024년 1-2월)에 대해 각각 위와 같은 방식으로 유동인구 데이터셋 생성
  <br> 'final_result_spring.csv', 'final_result_summer.csv', 'final_result_fall.csv', 'final_result_winter.csv'

<br>

```ruby
final_result_wide_spring = pd.read_csv("../data/result/final_result_spring.csv")
final_result_wide_summer = pd.read_csv("../data/result/final_result_summer.csv")
final_result_wide_fall = pd.read_csv("../data/result/final_result_fall.csv")
final_result_wide_winter = pd.read_csv("../data/result/final_result_winter.csv")

최종 = 관광지.merge(final_result_wide_spring, how="left", left_on="관광지명", right_on="관광지명")
최종 = 최종.merge(final_result_wide_summer, how="left", left_on="관광지명", right_on="관광지명")
최종 = 최종.merge(final_result_wide_fall, how="left", left_on="관광지명", right_on="관광지명")
최종 = 최종.merge(final_result_wide_winter, how="left", left_on="관광지명", right_on="관광지명")

# 유동인구가 없는 NaN 값을 0으로 채우기
최종.fillna(0, inplace=True)
최종.to_csv("final.csv", index=False)
```
- 관광지명과 각 계절의 유동인구 데이터를 합친 final.csv 데이터 생성

<br>

```ruby
pop = pd.read_csv("../data/final.csv")

place_cluster = blog_df.groupby('cluster')['place'].apply(lambda x: pd.Series(x)).reset_index(level=0)
place_cluster.columns = ['군집', 'place']

pop_cluster = pd.merge(pop, place_cluster, left_on='관광지명', right_on='place', how='left')
pop_cluster = pop_cluster.drop(['place'], axis=1)
```
- final.csv 데이터셋 불러오기
- 군집화된 결과를 데이터에 추가  

<br> <br>

### 3.6 관광지 추천 시스템

```ruby
# 대시보드 앱 생성
app = Dash(__name__)

# 성별, 연령대, 시간대 옵션
genders = ['남성', '여성']
ages = ['10대', '20대', '30대', '40대', '50대', '60대', '70대이상']
times = ['오전', '점심', '오후', '저녁', '심야']

# 군집별 설명
cluster_descriptions = {
    0: "1. 연극과 공연을 즐길 수 있는 곳",
    1: "2. 맛집, 먹거리가 많은 곳",
    2: "3. 작품과 전시를 관람할 수 있는 곳",
    3: "4. 아이와 같이 가기 좋은 볼거리가 많은 곳",
    4: "5. 역사를 느낄 수 있는 곳"
}

# 대시보드 레이아웃 정의
app.layout = html.Div(children=[
    html.H1(children="대전 중구 관광지 추천", style={'textAlign': 'center'}),
    
    html.Div(children=[
        html.Label('성별을 선택하세요'),
        dcc.Dropdown(id='gender', options=[{'label': g, 'value': g} for g in genders], value=None),
        
        html.Label('연령대를 선택하세요'),
        dcc.Dropdown(id='age', options=[{'label': a, 'value': a} for a in ages], value=None),
        
        html.Label('시간대를 선택하세요'),
        dcc.Dropdown(id='time', options=[{'label': t, 'value': t} for t in times], value=None),
        
        html.Label('날짜를 입력하세요 (yyyy-mm-dd)'),
        dcc.Input(id='date', type='text', value=None),
        
        html.Button('제출', id='submit-button', n_clicks=0)  # 제출 버튼 추가
    ]),
    
    html.Div(id='filtered-data', style={'margin-top': '20px'})  # 결과 출력 영역
])

# 콜백 정의
@app.callback(
    Output('filtered-data', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('gender', 'value'),
    Input('age', 'value'),
    Input('time', 'value'),
    Input('date', 'value')
)
def update_filtered_data(n_clicks, gender, age, time, date):
    # 입력된 날짜가 유효한지 확인
    try:
        if date and len(date) == 10:  # 'yyyy-mm-dd' 형식 확인
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
        else:
            return ""
    except ValueError:
        return ""

    # 사용자가 모든 선택을 완료하기 전에는 아무것도 표시하지 않음
    if not (n_clicks > 0 and gender and age and time and date):
        return ""

    # 휴일 또는 평일 계산
    kr_holidays = holidays.KR()
    holiday = '휴일' if date_obj.weekday() >= 5 or date_obj in kr_holidays else '평일'

    # 계절 계산
    month = date_obj.month
    if month in [3, 4, 5]:
        season = '봄'
    elif month in [6, 7, 8]:
        season = '여름'
    elif month in [9, 10, 11]:
        season = '가을'
    else:
        season = '겨울'
    
    # 필터링할 컬럼 이름 생성
    value = f"{season}_{gender}_{age}_{holiday}_{time}"
    
    # 데이터 필터링 및 순위 계산
    filtered_data = pop_cluster[['관광지명', '도로명주소', '군집', value]].copy()
    filtered_data['순위'] = filtered_data.groupby('군집')[value].rank(ascending=False)
    filtered_data = filtered_data.sort_values(by=['군집', '순위'], ascending=[True, True])

    # 군집별로 최대 2개의 관광지명과 주소 출력
    result = []
    for cluster, description in cluster_descriptions.items():
        # 해당 군집에 해당하는 데이터 필터링
        cluster_data = filtered_data[filtered_data['군집'] == cluster].head(2)

        if not cluster_data.empty:
            result.append(html.H3(description))
            result.append(html.Table([
                html.Thead(html.Tr([html.Th("관광지명"), html.Th("주소")])),  # 수정된 부분: html.Thead에 대한 괄호 맞춤
                html.Tbody([
                    html.Tr([
                        html.Td(cluster_data.iloc[i]['관광지명'], style={'padding-right': '20px'}),  # 관광지명과 주소 사이에 패딩 추가
                        html.Td(cluster_data.iloc[i]['도로명주소'])
                    ])
                    for i in range(len(cluster_data))
                ])
            ]))
            result.append(html.Br())  # 군집별 결과 사이에 줄바꿈 추가

    return result

# 앱 실행
if __name__ == '__main__':
    app.run_server(debug=True)
```

![Animation](https://github.com/user-attachments/assets/5083c8c9-29c3-4f9e-b810-07efce30cb6a)

- 파이썬의 Dash 라이브러리를 이용
- 사용자에게서 성별, 연령, 시간대와 방문하려는 날짜를 입력받음
- 앞서 생성한 데이터에서 군집별로 해당 성별, 연령, 시간대, 계절, 휴일평일 여부에 유동인구가 많은 관광지 2개씩 추천해주는 대시보드 생성  
<br>

## 4. 결론

**데이터 분석 및 문제 상황**

- 대전 중구의 관광객들은 성심당 위주의 방문과 소비 이루어짐
- 중구 내의 다른 매력적인 관광지들은 인지도 낮아 방문이 적음
- 관광객들에게 충분히 노출되지 않음
<br>

**해결 방안**

-  웹크롤링과 군집화를 통한 맞춤 관광지 추천 시스템 만들기
-  웹크롤링을 통해 수집한 블로그글을 바탕으로 관광지를 군집화
-  각 관광지를 테마별로 분류해 관광지의 특징을 반영한 군집 형성
<br>

**군집화 결과**
1. 연극, 공연 관련 관광지
2. 맛집, 먹거리 관련 관광지
3. 작품, 전시 관련 관광지
4. 아이와 가기 좋은 볼거리 많은 관광지
5. 역사 관련 관광지

:arrow_right: 사용자가 **성별, 연령대, 시간대, 방문날짜**를 입력하면, 이에 맞춰 적절한 관광지들을 군집별로 추천  
<br>

**활용 방안**  

- 성심당 대기줄 사이에 추천시스템과 연결한 QR코드 포스터 부착
- 관광객들이 성심당 줄을 기다리면서 대전 중구 관광지에 흥미를 가질 수 있도록 함  
<br>

**기대 효과**

- 성심당에 방문한 관광객들을 주변의 다른 관광지도 같이 방문하게 하면서 중구 전역에 걸친 관광지의 활성화 효과 기대
- 개인화된 맞춤 추천을 통해 사용자의 방문 만족도 극대화
- 새로운 명소들을 경험하게 해 대전에 대한 긍정적인 인식을 높이고, 대전의 재방문율을 높일 수 있음
- 대전 중구가 다양한 테마의 매력을 지닌 지역으로 알려지면, 빵여행 뿐만 아니라 새롭고 다양한 이유로 여행객들이 대전을 여행지로 고려하고 선택할 가능성이 높아질 것  
<br>

## 5. 참고자료

**사용 데이터셋**  

- 제공된 유동인구 데이터셋
- 행정안전부 제공 행정동코드 (https://www.mois.go.kr/frt/a01/frtMain.do)
- 네이버 블로그 전문 크롤링 사용
- 한국관광 데이터랩 대전 관광지 데이터셋 (대전_관광지.csv)  
  (출처: https://datalab.visitkorea.or.kr/datalab/portal)  
<br>

**라이브러리**

<img src="https://github.com/user-attachments/assets/774e618f-8eb2-408a-8287-5c988e24ad35" width="500"/> 
<br>

## 6. 모델 실행 환경

- **프로세서**: Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz   1.80 GHz
- **RAM**: 24.0GB
- **시스템**: 64비트 운영 체제, x64 기반 프로세서
